#include <iostream>
#include <armadillo>
#include <boost/regex.hpp>
#include <regex>
#include <functional>

using namespace std;

namespace elegant {
namespace npy {

template<typename T>
struct BaseTypeHelper
{
    using object_type = T;
    std::string numpyType();
    size_t bufferSize(const vector<size_t> &shape) {
        (void)shape;
        return sizeof(object_type);
    }
    T objectFromShape(const vector<size_t> &extents);
    //    char* fileToObjectBuffer(T &object);
    object_type fileToObject(const vector<size_t> &shape, const std::string& targetType,
                              std::function<void(char*)> callback);
    virtual bool canConvert(const std::string& targetType) {
        (void)targetType;
        return false;
    }
};

template<typename T>
struct TypeHelper : public BaseTypeHelper<T>
{
};

#define ELEGANT_NPY_REGISTER_SIMPLE_TYPE(type_name, numpy_type)\
    template<>\
    struct TypeHelper<type_name> : public BaseTypeHelper<type_name>\
{\
    std::string numpyType() {\
    return numpy_type;\
}\
}

ELEGANT_NPY_REGISTER_SIMPLE_TYPE(float, "f4");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(double, "f8");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(int32_t, "i4");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(int64_t, "i8");

template<typename eT>
struct TypeHelper<arma::Mat<eT>> : public BaseTypeHelper<arma::Mat<eT>>
{
    using object_type = arma::Mat<eT>;
    std::string numpyType() {
        return TypeHelper<eT>().numpyType();
    }
    object_type fileToObject(const vector<size_t> &shape, const std::string &targetType,
                             std::function<void(char*)> callback) {
        if(targetType != numpyType()) {
            if(targetType == "f4") {
                return arma::conv_to<object_type>::from(TypeHelper<arma::Mat<float>>().fileToObject(shape, targetType, callback));
            }
            if(targetType == "f8") {
                return arma::conv_to<object_type>::from(TypeHelper<arma::Mat<double>>().fileToObject(shape, targetType, callback));
            }
            if(targetType == "i4") {
                return arma::conv_to<object_type>::from(TypeHelper<arma::Mat<int32_t>>().fileToObject(shape, targetType, callback));
            }
            if(targetType == "i8") {
                return arma::conv_to<object_type>::from(TypeHelper<arma::Mat<int64_t>>().fileToObject(shape, targetType, callback));
            }
        }
        object_type object(shape[0], shape[1]);
        object = object.t();
        callback(reinterpret_cast<char*>(&object[0]));
        object = object.t();
        return object;
    }
    size_t bufferSize(vector<size_t> shape) {
        size_t product = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
        return TypeHelper<eT>().bufferSize(shape) * product;
    }
    bool canConvert(const string& targetType) {
        const string &t = targetType;
        if(t == "f4" || t == "f8" || t == "i4" || t == "i8") {
            return true;
        } else {
            return false;
        }
    }

    object_type m_temporary;
};

class Array
{
public:
    enum class Conversion {
        Relaxed,
        Strict
    };

    Array(string filename, Conversion conversionMode = Conversion::Relaxed)
        : m_conversionMode(conversionMode)
    {
        file.open(filename);

        string magicPrefix = "\x93NUMPY";
        int magicPrefixLength = magicPrefix.size();

        vector<char> magicBuffer(magicPrefixLength);
        file.read(&magicBuffer[0], magicBuffer.size());
        string resultingString(magicBuffer.begin(), magicBuffer.end());
        if(resultingString != magicPrefix) {
            throw std::runtime_error("The magic string is not correct");
        }
        uint8_t majorVersion = 0;
        uint8_t minorVersion = 0;
        file.read(reinterpret_cast<char*>(&majorVersion), 1);
        file.read(reinterpret_cast<char*>(&minorVersion), 1);
        cout << int(majorVersion) << "." << int(minorVersion) << endl;

        size_t headerLength = 0;
        if(majorVersion == 1 && minorVersion == 0) {
            uint16_t headerLength16 = 0;
            file.read(reinterpret_cast<char*>(&headerLength16), 2);
            headerLength = headerLength16;
        } else if(majorVersion == 2 && minorVersion == 0) {
            uint32_t headerLength32 = 0;
            file.read(reinterpret_cast<char*>(&headerLength32), 4);
            headerLength = headerLength32;
        }
        cout << int(headerLength) << endl;

        vector<char> headerBuffer(headerLength);
        file.read(&headerBuffer[0], headerLength);
        string header(headerBuffer.begin(), headerBuffer.end());

        // {'descr': '<i8', 'fortran_order': False, 'shape': (8,), }

        string::const_iterator start = header.begin();
        string::const_iterator end = header.end();

        // match parts on the form "'key': value,"
        smatch keyPairMatch;
        while(regex_search(start, end, keyPairMatch, regex("('(.*?)'\\s*?:\\s*((\\(.*?\\)|.*?)*?),)"))) {
            string key = keyPairMatch[2];
            string value = keyPairMatch[3];
            if(key == "descr") {
                smatch descrMatch;
                if(regex_search(value, descrMatch, regex("'(<|>)(.*?)'"))) {
                    string endian = descrMatch[1];
                    m_numpyType = descrMatch[2];
                    if(endian == ">") {
                        throw runtime_error("Big endian not supported");
                    }
                }
            }
            if(key == "fortran_order") {
                if(value == "False") {
                    m_fortranOrder = false;
                } else {
                    m_fortranOrder = true;
                    throw runtime_error("Fortran order is not supported");
                }
            }
            if(key == "shape") {
                smatch shapeMatch;
                string::const_iterator shapeStart = value.begin();
                string::const_iterator shapeEnd = value.end();
                while(regex_search(shapeStart, shapeEnd, shapeMatch, regex("([0-9]+?)(?:\\s*(?:,|\\)))"))) {
                    string shapeValue = shapeMatch[1];
                    m_shape.push_back(stoi(shapeValue));
                    shapeStart = shapeMatch[0].second;
                }
            }
            start = keyPairMatch[0].second; // move on to the next match
        }
    }

    template<typename T>
    operator T();

    template<typename T>
    T value();

private:
    string m_numpyType;
    vector<size_t> m_shape;
    vector<char> m_data;
    size_t m_byteCount = 0;
    ifstream file;
    bool m_fortranOrder = false;
    Conversion m_conversionMode = Conversion::Relaxed;

};

template<typename T>
T Array::value()
{
    TypeHelper<T> typeHelper;
    if(m_numpyType != typeHelper.numpyType()) {
        if(m_conversionMode == Conversion::Strict) {
            stringstream error;
            error << "Cannot convert from numpy type '" << m_numpyType << "' "
                  << "because your type expects '" << typeHelper.numpyType() << "'. ";
            if(typeHelper.canConvert(m_numpyType)) {
                error << "A conversion can be enabled automatically by changing the policy to ";
                error << "Array::Conversion::Relaxed. ";
            } else {
                error << "There is no known conversion between the two. ";
            }
            error << "The current conversion policy is Array::Conversion::Strict.";
            throw std::runtime_error(error.str());
        }
        if(!typeHelper.canConvert(m_numpyType)) {
            stringstream error;
            error << "Cannot convert from numpy type '" << m_numpyType << "' "
                  << "because your type expects '" << typeHelper.numpyType() << "' "
                  << "and no known conversion exists. "
                  << "The current conversion policy is Array::Conversion::Relaxed.";
            throw std::runtime_error(error.str());
        }
    }

    m_byteCount = typeHelper.bufferSize(m_shape);
    return typeHelper.fileToObject(m_shape, m_numpyType, [&](char* buffer) {
                                file.read(buffer, m_byteCount);
                            });
}

template<typename T>
Array::operator T()
{
    return value<T>();
}

Array load(string filename, Array::Conversion conversionMode = Array::Conversion::Relaxed) {
    return Array(filename, conversionMode);
}

}
}

using namespace std;
using namespace arma;
using namespace elegant;

int main()
{
    mat ma = npy::load("/home/svenni/tmp/test3.npy");
    cout << ma << endl;
    return 0;
}

