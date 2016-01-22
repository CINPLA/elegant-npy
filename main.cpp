#include <iostream>
#include <armadillo>
#include <boost/regex.hpp>
#include <regex>
#include <functional>

using namespace std;

namespace elegant {
namespace npy {

using callback_type = std::function<void(char*, size_t)>;

size_t elementCount(const vector<size_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

template<typename T, typename U>
struct BaseTypeHelper
{
    using object_type = T;
    std::string numpyType();
    object_type fileToObject(const vector<size_t> &shape, callback_type callback) {
        (void)(shape);
        (void)(callback);
        throw std::runtime_error("Conversion to this type is not supported");
    }
};

template<typename T, typename U>
struct TypeHelper : public BaseTypeHelper<T, U>
{
};

#define ELEGANT_NPY_REGISTER_SIMPLE_TYPE(type_name, numpy_type)\
    template<typename U>\
    struct TypeHelper<type_name, U> : public BaseTypeHelper<type_name, U>\
{\
    std::string numpyType() {\
    return numpy_type;\
}\
    size_t bufferSize(const vector<size_t> &shape) {\
    (void)shape;\
    return sizeof(type_name);\
}\
}

ELEGANT_NPY_REGISTER_SIMPLE_TYPE(float, "f4");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(double, "f8");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(int32_t, "i4");
ELEGANT_NPY_REGISTER_SIMPLE_TYPE(int64_t, "i8");

template<typename eT> struct TypeHelper<std::vector<eT>, bool> : public BaseTypeHelper<std::vector<eT>, bool> {};
template<typename eT, typename npyT>
struct TypeHelper<std::vector<eT>, npyT> : public BaseTypeHelper<std::vector<eT>, npyT>
{
    using object_type = std::vector<eT>;
    object_type fileToObject(const vector<size_t> &shape, callback_type callback) {
        if(std::is_same<eT, npyT>::value) {
            object_type object(elementCount(shape));
            callback(reinterpret_cast<char*>(&object[0]), elementCount(shape) * sizeof(eT));
            return object;
        } else {
            std::vector<npyT> sourceObject = TypeHelper<std::vector<npyT>, npyT>().fileToObject(shape, callback);
            object_type targetObject(elementCount(shape));
            copy(sourceObject.begin(), sourceObject.end(), targetObject.begin());
            return targetObject;
        }
    }

    object_type m_temporary;
};

template<typename eT> struct TypeHelper<arma::Mat<eT>, bool> : public BaseTypeHelper<arma::Mat<eT>, bool> {};
template<typename eT> struct TypeHelper<arma::Mat<eT>, int8_t> : public BaseTypeHelper<arma::Mat<eT>, int8_t> {};

template<typename eT, typename npyT>
struct TypeHelper<arma::Mat<eT>, npyT> : public BaseTypeHelper<arma::Mat<eT>, npyT>
{
    using object_type = arma::Mat<eT>;
    object_type fileToObject(const vector<size_t> &shape, callback_type callback) {
        if(std::is_same<eT, npyT>::value) {
            object_type object(shape[0], shape[1]);
            object = object.t();
            callback(reinterpret_cast<char*>(&object[0]), sizeof(eT) * elementCount(shape));
            object = object.t();
            return object;
        } else {
            return arma::conv_to<arma::Mat<eT>>::from(TypeHelper<arma::Mat<npyT>, npyT>().fileToObject(shape, callback));
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
                if(regex_search(value, descrMatch, regex("'(<|>|\\|)(.*?)'"))) {
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

    template<typename T, typename U>
    T valueFromTypeHelper();

private:
    string m_numpyType;
    vector<size_t> m_shape;
    vector<char> m_data;
    size_t m_byteCount = 0;
    ifstream file;
    bool m_fortranOrder = false;
    Conversion m_conversionMode = Conversion::Relaxed;

};

template<typename T, typename U>
T Array::valueFromTypeHelper()
{
    TypeHelper<T, U> typeHelper;
    return typeHelper.fileToObject(m_shape, [&](char* buffer, size_t byteCount) {
        file.read(buffer, byteCount);
    });
}

template<typename T>
T Array::value()
{
    //    if(m_numpyType != typeHelper.numpyType()) {
    //        if(m_conversionMode == Conversion::Strict) {
    //            stringstream error;
    //            error << "Cannot convert from numpy type '" << m_numpyType << "' "
    //                  << "because your type expects '" << typeHelper.numpyType() << "'. ";
    //            if(typeHelper.canConvert(m_numpyType)) {
    //                error << "A conversion can be enabled automatically by changing the policy to ";
    //                error << "Array::Conversion::Relaxed. ";
    //            } else {
    //                error << "There is no known conversion between the two. ";
    //            }
    //            error << "The current conversion policy is Array::Conversion::Strict.";
    //            throw std::runtime_error(error.str());
    //        }
    //        if(!typeHelper.canConvert(m_numpyType)) {
    //            stringstream error;
    //            error << "Cannot convert from numpy type '" << m_numpyType << "' "
    //                  << "because your type expects '" << typeHelper.numpyType() << "' "
    //                  << "and no known conversion exists. "
    //                  << "The current conversion policy is Array::Conversion::Relaxed.";
    //            throw std::runtime_error(error.str());
    //        } else {
    //            TypeHelper<T, float> typeHelper2;
    //            cout << typeHelper2.numpyType() << endl;
    //        }
    //    }

    if(false) {}
    else if(m_numpyType == "b1") { return valueFromTypeHelper<T, bool>(); }
    else if(m_numpyType == "f4") { return valueFromTypeHelper<T, float>(); }
    else if(m_numpyType == "f8") { return valueFromTypeHelper<T, double>(); }
    else if(m_numpyType == "i1") { return valueFromTypeHelper<T, int8_t>(); }
    else if(m_numpyType == "i2") { return valueFromTypeHelper<T, int16_t>(); }
    else if(m_numpyType == "i4") { return valueFromTypeHelper<T, int32_t>(); }
    else if(m_numpyType == "i8") { return valueFromTypeHelper<T, int64_t>(); }
    else if(m_numpyType == "u1") { return valueFromTypeHelper<T, uint8_t>(); }
    else if(m_numpyType == "u2") { return valueFromTypeHelper<T, uint16_t>(); }
    else if(m_numpyType == "u4") { return valueFromTypeHelper<T, uint32_t>(); }
    else if(m_numpyType == "u8") { return valueFromTypeHelper<T, uint64_t>(); }
    else {
        stringstream error;
        error << "Unknown npy type: " << m_numpyType << endl;
        throw std::runtime_error(error.str());
    }
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
    vector<float> ma = npy::load("/home/svenni/Dropbox/tmp/test3.npy");
    for(float a : ma) {
        cout << a << " ";
    }
    cout << endl;
    //    cout << ma << endl;
    return 0;
}

