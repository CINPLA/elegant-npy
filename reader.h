#ifndef READER_H
#define READER_H

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

#include "typehelper.h"

namespace elegant {
namespace npy {

class Reader
{
public:
    enum class Conversion {
        AllowLossy,
        RequireSame
    };

    Reader(std::string filename, Conversion conversionMode = Conversion::AllowLossy);

    template<typename T>
    operator T();

    template<typename T>
    T value();

    template<typename T, typename U>
    T valueFromTypeHelper();

    friend std::ostream& operator<<(std::ostream &out, const Reader &array);

    bool isFortranOrder() const;

    bool read(char* buffer, size_t byteCount) {
        m_file.read(buffer, byteCount);
        return true; // TODO check if read ok
    }

private:
    std::string m_fullNumpyType;
    std::string m_numpyType;
    std::vector<size_t> m_shape;
    std::vector<char> m_data;
    size_t m_byteCount = 0;
    std::ifstream m_file;
    bool m_isFortranOrder = false;
    Conversion m_conversionMode = Conversion::AllowLossy;
};

std::ostream &operator<<(std::ostream& out, const Reader& array);

template<typename T, typename U>
T Reader::valueFromTypeHelper()
{
    TypeHelper<T, U> typeHelper;
    if(!typeHelper.isLossyConvertible()) {
        std::stringstream error;
        error << "Cannot convert from numpy type '" << m_numpyType << "'. "
              << "The current conversion policy would allow it, but there is no known conversion available.";
        throw std::runtime_error(error.str());
    } else if(m_conversionMode == Conversion::RequireSame && !typeHelper.isSame()) {
        std::stringstream error;
        error << "Cannot convert from numpy type '" << m_numpyType << "'. "
              << "The current conversion policy requires equal types.";
        throw std::runtime_error(error.str());
    }
    return typeHelper.fromFile(m_shape, *this);
}

template<typename T>
T Reader::value()
{
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
        std::stringstream error;
        error << "Unknown npy type: " << m_numpyType << std::endl;
        throw std::runtime_error(error.str());
    }
}

template<typename T>
Reader::operator T()
{
    return value<T>();
}

}
}

#endif // READER_H
