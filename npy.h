#ifndef NPY_H
#define NPY_H

#include "reader.h"
#include "writer.h"
#include "typehelpers.h"

#include <string>

namespace elegant {
namespace npy {

Reader load(std::string filename, Reader::Conversion conversionMode = Reader::Conversion::AllowLossy) {
    return Reader(filename, conversionMode);
}

template<typename T>
bool save(std::string filename, const T &object) {
    const std::type_info& typeInfo = typeid(typename TypeHelper<T, void>::ElementType);
    size_t elementSize = sizeof(typename TypeHelper<T, void>::ElementType);
    Writer writer(filename, elementSize, typeInfo);
    TypeHelper<T, void> typeHelper;
    typeHelper.toFile(object, writer);

    return true;
}

}
}

#endif // NPY_H

