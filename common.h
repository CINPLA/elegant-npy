#ifndef COMMON_H
#define COMMON_H

#include <string>
#include <vector>
#include <algorithm>

namespace elegant {
namespace npy {

const std::string magicPrefix = "\x93NUMPY";
size_t elementCount(const std::vector<size_t> &shape);

}
}

#endif // COMMON

