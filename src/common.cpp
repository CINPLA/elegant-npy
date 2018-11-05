#include "common.h"

#include <numeric>

namespace elegant {
namespace npy {

size_t elementCount(const std::vector<size_t> &shape) {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
}

}
}
