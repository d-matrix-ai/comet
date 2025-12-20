#include <algorithm>
#include <numeric>
#include <functional>
#include "problem/dimsize.hpp"
#include "util/comet_assert.h"
namespace problem {
  DimSizeExpression operator/ (const DimSizeExpression& dim1, const DimSizeExpression& dim2) { 
    DimSizeExpression result(dim1.size(), 0);
    for (auto dim = 0; dim < dim1.size(); dim++) { 
      result[dim] = dim1[dim] / dim2[dim];
    }
    return result;
  }
  std::ostream& operator<<(std::ostream& out, const DimSizeExpression& dim_expression) { 
    out << dim_expression.print();
    return out;
  }

  std::string_view DimSizeExpression::print() const {
    std::string retval="";
    for (const auto& dim: get()) { 
      retval += std::string("[0, ") + std::to_string(dim) + "), ";
    }
    return retval;
  }

  
  uint64_t DimSizeExpression::resolve() const { 
    uint64_t accumulate = 1;
    for (uint64_t item: vec) {
      accumulate *= item;
    }
    return accumulate;
  }

  DimSizeExpression& DimSizeExpression::operator/=(const DimSizeExpression& rhs) { 
    COMET_ASSERT(rhs.size() == size(), "DimSizeExpressions are of unequal size");
    for (auto dim=0; dim<size(); dim++) {vec[dim] /= rhs[dim];}
    return *this;
  }
  DimSizeExpression& DimSizeExpression::operator+=(const DimSizeExpression& rhs) { 
    COMET_ASSERT(rhs.size() == size() || rhs.size() ==0, "DimSizeExpressions are of unequal size");
    if (rhs.size() == 0) return *this; // ignore the rhs if its a zero sized vector
    for (auto dim=0; dim<size(); dim++) {vec[dim] += rhs[dim];}
    return *this;
  }
}
