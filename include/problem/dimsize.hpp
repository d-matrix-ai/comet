#pragma once

#include <map>
#include <vector>
#include <string_view>

#include "util/comet_assert.h"
namespace problem{
  //FIXME:: this may need to become a vector of tuples for ordering purposes and to have residual, or become a vector of coordinates  
  class DimSizeExpression { 
    private:
      std::vector<uint32_t> vec;
    public:
      template <typename ...Tn> DimSizeExpression(Tn ...args) : vec(args...) {} //template <parameter-lis> class-declaration
      DimSizeExpression(std::initializer_list<uint32_t> list) : vec(list) {}
      const std::vector<uint32_t>& get() {return vec;}
      std::vector<uint32_t> get() const {return vec;}
      uint32_t size() const {return vec.size();}
      uint32_t& operator[](int idx) { return vec[idx];}
      uint32_t operator[](int idx) const {return vec[idx];}
      uint64_t resolve() const;
      DimSizeExpression& operator+=(const DimSizeExpression& rhs);
      DimSizeExpression& operator/=(const DimSizeExpression& rhs);
      std::vector<uint32_t>::iterator begin() {return vec.begin();}
      std::vector<uint32_t>::iterator end() {return vec.end();}
      std::vector<uint32_t>::const_iterator begin() const {return vec.cbegin();}
      std::vector<uint32_t>::const_iterator end() const {return vec.cend();}
      std::string_view print() const;
      std::vector<uint32_t>::iterator erase(std::vector<uint32_t>::iterator val) {return vec.erase(val);}
      
  }; 
  DimSizeExpression operator/ (const DimSizeExpression& dim1, const DimSizeExpression& dim2);
  std::ostream& operator<<(std::ostream& out, const DimSizeExpression& dim_expression);

}
