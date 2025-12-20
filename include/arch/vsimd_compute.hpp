#pragma once

#include "arch/arch_spec.hpp"
#include "problem/dimsize.hpp"
#include "arch/compute_base.hpp"
#include "util/factory_registry.hpp"

namespace arch {
  
  DEF_CLASS(VSIMDCompute, ComputeBase)
  {
    public:
      VSIMDCompute()=default;
      virtual ~VSIMDCompute() {};
      virtual void setSpec(ComputeSpec spec) {spec_ = spec;}
      virtual void setAttributes(YAML::Node& YNode);
      virtual uint64_t getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type);
      virtual uint64_t getIdealSpecLatency(problem::DimSizeExpression tilesize);
      virtual float getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type);

      //add functions for different SIMD operations like softmax, rowmax, rowsum
    private:
      static constexpr uint32_t kGEMMReductionFactor = 256;
      ComputeSpec spec_;
      YAML::Node node;
  };


}
