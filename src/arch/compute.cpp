#include "arch/compute.hpp"

namespace arch {
  REGISTER_CLASS(MACCompute, ComputeBase);  
  uint64_t MACCompute::getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type) {
    return spec_.computeLatency_.Get(); // FIXME:: this needs work 
  }
  uint64_t MACCompute::getIdealSpecLatency(problem::DimSizeExpression tilesize) {
    return spec_.computeLatency_.Get();
  }
  
  float MACCompute::getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type){
    return 0;
  }

  void MACCompute::setAttributes(YAML::Node& YNode) {
    node = YAML::Clone(YNode);
  }

}
