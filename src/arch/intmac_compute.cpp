#include "arch/intmac_compute.hpp"


namespace arch {
  REGISTER_CLASS(intmacCompute, ComputeBase);  
  uint64_t intmacCompute::getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type) {
    uint64_t op_latency=spec_.startupLatency_.Get();
    //TODO check if this will work for convolution 
    problem::DimSizeExpression op_expression = {8,8,8};

    return tilesize.resolve()*spec_.computeLatency_.Get(); //(1/256.0);

    // if (tilesize.size() == 3) { // should this be based on computation_attribute type
    //   if (comp_attr.specd) { 
    //     if (comp_attr.rmw) { // TODO should this be base on tensor stationary type 
    //       // K is only reduction dimension
    //       auto reduction_ops = comp_attr.reduction_factors[comp_attr.reduction_dimensions.front()] / kGEMMReductionFactor;
    //       if (reduction_ops == 16) { 
    //         op_latency += 16; // is hop latency 
    //       } else if (reduction_ops == 8) {
    //         op_latency += 12; // is hop latency 
    //       } else if (reduction_ops == 4) { 
    //         op_latency += 8; // is hop latency 
    //       } else if (reduction_ops == 2) { 
    //         op_latency += 6; // is hop latency 
    //       }
    //     }
    //   } 
    //   // is based on output reuse and weight reuse each TE in the MC gets 1, 32, 64
    //   op_latency += getIdealSpecLatency(tilesize);
    //   return op_latency;
    // }

    // return spec_.computeLatency_.Get(); // FIXME:: this needs work 
  }

  uint64_t intmacCompute::getIdealSpecLatency(problem::DimSizeExpression tilesize) { 
    if (tilesize.size() == 3) { // should this be based on computation_attribute type
      problem::DimSizeExpression op_expression = {8,8,8}; //THROUGHPUT PER COMPUTE LATENCY

      auto num_ops = tilesize.resolve() / op_expression.resolve();
      return num_ops * spec_.computeLatency_.Get();
    }
  }

  void intmacCompute::setAttributes(YAML::Node& YNode) {
    node = YAML::Clone(YNode);
  }

  float intmacCompute::getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type){
    return 0;
  }
  

}
