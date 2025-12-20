#include "arch/vsimd_compute.hpp"


namespace arch {
  REGISTER_CLASS(VSIMDCompute, ComputeBase); 
  void VSIMDCompute::setAttributes(YAML::Node& YNode) {
    node = YAML::Clone(YNode);
  }


  uint64_t VSIMDCompute::getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type){
    problem::DimSizeExpression op_expression = {1,2}; 

    // compute latencies from Synopsis DesignWare IP and latency fromula fro -> https://dl.acm.org/doi/pdf/10.1145/3696665#page=2.40
    uint64_t comp_latency;
    if(op_type == mapping::operation_t::DIV){
      comp_latency = 1;
    } else if(op_type == mapping::operation_t::EXP){
      comp_latency = 3;
    } else if(op_type == mapping::operation_t::ADD){
      comp_latency = 1;
    } else if(op_type == mapping::operation_t::MULT){
      comp_latency = 1;
    } else if(op_type == mapping::operation_t::MAX){
      comp_latency=1;
    } else if(op_type == mapping::operation_t::SQRT){
      comp_latency=1;
    } else if(op_type == mapping::operation_t::SOFTMAX){
      comp_latency=1;
    }  else if(op_type == mapping::operation_t::ROWSUM){
      comp_latency=std::ceil(std::log2(spec_.computeWidth_.Get())); // it takes log2(simd_width) cycles to reduce K numbers
    }  else if(op_type == mapping::operation_t::ROWMAX){
      comp_latency=std::ceil(std::log2(spec_.computeWidth_.Get())); // it takes log2(simd_width) cycles to to take max of K numbers
    } else {
      COMET_ASSERT(false, "Operation not supported in VSIMDCompute");
    }

    // uint64_t comp_latency = tilesize.resolve()/op_expression.resolve(); // 5 elements/cycle

    // 6 pipeline stages (inst fetch, decode, address gen, data read, execution, WB), therefore adding the time to fill the pipeline
    // https://dl.acm.org/doi/pdf/10.1145/3696665#page=2.40 --> taken from the code of this paper page16. pipeline step overhead
    // auto pipeline_overhead_latency = (6-1) + (spec_.computeWidth_.Get()-1);
   
    //remove the redundant dimensions from the tilesize
    float T=1;
    for(auto& dim: comp_attr.reduction_dimensions){
      T*=comp_attr.reduction_factors[dim];
    }

    // auto T = comp_attr.reduction_factors[comp_attr.reduction_dimensions.back()];

    auto pipeline_overhead_latency = 6-1; //6 stage pipeline but removing execution latency
    auto instruction_overhead = ceil(tilesize.resolve()/((float)spec_.computeWidth_.Get()*T)); //adding one cycle per additional instruction

    // return comp_latency + pipeline_overhead_latency + instruction_overhead;

    return comp_latency*instruction_overhead; //scaling to consider cases when instruction more than SIMD width is issued
  }

  float VSIMDCompute::getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type){
    float compute_energy=0;

    if(op_type == mapping::operation_t::DIV){
      compute_energy = 0.8;
    } else if(op_type == mapping::operation_t::EXP){
      compute_energy = 3.86;
    } else if(op_type == mapping::operation_t::ADD){
      compute_energy = 0.11;
    } else if(op_type == mapping::operation_t::MULT){
      compute_energy = 0.64;
    } else if(op_type == mapping::operation_t::MAX){
      compute_energy=0.0025;
    } else if(op_type == mapping::operation_t::SQRT){
      compute_energy=2.84;
    } else if(op_type == mapping::operation_t::SOFTMAX){
      compute_energy=1;
    } else {
      COMET_ASSERT(false, "Operation not supported in VSIMDCompute");
    }
    return compute_energy;
  }

  uint64_t VSIMDCompute::getIdealSpecLatency(problem::DimSizeExpression tilesize){
    uint64_t retval=0;

    return retval;
  }


  
}
