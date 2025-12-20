#include "arch/systolic_array.hpp"


namespace arch {
  REGISTER_CLASS(SystolicArrayCompute, ComputeBase);  
  uint64_t SystolicArrayCompute::getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type) {
    // uint64_t op_latency=spec_.startupLatency_.Get();
    //TODO check if this will work for convolution 

    uint64_t op_latency;

    auto rows = spec_.array_rows_.Get();
    auto cols = spec_.array_cols_.Get();
    auto num_sa_rows = spec_.num_sa_rows_.Get();
    auto num_sa_cols = spec_.num_sa_cols_.Get(); 
    auto T = comp_attr.reduction_factors[comp_attr.reduction_dimensions.back()];

    op_latency = 2*rows + cols + T - 2; // ScaleSim

    auto it=std::find(tilesize.begin(), tilesize.end(), T);

    if (it!=tilesize.end()){
      tilesize.erase(it); // remove only the first occurence of T
    } else {
      COMET_ASSERT(false, "T not found in tilesize in systolic array function");
    }

    uint64_t scale=1;

    size_t cnt=0;
    for(auto& val:tilesize){
      // scale*=std::ceil((double)val/(rows*num_sa_rows)); //FIXME::snegi assuming rows==cols //calculate number of folds in rows and column dimension
      if(cnt==0){
        scale*=std::ceil((double)val/(rows*num_sa_rows));
      } else if(cnt==1){
        scale*=std::ceil(double(val)/(cols*num_sa_cols));
      }
      cnt++;
    }

    return scale*op_latency;
    // return 20;
  }

  float SystolicArrayCompute::getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type){
    float compute_energy;

    //number of input cycles
    // auto T = comp_attr.reduction_factors[comp_attr.reduction_dimensions.back()];
    // auto total_compute_cyclces = getComputeLatency(tilesize, comp_attr, op_type);
    // auto energy_control = spec_.power_control_.Get()*spec_.num_sa_rows_.Get()*spec_.num_sa_cols_.Get()*total_compute_cyclces; 

    // auto energy_array = spec_.power_array_.Get()*spec_.num_sa_rows_.Get()*spec_.num_sa_cols_.Get()*T;

    // return energy_control + energy_array;

    auto comp_energy = tilesize.resolve()*0.2296; //pJ/MAC energy from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10714410

    return comp_energy;
  }

  uint64_t SystolicArrayCompute::getIdealSpecLatency(problem::DimSizeExpression tilesize) { 
    if (tilesize.size() == 3) { // should this be based on computation_attribute type
      problem::DimSizeExpression op_expression = {8,8,8}; //THROUGHPUT PER COMPUTE LATENCY

      auto num_ops = tilesize.resolve() / op_expression.resolve();
      return num_ops * spec_.computeLatency_.Get();
    }
  }

  void SystolicArrayCompute::setAttributes(YAML::Node& YNode) {
    node = YAML::Clone(YNode);
  }


}
