#pragma once

#include "compound_config/compound_config.hpp"
#include "arch/arch_spec.hpp"
#include "arch/network_spec.hpp"
#include "problem/dimsize.hpp"
#include "analysis/data_movement.hpp"

#include "mapping/mapping_utils.hpp"



namespace arch { 
  class ComputeBase {
    public:
      virtual ~ComputeBase() {};
      virtual uint64_t getComputeLatency(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type) = 0;
      virtual uint64_t getIdealSpecLatency(problem::DimSizeExpression tilesize) = 0;
      virtual float getComputeEnergy(problem::DimSizeExpression tilesize, mapping::computation_attributes comp_attr, mapping::operation_t op_type)=0;

      virtual void setSpec(ComputeSpec spec)=0;
      // set ynode for custom specs
      virtual void setAttributes(YAML::Node& YNode)=0;
  };

  class NetworkBase {
    public:
      virtual ~NetworkBase() {};
      virtual void setSpec(NetworkSpec& spec)=0;
      // virtual uint64_t GetLatency(analysis::DataMovementInfo& data_movement, bool reduction) = 0;
      virtual uint64_t GetLatency(analysis::DataMovementInfo& data_movement, bool reduction, float precision) = 0;

      virtual uint64_t GetEnergy(analysis::DataMovementInfo& data_movement, bool reduction, float precision) = 0;
      
      // virtual uint64_t GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor) = 0;

      virtual uint64_t GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type) = 0;
      
      virtual float GetLinkEnergy(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type) = 0;


      virtual uint64_t GetLatencyForFlit(uint32_t flit_cnt) = 0;
      virtual bool supportsBcast() = 0;
      virtual bool supportsReduction() = 0;
      virtual uint64_t getPortBW() = 0; 
      virtual uint64_t getNumPorts() = 0; // returns a links port count to the network
  };

}
