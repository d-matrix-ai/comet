#pragma once

#include "arch/network_spec.hpp"
#include "util/factory_registry.hpp"
#include "arch/compute_base.hpp"
#include "analysis/data_movement.hpp"

namespace arch {
  // This needs to be a registry
  DEF_CLASS(MeshNetwork, NetworkBase)
  {
    public:
      MeshNetwork()=default;
      virtual void setSpec(NetworkSpec& sp) {spec = sp;}
      // virtual uint64_t GetLatency(analysis::DataMovementInfo& data_movement, bool reduction); // this depends on network type
      virtual uint64_t GetLatency(analysis::DataMovementInfo& data_movement, bool reduction, float precision); // this depends on network type, precision in bytes
      virtual uint64_t GetEnergy(analysis::DataMovementInfo& mov_info, bool reduction, float precision);

      virtual uint64_t GetLatencyForFlit(uint32_t flit_cnt);
      // virtual uint64_t GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor);
      virtual uint64_t GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type);

      virtual float GetLinkEnergy(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type);      

      virtual bool supportsBcast() {return spec.bcast_support.Get();}
      virtual bool supportsReduction() {return spec.reduction_support.Get();}
      virtual uint64_t getPortBW();
      virtual uint64_t getNumPorts() { return 2;} // is flat mesh network
      const NetworkSpec& getSpec() {return spec;}
    private:
      NetworkSpec spec;
  };

}
