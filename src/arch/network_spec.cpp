#include "arch/network_spec.hpp"
#include "util/comet_assert.h"
namespace arch { 
  NetworkSpec::NetworkSpec(config::CompoundConfigNode network_config) { 
    std::string name;
    uint32_t value;
    if (network_config.getValue<uint32_t>("meshX", value)) { 
      mesh_dimensions.emplace_back(value);
      mesh = true;
    }
    if (network_config.getValue<uint32_t>("meshY", value)) { 
      COMET_ASSERT(mesh_dimensions.size() == 1, "Trying to define a meshY without a meshX");
      mesh_dimensions.emplace_back(value);
    } else if (mesh_dimensions.size() == 1) { 
      mesh_dimensions.emplace_back(1);
    }

    if (mesh.Get() == false) {mesh_dimensions = {1, 1};}
    
    if (network_config.getValue<uint32_t>("bw", value)) { 
      port_bw = value;
    } else {
      COMET_ERROR("BW not defined for network");
    }

    if (network_config.getValue<uint32_t>("hop_latency", value)) {
      hop_latency = value;
    } else hop_latency = 1;

    if (network_config.getValue<uint32_t>("efficiency", value)) {
      auto efficiency = static_cast<float>(value) / 100.0;
    } else efficiency = 1.0;

    bool bool_value;
    if (network_config.getValue<bool>("broadcast_support", bool_value)) { 
      bcast_support = bool_value;
    } else bcast_support = false;
    
    if (network_config.getValue<bool>("reduction_support", bool_value)) { 
      reduction_support = bool_value;
    } else reduction_support = false;
    
    if (network_config.getValue<uint32_t>("reduction_latency", value)) { 
      reduction_latency = value;
    } else reduction_latency = 0;
    
    std::vector<uint32_t> hierarchy_vals;
    if (network_config.getValue<std::vector<uint32_t>>("hierarchy_levels", hierarchy_vals)) {
      hierarchy_levels = std::move(hierarchy_vals);
    } else hierarchy_levels = std::vector<uint32_t>(0);


    if (network_config.getValue<uint32_t>("queuing_delay", value)) { 
      queuing_delay = value;
    } else queuing_delay = 0;

    if (network_config.getValue<uint32_t>("channel_width", value)) { 
      channel_width = value;
    } else channel_width = 32;    
    

  }

}
