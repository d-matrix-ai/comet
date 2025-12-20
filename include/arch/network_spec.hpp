#pragma once

#include <variant>

#include "arch/attribute.hpp"
#include "compound_config/compound_config.hpp"


namespace arch {
  struct NetworkSpec { 
    NetworkSpec()=default;
    NetworkSpec(config::CompoundConfigNode network_config);
    void print(std::ostream& out) const;
    Attribute<std::uint32_t> egress_nodes;
    Attribute<uint32_t> ingress_nodes;
    Attribute<uint32_t> hop_latency;
    Attribute<uint32_t> queuing_delay;
    Attribute<float> efficiency;
    Attribute<uint32_t> port_bw; // is bw of cycle
    Attribute<bool> mesh;
    Attribute<bool> bcast_support;
    Attribute<bool> reduction_support;
    Attribute<uint32_t> reduction_latency;
    std::vector<uint32_t> mesh_dimensions;
    std::vector<uint32_t> hierarchy_levels; //needed for hxbar but maybe can just use mesh dimensions
    Attribute<uint32_t> channel_width;
  };

}
