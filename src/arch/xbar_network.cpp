#include "arch/xbar_network.hpp"
#include <cmath>

namespace arch {
  REGISTER_CLASS(HXbarNetwork, NetworkBase);
  // uint64_t HXbarNetwork::GetLatency(analysis::DataMovementInfo& mov_info, bool reduction)  {
  uint64_t HXbarNetwork::GetLatency(analysis::DataMovementInfo& mov_info, bool reduction, float precision)  {

    auto multi_factor = mov_info.num_unique_tiles;
    
    if (multi_factor == 0) return 0;
    auto hierarchy = spec.hierarchy_levels;
    if (hierarchy.size() == 0) {COMET_ERROR("Hierarchial xbar doesnt have hierarchies");}
   
    auto max_path = hierarchy.size();
    auto def_latency = max_path * spec.hop_latency.Get();
    // to handle arbitration conflicts at xbar junctions
    if (!reduction) {
      //needed for fill efficiency if efficiency is 1 we can skip as if perfectly multicastable and BW portion is analyzed at the port level
      float derate_latency = 0;
      auto num_data = multi_factor;
      for (auto level: hierarchy) {
        // arbitration factor is number of children/ num of unique tiles
        derate_latency += (1/spec.efficiency.Get() - 1) * (static_cast<float>(level)) * (1/static_cast<float>(num_data));
        // num_unique tiles at next level is ceil (num_data / level)
        num_data = (num_data + level -1 )/ level;
      }

      auto total_latency = def_latency + static_cast<uint64_t>(derate_latency) * spec.hop_latency.Get(); 
      return total_latency;
    } else if (spec.reduction_support.Get()) {
      //reduction happens bottom up
      // num_tiles also needs to be prefix divided
      std::vector<uint32_t> num_tiles_at_each_level(hierarchy.size(), multi_factor);
      for (auto idx = 0; idx < hierarchy.size(); idx++) {
        if (idx = 0) continue;
        num_tiles_at_each_level[idx] = (num_tiles_at_each_level[idx-1] + hierarchy[idx-1] + 1) / hierarchy[idx-1];
      }
      auto reduction_latency = 0;
      for (auto idx = hierarchy.size()-1; idx >=0; idx --){
        auto reduction_factor = hierarchy[idx] / num_tiles_at_each_level[idx];
        reduction_latency += reduction_factor * spec.reduction_latency.Get();
      }
      return def_latency + reduction_latency;
    } else return def_latency;
  }

  uint64_t HXbarNetwork::GetEnergy(analysis::DataMovementInfo& mov_info, bool reduction, float precision){
    double energy=0;
    return energy;
  }

  float HXbarNetwork::GetLinkEnergy(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type){
    double energy=0;
    return energy;
  }


  uint64_t HXbarNetwork::GetLatencyForFlit(uint32_t flit_cnt) { 
    return flit_cnt * spec.hop_latency.Get() * spec.hierarchy_levels.size(); // always have to follow the max path
  }
  
  // uint64_t HXbarNetwork::GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor) { 
  uint64_t HXbarNetwork::GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type) { 
    auto multi_factor = data_movement.num_unique_tiles;
    auto size_of_data = data_movement.tile_access_size.resolve();
    // ensure that the spatial_factor is a multiple of the base xbar
    COMET_ASSERT(spatial_factor % spec.hierarchy_levels.back()==0, "In HierarchXbar we require spatial factor to be divisable by the leaf xbar node count");

    uint8_t path_cnt = 0;
    uint8_t node_cnt = 1;
    uint8_t max_nodes = spec.hierarchy_levels.size();
    for (auto itr = spec.hierarchy_levels.rbegin(); itr != spec.hierarchy_levels.rend(); itr++) {
      if (spatial_factor > node_cnt) path_cnt++;
      else break;
      node_cnt *= *itr;
    }
    
    uint64_t no_congestion_latency = 2 * path_cnt * spec.hop_latency.Get();
    uint64_t extra_network_latency = 0;
    bool allreduce = stype == mapping::stype_t::ALLREDUCE;
    bool allgather = stype == mapping::stype_t::ALLGATHER;
    bool gather = stype == mapping::stype_t::GATHER;
    bool reduction = stype == mapping::stype_t::REDUCTION;
    if (allreduce || allgather || gather) {
      // for broadcast part of congestion: extra arbitration cycles is number of nodes paricipating
      uint64_t conflicts=0;
      auto num_nodes = spatial_factor;
      auto num_packet = 1;
      for (auto idx =0; idx < path_cnt; idx++) {
        auto cur_leaf = spec.hierarchy_levels[max_nodes - idx - 1];
        uint8_t conflicting_nodes = 0;
        if (num_nodes > cur_leaf) conflicting_nodes = cur_leaf;
        else conflicting_nodes = num_nodes;
        conflicts += (conflicting_nodes - 1)*(num_packet);
        num_packet *= cur_leaf; 
        num_nodes /= cur_leaf;
      }
      extra_network_latency += conflicts * spec.hop_latency.Get();
    }
    if (spec.reduction_support.Get()) {
      if (reduction || allreduce) {
        auto data_reduced_at_each_hop = (size_of_data * multi_factor + spec.port_bw.Get() - 1) / spec.port_bw.Get();
        extra_network_latency += path_cnt * data_reduced_at_each_hop * spec.reduction_latency.Get();
      }
    }
    return no_congestion_latency + extra_network_latency;
  }

  uint64_t HXbarNetwork::getPortBW() {
    return spec.port_bw.Get();
  }

  uint64_t HXbarNetwork::getNumPorts() { 
    return 1; // xbar device has 1 port to router // TODO:: should we add a num_ports
  }
}
