#include "arch/network.hpp"
#include "arch/orion_power_model.hpp"
#include <cmath>
#include "util/simd_component_cost.hpp"

namespace arch { 
  REGISTER_CLASS(MeshNetwork, NetworkBase); 
  // uint64_t MeshNetwork::GetLatency(analysis::DataMovementInfo& mov_info, bool reduction) {
  // extern double power_summary_router(double channel_width, int input_switch, int output_switch, uint32_t hop, double trc, double tva, double tsa, double tst, double tl, double tenq, double Q, uint32_t mesh_edge);

  uint64_t MeshNetwork::GetLatency(analysis::DataMovementInfo& mov_info, bool reduction, float precision) { 
    // Assuming store and forward is free for network
    // Network always assumes store and forward -> XBAR should not
    // Memory time should take into account if network can support store and forward and use memory BW to send extra tiles
    // Hop overlapping is based on best_avg path * hop_count;
    auto multi_factor = mov_info.num_unique_tiles; //count of tile in projection set
    if (multi_factor == 0) return 0;
    std::vector<uint32_t> mesh_dimensions(2,1);
    auto mesh_xy = static_cast<uint32_t>(std::ceil(std::sqrt(mov_info.tile_count))); // for spatial nodes, tile_count will be equal to number of spatial children in mapping file
    if (spec.mesh.Get()) {
      mesh_dimensions = spec.mesh_dimensions;
    }else {
      mesh_dimensions = {mesh_xy, mesh_xy};
    }

    auto tile_xy = static_cast<uint32_t>(std::ceil(std::sqrt(multi_factor)));
    uint32_t tile_x;
    uint32_t tile_y;

    if (tile_xy >= mesh_dimensions[0]) { tile_x = mesh_dimensions[0];} 
    else if (tile_xy != 0) {
      tile_x = mesh_dimensions[0] / tile_xy; //doubt::snegi why will this ever happen?
    }
    
    if (tile_xy >= mesh_dimensions[1]) { tile_y = mesh_dimensions[1];}
    else if (tile_xy != 0) {
      tile_y = mesh_dimensions[1] / tile_xy;
    }

    auto best_case_hop_count = tile_x + tile_y - 1; //doubt::snegi is it "-1" because we have to consider target_mem->child_mem->hop also. Also why are we taking only the longest path? Would other short paths communicate in parallel?
    auto total_hop_latency = best_case_hop_count * spec.hop_latency.Get();
    auto parent_to_child_latency = total_hop_latency; 
    if (tile_xy == 0) parent_to_child_latency = 0;

    // if network supports reduction then resolve in network reduction
    //doubt::snegi what about broadcast? 
    if (reduction && spec.reduction_support.Get()) {
      auto reduction_latency = 0;
      if (tile_xy != 0) reduction_latency = best_case_hop_count * spec.reduction_latency.Get(); // spanning tree reduction does work for this //doubt::snegi
      parent_to_child_latency += reduction_latency;
    }
    //doubt::snegi what if the network doesn't support reduction, where are we reducing the tensor? we will manually read it multiple times, we have considered that cost of reading but don't we need to consider the network cost as well?
    
    //queuing delay https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10844846&tag=1 (page6)
    // Q2d/W2d*tenq
    int64_t queuing_delay = 0;
    if(mesh_dimensions[0]>1 &&mov_info.tile_count!=1) { //tile_count set to 1 when only 1 spatial unit is use

      auto W2d = spec.channel_width.Get(); 
      auto Q2d = multi_factor==1 ? mov_info.tile_access_size.resolve()*multi_factor: mov_info.tile_access_size.resolve() * (multi_factor-1);// -1 bcz 1 tile of data is already present at the node //doubt::snegi check with Manik //does broadcst and reduction also needs to be considered here? //should I multiply by multi_factor or not? 

      queuing_delay = ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();
    }
    return (parent_to_child_latency + queuing_delay); 
  }

  uint64_t MeshNetwork::GetEnergy(analysis::DataMovementInfo& mov_info, bool reduction, float precision) { 
    // Assuming store and forward is free for network
    // Network always assumes store and forward -> XBAR should not
    // Memory time should take into account if network can support store and forward and use memory BW to send extra tiles
    // Hop overlapping is based on best_avg path * hop_count;
    auto multi_factor = mov_info.num_unique_tiles; //count of tile in projection set
    if (multi_factor == 0) return 0;
    std::vector<uint32_t> mesh_dimensions(2,1);
    auto mesh_xy = static_cast<uint32_t>(std::ceil(std::sqrt(mov_info.tile_count))); // for spatial nodes, tile_count will be equal to number of spatial children in mapping file
    if (spec.mesh.Get()) {
      mesh_dimensions = spec.mesh_dimensions;
    }else {
      mesh_dimensions = {mesh_xy, mesh_xy};
    }

    auto tile_xy = static_cast<uint32_t>(std::ceil(std::sqrt(multi_factor)));
    uint32_t tile_x;
    uint32_t tile_y;

    if (tile_xy >= mesh_dimensions[0]) { tile_x = mesh_dimensions[0];} 
    else if (tile_xy != 0) {
      tile_x = mesh_dimensions[0] / tile_xy; //doubt::snegi why will this ever happen?
    }
    
    if (tile_xy >= mesh_dimensions[1]) { tile_y = mesh_dimensions[1];}
    else if (tile_xy != 0) {
      tile_y = mesh_dimensions[1] / tile_xy;
    }

    auto best_case_hop_count = tile_x + tile_y - 1-1; // extra -1 to ignore target+mem=>child_mem hop //doubt::snegi shouldn't hop count for energy calculation should have the hops for each spatial nodes? So for a 2x2 mesh hop_count has to be larger than 2
    auto total_hop_count = (mesh_dimensions[0]-1)*mesh_dimensions[1] + (mesh_dimensions[1]-1)*mesh_dimensions[0];
    /*
    auto total_hops = (mesh_dimensions[0]-1)*mesh_dimensions[1] + (mesh_dimensions[1]-1)*mesh_dimensions[0] //total edges on Y-axis + total edges on X-axis   
    
    */
    double total_energy=0;
   if(mesh_dimensions[0]>1 &&mov_info.tile_count!=1) { //tile_count set to 1 when only 1 spatial unit is use

      auto W2d = spec.channel_width.Get()*8; //converting to bits 
      auto Q2d = multi_factor==1 ? mov_info.tile_access_size.resolve()*multi_factor*8: mov_info.tile_access_size.resolve() * (multi_factor-1)*8;// -1 bcz 1 tile of data is already present at the node //doubt::snegi check with Manik //does broadcst and reduction also needs to be considered here? //should I multiply by multi_factor or not? 
      auto tenq = spec.queuing_delay.Get();
      // total_energy = power_summary_router(W2d, 5,5,best_case_hop_count,1,1,1,1,1,tenq,Q2d,mesh_dimensions[0]);
      total_energy = power_summary_router(W2d, 5,5,total_hop_count,1,1,1,1,1,tenq,Q2d,mesh_dimensions[0]);
    }

    return total_energy;
  }


  /* Collective Operation Algorithm related code*/
  // const std::map<mapping::stype_t, std::map<std::vector<uint32_t>, std::tuple<uint32_t, uint32_t>>> COLOP_HOPS_DATAMOVED = { //mesh_dimension, HOPS, DATAMOVED in bytes if the data at the source is also 1 byte or should be 1 byte in the end
  //   {mapping::stype_t::BROADCAST, {
  //     {{1, 1}, {0, 0}},
  //     {{2, 2}, {3, 3}},
  //     {{4, 4}, {20, 20}},
  //     {{16, 16}, {0, 0}},
  //   }},
  //   {mapping::stype_t::REDUCTION, {
  //     {{1, 1}, {0, 0}},
  //     {{2, 2}, {3, 3}},
  //     {{4, 4}, {20, 20}},
  //     {{16, 16}, {0, 0}},
  //   }},
  //   {mapping::stype_t::SCATTER, {
  //     {{1, 1}, {0, 0}},
  //     {{2, 2}, {3, 4}},
  //     {{4, 4}, {20, 48}},
  //     {{16, 16}, {0, 0}},
  //   }},
  //   {mapping::stype_t::GATHER, {
  //     {{1, 1}, {0, 0}},
  //     {{2, 2}, {3, 4}},
  //     {{4, 4}, {20, 48}},
  //     {{16, 16}, {0, 0}},
  //   }}
  //   }; 
  
  //cost after considering parallel communication
  const std::map<mapping::stype_t, std::map<std::vector<uint32_t>, std::tuple<uint32_t, uint32_t>>> COLOP_HOPS_DATAMOVED = { //mesh_dimension, HOPS, DATAMOVED in bytes if the data at the source is also 1 byte or should be 1 byte in the end
    {mapping::stype_t::BROADCAST, {
      {{1, 1}, {0, 0}},
      {{2, 2}, {2, 2}},
      {{4, 4}, {6, 6}},
      {{16, 16}, {30, 30}},
    }},
    {mapping::stype_t::REDUCTION, {
      {{1, 1}, {0, 0}},
      {{2, 2}, {2, 2}},
      {{4, 4}, {6, 6}},
      {{16, 16}, {30, 30}},
    }},
    {mapping::stype_t::SCATTER, {
      {{1, 1}, {0, 0}},
      {{2, 2}, {2, 2}},
      {{4, 4}, {6, 6}},
      {{16, 16}, {30, 30}},
    }},
    {mapping::stype_t::GATHER, {
      {{1, 1}, {0, 0}},
      {{2, 2}, {2, 2}},
      {{4, 4}, {6, 6}},
      {{16, 16}, {30, 30}},
    }}
    }; 


    // Mesh network works same for broadcast and reduction
  // uint64_t MeshNetwork::GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor) {
  uint64_t MeshNetwork::GetLinkLatency(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type) {

    auto multi_factor = data_movement.num_unique_tiles;
    if (multi_factor == 0) return 0;
    if(spatial_factor==1) return 0; //only working on single node
    std::vector<uint32_t> mesh_dimensions(2,1);
    auto mesh_xy = static_cast<uint32_t>(std::ceil(std::sqrt(spatial_factor)));
    if (spec.mesh.Get()) {
      mesh_dimensions = spec.mesh_dimensions;
      if (spec.mesh_dimensions[0] >= mesh_xy && spec.mesh_dimensions[1] >= mesh_xy) {
        mesh_dimensions = {mesh_xy, mesh_xy}; // is minimal
      } else if (spec.mesh_dimensions[0] >= mesh_xy && spec.mesh_dimensions[1] < mesh_xy) {
        auto mesh_y = spec.mesh_dimensions[1];
        auto mesh_x = (spatial_factor + mesh_y - 1) / mesh_y;
        mesh_dimensions = {mesh_x, mesh_y};
      } else if (spec.mesh_dimensions[0] < mesh_xy && spec.mesh_dimensions[1] >= mesh_xy) {
        auto mesh_x = spec.mesh_dimensions[1];
        auto mesh_y = (spatial_factor + mesh_x - 1) / mesh_x;
        mesh_dimensions = {mesh_x, mesh_y};
      }
    }else {
      mesh_dimensions = {mesh_xy, mesh_xy};
    }

    float trouter_latency=0, queuing_delay=0, reduction_op_latency=0;

    precision *=8; //converting precision to bits HISIM models needs it

    if((stype==mapping::stype_t::BROADCAST || stype==mapping::stype_t::GATHER || stype==mapping::stype_t::SCATTER || stype==mapping::stype_t::REDUCTION)) {
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(stype).at(mesh_dimensions);

      auto total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);

      trouter_latency = total_hops*spec.hop_latency.Get();

      auto W2d = spec.channel_width.Get(); 
      auto Q2d = data_movement.tile_access_size.resolve()*data_moved;

      if(stype==mapping::stype_t::REDUCTION && op_type!=mapping::operation_t::None){
        if(op_type==mapping::operation_t::ADD){
          reduction_op_latency = Q2d*spatial_factor*ADD_CYCLES/16.0; //width of SIMD unit FIXME::snegi in future get this directly from arch file
        } else if(op_type == mapping::operation_t::MAX){
          reduction_op_latency = Q2d*spatial_factor*MAX_CYCLES/16.0;
        } else if(op_type == mapping::operation_t::MULT){
          reduction_op_latency = Q2d*spatial_factor*MULT_CYCLES/16.0;
        }
      }
      

      if(stype==mapping::stype_t::SCATTER && data_movement.num_unique_tiles==1) Q2d = (float)Q2d/spatial_factor; 

      queuing_delay = ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();
 
    } else if(stype==mapping::stype_t::ALLREDUCE){
      //reduce followed by broadcast
      //reduce 
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::REDUCTION).at(mesh_dimensions);
      auto total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);

      trouter_latency = total_hops*spec.hop_latency.Get();

      auto W2d = spec.channel_width.Get(); 
      auto Q2d = data_movement.tile_access_size.resolve()*data_moved;


      queuing_delay = ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();

      if(op_type!=mapping::operation_t::None){
        if(op_type==mapping::operation_t::ADD){
          reduction_op_latency = Q2d*spatial_factor*ADD_CYCLES/16.0; //width of SIMD unit FIXME::snegi in future get this directly from arch file
        } else if(op_type == mapping::operation_t::MAX){
          reduction_op_latency = Q2d*spatial_factor*MAX_CYCLES/16.0;
        } else if(op_type == mapping::operation_t::MULT){
          reduction_op_latency = Q2d*spatial_factor*MULT_CYCLES/16.0;
        }
      }
      //broadcast
      hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::BROADCAST).at(mesh_dimensions);
      total_hops = std::get<0>(hop_datamoved);
      data_moved = std::get<1>(hop_datamoved);

      trouter_latency += total_hops*spec.hop_latency.Get();

      Q2d = data_movement.tile_access_size.resolve()*data_moved;
      queuing_delay += ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();
    } else if(stype==mapping::stype_t::ALLGATHER){
      //gather followed by broadcast
      //gather 
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::GATHER).at(mesh_dimensions);
      auto total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);

      trouter_latency = total_hops*spec.hop_latency.Get();

      auto W2d = spec.channel_width.Get(); 
      auto Q2d = data_movement.tile_access_size.resolve()*data_moved;


      queuing_delay = ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();

      //broadcast
      hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::BROADCAST).at(mesh_dimensions);
      total_hops = std::get<0>(hop_datamoved);
      data_moved = std::get<1>(hop_datamoved);

      trouter_latency += total_hops*spec.hop_latency.Get();

      Q2d = data_movement.tile_access_size.resolve()*data_moved;
      queuing_delay += ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();

    } else {
      COMET_ERROR("Unknown collective operation");
    }
    return trouter_latency + queuing_delay + reduction_op_latency;

/*
    uint32_t best_case_hop_cnt = std::max<uint32_t>(mesh_dimensions[0],1) - 1 + std::max<uint32_t>(mesh_dimensions[1],1) - 1;
    uint64_t no_congestion_latency = (best_case_hop_cnt) * spec.hop_latency.Get();
    uint64_t extra_network_latency = 0;
    float derate_factor = 1.0;
    bool allreduce = stype == mapping::stype_t::ALLREDUCE;
    bool allgather = stype == mapping::stype_t::ALLGATHER;
    bool reduction = stype == mapping::stype_t::REDUCTION;
    bool broadcast = stype == mapping::stype_t::BROADCAST;
    bool scatter = stype == mapping::stype_t::SCATTER;
    if ((reduction && spec.reduction_support.Get()) || (allreduce && spec.reduction_support.Get())) { // assuming MST 
      auto data_reduced_at_each_hop = (data_movement.tile_access_size.resolve() * spatial_factor + spec.port_bw.Get() - 1 ) / spec.port_bw.Get();
      extra_network_latency += best_case_hop_cnt * data_reduced_at_each_hop * spec.reduction_latency.Get();//doubt::snegi
    } 
    if ((allreduce) || (allgather)) {
      // max network conflicts is spatial_factor 
      extra_network_latency += spatial_factor * spec.hop_latency.Get(); // the memory time will take care of bandwidth portion of this //doubt::snegi
    }
    
    int64_t queuing_delay = 0;
    if(mesh_dimensions[0]>1) {

      auto W2d = spec.channel_width.Get(); 
      auto Q2d = data_movement.tile_access_size.resolve() * multi_factor;//doubt::snegi check with Manik //does broadcst and reduction also needs to be considered here? //should I multiply by multi_factor or not?

      queuing_delay = ((Q2d*precision)/ W2d) * spec.queuing_delay.Get();
    }



    return no_congestion_latency + extra_network_latency + queuing_delay;
*/
  }
  

  float MeshNetwork::GetLinkEnergy(mapping::stype_t stype, analysis::DataMovementInfo& data_movement, int spatial_factor, float precision, mapping::operation_t op_type){
    float total_energy=0;

    auto multi_factor = data_movement.num_unique_tiles;
    if (multi_factor == 0) return 0;
    if(spatial_factor==1) return 0; //only working on single node
    std::vector<uint32_t> mesh_dimensions(2,1);
    auto mesh_xy = static_cast<uint32_t>(std::ceil(std::sqrt(spatial_factor)));
    if (spec.mesh.Get()) {
      mesh_dimensions = spec.mesh_dimensions;
      if (spec.mesh_dimensions[0] >= mesh_xy && spec.mesh_dimensions[1] >= mesh_xy) {
        mesh_dimensions = {mesh_xy, mesh_xy}; // is minimal
      } else if (spec.mesh_dimensions[0] >= mesh_xy && spec.mesh_dimensions[1] < mesh_xy) {
        auto mesh_y = spec.mesh_dimensions[1];
        auto mesh_x = (spatial_factor + mesh_y - 1) / mesh_y;
        mesh_dimensions = {mesh_x, mesh_y};
      } else if (spec.mesh_dimensions[0] < mesh_xy && spec.mesh_dimensions[1] >= mesh_xy) {
        auto mesh_x = spec.mesh_dimensions[1];
        auto mesh_y = (spatial_factor + mesh_x - 1) / mesh_x;
        mesh_dimensions = {mesh_x, mesh_y};
      }
    }else {
      mesh_dimensions = {mesh_xy, mesh_xy};
    }

    uint32_t total_hops, Q2d;
    float reduction_op_energy=0;




    if((stype==mapping::stype_t::BROADCAST || stype==mapping::stype_t::GATHER || stype==mapping::stype_t::SCATTER || stype==mapping::stype_t::REDUCTION)) {
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(stype).at(mesh_dimensions);

      total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);

      Q2d = data_movement.tile_access_size.resolve()*data_moved;

      if(stype==mapping::stype_t::REDUCTION && op_type!=mapping::operation_t::None){
        if(op_type==mapping::operation_t::ADD){
          reduction_op_energy = Q2d*spatial_factor*ADD_ENERGY;
        } else if(op_type == mapping::operation_t::MAX){
          reduction_op_energy = Q2d*spatial_factor*MAX_ENERGY;
        } else if(op_type == mapping::operation_t::MULT){
          reduction_op_energy = Q2d*spatial_factor*MULT_ENERGY;
        }
      }

      if(stype==mapping::stype_t::SCATTER && data_movement.num_unique_tiles==1) Q2d = (float)Q2d/spatial_factor; //num_unique_tiles=1 for explicit collective but for implicit Q2d is already divided by spatial_factor 

    } else if(stype==mapping::stype_t::ALLREDUCE){
      //reduce followed by broadcast
      //reduce 
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::REDUCTION).at(mesh_dimensions);
      total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);


      Q2d = data_movement.tile_access_size.resolve()*data_moved;

      if(op_type!=mapping::operation_t::None){
        if(op_type==mapping::operation_t::ADD){
          reduction_op_energy = Q2d*spatial_factor*ADD_ENERGY;
        } else if(op_type == mapping::operation_t::MAX){
          reduction_op_energy = Q2d*spatial_factor*MAX_ENERGY;
        } else if(op_type == mapping::operation_t::MULT){
          reduction_op_energy = Q2d*spatial_factor*MULT_ENERGY;
        }
      }

      //broadcast
      hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::BROADCAST).at(mesh_dimensions);
      total_hops += std::get<0>(hop_datamoved);
      data_moved = std::get<1>(hop_datamoved);

      Q2d += data_movement.tile_access_size.resolve()*data_moved;

    } else if(stype==mapping::stype_t::ALLGATHER){
      //gather followed by broadcast
      //gather 
      auto hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::GATHER).at(mesh_dimensions);
      total_hops = std::get<0>(hop_datamoved);
      auto data_moved = std::get<1>(hop_datamoved);


      Q2d = data_movement.tile_access_size.resolve()*data_moved;

      //broadcast
      hop_datamoved = COLOP_HOPS_DATAMOVED.at(mapping::stype_t::BROADCAST).at(mesh_dimensions);
      total_hops += std::get<0>(hop_datamoved);
      data_moved = std::get<1>(hop_datamoved);


      Q2d += data_movement.tile_access_size.resolve()*data_moved;

    } else {
      COMET_ERROR("Unknown collective operation");
    }
    
    auto W2d = spec.channel_width.Get(); //just number of links
    Q2d *=precision*8; //precision is in bytes --> Q2d is in bits for HISIM
    auto tenq = spec.queuing_delay.Get();
    // total_energy = power_summary_router(W2d, 5,5,best_case_hop_count,1,1,1,1,1,tenq,Q2d,mesh_dimensions[0]);
    total_energy = power_summary_router(W2d, 5,5,total_hops,1,1,1,1,1,tenq,Q2d,mesh_dimensions[0]) + reduction_op_energy;

    return total_energy;
  }

  uint64_t MeshNetwork::GetLatencyForFlit(uint32_t flit_cnt) { 
    return flit_cnt * spec.hop_latency.Get();
  }

  uint64_t MeshNetwork::getPortBW() {
    return spec.port_bw.Get();
  }
  

}
