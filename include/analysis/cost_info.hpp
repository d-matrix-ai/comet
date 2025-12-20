#pragma once
#include <cstdint>
#include<vector>
#include "arch/arch_level.hpp"
#include "problem/utils.hpp"

using arch::ArchLevel;
using problem::TensorID; 

namespace analysis{
    struct node_cost{
        uint64_t compute_time=0;
        uint64_t stall=0;
    };

    struct target_child_cost{
        // std::vector<uint64_t> memory_window;
        uint64_t compute_time;

        std::vector<uint64_t> stall; //std::vector for number of tiles

        // target_child_cost(): memory_window(0),stall(2,0){}
    };
    
    // using TargetChildCostVec=std::vector<target_child_cost>; // std::vector for read and write ports
    
    //cost struct for target_child cost inside the function
    struct TargetChildCost {
        uint64_t compute_time=0;
        std::vector<int> mem_net_cycles;
        // std::vector<uint64_t> network_cycles;
        std::vector<int> stall;
        std::vector<int> compulsory_stall;
        std::vector<uint32_t> compute_time_vec;
        uint32_t reuse_factor=0; // bcz of this line in cost_walker.cpp  if(node_level_cost[arch_pair.first][idx].reuse_factor==0) node_level_cost[arch_pair.first][idx].reuse_factor=1;
        size_t tens_idx=0;
        bool hide_rw_latency=true; // if this is false then directly take the stall during merging, setting default to true

        int cummulative_mem_net_cycles=0;
        int cummulative_stall=0;

        uint64_t ramp_up=0;
        uint64_t ramp_down=0;

        uint32_t iterations=1;
        uint64_t access_count=1;
        uint64_t tile_size=1;
        double   noc_energy=0.0;

        // TargetChildCost(size_t num_tiles=0): mem_net_cycles(num_tiles, 0), stall(num_tiles,0), compulsory_stall(num_tiles,0), compute_time_vec(num_tiles,0){}

        TargetChildCost(size_t num_tiles=0): compute_time(0), mem_net_cycles(num_tiles, 0), stall(num_tiles,0), compulsory_stall(num_tiles,0), compute_time_vec(num_tiles,0), reuse_factor(0),tens_idx(0), hide_rw_latency(true), cummulative_mem_net_cycles(0), cummulative_stall(0), ramp_up(0), ramp_down(0), iterations(1), access_count(1), tile_size(1), noc_energy(0.0) {}

        // Add a proper copy constructor to avoid uninitialized members
        TargetChildCost(const TargetChildCost& other):
            compute_time(other.compute_time),
            mem_net_cycles(other.mem_net_cycles),
            stall(other.stall),
            compulsory_stall(other.compulsory_stall),
            compute_time_vec(other.compute_time_vec),
            reuse_factor(other.reuse_factor),
            tens_idx(other.tens_idx),
            hide_rw_latency(other.hide_rw_latency),
            cummulative_mem_net_cycles(other.cummulative_mem_net_cycles),
            cummulative_stall(other.cummulative_stall),
            ramp_up(other.ramp_up),
            ramp_down(other.ramp_down),
            iterations(other.iterations),
            access_count(other.access_count),
            tile_size(other.tile_size),
            noc_energy(other.noc_energy)
        {}

    };

    struct conflict_stalls{
        // int target_out_stall = 0;
        // int target_in_stall  = 0;

        // int child_out_stall  = 0;
        // int child_in_stall   = 0;

        std::vector<int> target_stall = std::vector<int>(2,0);
        std::vector<int> child_stall  = std::vector<int>(2,0);

    };

    struct DataMovementCostVec{
        std::vector<int> stall;
        std::vector<int> mem_net_cycles; //need this because we cannot just add stall from multiple branches and get the final stalls. Example: find final stall at a node where GB is the target and in child we have  multiple buffers
        // std::vector<int> network_cycles;

        int cummulative_mem_net_cycles;
        int cummulative_stall;

        uint64_t ramp_up;
        uint64_t ramp_down;
        uint64_t access_count;
        uint64_t tile_size;
        double noc_energy;

        DataMovementCostVec(size_t size=0): stall(size, 0), mem_net_cycles(size,0), cummulative_mem_net_cycles(0), cummulative_stall(0), ramp_up(0), ramp_down(0), access_count(0), tile_size(0), noc_energy(0) {}
    };

    using TargetChildCostVec = std::vector<TargetChildCost>; // vector for read and write port

    // using DataMovementCostVec = std::vector<DataMovementCost>;    // vector for # of iterations

    // using PortLevelCost = std::vector<DataMovementCostVec>; //vector for read write ports
    using PortLevelCost = std::vector<TargetChildCost>; //vector for read write ports


    // struct CostVec{
    //     uint64_t memory_window;
    //     std::vector<std::vector<int64_t>> stall; //outer vector for number of tiles and internal vector for in and out of memory stalls 
    //       
        
    // };

    // using CostVec=std::vector<Cost>; // std::vector for number of tiles

    using Time_t = std::vector<int64_t>; // 0 for read 1 for write

    // using CostVec = std::vector<Cost>;

    // using ArchLevelInfo = std::map<arch::LevelID, CostVec>; //target, child level 

    // using NodeLatencyInfo = std::map<const Node*, ArchLevelInfo>;    

    using ArchLevelCost=std::map<std::pair<ArchLevel*, ArchLevel*>, std::vector<TargetChildCostVec>>; //std::vector because from one Node we can have same target and child. Like GM->OB for Op1 and GM->OB for Op2
    using NodeLevelCost=std::map<ArchLevel*, TargetChildCostVec>;
    using TotalCost=std::map<ArchLevel*, Time_t>;


    // struct for energy
    struct access_count_struct{
        const ArchLevel* arch_level;
        uint64_t access_count;
        uint64_t tile_size;
        uint8_t precision;
        bool tensor_is_rw; // true->read_port, false->write_port
        bool target_child; //true->target, false->child
        bool is_compute;
        TensorID tid;
        bool rmw=false;
        uint32_t projected_spatial_count;
        std::string tag;
        uint64_t compute_time;
        mapping::operation_t op_type;
        double noc_energy=0;
        float compute_energy=0;

        access_count_struct(
            const ArchLevel* arch_level = nullptr,
            uint64_t access_count = 0,
            uint64_t tile_size = 0,
            uint8_t precision = 0,
            bool tensor_is_rw = true,
            bool target_child = true,
            bool is_compute = false,
            TensorID tid = TensorID(),
            bool rmw = false,
            uint32_t projected_spatial_count = 0,
            std::string tag = "",
            uint64_t compute_time = 0,
            mapping::operation_t op_type = mapping::operation_t::ADD,
            double noc_energy = 0.0,
            float compute_energy = 0.0f
        ) : arch_level(arch_level), access_count(access_count), tile_size(tile_size), precision(precision),
            tensor_is_rw(tensor_is_rw), target_child(target_child), is_compute(is_compute), tid(tid), rmw(rmw),
            projected_spatial_count(projected_spatial_count), tag(tag), compute_time(compute_time),
            op_type(op_type), noc_energy(noc_energy), compute_energy(compute_energy) {}


    };

    struct ColOp_struct{
        int64_t memory_time = 0;
        int64_t network_time = 0; 
        double read_energy = 0.0;
        double write_energy = 0.0;
        double network_energy = 0.0;

        // const ArchLevel* arch_level;
        // uint64_t tile_size;
        // uint8_t precision;
        // stype_t type; //collective operation type
        
    };

    struct energy_struct{
        const ArchLevel* arch_level = nullptr;
        double energy = 0.0;
    };

    using ArchLevelEnergy = std::map<const ArchLevel*, double>;

    using NOCEnergy = std::map<std::string, double>;//GBuff<->GBuff, OBuff<->OBuff NOC

    using EnergyTypes = std::variant<ArchLevelEnergy, NOCEnergy>;

    using AccessCount = std::map<const ArchLevel*, std::map<uint32_t, uint64_t>>;// ports, accesses
}