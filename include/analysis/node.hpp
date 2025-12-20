#pragma once


#include "mapping/mapping_utils.hpp"
#include "problem/workload.hpp"
#include "arch/arch_level.hpp"

using mapping::stype_t;
using mapping::operation_t;
using mapping::mapping_t;

using problem::TensorID;
using problem::DimensionID;
using problem::LoopOrder;

// using arch::ArchLevel;

namespace analysis{

    class arch::ArchLevel;
    
    // using analysis::LoopNestDescriptor;

    struct LoopNode{
        std::unordered_map<problem::TensorID, std::vector<size_t>> tid_idx;
        std::unordered_map<arch::ArchLevel*, std::vector<size_t>> arch_idx; //arch to vector index
        std::vector<problem::TensorID> tensors;
        std::vector<LoopNestDescriptor>  descriptor_; //(loopnest descriptor is a vector in vector. Outer vector is indexed by spatial dimensions and inner vector is indexed by DimID.)
        std::vector<problem::LoopOrder> order; // only matters for temporal loops
        std::vector<uint32_t> iteration_count;
        std::vector<uint32_t> sp_iteration_count;

        mapping_t type;
        std::vector<bool> tensor_is_rw;
        std::vector<arch::ArchLevel*> target;
        std::vector<arch::ArchLevel*> child;
        std::vector<bool> wb_output;
        std::vector<std::string> tags;
        std::vector<bool> dependent_tensor;
        std::map<std::pair<arch::ArchLevel*, arch::ArchLevel*>, std::vector<problem::TensorID>> target_child_tensor_map; 

        bool base_node;

        std::vector<bool> movement_evaluated;
        std::vector<bool> ramp_up_evaluated;
        std::vector<bool> cost_evaluated;

        std::vector<uint8_t> scale;
        std::vector<bool> rmw;

        uint8_t spatial_factor=1;
        
        std::map<problem::TensorID,HyperRectangleSet> prev_hr_set;

        std::vector<uint32_t> relative_timestep_to_dependent;

        bool spatial_node_exist=false;
        bool same_target_as_parent=false;
    };   

    using TensorSizeMap=std::map<problem::TensorID, std::vector<std::unordered_map<problem::DimensionID, uint32_t>>>;

    struct ColOpNode{
        arch::ArchLevel* target;
        arch::ArchLevel* child;
        std::string tag;
        uint32_t spatial_factor;

        operation_t reduction_op;
        stype_t type_;
        problem::DimensionID dimension;
        // std::string src;
        // std::string dest;
        std::vector<arch::ArchLevel*> src;
        std::vector<arch::ArchLevel*> dest;
        
        std::map<problem::TensorID, bool> tensor_is_rw;
        std::map<problem::TensorID, bool> wb_output;
        // std::vector<problem::TensorID> in_tensor; // here we have to know the input and output tensors since it is not defined in problem
        // std::vector<problem::TensorID> out_tensor;

        problem::TensorID in_tensor;
        problem::TensorID out_tensor;        

        uint8_t scale=1;
        
        // std::vector<std::vector<problem::TensorID>> tensors; //first vector is for global and local tensors, next vector is for number of tensors

        //comes from the calculation done during LoopNode creation
        TensorSizeMap in_tensor_size; //sizes for all the input tensors, std::vector bcz of spatial X and spatial Y
        TensorSizeMap out_tensor_size; //sizes for the output tensors

        //creating separate descriptors since input and output tensors IDs are same
        std::map<problem::TensorID, LoopNestDescriptor>  descriptor_input; //(loopnest descriptor is a vector in vector. Outer vector is indexed by spatial dimensions and inner vector is indexed by DimID.)        
        std::map<problem::TensorID, LoopNestDescriptor>  descriptor_output; //(loopnest descriptor is a vector in vector. Outer vector is indexed by spatial dimensions and inner vector is indexed by DimID.)        

    }; 


    struct OpNode{
        operation_t type; // connect to arch topology
        // std::vector<problem::TensorID> in_tensor; // we already know in_tensor and out_tensor from the problem.yaml file
        // std::vector<problem::TensorID> out_tensor;
        // std::vector<std::vector<problem::TensorID>> tensors;
        std::vector<problem::TensorID> tensors;

        // arch::ArchLevel* target;
        // arch::ArchLevel* child;
        std::string op_name; // connect to the problem file

        std::vector<problem::TensorID> in_tensors;
        problem::TensorID out_tensor;

        
        //comes from the calculation done during LoopNode creation
        // TensorSizeMap in_tensor_size; //std::vector because of spatial X and spatial Y dimension
        // TensorSizeMap out_tensor_size; //std::vector because of spatial X and spatial Y dimension
        // TensorSizeMap tensor_sizes;
        
        //parameters same as LoopNodes
        // std::map<problem::TensorID, bool> tensor_is_rw;
        // std::map<problem::TensorID, bool> rmw;
        // std::map<problem::TensorID, uint8_t> scale;



        // std::map<problem::TensorID, LoopNestDescriptor>  descriptor_; //(loopnest descriptor is a vector in vector. Outer vector is indexed by spatial dimensions and inner vector is indexed by DimID.)       // depricated, since there is a data movement node for OpNodes 
        
    };



}