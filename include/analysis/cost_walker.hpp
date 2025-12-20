#pragma once
#include <cstdint>
#include "mapping/mapping_utils.hpp"
#include "mapping/mapping.hpp"
#include "problem/dimsize.hpp"
#include <unordered_set>


using mapping::Visitor;
using mapping::Node;
using mapping::DataMovementTileNode;
using mapping::OperationNode;
using mapping::CollectiveOperationNode;
using mapping::InterTileBinding;
using mapping::stride_tilestep;
using mapping::computation_attributes;

using arch::ArchLevel;
using arch::LevelID;

using problem::DimSizeExpression;

using ChildNodes = std::variant<const Node*, std::vector<const Node*>>; // child nodes can be a single node or a vector of nodes for sharing binding



namespace analysis{


    struct cost_struct{
        std::map<const mapping::Node *, uint32_t> node_cost;

        std::map<const mapping::Node *, std::map<ChildNodes, analysis::NodeLevelCost>> arch_level_node_cost;

        std::map<const Node*, ArchLevelEnergy> arch_level_node_energy;

        std::map<const Node*, std::vector<access_count_struct>> access_count;

        std::map<const Node*, NOCEnergy> node_level_noc_energy;

        std::map<std::string, uint64_t> comp_time_map;

        std::map<std::string, float> compute_utilization_map;

        std::unordered_map<const Node*, uint64_t> parent_iteration_count;
        
    };

    struct RampUpInfo{
        std::unordered_set<TensorID> tid; //can be more than 1 tensor contributing to the rampup cost
        uint32_t rampup;
    };

    class NestAnalysis; //forward declaration to break the circular dependence

    // using ChildNodes = std::variant<const Node*, std::vector<const Node*>>; // child nodes can be a single node or a vector of nodes for sharing binding

    class CostWalker{
        private:
            NestAnalysis& analysis_;
            uint64_t steady_state_time;
        
        public:
            CostWalker(NestAnalysis& analysis): analysis_(analysis){}

            // std::map<const Node*, TotalCost> calculateSteadyStateTime();
            // std::map<const Node*, uint32_t> calculateSteadyStateTime();
            cost_struct calculateSteadyStateTime();

    };

    //IdealComputationTime
    class IdealComputationWalker: public mapping::Visitor{
        private:
            NestAnalysis& analysis_;
            uint64_t total_comp_time_;
            std::map<std::string, ComputeTilingMatrix> tileprimitive_map; // operation-name(problem file), tileprimitive
            std::map<std::string, computation_attributes> comp_attributes;
            std::map<std::string, mapping::operation_t> op_type_map;
            std::map<std::string, ArchLevel*> arch_level_map;

            std::map<std::string, uint64_t> comp_time_map;
            std::map<std::string, uint64_t> ideal_comp_time_map;

            // uint64_t comp_time;
            // uint64_t ideal_comp_time;

        public:
            IdealComputationWalker(NestAnalysis& analysis, uint64_t total_comp_time): analysis_(analysis), total_comp_time_(total_comp_time){}
            void visitDataMovementTileNode(const DataMovementTileNode* node) override;
            void visitOperationNode(const OperationNode* node) override;

            void calculate_ideal_computation_time();

            void resolve_computation_time();

            void tree_traversal_ideal_computation_latency(const Node* root) {root->accept(this);}
            std::map<std::string, float> calculate_compute_utilization();

            std::map<std::string, uint64_t> get_comp_time_map() {return comp_time_map;}
            // uint64_t get_comp_time() {return comp_time;}
            // uint64_t get_ideal_comp_time() {return ideal_comp_time;} 
    };     


    class SteadyStateWalker: public mapping::Visitor{
        private:
            NestAnalysis& analysis_;
            // std::map<const Node*, std::vector<InterTileBinding::type_t>> binding_map;

            // std::map<std::pair<const Node*, const Node*>, ArchLevelCost> parent_child_arch_level_cost_map;
            std::map<const Node*, std::map<ChildNodes, ArchLevelCost>> parent_child_arch_level_cost_map;

            std::map<const Node*, std::map<ChildNodes, NodeLevelCost>> parent_child_node_level_cost_map; // cost in terms of just the parent memory, so basically combining the cost from different parent-memory to child-memory into the parent-memory

            std::map<const Node*, std::map<ChildNodes, TargetChildCost>> parent_child_cost_map; // after merging cost from different memories at the node

            std::map<const Node*, std::map<ChildNodes, uint32_t>> parent_child_total_cost; // when node will have multiple children we need total cost per edge

            std::map<const Node*, uint32_t> node_cost;

            // std::map<std::pair<const Node*, std::vector<const Node*>>, uint32_t> memory_windows;
            
            // std::map< const Node*, std::map<std::vector<const Node*>, uint32_t>> memory_windows;
            std::map< const Node*, std::map<ChildNodes, uint32_t>> memory_windows;

            // std::unordered_map<const Node*, std::vector<uint32_t>> reuse_factor;
            std::unordered_map<const Node*, std::map<std::tuple<TensorID, std::string>, uint32_t>> reuse_factor;
            
            std::unordered_map<const Node*, std::map<std::tuple<TensorID, std::string>, uint32_t>> actual_iterations;

            std::unordered_map<const Node*, std::map<std::tuple<TensorID, std::string>, uint32_t>> rampupInfo;

            std::unordered_map<const Node*, std::map<std::tuple<TensorID, std::string>, uint32_t>> rampdownInfo;

            // std::map<const Node*, std::map<ArchLevel*, energy_struct>> energy_retval;
            std::map<const Node*, std::vector<access_count_struct>> access_count_map; //since at a node we can have multiple tensors going from a target -> child memory

            // std::map<const Node*, std::vector<energy_struct>> node_level_access_energy;

            std::map<const Node*, ArchLevelEnergy> node_level_access_energy;
            // std::map<const Node*, EnergyTypes> node_level_access_energy;
            std::map<const Node*, NOCEnergy> node_level_noc_energy;
            std::unordered_map<const Node*, uint64_t> parent_iteration_count; // for each node we need to know how many iterations are there in the parent node, this to get the latency breakdown

/*
            std::map<const Node*, ArchLevelCost> node_arch_level_cost_map; //std::vector of MemToMemTransferCost because  GB->o/pbuff can be there for GP1-P1 and GP1-P2 
            // std::map<std::pair<const Node*, const Node*>, TargetChildCost> parent_child_node_total_cost_map; //Tgp1_p1
            std::map<std::pair<const Node*, const Node*>, NodeLevelCost> parent_child_node_total_cost_map;
            // std::unordered_map<std::pair<const Node*, const Node*>, NodeLevelCost> parent_child_node_total_cost_map;

            // std::map<const Node*, std::map<ArchLevel*, TargetChildCostVec>> parent_child_node_total_cost_map;

            std::map<const Node*, NodeLevelCost> node_cost_map;
            std::map<const Node*, TotalCost> node_total_cost;
            Time_t write_port_stalls = {0,0}; // to keep track of only compute cost so that it can be subtracted from the total cost at the read port
            // std::map<const Node*, uint64_t> node_total_cost_map; //tp1
*/

        public:
            SteadyStateWalker(NestAnalysis& analysis): analysis_(analysis){}
            void visitDataMovementTileNode(const DataMovementTileNode* node) override;
            void visitOperationNode(const OperationNode* node) override;
            void visitCollectiveOperationNode(const CollectiveOperationNode* node) override;

            void tree_traversal_steady_state(const Node* root) {root->accept(this);}

            TargetChildCostVec get_target_child_cost(LevelID target_id, LevelID child_id, std::vector<DataMovementInfo> tensor_mov, LoopNode& loop_nodes, const mapping::operation_t& op_type, uint32_t child_cost, size_t tens_cnt, bool hide_rw_latency, bool run_single_iteration, std::vector<uint32_t>& memWindow, const DimSizeExpression& compute_tileprimitive=DimSizeExpression(), const computation_attributes& comp_attributes=computation_attributes());

            // TargetChildCost get_node1_node2_cost(ArchLevelCost& cost, uint64_t child_idx);
            // TargetChildCost get_dataMov_comp_node_cost(ArchLevelCost& cost, std::map<std::pair<ArchLevel*, ArchLevel*>, uint64_t>& child_cnt);
            // NodeLevelCost get_node1_node2_cost(ArchLevelCost& cost, std::map<std::pair<ArchLevel*, ArchLevel*>, uint64_t>& child_cnt, std::vector<arch::ArchLevel*> child_arch);
            
            // NodeLevelCost get_node_cost(std::map<std::pair<const Node*, const Node*>, NodeLevelCost>& parent_child_cost, std::vector<InterTileBinding::type_t>& binding, const Node* node);

            
            // uint32_t get_node_cost(ArchLevelCost cost_map, LoopNode& loop_nodes);
            
            NodeLevelCost get_node1_node2_cost(ArchLevelCost arch_level_cost, LoopNode& loop_nodes, std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>>& child_arch_level_port_usage);

            TargetChildCost get_node_cost(NodeLevelCost node_level_cost, LoopNode& loop_nodes, bool compute_child_node);

            uint32_t get_total_node_cost(TargetChildCost cost);


            // uint32_t get_node_cost_with_binding(std::map<ChildNodes, NodeLevelCost> parent_child_node_level_cost, std::map<ChildNodes, TargetChildCost> parent_child_cost, std::map<ChildNodes, uint32_t> parent_child_total_cost, const Node* parent_node, std::vector<InterTileBinding::type_t> bindings, LoopNode loop_nodes);

            uint32_t get_node_cost_with_binding(const Node* parent_node, std::vector<InterTileBinding::type_t> bindings, LoopNode loop_nodes);


            void bfs_conflict_map_constructor(const Node* root, std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>>& level_map, std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>>& num_updates_map, std::map<size_t, uint32_t>& num_iterations);

            InterTileBinding::type_t find_binding(const Node* node);

            uint32_t get_temporal_node_cost(std::map<ChildNodes, ArchLevelCost> child_arch_level_cost, LoopNode& loop_nodes);

            uint32_t get_temporal_node_cost1(const Node* node);

            // std::map<const Node*, TotalCost> get_node_level_cost(){return node_total_cost;}
            void get_node_level_cost();

            std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>> get_port_level_cost(ChildNodes child_nodes);
            
            void get_mem_win_vector(const Node* parent_node, ChildNodes child_nodes, std::vector<uint32_t>& mem_win_vec);

            // void estimate_energy(const Node* node, uint64_t parent_iterations);
            // void estimate_energy(const Node* node, std::map<TensorID, uint32_t> num_sp_parents={}, std::map<TensorID, uint32_t> iteration_count_map={});
            void estimate_energy(const Node* node, std::map<std::string, std::map<TensorID, uint32_t>> num_sp_parents={}, std::map<std::string, std::map<TensorID, uint32_t>> iteration_count_map={});

            std::vector<std::string> find_ops_on_left(const Node* node, TensorID tens_id);
            std::vector<TensorID> find_tens_on_left(const Node* node);
            // int64_t get_write_port_stall(){return write_port_stalls[1];}

            // functions to get the cost and energy maps
            std::map<const Node*, uint32_t> get_mapping_cost(){return node_cost;}

            std::map<const Node*, std::map<ChildNodes, NodeLevelCost>> get_archlevel_mapping_cost(){return parent_child_node_level_cost_map;}

            std::map<const Node*, ArchLevelEnergy> get_archlevel_energy(){return node_level_access_energy;}

            std::map<const Node*, NOCEnergy> get_noc_energy(){return node_level_noc_energy;}

            std::map<const Node*, std::vector<access_count_struct>> get_access_count_map(){return access_count_map;}

            std::unordered_map<const Node*, uint64_t> get_parent_iteration_count(){return parent_iteration_count;}
    };
    
}