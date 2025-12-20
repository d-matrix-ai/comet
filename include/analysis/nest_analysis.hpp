#pragma once

#include <map>
#include <unordered_map>
#include <variant>

#include "mapping/mapping.hpp"
#include "analysis/node.hpp"
#include "analysis/data_movement.hpp"
#include "analysis/cost_walker.hpp"

#include "analysis/node_types.hpp"
#include "analysis/cost_info.hpp"

using mapping::Node;
using mapping::DataMovementTileNode;
using mapping::OperationNode;
using mapping::CollectiveOperationNode;
using mapping::InterTileBinding;
using mapping::stride_tilestep;
using mapping::Visitor;

using mapping::mapping_t;
using mapping::stype_t;
using mapping::stype_to_string;

using mapping::ComputeTilingMatrix;


namespace analysis{

    //commented to test data movement only
    // class CostWalker; //forward declaration to break the circular dependence
    // class RampUpWalker;
    // class SteadyStateWalker;

    // using analysis::TotalCost;

    // use inline for function definitions in header file. This ensures that there will only be one instasnce of the function, even if the header is included in multiple translation units
    // other way is to define this function in the cpp file this ensures that there is only one definition of this function
    inline bool isStringInVector(const std::vector<std::string>& vec, const std::string& str){
        return std::find(vec.begin(), vec.end(), str) != vec.end();
    }

    // using analysis::LoopNode;
    // using NodeTypes = std::variant<std::vector<LoopNode>, ColOpNode, OpNode>; //vector of loopnodes because we have global and local tensors
    // using NodeTypes = std::variant<LoopNode, ColOpNode, OpNode>; //vector of loopnodes because we have global and local tensors

    // struct LoopDescriptor { 
    //     uint64_t end;
    //     uint64_t stride;
    // };
    // using LoopNestDescriptor = std::vector<std::map<problem::DimensionID, LoopDescriptor>>; // outer vec i indexed by spatial dimensions X and Y, inner map is indexed by DimID

    class NestAnalysis{
        private:
            problem::Workloads&       workloads_;
            mapping::Mapping&   mapping_;
            arch::Topology&     topology_;
            bool                calc_noc_energy_;


            void get_loopnest();
            void get_datamovement_info();
            // void get_cost();
            // std::map<const Node*, uint32_t> get_cost();
            cost_struct get_cost();

            // std::unordered_map<const Node*, std::vector<LoopNode>> configs; //vector of loopnodes because we have global and local tensors
            std::unordered_map<const Node*, NodeTypes> workload_mapping_graph; //vector of loopnodes because we have global and local tensors
            std::unordered_map<const Node*, NodeTypes> duplicate_tensors_workload_mapping_graph; //vector of loopnodes because we have global and local tensors

            // std::unordered_map<const Node*, LoopNode> workload_mapping_graph; //vector of loopnodes because we have global and local tensors

            
            // std::unordered_map<const Node*, std::vector<TensorMovementStruct>> datamovement_info; //FIXME:snegi should be a vector of TensorMovementStruct for global and local tensors but currently keeping it as it is // reason we have a map is because between two levels we can have multiple branches

            // std::unordered_map<const Node*, TensorMovementStruct> datamovement_info; //FIXME:snegi should be a vector of TensorMovementStruct for global and local tensors but currently keeping it as it is // reason we have a map is because between two levels we can have multiple branches

            std::unordered_map<const Node*, TensorMovementInfo> datamovement_info;
            std::unordered_map<const Node*, TensorMovementInfo> duplicate_tensors_datamovement_info;


        public:
            NestAnalysis(problem::Workloads& workloads, mapping::Mapping& mapping, arch::Topology& topology, bool calc_noc_energy): workloads_(workloads), mapping_(mapping), topology_(topology), calc_noc_energy_(calc_noc_energy){}

            friend class LoopNestConstructor;
            friend class DataMovementInfoConstructor;

            //commented to test data movement only
            friend class CostWalker;
            // friend class RampUpWalker;
            friend class SteadyStateWalker;
            friend class IdealComputationWalker;
            
            void analyze();

    };
// static_cast<const DataMovementTileNode*>(node)->level_to_tileMapping_global_tensors

    
    class LoopNestConstructor: public mapping::Visitor{
        NestAnalysis& analysis_;
        // stride_tilestep stride_tilesteps_global, stride_tilesteps_intermediate;
        // std::vector<stride_tilestep> stride_tilesteps_; //outer is for global and inner is for intermediate, std::vector for global and intermediate tensor
        stride_tilestep stride_tilesteps_; //outer is for global and inner is for intermediate, std::vector for global and intermediate tensor

        //need a datastructure to store the  output size of operations and Collective operations
        std::map<arch::ArchLevel*, std::map<problem::TensorID, std::vector<std::unordered_map<problem::DimensionID, uint32_t>>>> output_tensors_map_; // < at what arch level this tensor lives at, <which tensor is it, <dimensions, value of dimension>>>, getting all the dimension sizes because this will help us to generalize collective op output size depending on any collective operation. std::vector in the end because we have spatial X and spatial Y dimension

        void visitDataMovementTileNode(const DataMovementTileNode* node) override;
        void visitOperationNode(const OperationNode* node) override;
        void visitCollectiveOperationNode(const CollectiveOperationNode* node) override;

        public:
            // LoopNestConstructor(NestAnalysis& analysis): analysis_(analysis), stride_tilesteps_global(), stride_tilesteps_intermediate(){}
            // void construct(const Node* root) {root->accept(this, stride_tilesteps_global, stride_tilesteps_intermediate);}
            
            LoopNestConstructor(NestAnalysis& analysis): analysis_(analysis), stride_tilesteps_(){}
            void construct(const Node* root) {root->accept(this);}

            
    };

    class DataMovementInfoConstructor: public mapping::Visitor{
        NestAnalysis& analysis_;



        void visitDataMovementTileNode(const DataMovementTileNode* node) override;
        // void visitOperationNode(const OperationNode* node) override;
        void visitCollectiveOperationNode(const CollectiveOperationNode* node) override;

        // void ComputeTemporalInfo(problem::TensorID tid, const std::vector<LoopNode>& loop_node);
        void ComputeTemporalInfo(LoopNode& loop_nodes, const DataMovementTileNode* node);
        // void ComputeSpatialInfo(problem::TensorID tid, const std::vector<LoopNode>& loop_node);
        void ComputeSpatialInfo_fused(LoopNode& loop_nodes, const DataMovementTileNode* node, LoopNode& parent_loop_nodes);
        
        void ComputeSpatialInfo(LoopNode& loop_nodes, const DataMovementTileNode* node, LoopNode& parent_loop_nodes);

        std::vector<DataMovementInfo> createMovementInfo(std::map<problem::TensorID, LoopNestDescriptor>& loop_descriptor);

        public:
            DataMovementInfoConstructor(NestAnalysis& analysis): analysis_(analysis){}
            void construct(const Node* root) {root->accept(this);}


    };



}