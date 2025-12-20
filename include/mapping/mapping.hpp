#pragma once

#include <map>
#include <vector>
#include <string>
#include <iostream>

#include "compound_config/compound_config.hpp"

#include "analysis/node.hpp"
#include "problem/workload.hpp"
#include "mapping/mapping_utils.hpp"
// #include "arch/arch_level.hpp"
#include "arch/arch.hpp"
#include "analysis/cost_info.hpp"

using analysis::TotalCost;
using analysis::LoopNode;
using analysis::TensorSizeMap;

namespace mapping {


  class arch::Topology;
  
  //forward declaration
  class Node;
  class DataMovementTileNode;
  class OperationNode;
  class InterTileBinding;
  class CollectiveOperationNode;

  struct stride_tilestep{
    std::map<problem::TensorID, std::map<std::string, ComputeTilingMatrix>> stride; //std::vector for spatial X and spatial Y
    std::map<problem::TensorID, std::map<std::string, uint32_t>>        tilesteps;
    // std::map<std::pair<problem::TensorID,arch::ArchLevel>, bool> tensor_updated; //tensor's stride and tilestep updated at this arch level
  };


  class Visitor {
    protected:
        virtual void visitDataMovementTileNode(const DataMovementTileNode* node);
        virtual void visitOperationNode(const OperationNode* node);
        virtual void visitInterTileBinding(const InterTileBinding* node);
        virtual void visitCollectiveOperationNode(const CollectiveOperationNode* node);
        
        // virtual void visitColOpTileNode(const ColOpTileNode*);
        
        
        
        friend class DataMovementTileNode;
        friend class OperationNode;
        friend class InterTileBinding;
        friend class CollectiveOperationNode;
    public:
        virtual void run (const Node*);
  };


  class Node {
    public:
      enum type_t{
        DataMovementTileNode,
        OperationNode,
        CollectiveOperationNode,
        InterTileBinding
      };
    
    protected:
      static const std::map<type_t, std::string> type2name_; 
      Node::type_t type_;
      std::string  name_;
      mutable const Node* parent_ = nullptr;
      std::vector<const Node*> children_;
      // std::vector<Node*> children_;


    public:
      Node(type_t t, config::CompoundConfigNode config);
      type_t get_type() const{return type_;}
      std::string get_name() const{return name_;}
      void add_child(const Node* child);
      const std::vector<const Node*> get_children() const{ 
        return children_;
      }
      // std::vector<Node*> get_children(){ 
      //   return children_;
      // }
      std::vector<Node*> get_non_const_children();
      void set_children(const std::vector<const Node*>& children){
        children_ = children;
      }

      const Node* get_first_child() const {
        assert(children_.size());
        return children_.front();
      }

      void set_parent(const Node* parent) const {parent_ = parent;}
      inline const Node* get_parent() const {return parent_;}

      virtual void accept(Visitor* visitor) const=0;
      // virtual std::vector<stride_tilestep> constructLoopNest(std::vector<stride_tilestep>& stride_tilesteps) =0;
      virtual ~Node() {for (auto node: children_) delete node;}
      friend class Visitor;

      void printTree(const Node* node, std::map<const Node*, uint32_t> cost, const std::string& prefix = "", bool isLast = true);
      // void printTree(const Node* node, analysis::cost_struct cost, const std::string& prefix = "", bool isLast = true);

      void mark_nodes_with_colop_tensor(Node* root);
      void set_for_descendants(Node* node, problem::TensorID tid);
      bool check_colOp_descendant(Node* node, problem::TensorID tid);

  };

  using SpatialInfo = uint32_t;
  struct LevelTileMapping {
      LevelTileMapping()=default;
      mapping_t type;
      // stype_t stype=stype_t::BROADCAST;
      std::unordered_map<problem::TensorID, std::vector<size_t>> tid_idx;
      std::vector<problem::TensorID> tensors;
      std::vector<arch::ArchLevel*> target_;
      // std::vector<std::map<problem::TensorID, TilingMatrix>> tilings_; //outer vector is for spatial X and spatial Y
      std::vector<std::vector<ComputeTilingMatrix>> tilings_; //outer vector is for tensors and inner vector is for spatial X and spatial Y
      //FIXME:: this needs to be modified for serpent ordering as well
      std::vector<problem::LoopOrder> order;
      std::vector<arch::ArchLevel*> child_;
      std::vector<bool> wb_output;
      std::vector<std::string> tags; //which operation does the tensor belongs to  
      std::vector<uint8_t> scale;   
      std::string print(bool verbose) const;
      std::vector<bool> rmw;
      SpatialInfo spatial_info=0; // spatial_info, is num nodes for now and may need to also have reduction factor, 0 is for an all node operation  
      std::vector<uint64_t> itr_count; // tracks the iteration count
  };


  class DataMovementTileNode: public Node{
    private:
      // std::map<problem::TensorID, LevelTileMapping> level_to_tileMapping_global_tensors; //fill this if the tensor is in the global tensors in the problem yaml file, in comet leveltilemapping was a list because there we wanted to put both spatial and temporal loops for same tensors together, but here we have a seprate node for spatial node.
      // std::map<problem::TensorID, LevelTileMapping> level_to_tileMapping_intermediate_tensors; //fill this if the tensor is not in the global tensors -- need to map this to a particular Operation node as well
      // std::map<std::string, std::map<problem::TensorID, LevelTileMapping>> level_to_tileMapping_intermediate_tensors; //{ OpName1: {0:Mapping, 1:Mapping}, OpName2: {0:Mapping, 1:Mapping}}
      // size_t vector_size=2;
      // int vec_size=2;

      // std::vector<std::map<problem::TensorID, LevelTileMapping>> level_to_tileMapping_; //index0 is global tensor and index1 is local tensor
      
      // std::vector<LevelTileMapping> level_to_tileMapping_; //index0 is global tensor and index1 is local tensor
      LevelTileMapping level_to_tileMapping_;

    public:
      // variable to decide if the node is parent or it is to the right of collective operation at the same architecture level
      // currently a tensor in the node can be duplicated only twice
      bool has_tensor_from_colop=false;
      problem::TensorID tensor_from_colop;
      bool duplicate_tensor_exist=false;
      problem::DimensionID dim_id_from_colop;
      uint32_t factor_from_colop;

      DataMovementTileNode(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants);
      void accept(Visitor* visitor) const{visitor->visitDataMovementTileNode(this);}
      bool is_spatial() const {return level_to_tileMapping_.type==mapping_t::SPATIAL;}      
      mapping_t get_tile_type() const {return level_to_tileMapping_.type;}
      LevelTileMapping get_mapping() const{return level_to_tileMapping_;}
      LoopNode constructLoopNest(stride_tilestep& stride_tilesteps, std::vector<problem::TensorID>& tensors_updated, bool duplicate_tensor, problem::Workloads& workloads, const TensorSizeMap& tensor_size=TensorSizeMap()) const; //const at the end of l, workloads passsed to figure out the read-write tensors from the problem file

  };
// static_cast<const DataMovementTileNode*>(node)->level_to_tileMapping_global_tensors

  //moving problem::TensorID to OperationNode class so that it is compatible with DataMovement class.
  using TileSize = problem::DimSizeExpression;
  struct ComputeMapping {
    // TileSize compute_tilesize; // all dimension sizes needed for SW mapping on compute //tileprimitives
    // std::map<problem::TensorID, TileSize> tilesizes; // unprojected tilesize for the tensor for the SW mapping on compute
    // std::map<problem::TensorID, TileSize> projected_tilesizes; // projected tilesize for the tensor for SW mapping on compute
    // std::map<problem::TensorID, uint32_t> relative_tilesteps; // relative tilesetps to compare residency time for the tensor tiles
    // problem::TensorID min_tens_id; 
    // std::map<problem::TensorID, uint32_t> tensorComputeScaling_; // 

    void resolve(); 
    std::string print() const;
    computation_attributes comp_attr;
    // std::map<problem::TensorID,TilingMatrixMap> tileprimitives;
    // std::map<problem::TensorID,ComputeTilingMatrix> tileprimitives;
    ComputeTilingMatrix tileprimitives;

    std::map<problem::TensorID,ComputeTilingMatrix> tilesizes;
    std::map<problem::TensorID,ComputeTilingMatrix> projected_tilesizes;
    std::map<problem::TensorID,uint32_t> relative_tilesteps;
    std::map<problem::TensorID,uint32_t> tensorComputeScaling_; // 

    // computation_attributes comp_attribute;

  };

  // struct SIMDStruct{
  //   std::vector<problem::TensorID> global_tensors;
  //   std::vector<problem::TensorID> intermediate_tensors;

  // };
  struct CommonAttributes{
    operation_t type_; // can be softmax, gemm, max, conv operation
    std::string op_name_; // gemm1, gemm2, softmax1, rowmax etc from the problem.yaml file
    // arch::ArchLevel* target_;
    // arch::ArchLevel* child_;     
    // std::vector<std::vector<problem::TensorID>> gl_in_tensors; //first vector is for storing global and local tensors, next vector is for storing input tensors because there can be multiple inputs 
    std::vector<problem::TensorID> tensors; //vector is for storing input and tensors because there can be multiple inputs 
    std::map<problem::TensorID,ComputeTilingMatrix> tilesizes;
  };

  class OperationNode: public Node{
    private:
      std::shared_ptr<problem::Workload> p_workload;
      // std::map<problem::TensorID, ComputeMapping> computeMapping_global;
      // std::map<problem::TensorID, ComputeMapping> ComputeMapping_intermediate;
      // std::vector<std::map<problem::TensorID, ComputeMapping>> computeMapping_; //index0 is global tensor and index1 is local tensor
      // std::vector<ComputeMapping> computeMapping_;
      ComputeMapping computeMapping_;

      CommonAttributes common_attributes_;
      // SIMDStruct operation_description;
      // std::vector<std::vector<problem::TensorID>> simd_tensor_id_;// first vector is for storing global and local tensors, next vector is for storing input tensors because there can be multiple inputs
    public:
      OperationNode(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants);
      void accept(Visitor* visitor) const {visitor->visitOperationNode(this);}
      const std::shared_ptr<problem::Workload>& get_workload() const {return p_workload;}
      ComputeMapping get_compute_mapping() const {return computeMapping_;}
      // size_t get_vec_size() {return computeMapping_.size();}
      CommonAttributes get_common_attributes() const {return common_attributes_;}
      uint32_t relativeTileSteps(problem::TensorID tensor_id) const;
  };

  class InterTileBinding: public Node{
    public:
      enum type_t{
        sequential,
        sharing,
        parallel,
        pipeline,
        none
      };
      InterTileBinding::type_t type;      
      InterTileBinding(config::CompoundConfigNode config);
      void accept(Visitor* visitor) const {visitor->visitInterTileBinding(this);}
      InterTileBinding::type_t get_inter_tile_binding_type() const {return type;}

  };

  struct collective_op_struct{
    arch::ArchLevel* target;
    arch::ArchLevel* child;
    std::string       tag;
    stype_t type_;
    operation_t reduction_op;
    problem::DimensionID dimension;
    uint32_t spatial_factor;
    // std::string src;
    std::vector<arch::ArchLevel*> src;
    std::vector<arch::ArchLevel*> dest;
    problem::TensorID in_tensor;
    problem::TensorID out_tensor;
    bool wb_output;
    uint8_t scale;
  };

  class CollectiveOperationNode:public Node{
    public:
      collective_op_struct collective_op_description;
      void accept(Visitor* visitor) const {visitor->visitCollectiveOperationNode(this);}
      stype_t get_collective_op_type() const {return collective_op_description.type_;}

      CollectiveOperationNode(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads);

  };



  class Mapping {
    public:
      Mapping(config::CompoundConfigNode mapping_config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants): topology_(topology), workloads_(workloads) {
        std::cout<<"***** starting to parse mapping file ****"<<std::endl; 
        root= recursiveParseMapping(mapping_config, topology_, workloads_, constants);
        
        // std::cout<<"Begining validation of mapping on the architecture"<<std::endl;
        // validateMapping();
        // std::cout<<"Completed validation, the mapping is correct!!"<<std::endl;
        
        
        }

      //placeholder for getComputeMapping function
      //placeholder for getTensroMapping function

      Node* root=nullptr;
      arch::Topology& topology_;
      problem::Workloads& workloads_;

    

      //declaring Nodes as friend class so that they can access topology, workload variables
      friend class DataMovementTileNode;

    private:
      Node* recursiveParseMapping(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants);
      // void validateMapping();
  };


  class Validate: public mapping::Visitor{
        Node* node_;
        problem::Workloads& workloads_;
        arch::Topology& topology_;
    
        // std::unordered_map<const Node*, std::map<std::string,problem::DimSizeExpression>> tensor_sizes; 
        std::unordered_map<const Node*, std::map<problem::TensorID, std::map<std::string, ComputeTilingMatrix>>> node_level_tensor_sizes; 

        std::unordered_map<std::string, uint64_t> arch_level_occupancy;

        std::map<problem::TensorID, std::map<std::string, ComputeTilingMatrix>> tensor_sizes;

        void visitDataMovementTileNode(const DataMovementTileNode* node) override;
        void visitOperationNode(const OperationNode* node) override;
        void visitCollectiveOperationNode(const CollectiveOperationNode* node) override;

        ComputeTilingMatrix find_tensor_size(const Node* node, TensorID tid);

      public:
        Validate(Node* node, problem::Workloads& workloads, arch::Topology& topology): node_(node), workloads_(workloads), topology_(topology){}

        void fitsInMemory();

        void calculate_tensor_size(){node_->accept(this);}
  };

}
