#include <iostream>
#include <mapping/mapping.hpp>
#include "util/logger.hpp"


namespace mapping {

  using analysis::LoopNestDescriptor;

  const std::map<Node::type_t, std::string> Node::type2name_ = {
      {Node::DataMovementTileNode, "DataMovementTileNode"},
      {Node::OperationNode, "OperationNode"},
      {Node::CollectiveOperationNode, "CollectiveOpNode"},
      {Node::InterTileBinding, "InterTileBinding"}
    };



  void Node::add_child(const Node* child){
    assert(child !=nullptr);
    children_.push_back(child);
    child->set_parent(this);
  }

  //function to get non constant access to nodes in the mapping
  std::vector<Node*> Node::get_non_const_children(){
    std::vector<Node*> non_const_children;
    for(const Node* child:children_){
      non_const_children.push_back(const_cast<Node*>(child));
    }
    return non_const_children;
  }


  // Function to recursively print the tree
  // void Node::printTree(const Node* node, int depth) {
  //     if (!node) return;

  //     // Print current node with indentation according to depth
  //     for (int i = 0; i < depth; ++i) {
  //         std::cout << "  "; // Indentation (2 spaces per depth level)
  //     }

  //     // Print node information (type and name)
  //     std::cout << node->get_name() << std::endl;

  //     // Recursively print each child node
  //     const std::vector<const Node*>& children = node->get_children();
  //     for (const Node* child : children) {
  //         printTree(child, depth + 1);
  //     }
  // }
// Function to recursively print the tree in a tree-like format
  void Node::printTree(const Node* node, std::map<const Node*, uint32_t> cost, const std::string& prefix, bool isLast) {
      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      // Print node information (type and name)
      std::ostringstream oss;
      // auto my_map = node_level_cost[node];
      // oss << "{ ";
      // if (node->get_type()==mapping::)
      // Iterate over the map and append each key-value pair to the stringstream
      // for (auto it = my_map.begin(); it != my_map.end(); ++it) {
      //     if(it->second.size()==0) continue;
      //     oss << "\"" << it->first->getName() << "\": " << it->second[0] << ", "<<it->second[1];

      //     // Add a comma unless it's the last element
      //     if (std::next(it) != my_map.end()) {
      //         oss << ", ";
      //     }
      // }

      // for(auto&[node, cost]: cost){
      //   oss<< cost;
      // }

      oss<<cost[node];

      if(node->get_type()==Node::type_t::InterTileBinding || node->get_type()==Node::type_t::OperationNode){
        std::cout << node->get_name() << std::endl;
      } else {
        std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
      }
      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");

      // Recursively print each child node
      const std::vector<const Node*>& children = node->get_children();
      for (size_t i = 0; i < children.size(); ++i) {
          printTree(children[i], cost, childPrefix, i == children.size() - 1);
      }
  }
/*
  void Node::set_for_descendants(Node* node, problem::TensorID tid){

    if(!node) return;

    for(auto child: node->get_non_const_children()){
      if(child->get_type()==Node::DataMovementTileNode){
        auto child_node = static_cast<mapping::DataMovementTileNode*>(child);
        auto target = child_node->get_mapping().target_;      
        // if(child->get_type()==Node::DataMovementTileNode && target.find(tid)!=target.end()){
        if(target.find(tid)!=target.end()){
          child_node->has_tensor_from_colop = true;
          child_node->tensor_from_colop = tid;
        }
        set_for_descendants(child, tid);
      }
    }

  }
*/
  bool Node::check_colOp_descendant(Node* node, problem::TensorID tid){

    if(!node) return false;
    if(node->get_type()==Node::CollectiveOperationNode){
      auto colOp_node = static_cast<const mapping::CollectiveOperationNode*>(node);
      if(tid==colOp_node->collective_op_description.in_tensor){
        return true;
      } else {
        return false;
      }
    }
    for(auto child: node->get_non_const_children()){
      check_colOp_descendant(child, tid);
    }
  }

/*
  void Node::mark_nodes_with_colop_tensor(Node* root){
    if(!root) return;
    bool same_targets=false;
    auto data_mov_node = static_cast<mapping::DataMovementTileNode*>(root); // removed const from static_cast to make it modifiable 
    problem::TensorID tid;
    problem::DimensionID dim_id;
    // uint
    std::vector<problem::TensorID> child_wb_tensors;
    for(auto child: root->get_non_const_children()){
      if(child->get_type()==Node::CollectiveOperationNode){
        auto colOp_node = static_cast<const mapping::CollectiveOperationNode*>(child);
        auto child_target = colOp_node->collective_op_description.target;
        
        tid = colOp_node->collective_op_description.in_tensor;
        dim_id = colOp_node->collective_op_description.dimension;

        //check if the target in collective operation is same as the targets in the parent data movement node
        for(auto&[tens_id,arch]:data_mov_node->get_mapping().target_){
          if(arch==child_target){
            same_targets=true; 
            data_mov_node->has_tensor_from_colop=true;
            data_mov_node->tensor_from_colop = tid;
            data_mov_node->duplicate_tensor_exist = true;
            data_mov_node->dim_id_from_colop = dim_id;

            uint64_t factor=1;
            for(auto src:colOp_node->collective_op_description.src){
              factor*=src->getInstanceSize();
            }
            data_mov_node->factor_from_colop = factor;
            // data_mov_node->dim_id_from_colop = dim_id;
            break;
          }
        }
      }

      // even a DatamovementTile child node can do a collective operation at the OB level and write back the tensor
      // if(child->get_type()==Node::DataMovementTileNode && check_colOp_descendant(child, tid)){
      //   auto child_node = static_cast<mapping::DataMovementTileNode*>(child);

      //   for(auto&[tens_id, target]: child_node->get_mapping().target_){
          
      //     if(std::find(child_wb_tensors.begin(), child_wb_tensors.end(), tens_id)){
      //       // if the tensor in the child node is already written back by other 
      //     }


      //     if(child_node->get_mapping().wb_output[tens_id]) child_wb_tensors.push_back(tens_id);
      //   }

      // }

      // this flag can be used by mapper to make sure tensors size is correct in these nodes
      //if parent node and child node have same targets and current child node is a data movement node
      if(same_targets && child->get_type()==Node::DataMovementTileNode) {
        
        // also set the has_tensor_from_colop true for other nodes to the right of collective operation node that have same tensor as the collective operation
        // auto child_node = static_cast<mapping::DataMovementTileNode*>(child);
        auto child_node = static_cast<mapping::DataMovementTileNode*>(child);
        auto target = child_node->get_mapping().target_;
        if(target.find(tid)!=target.end()){//if tensor from collective op exist in the child node
          child_node->has_tensor_from_colop = true;
          child_node->tensor_from_colop = tid;

          set_for_descendants(child, tid);//if descendants have this tensor set the has_tensor .. variable true for them

        }
      }

      //recursively apply to each datamovementtile child node
      if(child->get_type()==Node::DataMovementTileNode) mark_nodes_with_colop_tensor(child);
    }
    
  }

*/
  void Visitor::visitDataMovementTileNode(const DataMovementTileNode* node){
    for(auto child: node->children_){
      child->accept(this);
    }
  }

  void Visitor::visitCollectiveOperationNode(const CollectiveOperationNode* node){
    return;
  }

  void Visitor::visitInterTileBinding(const InterTileBinding* node){
    return;
  }

  void Visitor::visitOperationNode(const OperationNode* node){
    return;
  }

  void Visitor::run(const Node* root){
    return;
  }


  // struct loopNest_tensorUpdated{
  //   std::vector<LoopNode> loop_node;
  //   std::vector<problem::TensorID> tensors_updated;
  // };

  // std::vector<LoopNode> DataMovementTileNode::constructLoopNest(std::vector<stride_tilestep>& stride_tilesteps, std::vector<problem::TensorID>& tensors_updated) const{






  LoopNode DataMovementTileNode::constructLoopNest(stride_tilestep& stride_tilesteps, std::vector<problem::TensorID>& tensors_updated, bool duplicate_tensor, problem::Workloads& workloads, const TensorSizeMap& tensor_size) const{
    LoopNode cur_node;

    for (auto tens_cnt=0; tens_cnt<level_to_tileMapping_.tensors.size(); tens_cnt++){

      auto tens_id = level_to_tileMapping_.tensors[tens_cnt];
      LoopNestDescriptor desc_vec;
      uint32_t                  iteration_count=1;
      uint32_t                  sp_iteration_count=1;

      auto tag           = level_to_tileMapping_.tags[tens_cnt];
      auto prev_tilestep = stride_tilesteps.tilesteps[tens_id][tag];

      bool tensor_rw_set = false;
      for (auto workload:workloads.workloads_){
          auto op_name = workload.first;
          auto tensor_map = workload.second->get_TensorNameMap();
          if(tensor_map.find(tens_id)!=tensor_map.end() && op_name==tag){
              cur_node.tensor_is_rw.emplace_back(workload.second->isTensorRW(tens_id)); //FIXME::snegi tensorRW might be overwritten by the next operations RW value, maybe add a tag
              tensor_rw_set = true;
          }

          // // fill in the dependent tensor
          // auto global_tensors = workloads.tensorNames_;
          // //if tens_id is not in the global_tensors then it is an intermediate tensor and it is the input of the operation that it is tagged to then we make dependent tensor as true
          // auto input_tensor = workloads.workloads_[tag]->ins_;
          // auto tens_is_input = std::find(input_tensor.begin(), input_tensor.end(), tensor_map[tens_id])!=input_tensor.end();
          // //if the tensor is a dependent tensor that the read-write latency cannot be hidden
          // //FIXME::snegi check if it has to be always the input tensor or even output tensor will have this compulsory stall
          // // if(global_tensors.find(tens_id)==global_tensors.end()&&tens_is_input){
          // if(global_tensors.find(tens_id)==global_tensors.end()){
          //   cur_node.dependent_tensor.emplace_back(true);
          // } else {
          //   cur_node.dependent_tensor.emplace_back(false);
          // }

      }  

      if(!tensor_rw_set){
        // this is from collective operation
        cur_node.tensor_is_rw.emplace_back(true);// if a tensor is tagged with a collective operation that means that tensor is a read-write tensor
      }    
      
      auto intermediate_tensors = workloads.intermediate_tensors;
      if((std::find(intermediate_tensors.begin(), intermediate_tensors.end(), tens_id)!=intermediate_tensors.end())){
        cur_node.dependent_tensor.emplace_back(true);
      } else {
        cur_node.dependent_tensor.emplace_back(false);
      }


      for(auto sp_idx=0; sp_idx<level_to_tileMapping_.tilings_[tens_cnt].size(); sp_idx++){
        std::map<problem::DimensionID, analysis::LoopDescriptor> desc_;

        for(auto dim_val:level_to_tileMapping_.tilings_[tens_cnt][sp_idx]){ // dim_id, val

          desc_[dim_val.first] = {.end = stride_tilesteps.stride[tens_id][tag][dim_val.first]*dim_val.second, .stride=stride_tilesteps.stride[tens_id][tag][dim_val.first]};

          stride_tilesteps.stride[tens_id][tag][dim_val.first]*=dim_val.second;

          if(level_to_tileMapping_.type == mapping::mapping_t::TEMPORAL){
            stride_tilesteps.tilesteps[tens_id][tag] *=dim_val.second;
            iteration_count *= dim_val.second;
          } else{
            //spatial iteration count
            sp_iteration_count *=dim_val.second;
          }
        }
        desc_vec.emplace_back(desc_);
      }
      
      cur_node.descriptor_.emplace_back(desc_vec);
      cur_node.relative_timestep_to_dependent.emplace_back(prev_tilestep);
      cur_node.iteration_count.emplace_back(iteration_count);
      cur_node.sp_iteration_count.emplace_back(sp_iteration_count);
      cur_node.arch_idx[level_to_tileMapping_.target_[tens_cnt]].emplace_back(tens_cnt);
      

    }
    cur_node.tid_idx   = level_to_tileMapping_.tid_idx;
    
    cur_node.tensors   = level_to_tileMapping_.tensors;
    cur_node.order     = level_to_tileMapping_.order;
    cur_node.type      = level_to_tileMapping_.type;
    cur_node.target    = level_to_tileMapping_.target_;
    cur_node.child     = level_to_tileMapping_.child_;
    cur_node.wb_output = level_to_tileMapping_.wb_output;
    cur_node.scale     = level_to_tileMapping_.scale;
    cur_node.rmw       = level_to_tileMapping_.rmw;
    cur_node.tags      = level_to_tileMapping_.tags;

    return cur_node;
  }

    //this function finds the binding associated with this node
    InterTileBinding::type_t find_binding(const Node* node){

        if(node==nullptr) return InterTileBinding::type_t::none;

        auto parent_node = node->get_parent();

        if(parent_node==nullptr) return InterTileBinding::type_t::none;

        auto num_children = parent_node->get_children().size();

        //if the parent node has only single parent
        if(num_children==1) return InterTileBinding::type_t::none; //return none if the parent has only one children
        

        const Node* left_node      = nullptr;
        const Node* left_of_left_node = nullptr;
        size_t cnt=0;
        for(auto& child: parent_node->get_children()){

            if(child==node && cnt==num_children-1){
                // if child is equal to the node and this is the last child then return the type of left_left node because left node will be the data movement node
                auto binding_node = static_cast<const InterTileBinding*>(left_of_left_node);
                return binding_node->type;
            }

            if(child==node){
                auto binding_node = static_cast<const InterTileBinding*>(left_node);
                return binding_node->type;
            }

            left_of_left_node = left_node;
            left_node         = child;
            cnt++;
        }
    }



  void Validate::visitDataMovementTileNode(const DataMovementTileNode* node){

    for(auto child:node->get_children()){
        child->accept(this);
    }


    if(node->get_parent()==nullptr){ //base case is when we are at root node
      // tensor_sizes[node]
      // auto mapping = node->get_mapping();
      // for(auto& [tid, target]: mapping.target_){

      auto tilemapping= node->get_mapping();
      //calculate tensor sizes
      for(auto tens_cnt=0; tens_cnt<tilemapping.tensors.size(); tens_cnt++){
        for(auto sp_idx=0; sp_idx<tilemapping.tilings_[tens_cnt].size(); sp_idx++){
          for(auto dim_val:tilemapping.tilings_[tens_cnt][sp_idx]){
            auto tag = tilemapping.tags[tens_cnt];
            auto tid = tilemapping.tensors[tens_cnt];
            tensor_sizes[tid][tag][dim_val.first] *=dim_val.second;
          }

        }
      }

      // for(auto&[tid, target]: tilemapping.target_){
      //   node_level_tensor_sizes[node][tid] = tensor_sizes[tid];
      // }



      // tensor_sizes[node][target->getName()]=
      // problem::DimSizeExpression full_rank_expression;

      auto& dim_sizes = workloads_.workloadsDimSizes_; //dimid, size map, contains all dimensions

      //basic check if the dimension sizes matches the given dimension sizes
      // for(auto&[tid, dim_val]:tensor_sizes){
      //FIXME::snegi add something so that this size check can be done at every mapping node
      for(auto cnt=0; cnt<tilemapping.tensors.size(); cnt++){
        auto tid = tilemapping.tensors[cnt];
        auto tag = tilemapping.tags[cnt];
        auto tilingmatrix = tensor_sizes[tid][tag];

        for(auto&[dim,val]: tilingmatrix){
          if(val!=dim_sizes[dim]){

            std::stringstream ss;
            ss<<"Dimension "<<workloads_.getDimName(dim) << " size does not match"<<" in operation "<<tag <<" node "<<node->get_name();

            COMET_ASSERT(false, ss.str());
          }
        }
      }

      // for(auto&[tid, tag_size]: tensor_sizes){
      //   if(std::find(tilemapping.tensors.begin(), tilemapping.tensors.end(), tid)==tilemapping.tensors.end()) continue; //this tensor doesn't exist at this level
      //   for(auto&[op_name, tilingmatrix]: tag_size){
      //     if(std::find(tilemapping.tags.begin(), tilemapping.tags.end(), op_name)==tilemapping.tags.end()) continue;// if tag doesn't match that means this tensor doesn't exist here
      //     for(auto&[dim,val]: tilingmatrix){
      //       if(val!=dim_sizes[dim]){

      //         std::stringstream ss;
      //         ss<<"Dimension "<<workloads_.getDimName(dim) << " size does not match";

      //         COMET_ASSERT(false, ss.str());
      //       }
      //     }
      //   }
      // }

    } else {
      //for non root node
      auto tilemapping= node->get_mapping();
      //calculate tensor sizes
      for(auto tens_cnt=0; tens_cnt<tilemapping.tensors.size(); tens_cnt++){
        for(auto sp_idx=0; sp_idx<tilemapping.tilings_[tens_cnt].size(); sp_idx++){
          for(auto dim_val:tilemapping.tilings_[tens_cnt][sp_idx]){
            auto tag = tilemapping.tags[tens_cnt];
            auto tid = tilemapping.tensors[tens_cnt];
            tensor_sizes[tid][tag][dim_val.first] *=dim_val.second;
          }

        }
      }
      if(tilemapping.type==mapping_t::SPATIAL) return; // update occupancy only at the temporal node

      auto binding = find_binding(node);
      if(binding==InterTileBinding::type_t::none){
        //if binding is none, then maybe this parent only has single children so check the binding of the parent. 2 GEMMs without any 
        binding = find_binding(node->get_parent());
      }

      std::unordered_map<std::string, uint64_t> node_occupancy;
      
      std::map<TensorID, std::map<std::string, uint64_t>> node_occupancy_tid;

      // calculate the occupancy of this archLevel
      // std::vector<TensorID> tensors_seen;
      for(int cnt=0; cnt<tilemapping.target_.size(); cnt++){
        
        // get size after projecting multi-dimensional tensor on the relavant dimensions
        auto& tag = tilemapping.tags[cnt];
        auto& tid = tilemapping.tensors[cnt];
        // if(std::find(tensors_seen.begin(), tensors_seen.end(), tid)!=tensors_seen.end()) continue; // this tensor size has already been added in the previous iteration

        auto& workload = workloads_.workloads_[tag];
        auto tensor_size = workload->projectDimExpressionOnTensor(tensor_sizes[tid][tag], tid);
        auto arch_level = tilemapping.target_[cnt];
        auto scale = (float)tilemapping.scale[cnt]/8.0; //converting bits to bytes
        //FIXME::snegi add condition for duplicate tensors 
        auto arch_level_name = arch_level->getName();

        // node_occupancy[arch_level_name] += tensor_size.resolve()*scale;

        if(node_occupancy_tid.count(tid) && node_occupancy_tid[tid].count(arch_level_name)){
          //tid and arch_level exist so take max
          node_occupancy_tid[tid][arch_level_name] = std::max((double)node_occupancy_tid[tid][arch_level_name], tensor_size.resolve()*scale);
        } else{
          node_occupancy_tid[tid][arch_level_name] += tensor_size.resolve()*scale;
        }
        // tensors_seen.push_back(tid);
      }

      for(auto&[tid, arch_level_map]: node_occupancy_tid){
        for(auto&[arch_level_name, size]: arch_level_map){
          node_occupancy[arch_level_name] += size;
        }
      }

      for(auto&[arch_level_name, size]:node_occupancy){
        // only update the occupancy at the temporal nodes
        if(arch_level_occupancy.find(arch_level_name)!=arch_level_occupancy.end() && binding != InterTileBinding::type_t::none){
          // this arch_level exist in the occupancy map
          // get the binding at this level, if there is no binding at this level get the binding of the parent

          if(binding == InterTileBinding::type_t::sequential){
            arch_level_occupancy[arch_level_name] = std::max(arch_level_occupancy[arch_level_name], (uint64_t)size);
          }
          else if (binding == InterTileBinding::type_t::pipeline){
            arch_level_occupancy[arch_level_name] = arch_level_occupancy[arch_level_name] + size;
          }
        } else {
          arch_level_occupancy[arch_level_name] +=size; // this is just the number of elements, if the arch_level doesn't exist map is initialized with zero and then the  tensor_size.resolve is added to the map
        }        

      }

      // for(auto&[tid, target]: tilemapping.target_){
      //   node_level_tensor_sizes[node][tid] = tensor_sizes[tid];
      // }

      // if the node is a temporal node calculate the total size occupied by the tensors 
      // consider binding and write back


    }



  }

  void Validate::visitOperationNode(const OperationNode* node){

    auto common_attributes = node->get_common_attributes();

    auto op_name = common_attributes.op_name_;

    if(node->get_common_attributes().type_==operation_t::GEMM){
      auto compute_map = node->get_compute_mapping();

      for(const auto& pair: compute_map.tilesizes){
        tensor_sizes[pair.first][op_name] = pair.second;
      }
    }
    // else if(node->get_common_attributes().type_==operation_t::SOFTMAX){
    else {
      //all other operations 
      auto workload = node->get_workload(); 
      auto in_tid   = workload->getTensorID(workload->ins_.front());//softmax has only one input
      auto out_tid = workload->getTensorID(workload->out_);
      //only initialize the size of output tensor from non-gemm operation nodes

      // if(tensor_sizes.size()==0){
      //   for(auto&[tid, ts]:common_attributes.tilesizes){
      //     tensor_sizes[tid] = ts;
      //   }
      // } else{
      //   tensor_sizes[out_tid] = tensor_sizes[in_tid];
      // }
      // for (auto& [tid, ts]: common_attributes.tilesizes){
      //   if(tid==out_tid){
      //     // tensor_sizes[tid] = ts;
      //     tensor_sizes[tid] = tensor_sizes[in_tid];
      //   }
      // }

      for(auto&[tid, ts]:common_attributes.tilesizes){
        tensor_sizes[tid][op_name] = ts;
      }


    }

  }

  ComputeTilingMatrix Validate::find_tensor_size(const Node* node, TensorID tid){

    auto parent_node = node->get_parent();
    ComputeTilingMatrix retval;

    for(auto& c: parent_node->get_children()){
      if(retval.size()!=0) break; //already found the size of tensor in prev iteration
      if(c==node) break;
      if(c->get_type()==Node::CollectiveOperationNode && c!=node) continue;
      if(c->get_type()==Node::InterTileBinding)continue;

      auto data_mov_node = static_cast<const DataMovementTileNode*>(c);

      auto tensors = data_mov_node->get_mapping().tensors;

      if(std::find(tensors.begin(), tensors.end(), tid)!=tensors.end()){
        //found the tensor

        for(auto&[tag, size]: tensor_sizes[tid]){
          //before the collective operation all nodes should have the same tensor size
          retval = size;
          break;
        }

      }

    }

    COMET_ASSERT(retval.size()!=0, "could not find tensor produced on the left of this node");

    return retval;


  }

  void Validate::visitCollectiveOperationNode(const CollectiveOperationNode* node){


    auto col_op_desc = node->collective_op_description;

    auto in_tensor_size = find_tensor_size(node, col_op_desc.in_tensor);

    ComputeTilingMatrix out_tensor_size = in_tensor_size;

    uint64_t num_src_devices=1;
    for (auto src: col_op_desc.src){
        num_src_devices*=src->getInstanceSize(); // number of source devices from which the data is collected from
    }
    uint64_t num_dest_devices=1;
    for (auto dest: col_op_desc.dest){
        num_dest_devices*=dest->getInstanceSize();
    }

    if(col_op_desc.type_==stype_t::GATHER || col_op_desc.type_==stype_t::ALLGATHER){

      for(auto&[dimid,val]: out_tensor_size){
        if(col_op_desc.dimension==dimid){
          val*=num_src_devices;
        }
      }
    } else if(col_op_desc.type_==stype_t::SCATTER){

      for(auto&[dimid,val]: out_tensor_size){
        if(col_op_desc.dimension==dimid){
          val/=(float)num_dest_devices;
        }
      }

    }
    //  else if(col_op_desc.type_==stype_t::REDUCTION || col_op_desc.type_==stype_t::ALLREDUCE){ //reduction should not change the size

    //   for(auto&[dimid, val]: out_tensor_size){
    //     if(col_op_desc.dimension==dimid){
    //       val*=num_src_devices;
    //     }
    //   }

    // }


    tensor_sizes[col_op_desc.in_tensor][col_op_desc.tag] = out_tensor_size;

    return;
    // auto col_op_desc = node->collective_op_description;
    // auto in_tensor_size = tensor_sizes[col_op_desc.in_tensor];

    //   uint64_t num_src_devices=1;
    //   for (auto src: col_op_desc.src){
    //       num_src_devices*=src->getInstanceSize(); // number of source devices from which the data is collected from
    //   }
    //   uint64_t num_dest_devices=1;
    //   for (auto dest: col_op_desc.dest){
    //       num_dest_devices*=dest->getInstanceSize();
    //   }

    //   if(col_op_desc.type_==stype_t::GATHER || col_op_desc.type_==stype_t::ALLGATHER){

    //     for(auto& [dim, size]: in_tensor_size){
    //       if(dim==col_op_desc.dimension){
    //         tensor_sizes[col_op_desc.in_tensor][dim] *=num_src_devices;
    //       }
    //     }
    //   }
      // else if(col_op_desc.type_==stype_t::BROADCAST || col_op_desc.type_==stype_t::SCATTER){
      //   for(auto& [dim, size]: in_tensor_size){
      //     if(dim==col_op_desc.dimension){
      //       tensor_sizes[col_op_desc.in_tensor][dim] /=num_src_devices;
      //     }
      //   }
      // }
      // broadcast or scatter does not change the tensor size ---> 
      // REDUCE or ALL-REDUCE does not change the tensor size

  }

  void Validate::fitsInMemory(){

    // for(auto const& arch_name: topology_.getLevels()){
    //   if(arch_name == "DRAM") continue; // assuming everything can fit in DRAM
    //   auto capacity = topology_.getArchLevel(arch_name)->instanceCapacity();
    //   COMET_ASSERT(arch_level_occupancy[arch_name] <= capacity, "Tensors do not fit in " << arch_name << " Capacity: " << capacity << " Bytes, Requested: " << arch_level_occupancy[arch_name] << " Bytes");
                                                                                                                                                       
    // }
    std::ostringstream error_msg;
    bool all_fit = true;

    for (const auto& arch_name : topology_.getLevels()) {
        if (arch_name == "DRAM") continue; // assuming everything can fit in DRAM

        auto capacity = topology_.getArchLevel(arch_name)->instanceCapacity();
        auto requested = arch_level_occupancy[arch_name];

        if (requested > capacity) {
            all_fit = false;
            error_msg << "Tensors do not fit in " << arch_name 
                      << " | Capacity: " << capacity << " Bytes"
                      << " | Requested: " << requested << " Bytes\n";
        }
    }
    // Single assert after checking all levels
    COMET_ASSERT(all_fit, error_msg.str());


  }

  // std::string LevelTileMapping::print(bool verbose) const { 
  //   std::string debug_statement="";
  //   if (!verbose) { 
  //     std::string type_s = (type == mapping_t::SPATIAL) ? "Spatial" : "Temporal";
  //     std::string child_s = (child_ == nullptr) ? "Null" : child_->getName();
  //     debug_statement += type_s + " tile is mapped at " + target_->getName() + " with child " + child_s + ".";
  //     debug_statement += "Loop Coordinates are \n";
  //     for (auto& mesh : tilings_) {
  //       auto dim_id = 0;
  //       for (auto& coordinate: mesh) { 
  //         debug_statement += "Dimension:" + std::to_string(dim_id) + " -> [0," + std::to_string(coordinate) + "), ";
  //         dim_id++;
  //       }
  //       debug_statement += "\n";
  //     }
  //   } // TODO:: move to fmt and hav verbose
  //   return debug_statement;
  // }


  
}
