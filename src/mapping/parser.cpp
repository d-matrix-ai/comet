#include <regex>
#include <numeric>
#include "mapping/mapping.hpp"
#include "util/logger.hpp"

namespace mapping {
  int counter=0;
  void tolower(std::string& str){
      std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {return std::tolower(c);});
  }

  Node::Node(Node::type_t t, config::CompoundConfigNode config):type_(t){
    name_ = type2name_.at(type_);
    name_ += "::";
    name_ += std::to_string(counter);
    counter++;
  }


  Node* Mapping::recursiveParseMapping(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants) {
    std::string node_type;

    if(!config.lookupValue("node_type", node_type)){
      COMET_ERROR("node-type is not specified in mapping file");
    }
    tolower(node_type);
    std::cout<<"**** Creating "<<node_type<<std::endl;
    Node* node = nullptr;
    if (node_type == "datamovement_tile"){
      // std::cout<<"before"<<std::endl;
      // std::cout<<node<<std::endl;
      node = new DataMovementTileNode(config, topology, workloads, constants);
      // std::cout<<node<<std::endl;
      // std::cout<<"after"<<std::endl;

    }
    else if(node_type == "operation"){
      node = new OperationNode(config, topology, workloads, constants);
    }
    else if(node_type == "inter_tile_binding"){
      node = new InterTileBinding(config);
    }
    else if (node_type=="collectiveoperation"){
      node = new CollectiveOperationNode(config, topology, workloads);
    }
    else {
      COMET_ERROR(node_type << " is not a valid type****");
    }
    std::cout<<"**** Finished "<<node_type<< " " << node->get_name()<<std::endl;

    assert(node != nullptr);

    if (config.exists("subtree")){
      config = config.lookup("subtree");
      if (config.isList()){
        for (int i=0; i<config.getLength(); i++){
          node->add_child(recursiveParseMapping(config[i], topology, workloads, constants));
        }
      }
      else {
        node->add_child(recursiveParseMapping(config, topology, workloads, constants));
      }
    }
    
    return node;


  }
  
  arch::ArchLevel* GetTilingLevel(config::CompoundConfigNode loop_node, arch::Topology& topology, std::string find_string) { 
    std::string target;
    COMET_ASSERT(loop_node.lookupValue(find_string, target), "LOOP NODE HAS NO Target");
    return topology.getArchLevel(target);  
  }

  arch::ArchLevel* GetChild(config::CompoundConfigNode loop_node, arch::Topology& topology, arch::ArchLevel* target) { 
    std::string child;
    if (loop_node.lookupValue("child", child)) { 
      bool ischild_or_same = target->isChild(child);
      ischild_or_same |= (target->getName() == child);
      COMET_ASSERT(ischild_or_same, "Loop Node has an invalid child " << child << " for given target " << target->getName()); 
      return topology.getArchLevel(child);
    }else return nullptr;
  }

  struct tensor_memory_struct{
    std::vector<problem::TensorID> global_tensors;
    std::vector<problem::TensorID> intermediate_tensors;
    std::vector<std::pair<arch::ArchLevel*, arch::ArchLevel*>> target_child_global;
    std::vector<std::pair<arch::ArchLevel*, arch::ArchLevel*>> target_child_intermediate;
  };
  
  bool isStringInVector(const std::vector<std::string>& vec, const std::string& str){
    return std::find(vec.begin(), vec.end(), str) != vec.end();
  }
  // tensor_memory_struct GetTensorsInTile(config::CompoundConfigNode loop_node, problem::Workloads& workloads, arch::Topology& topology, bool get_target_child) { 
  //   auto tensors = loop_node.lookup("tensors");
  //   tensor_memory_struct all_tensors_memory;
  //   if (tensors.isArray()) { 
  //     std::vector<problem::TensorID> tensors(0);
  //     std::vector<std::string> tensor_strings;
  //     loop_node.lookupArrayValue("tensors", tensor_strings);
  //       std::vector<std::string> target_strings;
  //       std::vector<std::string> child_strings;
        
  //     if (get_target_child){
  //       loop_node.lookupArrayValue("target", target_strings);
  //       loop_node.lookupArrayValue("child", child_strings);
  //     }
  //     int cnt=0;
  //     for (auto& tens : tensor_strings) {
  //       if (isStringInVector(workloads.ins_, tens)||isStringInVector(workloads.outs_, tens)){
  //         //tensor is defined globally
  //         all_tensors_memory.global_tensors.emplace_back(workloads.getTensorID(tens));
          
  //         if(get_target_child){
  //           auto target=topology.getArchLevel(target_strings[cnt]);
  //           auto child=topology.getArchLevel(child_strings[cnt]);
  //           all_tensors_memory.target_child_global.emplace_back(std::make_pair(target,child));
  //         }
  //       } 
  //       else {
  //         //look tens belongs to which workload
  //         // for (auto& w : workloads.workloads_){
  //         //   if (isStringInVector(*w.second.ins_, tens)||isStringInVector(*w.second.outs_,tens)){
  //         //     all_tensors.intermediate_tensors[*w.second.get_name()].emplace_back(*w.second.getTensorID(tens));
  //         //   }
  //         // }
  //         all_tensors_memory.intermediate_tensors.emplace_back(workloads.getTensorID(tens));

  //       }
  //     }
  //     return all_tensors_memory;
  //   }
  // }

  struct tensor_arch_level_struct{
    std::vector<problem::TensorID> tensors;
    std::vector<bool> wb_outputs;
    std::vector<bool> rmw;
    std::vector<std::string> tags;
    std::vector<uint8_t> scale; 
    // std::vector<std::pair<arch::ArchLevel*, arch::ArchLevel*>> target_child;
    std::vector<ArchLevel*> target;
    std::vector<ArchLevel*> child;

  };

  tensor_arch_level_struct GetTensorsInTile(config::CompoundConfigNode loop_node, problem::Workloads& workloads, arch::Topology& topology, bool get_target_child) { 
    auto tensors = loop_node.lookup("tensors");

    tensor_arch_level_struct all_tensors_archlevel;
    if (tensors.isArray()) { 
      // std::vector<problem::TensorID> tensors(0);
      std::vector<std::string> tensor_strings;
      loop_node.lookupArrayValue("tensors", tensor_strings);
      std::vector<std::string> target_strings;
      std::vector<std::string> child_strings;

      bool wb_exist=false;
      all_tensors_archlevel.wb_outputs.resize(tensor_strings.size(),false);
      if(loop_node.exists("wb_output")){
        all_tensors_archlevel.wb_outputs=loop_node.get<std::vector<bool>>("wb_output");
        wb_exist=true;
      }

      bool rmw_exist=false;
      all_tensors_archlevel.rmw.resize(tensor_strings.size(),false);
      if(loop_node.exists("rmw")){
        all_tensors_archlevel.rmw=loop_node.get<std::vector<bool>>("rmw");
        rmw_exist=true;
      }

      bool scale_exist=false;
      all_tensors_archlevel.scale.resize(tensor_strings.size(),1);
      if(loop_node.exists("precision")){
        all_tensors_archlevel.scale=loop_node.get<std::vector<uint8_t>>("precision");
        scale_exist=true;
      }
      bool tag_exist=false;
      // std::vector<std::string> tags;
      all_tensors_archlevel.tags.resize(tensor_strings.size(),"None");
      if(loop_node.exists("tag")){
        all_tensors_archlevel.tags=loop_node.get<std::vector<std::string>>("tag");
        tag_exist=true;
      }
        
      if (get_target_child){
        loop_node.lookupArrayValue("target", target_strings);
        loop_node.lookupArrayValue("child", child_strings);
        COMET_ASSERT(tensor_strings.size()==target_strings.size(), "Mapping should have same number of tensors and targets");
        COMET_ASSERT(tensor_strings.size()==child_strings.size(), "Mapping should have same number of tensors and children");
      }

      int cnt=0;
      for (auto& tens : tensor_strings) { 
        auto tid=workloads.getTensorID(tens);
        all_tensors_archlevel.tensors.emplace_back(tid);
        // if (wb_exist){
        //   all_tensors_archlevel.wb_outputs[tid] = wb_outputs[cnt];
        // }
        // if (rmw_exist){
        //   all_tensors_archlevel.rmw[tid] = rmw[cnt]; 
        // }
        // if(tag_exist){
        //   all_tensors_archlevel.tags[tid] = tags[cnt];
        //   tolower(all_tensors_archlevel.tags[tid]);
        // }
        // if(scale_exist){
        //   all_tensors_archlevel.scale[tid] = scale[cnt];
        // }
        if(get_target_child){
          auto target=topology.getArchLevel(target_strings[cnt]);
          auto child=topology.getArchLevel(child_strings[cnt]);
          all_tensors_archlevel.target.emplace_back(target);
          all_tensors_archlevel.child.emplace_back(child);
          // all_tensors_archlevel.target_child.emplace_back(std::make_pair(target,child));
        }
        cnt++;        
      }

      //reference size
      auto ref_size = all_tensors_archlevel.tensors.size();
      if(all_tensors_archlevel.target.size()!=ref_size && get_target_child){
        COMET_ASSERT(false, "Target vec is not same size as tensor vec");
      } else if(all_tensors_archlevel.tags.size()!=ref_size){
        COMET_ASSERT(false, "Tags vec is not same size as tensor vec");
      } else if(all_tensors_archlevel.child.size()!=ref_size && get_target_child){
        COMET_ASSERT(false, "Child vec is not same size as tensor vec");
      } else if(all_tensors_archlevel.scale.size()!=ref_size){
        COMET_ASSERT(false, "Scale vec is not same size as tensor vec");
      } else if(all_tensors_archlevel.rmw.size()!=ref_size){
        COMET_ASSERT(false, "RMW vec is not same size as tensor vec");
      } else if(all_tensors_archlevel.wb_outputs.size()!=ref_size){
        COMET_ASSERT(false, "Wb_output vec is not same size as tensor vec");
      }


      return all_tensors_archlevel;
    }


    // else { 
    //   std::string tensor_string;
    //   COMET_ASSERT(loop_node.lookupValue("tensors", tensor_string), "unable to parse tensor strings");
    //   COMET_ASSERT(tensor_string == "all", "CURRENTLY COMET only supports all as a tensor string in the tensor directive");
    //   return workload.getTensors();
    // }
  }

  std::vector<arch::ArchLevel*> getsrc_dest(config::CompoundConfigNode config, std::string src_dest_string, arch::Topology& topology){
    std::vector<std::string> src_dest;
    config.lookupArrayValue(src_dest_string, src_dest);

    std::vector<arch::ArchLevel*> sd_vec;
    // auto target=topology.getArchLevel(target_strings[cnt]);
    for(auto sd: src_dest){
      sd_vec.emplace_back(topology.getArchLevel(sd));
    }
    return sd_vec;

  }

  std::vector<std::string> splitString(const std::string& str, char delimiter) {
      std::vector<std::string> result;
      std::istringstream ss(str);
      std::string token;
      while (std::getline(ss, token, delimiter)) {
          token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end()); //remove space from any factor that is read
          result.push_back(token);
      }
      return result;
  }

  ComputeTilingMatrix parseComputeTilingCoordinates(const std::string& tiling_string, problem::Workloads& workloads, config::CompoundConfigNode constants) {
      // std::map<std::string, std::vector<std::string>> tiling_map;
      ComputeTilingMatrix tiling_map;
      // std::regex re(R"(\s*([A-Z]+)\s*=\s*\[([^\]]+)\])");
      // std::regex re(R"(([A-Z]+)=([A-Z_]+)\s*([A-Z]+)=([A-Z_]+)\s*([A-Z]+)=([A-Z_]+))");
      // std::regex re(R"(([A-Z]+)=([A-Z_]+|\d+))");
      std::regex re(R"((.)=([^ ]*))");
      std::smatch sm;
      std::string::const_iterator searchStart(tiling_string.cbegin());
      
      while (std::regex_search(searchStart, tiling_string.cend(), sm, re)) {
          std::string key = sm[1];
          std::string value_string = sm[2];

          uint32_t factor_value;
          if (constants.exists(value_string)){
            constants.lookupValue(value_string, factor_value);
          } else {
            factor_value=std::stoul(value_string);
          }

          tiling_map[workloads.getDimID(key)] = factor_value;
          searchStart = sm.suffix().first;
      }

      return tiling_map;
  }


  TilingMatrixMap parseTilingCoordinates(const std::string& tiling_string, problem::Workloads& workloads, config::CompoundConfigNode constants) {
      // std::map<std::string, std::vector<std::string>> tiling_map;
      TilingMatrixMap tiling_map;
      std::regex re(R"(\s*([A-Z]+)\s*=\s*\[([^\]]+)\])");
      std::smatch sm;
      std::string::const_iterator searchStart(tiling_string.cbegin());
      
      while (std::regex_search(searchStart, tiling_string.cend(), sm, re)) {
          std::string key = sm[1];
          std::string value_string = sm[2];
          std::vector<std::string> values = splitString(value_string, ',');
          problem::DimSizeExpression factor_array(values.size(),0);
          uint8_t id=0;
          for(auto v:values){
            uint32_t end;
            if (constants.exists(v)){
              constants.lookupValue(v, end);
            } else {
              end=std::stoul(v);
            }
            factor_array[id]=end;
            id++;
          }
          tiling_map[workloads.getDimID(key)] = factor_array;
          searchStart = sm.suffix().first;
      }

      return tiling_map;
  }

  SpatialTilingMatrixMap GetTileFactors(config::CompoundConfigNode loop_node, problem::Workloads& workloads, config::CompoundConfigNode constants) {
    auto factors = loop_node.lookup("factors");
    COMET_ASSERT(factors.isArray(), "COMET currently requires factors to be listed");
    auto factor_lengths = factors.getLength();
    // TilingMatrixMap tiling_matrix;
    SpatialTilingMatrixMap tiling_matrix(factor_lengths);
    for (auto fac=0; fac< factor_lengths; fac++) {
      
      auto fac_term = factors[fac].resolve();

      // std::map<std::string, std::vector<std::string>> temp= parseTilingCoordinates(fac_term);

      tiling_matrix[fac] = parseTilingCoordinates(fac_term, workloads, constants);
    }
    return tiling_matrix;
  }



  problem::LoopOrder GetLoopOrder(config::CompoundConfigNode loop_node, problem::Workloads& workloads, config::CompoundConfigNode constants) { 
   problem::LoopOrder order(0);
   std::string order_buffer;
   std::string value_string;

          // std::string value_string = sm[2];

          // uint32_t factor_value;
          // if (constants.exists(value_string)){
          //   constants.lookupValue(value_string, factor_value);
          // } else {
          //   factor_value=std::stoul(value_string);
          // }
    loop_node.getValue<std::string>("permutation", value_string);

    if(constants.exists(value_string)){
      //permutation is a parameter
      constants.lookupValue(value_string, order_buffer);
    } else{
      order_buffer = value_string;
    }

    std::istringstream iss(order_buffer);
    char token;
    while (iss >> token)
    {
      auto dimension = workloads.getDimID(std::string(1, token)); // note: can fault.
      order.push_back(dimension);
    }
    return order;
  }

  //  if (loop_node.getValue<std::string>("permutation", order_buffer)) {
  //    std::istringstream iss(order_buffer);
  //    char token;
  //    while (iss >> token)
  //    {
  //      auto dimension = workloads.getDimID(std::string(1, token)); // note: can fault.
  //      order.push_back(dimension);
  //    }
  //    return order;
  //  } else {
  //    order.resize(workloads.getDimensionCount());
  //    std::iota(std::begin(order), std::end(order), 0); //fills a range of elements with sequentially increasing values, (::begin ->iterator pointing to beginning of the range to be filled)
  //    return order;
  //  }
  // }

  // struct tensor_target_child_struct{
  //   tensor_struct tensors;
  //   std::vector<arch::ArchLevel*> targets;
  //   std::vector<arch::ArchLevel*> childs;
  // };

  // tensor_target_child_struct GetTensorTargetChild(config::CompoundConfigNode config, arch::Toplogy& topology, problem::Workloads& workloads){
  //   tensor_target_child_struct 
  // }

  DataMovementTileNode::DataMovementTileNode(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants):Node(Node::DataMovementTileNode, config){
    // return;
    std::string type;
    assert(config.lookupValue("type", type));
    COMET_LOG(logger::DEBUG, "{} Mapping", type);
    // type_ = type;
    // auto target = GetTilingLevel(config, topology);
    // auto child = GetChild(config, topology, target);
    auto all_tensors = GetTensorsInTile(config, workloads, topology, true);

    auto factors = GetTileFactors(config, workloads, constants);
    auto order = GetLoopOrder(config, workloads, constants);//vector of dimensionID

    // std::cout<<"loop starting"<<std::endl;

    COMET_ASSERT(factors.front().size()==order.size(), "Number of dimensions in factor should match the number of dimensions in permutation");

    //also check that factors are defined for each tensor
    for (auto factor_spatial_dim: factors){
      for(auto&[dim,val]:factor_spatial_dim){
        COMET_ASSERT(val.size()==all_tensors.tensors.size(), "Factor should be specified for each tensor");
      }
    }

    //loop over all the global tensors
    int tens_cnt=0;
    // std::map<problem::TensorID, LevelTileMapping> tile;
    LevelTileMapping tile;
    //initialize tilings_ vector of size number of spatial dimensions
    tile.target_ = all_tensors.target;
    tile.child_ = all_tensors.child;
    tile.tensors = all_tensors.tensors;
    tile.wb_output = all_tensors.wb_outputs;
    tile.rmw = all_tensors.rmw;
    tile.tags = all_tensors.tags;
    tile.scale = all_tensors.scale;

    // tile.tilings_.resize(factors.size());
    for (auto tens: all_tensors.tensors){

      // LevelTileMapping tile;
      // TilingMatrix factor_per_tensor(order.size()); ===> this is wrong we need a map for factors dimensionID (key) -> Tilingvalue (value)
      std::vector<ComputeTilingMatrix> spatial_factor_per_tensor; //FIXME::snegi maybe change this to unordered map because the dimension indices are coming in order after insertion
      
      // level_to_tileMapping_global_tensors.emplace(tens, &tile);
      tile.tid_idx[tens].emplace_back(tens_cnt);
      if (type == "temporal") {
        tile.type = mapping_t::TEMPORAL;
      } else {
        tile.type = mapping_t::SPATIAL;
      }
      int spatial_cnt=0;
      for (auto factor_spatial_dim: factors){ //loop over the spatial X and Y dimensions
        ComputeTilingMatrix factor_per_tensor;
        for (auto dim: order){
          factor_per_tensor[dim] = factor_spatial_dim[dim][tens_cnt]; //order of factors is same as permutation order, FIXME::snegi factor_per_tensor should be an unordered map?
        }
        // for(const auto& pair: factor_spatial_dim){
        //   factor_per_tensor[pair.first]=pair.second[tens_cnt];
        // }
        // tile.tilings_[spatial_cnt][tens] = factor_per_tensor;
        spatial_factor_per_tensor.emplace_back(factor_per_tensor);
        spatial_cnt++;
      }

      tile.tilings_.emplace_back(spatial_factor_per_tensor);
      
      tile.order.emplace_back(order);

      //add something for attributes
      // COMET_LOG(logger::DEBUG, "NewTile::\n {}", tile.print(false));
      // level_to_tileMapping_global_tensors[tens] = std::move(tile);

      tens_cnt++;
    }

    level_to_tileMapping_ = std::move(tile);

    // level_to_tileMapping_.wb_output = all_tensors.wb_outputs;
    // level_to_tileMapping_.rmw       = all_tensors.rmw;
    // level_to_tileMapping_.tags      = all_tensors.tags;
    // level_to_tileMapping_.scale     = all_tensors.scale;

    // // std::cout<<"loop complete"<<std::endl;
    // // std::cout<<"loop complete"<<std::endl;

    // //loop over all the local tensors from different operations
    // tens_cnt=0;
    // // std::map<problem::TensorID, LevelTileMapping> tile_intermediate;
    // LevelTileMapping tile_intermediate;
    // for (auto tens: all_tensors.intermediate_tensors){

    //   // LevelTileMapping tile;
    //   // TilingMatrix factor_per_tensor(order.size());
    //  ComputeTilingMatrix factor_per_tensor;

    //   // level_to_tileMapping_global_tensors.emplace(tens, &tile);

    //   if (type == "temporal") {
    //     tile_intermediate.type = mapping_t::TEMPORAL;
    //   } else {
    //     tile_intermediate.type = mapping_t::SPATIAL;
    //   }
    //   tile_intermediate.target_[tens] = all_tensors.target_child_intermediate[tens_cnt].first;
    //   tile_intermediate.child_[tens] = all_tensors.target_child_intermediate[tens_cnt].second;

    //   int spatial_cnt=0;
    //   for (auto factor_spatial_dim: factors){ //loop over the spatial X and Y dimensions
    //     tile_intermediate.tilings_.emplace_back();
    //     for (auto dim: order){
    //       factor_per_tensor[dim] = factor_spatial_dim[dim][tens_cnt]; //order of factors is same as permutation order
    //     }
    //     // for(const auto& pair: factor_spatial_dim){
    //     //   factor_per_tensor[pair.first]=pair.second[tens_cnt];
    //     // }
    //     tile_intermediate.tilings_[spatial_cnt][tens] = factor_per_tensor;
    //   }

      
    //   tile_intermediate.order[tens] = order;

    //   //add something for attributes
    //   // COMET_LOG(logger::DEBUG, "NewTile::\n {}", tile_intermediate.print(false));
    //   // level_to_tileMapping_global_tensors[tens] = std::move(tile);

    //   tens_cnt++;
    // }

    // level_to_tileMapping_.emplace_back(std::move(tile_intermediate));

    // int tens_cnt=0;
    // for (auto tens: all_tensors.intermediate_tensors){

    //   LevelTileMapping tile;
    //   TilingMatrix factor_per_tensor(order.size());
      
    //   // level_to_tileMapping_global_tensors.emplace(tens, &tile);

    //   if (type == "temporal") {
    //     tile.type = mapping_t::TEMPORAL;
    //   } else {
    //     tile.type = mapping_t::SPATIAL;
    //   }
    //   tile.target_ = target;
    //   tile.child_ = child;

    //   for (auto factor_spatial_dim: factors){ //loop over the spatial X and Y dimensions
    //     for (auto dim: order){
    //       factor_per_tensor[dim] = factor_spatial_dim[dim][tens_cnt]; //order of factors is same as permutation order
    //     }
    //     // for(const auto& pair: factor_spatial_dim){
    //     //   factor_per_tensor[pair.first]=pair.second[tens_cnt];
    //     // }
    //   }

    //   tile.tilings_ = factor_per_tensor;
    //   tile.order = order;

    //   //add something for attributes
    //   COMET_LOG(logger::DEBUG, "NewTile::\n {}", tile.print(false));
    //   level_to_tileMapping_intermediate_tensors[tens] = std::move(tile);

    //   tens_cnt++;
    // }

  }


  OperationNode::OperationNode(config::CompoundConfigNode compute_mapping, arch::Topology& topology, problem::Workloads& workloads, config::CompoundConfigNode constants):Node(Node::OperationNode, compute_mapping){
    //common parameters in all operations
    std::string type;
    compute_mapping.lookupValue("type", type);
    tolower(type);
    
    // //set target
    // std::string target;
    // COMET_ASSERT(compute_mapping.lookupValue("target", target), "ComputeMapping doesnt have target directive");
    // common_attributes_.target_ = topology.getArchLevel(target);
    // COMET_ASSERT(common_attributes_.target_->isCompute(), "ComputeMapping's Target is not a compute node in the graph");

    // //set child
    // std::string child;
    // COMET_ASSERT(compute_mapping.lookupValue("child", child), "ComputeMapping doesnt have child directive");
    // common_attributes_.child_ = topology.getArchLevel(child);

    //set op_name
    //tag related to the name of the operation in problem file

    std::string name;
    COMET_ASSERT(compute_mapping.lookupValue("name", name), "ComputeMapping doesnt have name directive");
    // tolower(name);
    common_attributes_.op_name_ = name;
    p_workload = workloads.workloads_.at(name);

    if (type=="gemm"){
      common_attributes_.type_=operation_t::GEMM;
    }
    else if (type=="convolution"){
      common_attributes_.type_=operation_t::CONVOLUTION;
    }
    else if (type=="softmax"){
      common_attributes_.type_=operation_t::SOFTMAX;
    } else if(type=="max"){
      common_attributes_.type_=operation_t::MAX;
    } else if(type=="exp"){
      common_attributes_.type_=operation_t::EXP;
    } else if(type=="div"){
      common_attributes_.type_=operation_t::DIV;
    } else if(type=="add"){
      common_attributes_.type_=operation_t::ADD;
    } else if(type=="mult"){
      common_attributes_.type_=operation_t::MULT;
    } else if(type=="sqrt"){
      common_attributes_.type_=operation_t::SQRT;
    } else if(type=="rowsum"){
      common_attributes_.type_ = operation_t::ROWSUM;
    } else if(type=="rowmax"){
      common_attributes_.type_ = operation_t::ROWMAX;
    }
    else{
      COMET_ASSERT(true,"Enter a valid operation type");
    }

    //get tensors
    auto all_tensors = GetTensorsInTile(compute_mapping, workloads, topology, false);//don't need to get target and child here

    // common_attributes_.gl_in_tensors.emplace_back(all_tensors.global_tensors);
    // common_attributes_.gl_in_tensors.emplace_back(all_tensors.intermediate_tensors);
    
    common_attributes_.tensors = all_tensors.tensors;

    //tileprimitives
    std::string tileprimitive_string;
    COMET_ASSERT(compute_mapping.lookupValue("TilePrimitive", tileprimitive_string), "ComputeMapping does not have TilePrimitive directive");
    auto tp = parseComputeTilingCoordinates(tileprimitive_string, workloads, constants);
    ComputeMapping cMap_global;
    //assign tile primitives to the map
    // ComputeMapping cMap_global;
    cMap_global.tileprimitives = tp; // all tensors have same tile primitives

    if (common_attributes_.type_==operation_t::GEMM){      
      COMET_ASSERT(compute_mapping.exists("TileSteps"), "Compute Mapping doesnt have tilesteps directive");
      auto scale_exists = compute_mapping.exists("Scale");
      auto tilesteps = compute_mapping.lookup("TileSteps");

      computation_attributes comp_attribute;

      if (compute_mapping.exists("compute_attributes")){
        auto attr = compute_mapping.lookup("compute_attributes");
        comp_attribute.specd = attr.get<bool>("specd");
        comp_attribute.rmw   = attr.get<bool>("rmw");
        // auto reduction_dims = attr.get<std::vector<std::string>>("reduction_dimensions");
        // auto reduction_factors = attr.get<std::vector<uint32_t>>("reduction_factors");
        // std::vector<std::string> reduction_dims;
        // std::vector<uint32_t>    reduction_factors; 
        auto red_dim_string = attr.get<std::vector<std::string>>("reduction_dimensions");

        auto red_fac_string = attr.get<std::vector<std::string>>("reduction_factors");

        COMET_ASSERT(red_dim_string.size() == red_fac_string.size(), "Reduction Factors and dimensions are unmatched in the compute attributes");

        for(auto idx=0; idx<red_dim_string.size(); idx++){
          std::string red_dim_value;
          if(constants.exists(red_dim_string[idx]) && red_dim_string[idx].size()!=1){
            constants.lookupValue(red_dim_string[idx], red_dim_value);
          } else{
            red_dim_value = red_dim_string[idx];
          }
          auto dimID = workloads.getDimID(red_dim_value);
          comp_attribute.reduction_dimensions.emplace_back(dimID);
          
          uint32_t red_fac_value;
          if(constants.exists(red_fac_string[idx])){
            constants.lookupValue(red_fac_string[idx], red_fac_value);
          } else {
            red_fac_value = std::stoi(red_fac_string[idx]);
          }
          comp_attribute.reduction_factors[dimID] = red_fac_value;

        }

        // if (constants.exists(red_dim_string)){
        //   constants.lookupValue(value_string, factor_value);
        // } else {
        //   factor_value=std::stoul(value_string);
        // }


        // COMET_ASSERT(reduction_dims.size() == reduction_factors.size(), "Reduction Factors and dimensions are unmatched in the compute attributes");
        // for (auto idx = 0; idx < reduction_dims.size(); idx++) {
        //   auto dimID = workloads.getDimID(reduction_dims[idx]);
        //   comp_attribute.reduction_dimensions.emplace_back(dimID);
        //   comp_attribute.reduction_factors[dimID] = reduction_factors[idx]; 
        // }
      }


      // auto all_tensors = GetTensorsInTile(compute_mapping, workloads, topology, false);//don't need to get target and child here

      int tens_cnt=0;
      //global tensors
      // std::map<problem::TensorID, ComputeMapping> cMap_global;

      cMap_global.comp_attr=comp_attribute;
      

      for (auto tens: all_tensors.tensors){
        //tilesizes
        std::string tensor_name=workloads.getTensorName(tens);
        std::string tilestep_string;
        COMET_ASSERT(tilesteps.lookupValue(tensor_name.c_str(), tilestep_string), "ComputeMapping doesnt have a tilestep for tensor with name " << tensor_name);
        cMap_global.tilesizes[tens] = parseComputeTilingCoordinates(tilestep_string, workloads, constants);

        //placeholder for projectedExpressionOnTensor

        //relative tilesteps = tilesizes/tileprimitives , loop over every dimension and calculate relative tilesteps for every dimension
        // uint32_t accumulation=1;
        // for(auto ts: cMap_global.tilesizes[tens]){
        //   accumulation *= (ts.second/cMap_global.tileprimitives[tens][ts.first]).resolve();
        // }
        // cMap_global.relative_tilesteps[tens] = accumulation;

        uint32_t accumulation=1;
        // for (auto ts: cMap_global.tilesizes[tens]){
        //   accumulation *= (ts.second)/cMap_global.tileprimitives[ts.first];
        // }
        for(auto&[dimid,val] :cMap_global.tileprimitives){
          accumulation *= cMap_global.tilesizes[tens][dimid]/val;
        }
        cMap_global.relative_tilesteps[tens] = accumulation;

        if(scale_exists){
          uint32_t scale;
          std::string scale_string;
          compute_mapping.lookup("Scale").lookupValue(tensor_name.c_str(), scale_string);
          constants.lookupValue(scale_string, scale);
          cMap_global.tensorComputeScaling_[tens] = scale;
        } else{
          cMap_global.tensorComputeScaling_[tens] = 1;
        }

        // computeMapping_global[tens] = std::move(cMap_global);
      }    
    // } else if (common_attributes_.type_==operation_t::SOFTMAX) {
    } else {
      COMET_ASSERT(compute_mapping.exists("TileSteps"), "Compute Mapping doesnt have tilesteps directive");

      computation_attributes comp_attribute;

      if (compute_mapping.exists("compute_attributes")){
        auto attr = compute_mapping.lookup("compute_attributes");
        comp_attribute.specd = attr.get<bool>("specd");
        comp_attribute.rmw   = attr.get<bool>("rmw");

        auto red_dim_string = attr.get<std::vector<std::string>>("reduction_dimensions");

        auto red_fac_string = attr.get<std::vector<std::string>>("reduction_factors");

        COMET_ASSERT(red_dim_string.size() == red_fac_string.size(), "Reduction Factors and dimensions are unmatched in the compute attributes");

        for(auto idx=0; idx<red_dim_string.size(); idx++){
          std::string red_dim_value;
          if(constants.exists(red_dim_string[idx]) && red_dim_string[idx].size()!=1){
            constants.lookupValue(red_dim_string[idx], red_dim_value);
          } else{
            red_dim_value = red_dim_string[idx];
          }
          auto dimID = workloads.getDimID(red_dim_value);
          comp_attribute.reduction_dimensions.emplace_back(dimID);
          
          uint32_t red_fac_value;
          if(constants.exists(red_fac_string[idx])){
            constants.lookupValue(red_fac_string[idx], red_fac_value);
          } else {
            red_fac_value = std::stoi(red_fac_string[idx]);
          }
          comp_attribute.reduction_factors[dimID] = red_fac_value;

        }
      }
      cMap_global.comp_attr=comp_attribute;


      for (auto tens: all_tensors.tensors){
        auto tilesteps = compute_mapping.lookup("TileSteps");
        //tilesizes
        std::string tensor_name=workloads.getTensorName(tens);
        std::string tilestep_string;
        COMET_ASSERT(tilesteps.lookupValue(tensor_name.c_str(), tilestep_string), "ComputeMapping doesnt have a tilestep for tensor with name " << tensor_name);
        common_attributes_.tilesizes[tens] = parseComputeTilingCoordinates(tilestep_string, workloads, constants);        
      }
    }

    computeMapping_ = std::move(cMap_global);


    // don't need any of the specifics because it is common for all workloads
    // else if (common_attributes_.type_==operation_t::SOFTMAX) {
    //   auto all_tensors=GetTensorsInTile(compute_mapping, workloads, topology, false);

    //   // operation_description.global_tensors=all_tensors.global_tensors;
    //   // operation_description.intermediate_tensors=all_tensors.global_tensors;
    //   simd_tensor_id_.emplace_back(all_tensors.global_tensors);
    //   simd_tensor_id_.emplace_back(all_tensors.intermediate_tensors);
    // }


   
  }

  InterTileBinding::InterTileBinding(config::CompoundConfigNode config):Node(Node::InterTileBinding, config){
    std::string type_s;
    config.lookupValue("binding", type_s);
    tolower(type_s);
    if (type_s.find("seq") != std::string::npos) {
        type = sequential;
        name_ += "::Sequential";
    }
    else if (type_s.find("para") != std::string::npos) {
        type = parallel;
        name_ += "::Parallel";
    }
    else if (type_s.find("pipe") != std::string::npos) {
        type = pipeline;
        name_ += "::Pipeline";
    }
    else if (type_s.find("shar") != std::string::npos) {
        type = sharing;
        name_ += "::Sharing";
    }
    else {COMET_ERROR("InterTileBinding type error. Should have a sequential/parallel type");}  

  }

  CollectiveOperationNode::CollectiveOperationNode(config::CompoundConfigNode config, arch::Topology& topology, problem::Workloads& workloads):Node(Node::CollectiveOperationNode, config){
    //target
    collective_op_description.target = GetTilingLevel(config, topology, "target");
    //child
    collective_op_description.child = GetTilingLevel(config, topology, "child");

    //type
    std::string type_string;
    config.lookupValue("type", type_string);   
    if (type_string=="broadcast"){
      collective_op_description.type_ = stype_t::BROADCAST; 
    }
    else if (type_string=="reduction"){
      collective_op_description.type_ = stype_t::REDUCTION;
    }
    else if (type_string=="allgather"){
      collective_op_description.type_ = stype_t::ALLGATHER;
    }
    else if (type_string=="allreduce"){
      collective_op_description.type_ = stype_t::ALLREDUCE;
    }
    else if (type_string=="scatter"){
      collective_op_description.type_ = stype_t::SCATTER;
    }
    else if (type_string=="gather"){
      collective_op_description.type_ = stype_t::GATHER;
    }
    else{
      COMET_ASSERT(true, "Enter valid collective operation type");
    }
    
    //operation-name for the reduction collective
    std::string reduc_op;
    if(config.exists("reduction_op")){
      config.lookupValue("reduction_op", reduc_op);
      tolower(reduc_op);

      if(reduc_op=="max"){
        collective_op_description.reduction_op = operation_t::MAX;
      } else if(reduc_op=="add" || reduc_op=="sub"){
        collective_op_description.reduction_op = operation_t::ADD;
      } else if(reduc_op=="mult"){
        collective_op_description.reduction_op = operation_t::MULT;
      }

    }

    //dimension
    std::string dim;
    config.lookupValue("dimension", dim); //might need this to calculate output tensor size
    collective_op_description.dimension = workloads.getDimID(dim);

    //tag
    std::string tag;
    if(config.exists("tag")) config.lookupValue("tag", tag);
    collective_op_description.tag = tag;

    uint32_t spatial_factor=0;
    if(config.exists("spatial_factor")){
      config.lookupValue("spatial_factor", spatial_factor);
    }
    collective_op_description.spatial_factor = spatial_factor;

    uint32_t scale=1;
    if(config.exists("precision")){
      config.lookupValue("precision", scale);
    }
    collective_op_description.scale = scale;

    //src
    // std::string src;
    // config.lookupValue("src", src);
    // config.lookupArrayValue("src", src);

    collective_op_description.src = getsrc_dest(config, "src", topology);
    
    // (config, src, topology);

    //dest
    // std::string dest;
    // config.lookupValue("dest", dest);
    collective_op_description.dest = getsrc_dest(config, "dest", topology);
    // (config, dest, topology);

    //in_tensor
    std::string tensor;
    config.lookupValue("in_tensor", tensor);
    collective_op_description.in_tensor = workloads.getTensorID(tensor);
    //out_tensor
    config.lookupValue("out_tensor", tensor);
    collective_op_description.out_tensor = workloads.getTensorID(tensor);

    //wb_output
    // config.lookupValue("wb_output", )
    collective_op_description.wb_output = config.get<bool>("wb_output");

  }

}