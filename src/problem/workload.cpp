
#include "problem/workload.hpp"
#include "util/comet_assert.h"
#include "util/logger.hpp"
#include <algorithm>
#include <numeric>
#include <functional>
#include <regex>

namespace problem {

  void tolower(std::string& str){
      std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {return std::tolower(c);});
  }

  std::vector<std::string> Split(const std::string& buffer) {
      std::vector<std::string> retval;
      // std::regex re("([A-Za-z]+)[,]*[[:space:]]*", std::regex::extended);
      std::regex re("([A-Za-z0-9]+)[,]*[[:space:]]*", std::regex::extended); //to support alpha numeric names for tensors
      std::smatch sm;
      std::string str = std::string(buffer);
      str = str.substr(0, str.find("#"));

      while (std::regex_search(str, sm, re))
      {
          retval.push_back(sm[1]);
          str = sm.suffix().str();
      }
      return retval;
  }

  const ShapeID Workload::newShape(const std::string& shape_name) { 
    if (shapeNamesToID_.find(shape_name) == shapeNamesToID_.end()) { 
      shapeNames_[shape_count] = shape_name;
      shapeNamesToID_[shape_name] = shape_count;
      return shape_count++;
    } else {
      COMET_ASSERT(false, "Shape with name " << shape_name << " already exists");
      return 0;
    }
  }

  void Workload::newDimension(const std::string& dim_name, std::map<std::string, DimensionID> global_dim_ids) { 
    //get the dimension ids from global dimension map
    if(global_dim_ids.find(dim_name) != global_dim_ids.end()){ //dim_names exist in global_dims
      auto dim_id = global_dim_ids[dim_name];
      dimensionIDToDimName_[dim_id] = dim_name;
      dimNamesToDimensionID_[dim_name] = dim_id;
      dim_count++;
    } else{
      COMET_ASSERT(false, "dimension "<< dim_name <<"does not exist in global dimension map");
    }
    // if (dimNamesToDimensionID_.find(dim_name) == dimNamesToDimensionID_.end()) { 
    //   dimensionIDToDimName_[dim_count] = dim_name;
    //   dimNamesToDimensionID_[dim_name] = dim_count;
    //   dim_count++;
    // } else { 
    //   COMET_ASSERT(false, "dimension with " << dim_name << " already exists");
    // }
  }

  void Workload::addDimensionToShape(const ShapeID shapeid, const DimensionID dimID) { 
    shapeDimensions_[shapeid].emplace_back(dimID);
  }

  const TensorID Workload::newTensor(const std::string& tensor_name, std::map<std::string, TensorID> global_tensor_ids) { 
    TensorID tensor_id;
    if (global_tensor_ids.find(tensor_name) != global_tensor_ids.end()){
      tensor_id = global_tensor_ids[tensor_name];
      tensorNames_[tensor_id] = tensor_name;
      tensorNamesToTensorID_[tensor_name] = tensor_id;
      tensor_count++;
      common_tensor_count++;
    }
    else if (tensorNamesToTensorID_.find(tensor_name) == tensorNamesToTensorID_.end()) { 
      tensor_id = global_tensor_ids.size() + tensorNames_.size()-common_tensor_count;
      // tensorNames_.size()?  global_tensor_ids.size() + tensorNames_.size()-common_tensor_count: global_tensor_ids.size() + tensorNames_.size();
      tensorNames_[tensor_id] = tensor_name;
      tensorNamesToTensorID_[tensor_name] = tensor_id;
      // TensorRW.emplace_back(false);
      tensor_count++;
      // return tensor_count++;
    } else {
      COMET_ASSERT(false, "Tensor with name " << tensor_name << " already exists;");
      return 0;
    }
    TensorRW[tensor_id] = false;
    return tensor_id;
  }

  void Workload::SetRWForTensor(const TensorID tensor_id) { 
    TensorRW[tensor_id] = true;
  }

  void Workload::AddProjection(const TensorID tensor_id, Projection&& proj) { 
    projection_map_[tensor_id] = proj; 
  }

  // void Workload::parse(config::CompoundConfigNode problem_config) { 
  //   auto problem_node = problem_config.lookup("problem");
  //   if (!problem_node.isList()) { 
  //     parseProblem(problem_node);
  //   } else {
  //     throw std::logic_error("COMET does not support multiple problems yet");
  //     //for (int prob=0; prob < problem_config.isList(); prob++) {
  //     //  parseInstance(problem_config[i]);
  //     //}
  //   }
  // }

  std::string Workload::parseProblem(config::CompoundConfigNode problem_config, ComputeTilingMatrix& dim_sizes, std::map<std::string, DimensionID> global_dim_ids, std::map<std::string, TensorID> global_tensor_ids){ 

    std::string name;
    COMET_ASSERT(problem_config.lookupValue("name", name), "No name specified for an op");
    // tolower(name);
    set_name(name);    
    std::string ins, out;
    COMET_ASSERT(problem_config.lookupValue("ins", ins), "No ins specified for op "<<name <<".");
    COMET_ASSERT(problem_config.lookupValue("out", out), "No out specified for op "<<name <<".");
    set_io(Split(ins), Split(out));

    std::vector<std::string> dim_names{};
    problem_config.lookupArrayValue("dimensions", dim_names);
    for (const auto& dim_name: dim_names) { 
      newDimension(dim_name, global_dim_ids); // TODO
      // addDimensionToShape(shapeID, dimID); // TODO
      // COMET_LOG(logger::DEBUG, "Adding dim_name {} as dim_id {}", dim_name, dimID);
    }

    //find coefficients
    // problem_config.loook

   
    auto tensors_cfg = problem_config.lookup("tensors");

    COMET_ASSERT(tensors_cfg.isList(), "Tensors should be in an Array form");

    // TODO::snegi add a for loop above this for operations
    for (auto tens = 0; tens< tensors_cfg.getLength(); tens++) {
      Projection tensor_proj;
      auto tensor_cfg = tensors_cfg[tens];
      std::string tensor_name;
      COMET_ASSERT(tensor_cfg.lookupValue("name", tensor_name), "TensorID " << tens << " does not have a name");
      auto tens_id = newTensor(tensor_name, global_tensor_ids); // TODO
      bool is_rw = false;
      tensor_cfg.getValue<bool>("read-write", is_rw);
      if (is_rw) SetRWForTensor(tens_id);
      
      auto tensor_proj_cfg = tensor_cfg.lookup("projection");
      COMET_ASSERT(tensor_proj_cfg.isList(), "COMET currently only supports array of projections please change workload.cpp to support otherwise"); //TODO:: Add String support for ISL MAP
      for (auto proj = 0; proj < tensor_proj_cfg.getLength(); proj++) { 
        auto dimension = tensor_proj_cfg[proj];
        ProjectionExpression ProjExp;
        for (auto t = 0; t < dimension.getLength(); t++) {
          auto term = dimension[t];
          COMET_ASSERT(term.isArray(), "COMET only suports projection terms to be arrays, please change workload.cpp to support otherwise");
          std::vector<std::string> projDimNames;
          term.getArrayValue(projDimNames);
          
          // if (term.getLength()==1){
          COMET_ASSERT(term.getLength()==1, "COMET only supports Dimension names in the projection term, please change workload.cpp/hpp to support otherwise"); // discuss::snegi -- so we can't support W*stride? SOP?
          const std::string& projDimName = projDimNames[0];
          auto dim_id = dimNamesToDimensionID_.at(projDimName); // discuss::snegi just adding the index from global might create an issue now, we might need indices from local tensors for the operations
          // }
          // else if (term.getLength==2){
          //   //problem has both dimension and coefficient
          //   const std::string& projDimName = projDimNames[0];
          //   const std::string& coeff_name  = projDimNames[1];
          //   auto dim_id = dimNamesToDimensionID_.at(projDimName);

          // }
          ProjExp.push_back(dim_id);
        }
        tensor_proj.push_back(ProjExp);
      }
      AddProjection(tens_id, std::move(tensor_proj)); // TODO 
    }

    DimSizeExpression DimSizes(dim_names.size(), 0);
    int cnt=0;
    for(auto& [dimName, dim_id]: dimNamesToDimensionID_){
      if(dim_sizes.find(dim_id)!=dim_sizes.end()){
        DimSizes[cnt]= dim_sizes.at(dim_id);
      } else{
        COMET_ASSERT(false, dimName<<" does not exist in global dimensions");
      }
      cnt++;
    }
    SetDimSizes(std::move(DimSizes));
    
    // DimSizeExpression DimSizes(getDimensionsForShape(shapeID), 0); // TODO
    // auto instance_cfg = problem_config.lookup("instance");
    // auto instance_cfg_yaml = instance_cfg.getYNode();
    // for (uint8_t dim=0; dim < getDimensionsForShape(shapeID); dim++) { 
    //   const std::string& dimName = getDimName(dim); // Todo
    //   COMET_ASSERT(instance_cfg.getValue<uint32_t>(dimName.c_str(), DimSizes[dim]), "Workload Instance does not have all dimensions");
    // }
    // SetDimSizes(std::move(DimSizes));

    return name;

  }

  // std::string ParseWorkload(config::CompoundConfigNode config, problem::Workload& workload){
  //   std::string name;
  //   COMET_ASSERT(config.lookupValue("name", name), "No name specified for an op");
  //   workload.set_name(name);

  // }



  void Workloads::ParseWorkloads(config::CompoundConfigNode config, config::CompoundConfigNode constants){
    if (config.exists("io")){
      auto io = config.lookup("io");
      std::string ins, outs;
      COMET_ASSERT(io.lookupValue("ins", ins), "No ins property in problem::io");
      COMET_ASSERT(io.lookupValue("outs", outs), "No outs property in problem::io");
      set_io(Split(ins), Split(outs));
    }
    else {
      COMET_ERROR("No io found in problem.");
    }

    std::vector<std::string> dims;
    if (config.exists("dimensions")) {
      config.lookupArrayValue("dimensions", dims);
    }
    else {
      COMET_ERROR("no dimensions passed for the workloads");
    }

    for (const auto& dim_name: dims) { 
      newDimension(dim_name);
      dim_count++;
    }    

    if (config.exists("instance")){
      auto dim_sizes = config.lookup("instance");
      for (auto dim: dims) {
        COMET_ASSERT(dim_sizes.exists(dim), "no instance passed for axis" << dim << ".");
        std::string _tmp;
        dim_sizes.lookupValue(dim, _tmp);
        uint32_t dim_size;
        if(constants.exists(_tmp)){
          constants.lookupValue(_tmp, dim_size);
        } else{
          dim_size = std::stoi(_tmp);
        }
        set_dim_sizes(dim, dim_size);
      }
      // DimSizeExpression DimSizes(dims.size(), 0); // TODO
      // auto instance_cfg = config.lookup("instance");
      // auto instance_cfg_yaml = instance_cfg.getYNode();
      // for (uint8_t dim=0; dim < dims.size(); dim++) { 
      //   const std::string& dimName = getDimName(dim); // Todo
        // COMET_ASSERT(instance_cfg.getValue<uint32_t>(dimName.c_str(), DimSizes[dim]), "Workload Instance does not have all dimensions");
      // }
      // SetDimSizes(std::move(DimSizes)); 
    }

    if (config.exists("operations")) {
      auto ops = config.lookup("operations");
      for (int i=0; i<ops.getLength(); ++i){
        std::shared_ptr<Workload> p_workload(new Workload());
        std::string name = p_workload->parseProblem(ops[i], workloadsDimSizes_, dimNamesToDimensionID_, tensorNamesToTensorID_);
        assert(add_workload(name, p_workload));
      }
    }
    else{
        std::shared_ptr<Workload> p_workload(new Workload());
        std::string name = p_workload->parseProblem(config, workloadsDimSizes_, dimNamesToDimensionID_, tensorNamesToTensorID_); //no operations specified implies we have a single operation
        assert(add_workload(name, p_workload));
    }

    for(auto&[tname,tid]:tensorNamesToTensorID_){
      if((std::find(ins_.begin(), ins_.end(), tname)==ins_.end()) && (std::find(outs_.begin(), outs_.end(), tname)==outs_.end())){
        //tname doesn't exist in the global input
        intermediate_tensors.push_back(tid);
      }
    }

  }

  void Workloads::set_dim_sizes(const std::string dim, uint32_t dim_size){
    assert(dimNamesToDimensionID_.count(dim));
    workloadsDimSizes_[dimNamesToDimensionID_[dim]] = dim_size;
  }

  bool Workloads::add_workload(const std::string &name, std::shared_ptr<Workload>& workload){
    COMET_ASSERT(!workloads_.count(name), "Duplicate op named " << name << ". Drop all but the first.");

    for (auto& kv: workload->dimNamesToDimensionID_) {
      COMET_ASSERT(dimNamesToDimensionID_.count(kv.first), "Op:"<< name << "'s dimension " << kv.first <<" is not declared in global scope.");
    } 
    //snegi::discuss do we need to check the order? 
    for (auto& kv: workload->tensorNames_) {
      std::string access_pattern_name = kv.second;
      // if ()
      // if(tensorNamesToTensorID_[access_pattern_name]){ //FIXME::snegi intermediate tensors have same ids
      if(tensorNamesToTensorID_.count(access_pattern_name)){// if the tensor exist in global tensor map continue
        continue;
      }
      // auto proj = workload->getProjection(kv.first);
      // auto& id   = tensor_count;
      auto id = kv.first;//get id from the workload
      // assert (id == Projections.size());
      // Projections.emplace_back();
      // auto& new_proj = Projections.back();
      // for (auto& expr: proj){
      //   new_proj.emplace_back();
      //   auto& new_expr = new_proj.back(); 
      //   for (auto& term: expr){
      //     CoefficientID new_coeff_id = term.first == workload->NumCoefficients ? -1: coefficientNameToID[workload->coefficientIDToName.at(term.first)];

      //     DimensionID new_dim_id = dimNamesToDimensionID_[workload->dimensionIDToDimName_.at(term.second)];
      //     new_expr.emplace_back(new_coeff_id, new_dim_id);
      //   }
      // }

      
      // TensorRW.push_back(workload->TensorRW[kv.first]);
      TensorRW[id] = workload->TensorRW[kv.first];

      tensorNames_[id] = access_pattern_name;
      tensorNamesToTensorID_[access_pattern_name] = id;
      // id++;
      tensor_count++;
    }

    // for (auto& kv: workload->CoefficientNamesToID_) {
    //   std::string coeff_name = kv.first;
    //   CoefficientIDToName_[coefficient_count] = coeff_name;
    //   CoefficientNamesToID_[coeff_name] = coefficient_count++;
    // } 

    workloads_[name] = std::move(workload);
    
    return true;
    
  }

  const std::string& Workload::ToString() {
    std::string ss="";
    ss =  "Workload Shape with name " + shapeNames_[0] + " with Dimensions: ";
    for (const auto& dim: dimNamesToDimensionID_ ) { 
      ss += dim.first + "(" + std::to_string(DimSizes_[dim.second]) + ")" + ", ";
    }
    ss += "\n Has tensors: \n";
    for (const auto& tensor: tensorNamesToTensorID_) { 
      ss += tensor.first + "\n";
    }
    return ss;
  }

  std::ostream& operator << (std::ostream& out, Workload& workload) {
    out << workload.ToString();
    return out;
  }

  TensorID Workload::getTensorID(std::string tensor_name) const { 
    std::map<std::string, TensorID>::const_iterator iter = tensorNamesToTensorID_.find(tensor_name);
    if (iter != tensorNamesToTensorID_.end()) return iter->second;
    else { 
      COMET_ERROR("Workload does not have tensor with name " << tensor_name);
      return 0;
    }
  }

  std::vector<TensorID> Workload::getTensors() const {
    std::vector<TensorID> vec(tensor_count);
    std::iota(vec.begin(), vec.end(), 0);
    return vec;
  }

  std::vector<DimensionID> Workload::getDimensionsVector() const { 
    std::vector<DimensionID> dims(dim_count);
    std::iota(dims.begin(), dims.end(), 0);
    return dims;
  }
  
  DimensionID Workload::getDimID(std::string dimension_name) const { 
    std::map<std::string, DimensionID>::const_iterator iter = dimNamesToDimensionID_.find(dimension_name);
    if (iter != dimNamesToDimensionID_.end()) return iter->second;
    else { 
      COMET_ERROR("Workload does not have Dimension with name " << dimension_name);
      throw std::logic_error("Bad dimension name");
      return 0;
    }
  }

  // DimSizeExpression Workload::projectDimExpressionOnTensor(const DimSizeExpression& full_rank_expression, TensorID tensor_id, ShapeID shape_id) const { 
  //   auto order_of_tensor = projection_map_.at(tensor_id).size();
  //   DimSizeExpression retval(order_of_tensor);
  //   for (unsigned dim=0; dim < order_of_tensor; dim++) { 
  //     for (const auto& term: projection_map_.at(tensor_id).at(dim)) { 
  //       retval[dim] += full_rank_expression[term];
  //     } 
  //   }
  //   return retval;
  // }

  DimSizeExpression Workload::projectDimExpressionOnTensor(ComputeTilingMatrix& dim_sizes, TensorID tensor_id){

    DimSizeExpression retval(projection_map_.at(tensor_id).size());

    for(auto cnt=0; cnt<projection_map_.at(tensor_id).size();cnt++){
    // for(auto cnt=0; cnt<projection_map_.at(tensor_id).size(); cnt++){
    //   auto dim_list = projection_map_.at(tensor_id).at(cnt);
    //   for(auto& dim: dim_list){
    //   }
      // if the dimension was [[M],[N]]--> for convolution add up the sizes
      for(auto& term:projection_map_.at(tensor_id).at(cnt)){
        
        // if(dim_sizes.find(term) != dim_sizes.end()){
        //   std::cout << "key exist" << std::endl;
        // } else {
        //   std::cout << "key doesn't exist" << std::endl;
        // }


        // std::cout << "term = " << term << " at address " << &term << std::endl;
        // std::cout<< "dim_size val at term = " << dim_sizes[term] << std::endl;
        retval[cnt] += dim_sizes[term];
        // retval[term] += dim_sizes.at(term);
      }
    }

    return retval;
  }


  bool Workload::tensorHasDimensionRank(TensorID tensor_id, ShapeID shape_id, DimensionID dim_id) const { 
    auto order_of_tensor = projection_map_.at(tensor_id).size();
    analysis::Point retval(order_of_tensor);
    auto ignore = shape_id;
    for (unsigned dim=0; dim < order_of_tensor; dim++) { 
      for (const auto& term: projection_map_.at(tensor_id).at(dim)) { 
        if (term == dim_id) return true;
      } 
    }
    return false;
  }

  //compound op related functions
  void Workloads::set_io(const std::vector<std::string>& ins, const std::vector<std::string>& outs) {
    ins_ = ins;
    outs_ = outs;
  }

  void Workloads::newDimension(const std::string& dim_name) { 
    if (dimNamesToDimensionID_.find(dim_name) == dimNamesToDimensionID_.end()) { 
      dimensionIDToDimName_[dim_count] = dim_name;
      dimNamesToDimensionID_[dim_name] = dim_count;
      // return dim_count++;
    } else { 
      COMET_ASSERT(false, "dimension with " << dim_name << " already exists");
    }
  }

  DimensionID Workloads::getDimID(std::string dimension_name) const { 
    std::map<std::string, DimensionID>::const_iterator iter = dimNamesToDimensionID_.find(dimension_name);
    if (iter != dimNamesToDimensionID_.end()) return iter->second;
    else { 
      COMET_ERROR("Workload does not have Dimension with name " << dimension_name);
      throw std::logic_error("Bad dimension name");
      return 0;
    }
  }  


  void Workload::set_io(const std::vector<std::string>& ins, const std::vector<std::string>& outs) {
    ins_ = ins;
    out_ = outs.front();
  }

  TensorID Workloads::getTensorID(std::string tensor_name) const { 
    std::map<std::string, TensorID>::const_iterator iter = tensorNamesToTensorID_.find(tensor_name);
    if (iter != tensorNamesToTensorID_.end()) return iter->second;
    else { 
      COMET_ERROR("Workloads does not have tensor with name " << tensor_name);
      return 0;
    }
  }


}
