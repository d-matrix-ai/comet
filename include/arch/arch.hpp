#pragma once

#include "compound_config/compound_config.hpp"
#include "arch/arch_level.hpp"
#include "arch/arch_spec.hpp"
#include "arch/inc_utils.hpp"
#include "analysis/cost_info.hpp"
#include "arch/network.hpp"
#include "arch/compute_base.hpp"
#include "analysis/data_movement.hpp"
// #include "analysis/nest_analysis.hpp"
#include "analysis/node_types.hpp"
#include "analysis/cost_info.hpp"

namespace arch {

// struct analysis::Cost;


using ComputeNodeID = size_t;
using analysis::DataMovementInfo;
// using analysis::NodeTypes;
using analysis::LoopNode;
using analysis::ColOpNode;
using analysis::DataMovementCostVec;
using analysis::ColOp_struct;

using problem::TensorID;

class Topology {
  public:
    void ParseComponents(config::CompoundConfigNode component_config, bool compute_or_memory);
    ArchLevel* ParseHierarchy(config::CompoundConfigNode level_config, ArchLevel* parent_level);
    void ParseConfig(config::CompoundConfigNode full_config);
    ComputeNodeID getComputeID(std::string compute_level_name); // will error out if parameter is not a compute level
    size_t getNumComputeLevels() { return comp_nodes;}
    ArchLevel* getArchLevel(const std::string& level_name) const;
    
    Topology(config::CompoundConfigNode arch_config): comp_nodes(0), level_id(0) { ParseConfig(arch_config); }
    std::vector<std::string> getLevels() const { 
      std::vector<std::string> level_names;
      for (auto const& level: levels_) { 
        level_names.push_back(level.first);
      }
      return level_names;
    }

    const std::vector<LevelID>& getRoots() {return root_ids;} 
    bool isCompute(LevelID id) { return levelIDs_.at(id)->isCompute();}
   
    std::vector<LevelID> getCompIDs() const { return compute_ids;}
    std::shared_ptr<ComputeBase> getCompute(LevelID id) { return computes.at(id);}
    LevelID getLevelID(std::string name) const { return levels_.at(name)->getLevelID(); }
    uint64_t getTotalNodes(LevelID id) const;

    const ArchLevel* getLevel(LevelID id) {return levelIDs_.at(id);}

    const std::set<LevelID>& getParents(LevelID id) {return levelIDs_.at(id)->getParents();}
    const std::set<LevelID>& getChildren(LevelID id) {return levelIDs_.at(id)->getChildren();}
   

    DataMovementCostVec Evaluate(LevelID target_id, LevelID child_id, std::vector<DataMovementInfo>& tensor_movement, std::vector<uint32_t> no_stall_time, LoopNode& loop_nodes, size_t tens_cnt, bool hide_rw_latency, bool run_single_iteration, uint32_t reuse_factor, bool different_iterations);     

    ColOp_struct EvaluateCollectiveOperation(LevelID target_id, LevelID child_id, std::vector<std::vector<DataMovementInfo>>& mov_info, ColOpNode& node, size_t tens_cnt);



  private:
    ComputeNodeID comp_nodes;
    LevelID level_id;
    std::map<std::string, ComputeNodeID> computeNodes_;
    std::map<std::string, LevelSpec> specs_;
    std::map<std::string, ArchLevel*> levels_;
    std::map<LevelID, ArchLevel*> levelIDs_;
    std::map<LevelID, std::shared_ptr<ComputeBase>> computes;

    std::vector<LevelID> root_ids;
    std::vector<LevelID> compute_ids;
};



}

