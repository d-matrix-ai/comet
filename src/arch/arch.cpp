#include "arch/arch.hpp"
#include "util/comet_assert.h"
#include "util/logger.hpp"
namespace arch {
  void Topology::ParseConfig(config::CompoundConfigNode arch_config) {
    if (arch_config.exists("ComputeTiles")) { 
      ParseComponents(arch_config.lookup("ComputeTiles"), true);
    } 
    if (arch_config.exists("MemoryTypes")) { 
      ParseComponents(arch_config.lookup("MemoryTypes"), false);
    }
    if (arch_config.exists("Hierarchy")) { 
      auto root_levels = arch_config.lookup("Hierarchy");
      for (int i=0; i < root_levels.getLength(); i++) {
        auto first_level = ParseHierarchy(root_levels[i], nullptr);  
      }
    } 
    
  } 

  void Topology::ParseComponents(config::CompoundConfigNode component_config, bool compute_or_memory) { 
    assert(component_config.isList());
    for (int i=0; i < component_config.getLength(); i++) {
      std::string component_name;
      component_config[i].getValue<std::string>("name", component_name);
      assert(specs_.find(component_name) == specs_.end());
      if (compute_or_memory) {
        LevelSpec v{std::in_place_type<ComputeSpec>, component_config[i]};
        specs_.emplace(component_name, v);
      } else { 
        LevelSpec v{std::in_place_type<MemorySpec>, component_config[i]};
        specs_.emplace(component_name, v);
      }
    }
  }

  ComputeNodeID Topology::getComputeID(std::string compute_level_name) { 
    auto itr = computeNodes_.find(compute_level_name);
    COMET_ASSERT(itr != computeNodes_.end(), "Trying to find computeNodeID for a level thats not compute " << compute_level_name);
    return itr->second;
  }
  
  //slightly recursive builder of hierarchy
  ArchLevel* Topology::ParseHierarchy(config::CompoundConfigNode arch_config, ArchLevel* parent_level) {
    ArchLevel* curLevel = new ArchLevel();
    assert(arch_config.exists("name"));
    std::string curName;
    arch_config.lookupValue("name", curName);
    if (parent_level == nullptr) {
      curLevel->setRoot(); // if parent is a root then we can have different networks for same memory level connecting to the different parents
    } else {
      // ensure attributes
      assert(arch_config.exists("attributes"));
    }
    // add level to the levels_ vector 
    bool has_connects = arch_config.exists("connects");
    bool has_fills = arch_config.exists("fills");
    bool has_drains = arch_config.exists("drains");

    // do compute base
    if(arch_config.exists("is_compute")) {
      curLevel->setCompute();
      // check if current level is already a leaf node in the tree elsewhere;
      // TODO:: currently we can only support exact same leaf node, we dont allow for partial slicing of leaf nodes 
      if (levels_.find(curName) != levels_.end()) {
        return levels_.at(curName);
      }
      computeNodes_[curName] = comp_nodes++; // define the compute node
    }

    // error check to ensure no name is repeated if parent level is not a root
    if (parent_level != nullptr) { 
        if (parent_level->isRoot()) { 
          if (levels_.find(curName) != levels_.end()) { // if a (root+1) level already exists in levels return 
            return levels_.at(curName);
          }
        }
    }
    std::string typeName;
    assert(arch_config.exists("type"));
    arch_config.lookupValue("type", typeName);
    assert(specs_.find(typeName) != specs_.end());
    curLevel->setSpec(specs_.at(typeName));
    curLevel->setLevelID(level_id);

    assert(levels_.find(curName) == levels_.end());
    if (arch_config.exists("attributes")) { 
      auto attr_cfg = arch_config.lookup("attributes");
      bool link_transfers = false;
      if (attr_cfg.getValue<bool>("link_transfers", link_transfers)) {
        if (link_transfers) { 
          auto netspec = NetworkSpec(attr_cfg); // FIXME
          std::string network_type;
          attr_cfg.lookupValue("connection_type", network_type);
          if (network_type == "null") {network_type = "MeshNetwork";}
          auto network = FactoryBase<NetworkBase>::getInstance().create_object(network_type);
          network->setSpec(netspec);
          curLevel->setLinkNetwork(std::move(network));
        }
      }
    }

    // add to structures 
    if (curLevel->isRoot()) root_ids.emplace_back(level_id);
    if (curLevel->isCompute()) {
      std::string compute_type = typeName + "Compute"; // TODO:: could this be better
      auto comp_ptr = FactoryBase<ComputeBase>::getInstance().create_object(compute_type);
      comp_ptr->setSpec(std::get<ComputeSpec>(curLevel->getSpec()));
      computes[level_id] = (comp_ptr);
      compute_ids.emplace_back(level_id);
    }
    levelIDs_[level_id++] = curLevel;
    levels_[config::parseName(curName)] = curLevel;
    // setting name
    curLevel->setName(config::parseName(curName));
    // do connections
    COMET_LOG(logger::INFO, "Level::{}({}) has F/D/C {}{}{}", config::parseName(curName), curName,  has_fills, has_drains, has_connects); 
    if (has_connects) {
      COMET_LOG(logger::INFO, "    Parsing Connects");
      auto connections = arch_config.lookup("connects");
      assert(connections.isList());
      // parse connections and build the connection vector in th level
      for (int i=0; i < connections.getLength(); i++) {
        auto cur_connect = connections[i];
        auto connect_level = ParseHierarchy(connections[i], curLevel);
        curLevel->AddConnection(connect_level);
        // connect the parent 
        connect_level->AddParent(true, curLevel);
        connect_level->AddParent(false, curLevel);

        auto connection_attributes = cur_connect.lookup("attributes");
        auto netspec = NetworkSpec(connection_attributes); // FIXME
        std::string network_type;
        connection_attributes.lookupValue("connection_type", network_type);
        if (network_type == "null") {network_type = "MeshNetwork";}
        auto network = FactoryBase<NetworkBase>::getInstance().create_object(network_type);
        network->setSpec(netspec);

        connect_level->AddParentNetwork(curLevel->getLevelID(), network);
        curLevel->AddChildNetwork(connect_level->getLevelID(), network);
      }
    }
    if (has_drains) {
      auto drains = arch_config.lookup("drains");
      assert(drains.isList());
      for (int i =0; i < drains.getLength(); i++) {
        auto cur_drain = drains[i];
        COMET_LOG(logger::INFO, "    Parsing Drains");
        auto drain_level = ParseHierarchy(cur_drain, curLevel);
        curLevel->AddChild(true, drain_level);
        drain_level->AddParent(false, curLevel);
        auto connection_attributes = cur_drain.lookup("attributes");
        auto netspec = NetworkSpec(connection_attributes); // FIXME
        std::string network_type;
        connection_attributes.lookupValue("connection_type", network_type);
        if (network_type == "null") {network_type = "MeshNetwork";}
        auto network = FactoryBase<NetworkBase>::getInstance().create_object(network_type);
        network->setSpec(netspec);

        drain_level->AddParentNetwork(curLevel->getLevelID(), network);
        curLevel->AddChildNetwork(drain_level->getLevelID(), network);
        // TODO:: setupnetwork as for connect level
      } 
    }
    if (has_fills) { 
      auto fills = arch_config.lookup("fills");
      assert(fills.isList());
      for (int i=0; i < fills.getLength(); i++) {
        auto cur_fill = fills[i];
        COMET_LOG(logger::INFO, "    Parsing Fills");
        auto fillLevel = ParseHierarchy(cur_fill, curLevel);
        curLevel->AddChild(false, fillLevel);
        fillLevel->AddParent(true, curLevel);
        auto connection_attributes = cur_fill.lookup("attributes");
        auto netspec = NetworkSpec(connection_attributes); // FIXME
        std::string network_type;
        connection_attributes.lookupValue("connection_type", network_type);
        if (network_type == "null") {network_type = "MeshNetwork";} //TODO should this be an error
        auto network = FactoryBase<NetworkBase>::getInstance().create_object(network_type);
        network->setSpec(netspec);

        fillLevel->AddParentNetwork(curLevel->getLevelID(), network);
        curLevel->AddChildNetwork(fillLevel->getLevelID(), network);
      }
    }

    if (arch_config.exists("attributes")) { 
      curLevel->setAttributes(arch_config.lookup("attributes")); 
    }

    return curLevel;

    // TODO:: do sanity checks based on sizing in name and sizing in mesh coordinated
    //
  } // parsehierarchy

  ArchLevel* Topology::getArchLevel(const std::string& level_name) const { 
    std::map<std::string, ArchLevel*>::const_iterator iter = levels_.find(level_name);
    if (iter != levels_.end()) return iter->second;
    else { 
      COMET_ERROR("Level:: " << level_name << "does not match to a level name");
      return nullptr;
    }
  }
  
  uint64_t Topology::getTotalNodes(LevelID id) const { 
    auto level = levelIDs_.at(id);
    auto num_nodes = level->getInstanceSize();
    while (!level->isRoot()) { 
      auto parents = level->getParents();
      auto parent_nodes = 0;
      LevelID parent_id;
      for (auto parent: parents) { 
        auto cur_nodes = (levelIDs_.at(parent))->getInstanceSize();
        if (cur_nodes > parent_nodes) { // choose the max as we assume correct by input
          parent_id = parent;
          parent_nodes = cur_nodes;
        }
      }
      num_nodes *= parent_nodes;
      level = levelIDs_.at(parent_id);
    }
    return num_nodes;
  }

}
