#include "analysis/data_movement.hpp"
#include "arch/arch_level.hpp"
#include "util/logger.hpp"
#include <sstream>
namespace arch {

void ArchLevel::AddChild(bool drain_or_fill, ArchLevel* child) {
  std::string type = (drain_or_fill) ? "drain" : "fill";
  COMET_LOG(logger::INFO, "ArchLevel:{} adding a {} child with name {}", name_, type, child->getName());
  if (drain_or_fill) drain_levels_.push_back(child);
  else fill_levels_.push_back(child);
  child_set.emplace(child->getLevelID());
}

void ArchLevel::AddParent(bool read_or_write, ArchLevel* parent) {
  std::string type = (read_or_write) ? "read" : "write";
  COMET_LOG(logger::INFO, "ArchLevel:{} adding a {} parent with name {}", name_, type, parent->getName());
  if (read_or_write) read_levels_.push_back(parent);
  else write_levels_.push_back(parent);
  parent_set.emplace(parent->getLevelID());
}

void ArchLevel::setAttributes(config::CompoundConfigNode attribute_cfg) { 
  if (attribute_cfg.exists("mesh")) {
    std::vector<std::string> meshVal;
    COMET_ASSERT(attribute_cfg.lookupArrayValue("mesh", meshVal), "Mesh was not in an array form");
    instance_array_.resize(0); // clear initial instance_array;
    std::transform(meshVal.begin(), meshVal.end(), std::back_inserter(instance_array_), 
        [](const std::string& str) { return static_cast<uint32_t>(std::stoul(str)); }); // return the instance array;
  } else {
    if (attribute_cfg.exists("meshX")) { 
      instance_array_[0] = attribute_cfg.get<uint32_t>("meshX");
    }
    if (attribute_cfg.exists("meshY")) { 
      instance_array_[1] = attribute_cfg.get<uint32_t>("meshY");
    }
  }
  std::stringstream debug_ss;
  debug_ss << "For Level Instance Mesh " << getName();
  for (auto size: instance_array_) { 
    debug_ss << size << " , " ;
  }
  debug_ss << std::endl;
  COMET_LOG(logger::INFO, "{}", debug_ss.str());
}

bool ArchLevel::isChild(std::string& child_name) const { 
  for (auto child: drain_levels_) { 
    if (child->getName() == child_name) return true;
  }
  for (auto child: fill_levels_) { 
    if (child->getName() == child_name) return true;
  }
  return false;
}
// FIXME::MANIKS do these need to be vectors or can they be sets
bool ArchLevel::isChild(const ArchLevel* child_ptr) const { 
  if (std::find_if(drain_levels_.begin(), drain_levels_.end(), [&child_ptr](const auto val) { return val->getLevelID() == child_ptr->getLevelID();}) != drain_levels_.end()) return true;
  else if (std::find_if(fill_levels_.begin(), fill_levels_.end(), [&child_ptr](const auto val) { return val->getLevelID() == child_ptr->getLevelID();}) != fill_levels_.end()) return true;
  return false;
}

uint64_t ArchLevel::instanceCapacity() const { 
  uint64_t capacity=0;
  std::visit([&capacity](const auto& spec){capacity = spec.getCapacity();}, spec_);
  return capacity;
}

uint32_t ArchLevel::getInstanceSize() const { 
  uint32_t retval = instance_array_.front();
  for (auto cnt = 1; cnt < base_dimensions; cnt++) { 
    retval *= instance_array_[cnt];
  }
  return retval;
}

void ArchLevel::AddChildNetwork(arch::LevelID id, std::shared_ptr<NetworkBase> network) { 
  auto itr = child_networks.find(id);
  COMET_ASSERT(itr == child_networks.end(), "Adding a Child network for a previously existing child");
  child_networks.emplace(std::make_pair(id, network));
}

void ArchLevel::AddParentNetwork(arch::LevelID id, std::shared_ptr<NetworkBase> network) { 
  auto itr = parent_networks.find(id);
  COMET_ASSERT(itr == parent_networks.end(), "Adding a Child network for a previously existing child");
  parent_networks.emplace(std::make_pair(id, network));
}

std::shared_ptr<NetworkBase> ArchLevel::getNetwork(bool child_or_parent, arch::LevelID id) { 
  if (id == id_) return link_network;
  auto& net_map = (child_or_parent) ? child_networks : parent_networks;
  auto itr = net_map.find(id);
  COMET_ASSERT(itr != net_map.end(), "No Network for level_id " << id << " for target " << getLevelID());
  return itr->second;
}

bool ArchLevel::supportsReduction() const {
  if (isCompute()) return false;
  else {
    auto spec = std::get<MemorySpec>(getSpec());
    return spec.reduction_support.Get();
  }
}

}
