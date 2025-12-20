#pragma once

#include "arch/arch_spec.hpp"
#include "arch/compute_base.hpp"
#include "arch/network_spec.hpp"
#include "compound_config/compound_config.hpp"
#include "util/comet_assert.h"
#include "arch/inc_utils.hpp"
#include "util/simd_component_cost.hpp"

// Arch level Class to construct levelling in the arch to be able to define connection approproately
namespace arch {

// using arch::NetworkBase;

// class NetworkBase {};

class Network;
class ArchLevel
{
  public:
    ArchLevel(): name_(""), instance_array_(2,1), is_compute_(false), is_root_(false), has_link_transfers_(false), drain_levels_(0,0), fill_levels_(0,0), read_levels_(0,0), write_levels_(0,0) , base_dimensions(2){}
    //ArchLevel(const ArchLevel& other);
    //ArchLevel(const ArchLevel&& other);
    //ArchLevel& operator=(const ArchLevel& other);
    //ArchLevel& operator=(ArchLevel&& other);

    void AddChild(bool drain_or_fill, ArchLevel* child);
    void AddConnection(ArchLevel* child) { AddChild(false, child); AddChild(true, child);}
    void AddParent(bool read_or_write, ArchLevel* parent);
    void setCompute() {is_compute_ = true;}
    void setSpec(LevelSpec& level_spec) {spec_ = level_spec;}
    bool isChild(std::string& child_name) const;
    bool isChild(const ArchLevel* child_ptr) const;

    void ResolveSpecs();
    LevelSpec getSpec() const { return spec_;}

    void setRoot() {is_root_ = true;}
    bool isRoot() {return is_root_;}
    
    void setPassthrough() {is_passthrough_ = true;}
    bool isPassthrough() {return is_passthrough_;}

    bool isCompute() const {return is_compute_;}

    std::string getName() const {return name_;}
    void setName(std::string name) {name_ = name;}

    void setLevelID(LevelID level_id) { id_ = level_id;}
    LevelID getLevelID() const {return id_;}

    void setAttributes(config::CompoundConfigNode attribute_config);
    void AddParentNetwork(arch::LevelID, std::shared_ptr<arch::NetworkBase> network);
    void AddChildNetwork(arch::LevelID level, std::shared_ptr<arch::NetworkBase> network);
    void setLinkNetwork(std::shared_ptr<arch::NetworkBase> network) {link_network = std::move(network);}
    std::shared_ptr<arch::NetworkBase> getLinkNetwork() {return link_network;}

    std::shared_ptr<arch::NetworkBase> getNetwork(bool child_or_parent, arch::LevelID);
    const std::set<LevelID>& getParents() { return parent_set;}
    const std::set<LevelID>& getChildren() { return child_set;}
    
    uint64_t instanceCapacity() const;
    uint32_t getInstanceSize() const; // returns the size information for 

    bool supportsReduction() const;
  private:
    LevelID id_;
    std::vector<uint64_t> instance_array_; // each entry is a new dimension
    uint8_t base_dimensions; // dimension to know what is the marker on which instance_array is for lowest level node
    LevelSpec spec_;
    std::string name_;

    bool is_compute_;
    bool is_root_;
    bool is_passthrough_;
    bool has_link_transfers_;
    
    std::map<arch::LevelID, std::shared_ptr<arch::NetworkBase>> child_networks;
    std::map<arch::LevelID, std::shared_ptr<arch::NetworkBase>> parent_networks;
    std::shared_ptr<arch::NetworkBase> link_network;

    std::set<arch::LevelID> child_set;
    std::set<arch::LevelID> parent_set;

    //std::shared_ptr<Compute> compute = nullptr; TODO:: make compute factory

    std::vector<ArchLevel*> drain_levels_; // level that drains into this level (every instance in instance array is connected to an instance of the drain level)
    std::vector<ArchLevel*> fill_levels_; // level that is filled by this level (every instance in instance array is connected to an instance of the fill level)
    std::vector<ArchLevel*> read_levels_; // level that is read by this level is a 1:1 connection, this would be fill_level of the read_level
    std::vector<ArchLevel*> write_levels_; // level that is backing memory for this level, so for writable tensors we have to write back


};

}





