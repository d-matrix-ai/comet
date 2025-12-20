#pragma once

#include <variant>

#include "arch/attribute.hpp"
#include "compound_config/compound_config.hpp"
namespace arch { 
struct MemorySpec
{
  private:
    void parseConfig(config::CompoundConfigNode spec_config);
  public:
    MemorySpec()=default;
    MemorySpec(config::CompoundConfigNode spec_config) {parseConfig(spec_config); }
    void print(std::ostream& out) const;
    std::string name_="";
    Attribute<bool> parent_child_ports_shared_;
    Attribute<std::uint32_t> parent_read_bw_;
    Attribute<std::uint32_t> parent_write_bw_;
    Attribute<std::uint32_t> child_read_bw_;
    Attribute<std::uint32_t> child_write_bw_;
    Attribute<std::uint32_t> link_read_bw_;
    Attribute<std::uint32_t> link_write_bw_;
    Attribute<bool> double_buffered_;
    Attribute<std::uint32_t> ports_; // per parent/child/link
    Attribute<std::uint64_t> width_;
    Attribute<std::uint64_t> depth_;
    Attribute<bool> reduction_support;
    Attribute<uint32_t> reduction_cost=1;
    std::uint64_t capacity_;  
    Attribute<bool> dma_contiguous_;

    Attribute<double> parent_read_energy_;
    Attribute<double> parent_write_energy_;
    Attribute<double> child_read_energy_;
    Attribute<double> child_write_energy_;



    uint64_t getCapacity() const {return capacity_;}
  
};

struct ComputeSpec
{
  private:
    void parseConfig(config::CompoundConfigNode spec_config);
  public:
    ComputeSpec()=default;
    ComputeSpec(config::CompoundConfigNode spec_config) {parseConfig(spec_config);}
    void print(std::ostream& out) const;

    uint64_t getCapacity() const {return capacity_.Get();}

    std::string name_="";
    Attribute<std::uint8_t> datawidth_;
    Attribute<std::uint64_t> capacity_;
    Attribute<std::uint64_t> computeWidth_;
    Attribute<std::uint64_t> computeOpC_;
    Attribute<std::uint64_t> computeLatency_;
    Attribute<std::uint64_t> startupLatency_;
    Attribute<float> compute_energy_;
    Attribute<std::uint64_t> array_rows_;
    Attribute<std::uint64_t> array_cols_;
    Attribute<std::uint64_t> num_sa_rows_;
    Attribute<std::uint64_t> num_sa_cols_;
    Attribute<float> power_array_;
    Attribute<float> power_control_;
    Attribute<float> accum_latency_;

    Attribute<double> accumulator_energy_;

  };

using LevelSpec = std::variant<MemorySpec, ComputeSpec>;    
std::ostream& operator<<(std::ostream& out, const LevelSpec& spec);
}
