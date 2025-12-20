#include "arch/arch_spec.hpp"
namespace arch {

void MemorySpec::parseConfig(config::CompoundConfigNode spec_config) { 
  std::string name;
  if (spec_config.lookupValue("name", name)) { 
    name_ = config::parseName(name);
  }
  assert(spec_config.exists("attributes"));
  auto attributes = spec_config.lookup("attributes");
  uint64_t width=1;
  uint64_t depth=1;
  uint64_t ports=1;
  if (attributes.exists("width")) {
    attributes.getValue<uint64_t>("width", width);
  }
  width_ = width;
  if (attributes.exists("depth")) {
    attributes.getValue<uint64_t>("depth", depth);
  }
  depth_ = depth;
  // store per instance capacity in B
  capacity_ = width * depth;
  
  if (attributes.exists("ports")) { 
    attributes.getValue<uint64_t>("ports", ports);
  }
  ports_ = ports;
  
  bool bvalue = false;
  if (attributes.exists("doublebuffered")) {
    attributes.getValue<bool>("doublebuffered", bvalue);
  }
  double_buffered_ = bvalue;

  if (attributes.getValue<bool>("reduction", bvalue)) {
    reduction_support = bvalue;
  } else reduction_support = false;

  bool port_shared = false;
  if (attributes.exists("parent_child_ports_shared")){
    attributes.getValue<bool>("parent_child_ports_shared", port_shared);
  }
  parent_child_ports_shared_ = port_shared;

  //energy parameters
  float parent_read_energy(0), parent_write_energy(0), child_read_energy(0), child_write_energy(0);

  if(attributes.exists("parent_read_energy")){
    attributes.getValue<float>("parent_read_energy", parent_read_energy);
  }
  parent_read_energy_ = parent_read_energy;

  if(attributes.exists("parent_write_energy")){
    attributes.getValue<float>("parent_write_energy", parent_write_energy);
  }
  parent_write_energy_ = parent_write_energy;

  if(attributes.exists("child_read_energy")){
    attributes.getValue<float>("child_read_energy", child_read_energy);
  }
  child_read_energy_ = child_read_energy;

  if(attributes.exists("child_write_energy")){
    attributes.getValue<float>("child_write_energy", child_write_energy);
  }
  child_write_energy_ = child_write_energy;

  //bw
  double link_read_bw(0), link_write_bw(0), child_read_bw(0), child_write_bw(0), parent_read_bw(0), parent_write_bw(0);
  if (attributes.exists("link_read_bw")) {
    attributes.getValue<double>("link_read_bw", link_read_bw);
  }
  link_read_bw_ = link_read_bw;

  if (attributes.exists("link_write_bw")) {
    attributes.getValue<double>("link_write_bw", link_write_bw);
  }
  link_write_bw_ = link_write_bw;

  if (attributes.exists("child_read_bw")) { 
    attributes.getValue<double>("child_read_bw", child_read_bw);
  }
  child_read_bw_ = child_read_bw;

  if (attributes.exists("child_write_bw")) {
    attributes.getValue<double>("child_write_bw", child_write_bw);
  }
  child_write_bw_ = child_write_bw;
  
  if (attributes.exists("parent_read_bw")) {
    attributes.getValue<double>("parent_read_bw", parent_read_bw);
  }
  parent_read_bw_ = parent_read_bw;

  if (attributes.exists("parent_write_bw")) {
    attributes.getValue<double>("parent_write_bw", parent_write_bw);
  }
  parent_write_bw_ = parent_write_bw;

  double generic_bw;
  if (attributes.exists("bw")) { 
    attributes.getValue<double>("bw", generic_bw);
    parent_write_bw_ = generic_bw;
    parent_read_bw_ = generic_bw;
    child_read_bw_ = generic_bw;
    child_write_bw_ = generic_bw;
    link_read_bw_ = generic_bw;
    link_write_bw_ = generic_bw;
  }
  bvalue = false;
  if (attributes.getValue<bool>("dma_contiguous", bvalue)) { 
    dma_contiguous_ = bvalue;
  } else dma_contiguous_ = false; 
}

void MemorySpec::print(std::ostream& out) const {
  out << "MemorySpec::"<< name_ << " Capacity=" << capacity_ << " Width:" << width_ << " Depth:" << depth_ << " doubleBuffered:" << double_buffered_ << std::endl;
  out << " BW[PR/W, CR/W, LR/W]" ;
  out << parent_read_bw_ << " " << parent_write_bw_ << std::endl; 
  out << child_read_bw_ << " " << child_write_bw_ << std::endl; 
  out << link_read_bw_ << " " << link_write_bw_ << std::endl; 
}

void ComputeSpec::parseConfig(config::CompoundConfigNode spec_config) {
  std::string name="";
  assert(spec_config.exists("name"));
  if (spec_config.lookupValue("name", name)) { name_ = config::parseName(name);}

  assert(spec_config.exists("attributes"));
  auto attributes = spec_config.lookup("attributes");
  uint64_t capacity(0), datawidth(0), computeWidth(8), computeOpC(0), computeLatency(0); float compute_energy(0);
  uint64_t array_rows=32;
  uint64_t array_cols=32;
  uint64_t num_sa_rows=1;
  uint64_t num_sa_cols=1;

  assert(attributes.exists("computeOpC"));
  assert(attributes.exists("computeLatency"));

  if (attributes.exists("capacity")) {
    attributes.getValue<uint64_t>("capacity", capacity);
  }
  capacity_ = capacity;

  if (attributes.exists("datawidth")) {
    attributes.getValue<uint64_t>("datawidth", datawidth);
  }
  datawidth_ = datawidth;

  attributes.getValue<uint64_t>("computeOpC", computeOpC);
  computeOpC_ = computeOpC;
  
  attributes.getValue<uint64_t>("computeLatency", computeLatency);
  computeLatency_ = computeLatency;

  //energy
  if(attributes.exists("compute_energy")){
    attributes.getValue<float>("compute_energy", compute_energy);
  }
  compute_energy_ = compute_energy;

  if (attributes.exists("computeWidth")) {
      attributes.getValue<uint64_t>("computeWidth", computeWidth);
  }
  computeWidth_ = computeWidth;
  
  if(attributes.exists("array_rows")){
    attributes.getValue<uint64_t>("array_rows", array_rows);
  }
  array_rows_ = array_rows;

  if(attributes.exists("array_cols")){
    attributes.getValue<uint64_t>("array_cols", array_cols);
  }
  array_cols_ = array_cols;

  if(attributes.exists("num_sa_rows")){
    attributes.getValue<uint64_t>("num_sa_rows", num_sa_rows);
  }
  num_sa_rows_ = num_sa_rows;

  if(attributes.exists("num_sa_cols")){
    attributes.getValue<uint64_t>("num_sa_cols", num_sa_cols);
  }
  num_sa_cols_ = num_sa_cols;



  uint64_t startupLatency=0;
  attributes.getValue<uint64_t>("startupLatency", startupLatency);
  startupLatency_ = startupLatency;

  uint64_t bw;

  assert(computeOpC!=0);
  assert(computeLatency!=0);

  //compute energy for systolic array using power from HISIM https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10844846

  double p_pe = 0.0929; //Power of MAC unit in mW
  double p_pe_toprow = 0.0551; // Power of first (top) row MAC unit in mW
  double p_pe_cal = 0.0326; // Power calibration value in mW

  //control logic power
  double p_control = 1.614 + array_rows_.Get()*0.04557; // power of control circuit in mW
  if(name=="SystolicArray"){
    // compute_energy_ = (p_pe*(array_rows_.Get()*array_cols_.Get() - array_cols_.Get()) + p_pe_toprow*array_cols_.Get() + p_pe_cal + p_control)*num_sa_rows_.Get()*num_sa_cols_.Get(); //pJ 
    power_array_ = p_pe*(array_rows_.Get()*array_cols_.Get() - array_cols_.Get()) + p_pe_toprow*array_cols_.Get() + p_pe_cal;
    power_control_ = p_control;
  }
  double accum_energy = 8.97*0.49; //pJ from HISIM

  accumulator_energy_ = accum_energy;
  accum_latency_ = 1.0;
}

void ComputeSpec::print(std::ostream& out)  const{ 
  out << "ComputSpec::" << name_ << " ComputeOpC:" << computeOpC_ << " ComputeLatency:" << computeLatency_ << " computeWidth:" << computeWidth_ ;
  out << " ComputeMemory::" << capacity_ << " Datawidth:" << datawidth_ << std::endl;
}

std::ostream& operator << (std::ostream& out, const LevelSpec& spec) {
  std::visit([&out](auto&& spec){spec.print(out);}, spec);
  return out;
}
}








