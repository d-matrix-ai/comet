#pragma once
#include "analysis/hyperrectangle.hpp"
#include "arch/inc_utils.hpp"

namespace analysis{

  struct DataMovementInfo {
  Point tile_access_size; // size can be described with point expression, just co
  uint32_t num_unique_tiles = 0; // spatial loop needs this information  -> can be used for reduction and multicast factor calculation
                              // hop count also falls out of this
                              // temporal reuse will also use this -> for a temporal loop, num_unique_tiles = 1/0 : 0 -> temporal reuse
  uint32_t tile_count = 0;
  //following are only used in temporal loops
  uint32_t link_transfers = 0; // number of link_transfers possible between children (will be 0 for first tile )
  uint64_t timesteps_for_tile = 0; // timesteps needed for a tile to be moved to the dependent
  };


  using TensorMovementInfo = std::vector<std::vector<DataMovementInfo>>; //outer vector for number of tensors and inner vector for number of tiles for that tensor


}
