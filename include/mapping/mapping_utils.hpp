#pragma once
#include <vector>
#include <string>
#include <iostream>

#include "arch/attribute.hpp"
#include "problem/utils.hpp"
#include "problem/dimsize.hpp"

namespace mapping { 
  enum class mapping_t { SPATIAL=1, TEMPORAL=2};
  enum class stype_t {BROADCAST=0, REDUCTION=1, ALLGATHER=2, ALLREDUCE=3, SCATTER=4, GATHER=5};

  inline std::string stype_to_string(stype_t stype){
    if(stype == stype_t::BROADCAST) return "BROADCAST";
    else if(stype==stype_t::REDUCTION) return "REDUCTION";
    else if(stype==stype_t::ALLGATHER) return "ALLGATHER";
    else if(stype==stype_t::ALLREDUCE) return "ALLREDUCE";
    else if(stype==stype_t::SCATTER) return "SCATTER";
    else if(stype==stype_t::GATHER) return "GATHER";
  }
  // inline std::unordered_map<stype_t, std::string> stypeToStringMap = {
  //   {stype_t::BROADCAST, "BROADCAST"},
  //   {stype_t::REDUCTION, "REDUCTION"},
  //   {stype_t::ALLGATHER, "ALLGATHER"},
  //   {stype_t::ALLREDUCE, "ALLREDUCE"},
  //   {stype_t::SCATTER, "SCATTER"},
  //   {stype_t::GATHER, "GATHER"}
  // };

  enum class operation_t {None=0,GEMM=1, SOFTMAX=2, CONVOLUTION=3, MAX=4, EXP=6, DIV=7, MULT=8, SQRT=9, ADD=10, ROWSUM=11, ROWMAX=12};

  struct computation_attributes {
    bool specd = false;
    bool rmw = false;
    std::vector<problem::DimensionID> reduction_dimensions;
    std::map<problem::DimensionID, uint32_t> reduction_factors;
  };
  using LoopCoordinate = std::pair<problem::DimensionID, uint32_t>; // can be expressed as just uint32_t given the workload's knowledge
  using TilingMatrix = std::vector<problem::DimSizeExpression>; //outer vector is for spatial X and Y dimension
  using ComputeTilingMatrix = std::unordered_map<problem::DimensionID, uint32_t>;
  using TilingMatrixMap = std::unordered_map<problem::DimensionID, problem::DimSizeExpression>; //we have dimensionID here because various operations have different dimensions, DimSizeExpression because in our Mapping file we give tiling factors as a list of tilings for all the tensors at that level
  using SpatialTilingMatrixMap = std::vector<TilingMatrixMap>; // OUTSIDE vector because we can have factors for X and Y spatial dimensions

  inline uint32_t resolveMap(ComputeTilingMatrix map) {
    uint32_t retval=1;
    for (auto& pair : map) {
        retval *= pair.second;
    }
    return retval;
  }


}
