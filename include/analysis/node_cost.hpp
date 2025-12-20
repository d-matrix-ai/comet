#pragma once
#include <cstdint>
#include<vector>
#include "arch/arch_level.hpp"
#include "analysis/cost_info.hpp"

using arch::ArchLevel;


namespace analysis{

    using ArchLevelCost=std::map<std::pair<ArchLevel*, ArchLevel*>, std::vector<TargetChildCostVec>>; //std::vector because from one Node we can have same target and child. Like GB->buffer for Op1 and GB->buffer for Op2
    using NodeLevelCost=std::map<ArchLevel*, TargetChildCostVec>;
    using TotalCost=std::map<ArchLevel*, Time_t>;


}