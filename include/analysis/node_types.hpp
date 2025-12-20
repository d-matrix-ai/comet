#include "analysis/node.hpp"

namespace analysis{
    using NodeTypes = std::variant<LoopNode, ColOpNode, OpNode>; //vector of loopnodes because we have global and local tensors
}