#pragma once
#include <list>
#include <vector>
namespace problem{ 
  using ShapeID = size_t;
  using TensorID = size_t;
  using DimensionID = size_t;
  using CoefficientID = size_t;
  
  using ProjectionTerm = DimensionID;
  using ProjectionExpression = std::list<ProjectionTerm>;
  using Projection = std::vector<ProjectionExpression>;
  using LoopOrder = std::vector<problem::DimensionID>;
}
