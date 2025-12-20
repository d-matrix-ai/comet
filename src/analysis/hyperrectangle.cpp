#include <numeric>
#include <sstream>
#include "analysis/hyperrectangle.hpp"

namespace analysis { 
  std::pair<Point, Point> getMinMax(const LoopNestDescriptor& descriptor) { 
    //FIXME:: assure all descriptors are of same dimension
    auto dim_cnt = descriptor[0].size(); // is dimension count;
    Point min(dim_cnt);
    const auto& last_descriptor = descriptor.back();
    std::vector<Coordinate> max_point(dim_cnt);

    //descriptor is a map
    uint32_t cnt=0;
    for(auto& pair: last_descriptor){
      max_point[cnt] = pair.second.end;
      cnt++;
    }

    // for (uint32_t dim=0; dim< dim_cnt; dim++) {
    //   max_point[dim] = last_descriptor[dim].end; // the way descriptors are built the end should be the exclusive point in the space
    // }
    Point max(std::move(max_point));
    return std::move(std::make_pair(min, max));
  }

  std::string Point::Print() const {
    bool needs_comma = false;
    std::string retval = "[ ";
    for (auto coord: coordinates_) {
      if (needs_comma) retval += ", ";
      retval += std::to_string(coord);
      needs_comma = true;
    }
    return retval;
  }

  uint64_t Point::resolve() const {
    return std::accumulate(std::begin(coordinates_), std::end(coordinates_), 1, std::multiplies<uint64_t>());
  }

  Point& Point::operator=(Point other) { 
    coordinates_ = other.coordinates_;
    return *this;
  }

  Point operator-(const Point& lhs, const Point& rhs) { 
    Point retval(lhs.getRank());
    for (uint32_t dim=0; dim < retval.getRank(); dim++) { 
      retval[dim] = lhs[dim] - rhs[dim];
    }
    return retval;
  }
  Point operator+(const Point& lhs, const Point& rhs) { 
    Point retval(lhs.getRank());
    for (auto dim=0; dim < retval.getRank(); dim++) { 
      retval[dim] = lhs[dim] + rhs[dim];
    }
    return retval;
  }
  
  bool operator<=(const Point& lhs, const Point& rhs) { 
    bool is_less = true;
    for (auto dim=0; dim< lhs.getRank(); dim++) {
      is_less &= lhs[dim] <= rhs[dim];
    }
    return is_less;
  }

  bool operator>=(const Point& lhs, const Point& rhs) { 
    bool is_more=true;
    for (auto dim=0; dim< lhs.getRank(); dim++) {
      is_more &= lhs[dim] >= rhs[dim];
    }
    return is_more;
  }
  
  bool operator==(const Point& lhs, const Point& rhs) { 
    bool is_eq=(lhs.getRank() == rhs.getRank());
    for (auto dim=0; dim< lhs.getRank(); dim++) {
      is_eq &= lhs[dim] == rhs[dim];
    }
    return is_eq;
  }

  std::ostream& operator<<(std::ostream& out, const Point& p) { 
    out << p.Print();
  }

  bool HyperRectangle::hasPoint(const Point& p) const{
    bool less = p <= max;
    bool more = p >= min;
    return less && more;
  }

  bool HyperRectangle::hasOverlap(const HyperRectangle& tile) const{ 
    if (hasPoint(tile.getMax()) && hasPoint(tile.getMin())) return true;
    else return false; // working only with full overlap of tiles FIXME:: this needs to be changed;
  }

  bool operator==(const HyperRectangle& lhs, const HyperRectangle& rhs) { 
    return (lhs.getMin() == rhs.getMin()) && (lhs.getMax() == rhs.getMax());
  }
      
  bool HyperRectangleWalker::walk() {
    int retval=1;
    for(auto i=0; i<order.size();i++){ // this code is just for checking when one of the tensors factors are not given, like I removed softmax from mapping file so NO factors for tensor D. FIXME::snegi remove this later
      retval+=max_point[i];
    }
    if(retval==1){
      return false;
    }
    for (auto i = 0; i < order.size(); i++) { 
      // auto dim = order[i]; // this will not work if we have 5 dimensions at global level but current operation only uses 3 dimensions which can have random dimension ID FIXME::snegi verify if the change below is correct or not. Basically I have filled all the strides, cur_point, max_point vectors in the dimension order
      auto dim=i;
      cur_point[dim] += strides[dim];
      if (cur_point[dim] >= max_point[dim]) { cur_point[dim] = base_point[dim];}
      else return true;
    }
    return false;
  }

  HyperRectangleSet::HyperRectangleSet(LoopNestDescriptor loop_nest_descriptor): set_(0) {
    auto& max_loop = loop_nest_descriptor.back();
    auto dim_cnt = max_loop.size();
    Point offset(dim_cnt);
    problem::LoopOrder order(dim_cnt);
    std::iota(std::begin(order), std::end(order), 0);
    WalkAndAdd(offset, loop_nest_descriptor, order);
  }

  // for temporal loops 
  HyperRectangleSet::HyperRectangleSet(Point offset, LoopNestDescriptor loop_nest_descriptor, problem::LoopOrder order) {
    auto& max_loop = loop_nest_descriptor.back(); // back is the max loop
    auto dim_cnt = max_loop.size();
    // problem::LoopOrder order(dim_cnt);
    // std::iota(std::begin(order), std::end(order), 0);
    WalkAndAdd(offset, loop_nest_descriptor, order);
  }

  void HyperRectangleSet::WalkAndAdd(Point& offset, LoopNestDescriptor& loop_nest_descriptor, problem::LoopOrder& loop_order) {
    auto& max_loop = loop_nest_descriptor.back();
    auto& min_loop = loop_nest_descriptor.front();
    auto dim_cnt = max_loop.size();
    //make two sets of points
    Point min(offset);
    Point max(offset);
    Point min_max(offset);
    Point max_max(offset);

    std::vector<Coordinate> strides(dim_cnt);
    // create the min/max sets for the volume we are going to walk
    // for (auto dim =0; dim < dim_cnt; dim++) {
    uint32_t cnt=0; 
    for (auto dim: loop_order){
      max[cnt] = offset[cnt] + min_loop[dim].stride; //basepoint of walker //offset should also be indexed by cnt since offset is already sorted with order before
      max_max[cnt] = offset[cnt] + max_loop[dim].end + min_loop[dim].stride;
      min_max[cnt] = max_max[cnt] - min_loop[dim].stride;
      strides[cnt] = min_loop[dim].stride;
      cnt++;
    }
    HyperRectangleWalker min_walker(min, min_max, strides, loop_order); //max_point of min is stride of min_loop (min_max) because we will not walk the strides over time rather they will be mapped spatially
    HyperRectangleWalker max_walker(max, max_max, strides, loop_order);
    bool walkable = true;
    do {
      min = min_walker.get();
      max = max_walker.get();
      if (walkable) AddHR(std::move(min), std::move(max));
      auto min_walkable = min_walker.walk();
      auto max_walkable = max_walker.walk();
      COMET_ASSERT(min_walkable == max_walkable, "HYPERRECTANGLESET, min_walker and max_walker dont have save value");
      walkable = min_walkable & max_walkable;
    } while (walkable);
  }

  // FIXME:: currently comet requires HR to be disjoint for all rank -> Will need to be extended for future proofing and uneven spaces
  void HyperRectangleSet::AddHR(Point min, Point max) { 
    for (auto& hr: set_) {
      // TODO:: do we have to worry about overlapping but not inclusive 
      bool is_inside = hr.hasPoint(min) && hr.hasPoint(max);
      //hr.checkOverlap(min, max);  // FIXME::MANIKS:: does it matter if it overlaps or can we assume at this memory level we are not trying to optimize tile overlaps unless its full overlap
      if (is_inside) return;
    }

    set_.push_back(HyperRectangle(min, max));
  }
  void HyperRectangleSet::AddHR(HyperRectangle hr) { 
    set_.push_back(hr);
  }

  uint32_t HyperRectangleSet::Count() const {return set_.size();}

  bool HyperRectangleSet::hasOverlap(const HyperRectangle& tile) const{
    for (const auto& hr: set_) { 
      if (hr.hasOverlap(tile)) return true;
    }
    return false;
  }

  std::string HyperRectangleSet::print() const {
    std::stringstream debug_ss;
    int tile_cnt = 0;
    debug_ss << "HRSET::" << std::endl;
    for (const auto& hr: set_) {
      debug_ss << "Tile:" << tile_cnt << " Min:" << hr.getMin().Print() << " Max:" << hr.getMax().Print() << std::endl;
      tile_cnt++;
    }
    return debug_ss.str();
  }


  int findIndex(const problem::LoopOrder& vec, problem::ProjectionTerm value) {
      auto it = std::find(vec.begin(), vec.end(), value);
      if (it != vec.end()) {
          return std::distance(vec.begin(), it);
      } else {
          return -1; // Return -1 if the value is not found
      }
  }

  //TODO:: fuse dimsize expression and point if they are going to be the same
  Point ProjectPoint(const Point& point, problem::Projection projection, problem::LoopOrder order) { 
    auto order_of_tensor = projection.size();
    analysis::Point retval(order_of_tensor); //this can have more points from other dimensions of various tensors we have
    for (unsigned dim=0; dim < order_of_tensor; dim++) { 
      for (const auto& term: projection.at(dim)) { //projection is a vector, term is dimension id 
        
        // find index of term in order
        auto index = findIndex(order, term);

        retval[dim] += point[index];
      } 
    }
    return retval;
  }


  HyperRectangle ProjectHR(const HyperRectangle& hr, problem::Projection projection, problem::LoopOrder order) { 
    HyperRectangle retval(ProjectPoint(hr.getMin(), projection, order), ProjectPoint(hr.getMax(), projection, order));
    return retval;
  }
  
  HyperRectangleSet ProjectHRSet(const HyperRectangleSet& hrset, problem::Projection projection, problem::LoopOrder order) { 
    HyperRectangleSet retval;
    for (const auto& hr: hrset) { //TODO
      auto proj_hr = ProjectHR(hr, projection, order);
      retval.AddHR(proj_hr.getMin(), proj_hr.getMax());
    }
    // some checks
    auto size = retval[0].getDelta(); // is just max - min
    // ensure sizes are the same after projection 
    for (const auto& hr: retval) { 
      auto size_inc = hr.getDelta();
      COMET_ASSERT(size_inc == size, "HR's Have different sizes HR0: "<< size << " HR1:" << size_inc);
      size = size_inc;
    }
    return retval;
  }

  // should be used by projected HR sets
  HyperRectangleSet DiffTemporalHRSets(const HyperRectangleSet& rhs, const HyperRectangleSet& lhs) { 
    // should be used for same sizes sets
    COMET_ASSERT(rhs.Count() == lhs.Count(), "HR sets being compared across timeSteps  have different size");
    HyperRectangleSet retval;
    for( auto cnt=0; cnt < rhs.Count(); cnt++) {
      if (lhs.hasOverlap(rhs[cnt])) retval.AddHR(rhs[cnt]);
    }
    return retval;
  }

  
}


  



  