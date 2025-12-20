#pragma once
#include <memory>
#include <algorithm>
#include "problem/dimsize.hpp"
#include "problem/utils.hpp"



namespace analysis{
    struct LoopDescriptor {
        uint64_t end;
        uint64_t stride;
    };
  using LoopNestDescriptor = std::vector<std::map<problem::DimensionID, LoopDescriptor>>; // outer vec is indexed by spatial dimensions X and Y, inner map is indexed by DimID, // need map internally because there are multiple workloads so at leaf level different workloads can have different dimensions involved with them. Take example of two GEMMs in GEMM1 MNK is involved and in GEMM2 MLN is involved

  using Coordinate = int; // -1 means unused -> Dimension ID is the order
  class Point {
    protected:
      std::vector<Coordinate> coordinates_;
      // std::map<problem::DimensionID, Coordinate> coordinates_;
    public:
      Point(): coordinates_(0){}
      Point(uint32_t dimension_cnt) : coordinates_(dimension_cnt, 0) {}
      Point(std::vector<Coordinate>&& coordinates): coordinates_(coordinates) {}
      Point(Point& point): coordinates_(point.coordinates_) {}
      Point(Point&& point): coordinates_(std::move(point.coordinates_)) {}
      Point(const Point& point): coordinates_(point.coordinates_) {}
      
      Point& operator= (Point other);
      friend void swap(Point& first, Point& second);
      
      uint32_t getRank() const {return coordinates_.size();}
      std::vector<Coordinate> get() { return coordinates_;}
      Coordinate& operator[](int index) {return coordinates_[index];}
      Coordinate operator[](int index) const {return coordinates_[index];}
      std::string Print() const;
      uint64_t resolve() const;
      
  }; 
  Point operator-(const Point& lhs, const Point& rhs) ; 
  Point operator+(const Point& lhs, const Point& rhs) ; 
  
  bool operator<=(const Point& lhs, const Point& rhs) ; 

  bool operator>=(const Point& lhs, const Point& rhs) ;
  bool operator==(const Point& lhs, const Point& rhs) ;
  std::ostream& operator<<(std::ostream& out, const Point& p);

  std::pair<Point, Point> getMinMax(LoopNestDescriptor& descriptor);
  // class to form the tiles in a given time step
  // takes mesh size -> vector
  // and a Point vector<(dimension, index, stride)>
  // has utility functions to compare Hyperrectangles for finding access/multicast/reduction/linktransfer

  class HyperRectangle { 
    protected:
      Point min;
      Point max;
      problem::LoopOrder order;
    public:
      HyperRectangle() : min(0), max(0) {}
      HyperRectangle(Point&& min, Point&& max): min(min), max(max) {}
      HyperRectangle(Point& min, Point& max): min(min), max(max) {}//loop_coordinates);
      HyperRectangle(const HyperRectangle& hr) : min(hr.min), max(hr.max) {}
      Point getMin() const {return min;}
      Point getMax() const {return max;}
      void setMin(Point new_min) {min = new_min;}
      void setMax(Point new_max) {max = new_max;}
      Point getDelta() const { return max - min;}
      bool hasPoint(const Point& p) const;
      bool hasOverlap(const HyperRectangle& tile) const;
      void checkOverlap();
  };

  bool operator==(const HyperRectangle& rhs, const HyperRectangle& lhs);

  class HyperRectangleWalker { 
    private:
      Point base_point;
      Point cur_point;
      Point max_point;
      std::vector<Coordinate> strides;
      problem::LoopOrder order;
      Coordinate cur_dim_idx;
    public:
      HyperRectangleWalker(Point base_point, Point max_point, std::vector<Coordinate> strides, problem::LoopOrder order) : base_point(base_point), cur_point(base_point), max_point(max_point), strides(strides) , order(order) {}
      bool walk();
      Point get() {return cur_point;}
  };

  class HyperRectangleSet {
    private:
      std::vector<HyperRectangle> set_;

    public:
      HyperRectangleSet(): set_(0) {}
      HyperRectangleSet(uint32_t aahr_count) : set_(aahr_count) {}
      HyperRectangleSet(LoopNestDescriptor loop_nest_descriptor);
      HyperRectangleSet(Point offset, LoopNestDescriptor loop_nest_descriptor, problem::LoopOrder order); // form the tiles based on the loop nest
      HyperRectangle& operator[](uint32_t index) {return set_[index];}
      HyperRectangle operator[](uint32_t index) const {return set_[index];}
      std::vector<HyperRectangle>::iterator begin() {return set_.begin();}
      std::vector<HyperRectangle>::iterator end() {return set_.end();}
      std::vector<HyperRectangle>::const_iterator begin() const {return set_.cbegin();}
      std::vector<HyperRectangle>::const_iterator end() const {return set_.cend();}

      void WalkAndAdd(Point& offset, LoopNestDescriptor& loop_nest_descriptor, problem::LoopOrder& order);
      void AddHR(Point min, Point max);
      void AddHR(HyperRectangle hr);
      uint32_t Count() const;
      bool hasOverlap(const HyperRectangle& tile) const;
      std::string print() const;
  };

  // helper class for uniform hyperrectangles which can be expressed with base and loop nest -> this should be the common case but above is easier to visualize
  //class UniformHyperRectangleSet { 
  //  protected:
  //    HyperRectangle base;
  //};


  Point ProjectPoint(const Point& point, problem::Projection projection, problem::LoopOrder order);
  HyperRectangle ProjectHR(const HyperRectangle& Point, problem::Projection projection, problem::LoopOrder order);
  HyperRectangleSet ProjectHRSet(const HyperRectangleSet& set, problem::Projection projection, problem::LoopOrder order);

  HyperRectangleSet DiffTemporalHRSets(const HyperRectangleSet& rhs, const HyperRectangleSet& lhs);



}