#pragma once

#include <map>
#include <vector>

#include "util/comet_assert.h"
#include "compound_config/compound_config.hpp"
#include "analysis/hyperrectangle.hpp"
#include "problem/dimsize.hpp"
#include "problem/utils.hpp"
#include "mapping/mapping_utils.hpp"
#include "util/simd_component_cost.hpp"

using mapping::ComputeTilingMatrix;

namespace problem { 


  //DimSizeExpression  -> wrapper around std::vector of size dimensionCount

  //class Workloads; //forward declaration

  class Workload {
    public:
      Workload() {
        dim_count=0;
        shape_count = 0;
        tensor_count = 0;
        common_tensor_count=0;
        // parse(problem_config);
      }
      
      std::vector<std::string> ins_;
      std::string out_;
      std::string name_; 
      
      int getDimensionCount() const {return dim_count;}
      int getTensorCount() const {return tensor_count;}
      std::vector<DimensionID> getDimensionsVector() const;
      int getFlattenedDimensions() const;
      size_t getDimensionsForShape(ShapeID shape_id) {return shapeDimensions_[shape_id].size();}
      TensorID getTensorID(std::string tensor_name) const;
      std::vector<TensorID> getTensors() const;
      std::string getDimName(DimensionID dim_id) {return dimensionIDToDimName_.at(dim_id);}
      std::string getTensorName(TensorID tens_id) {return tensorNames_.at(tens_id);}
      DimensionID getDimID(std::string dimension_name) const;
      const std::string& ToString();
      DimSizeExpression getTensorSize(TensorID tens_id) const {return DimSizes_;} // current usecaseis just to get full dimension size expression -> analysis will need to understand projection
      DimSizeExpression getInstanceSize() const {return DimSizes_;}
      std::map<std::string, DimensionID> getDimensionMap(){return dimNamesToDimensionID_;}
      std::map<TensorID, std::string> get_TensorNameMap(){return tensorNames_;}

      std::map<std::string, TensorID> get_TensorIDMap(){return tensorNamesToTensorID_;}

      // DimSizeExpression projectDimExpressionOnTensor(const DimSizeExpression& full_rank_expression, TensorID tensor_id, ShapeID shape_id) const ;
      DimSizeExpression projectDimExpressionOnTensor(ComputeTilingMatrix& dim_sizes, TensorID tensor_id) ;

      bool tensorHasDimensionRank(TensorID tensor_id, ShapeID shape_id, DimensionID dim_id) const;

      Projection getProjection(TensorID tensor_id) {return projection_map_.at(tensor_id);}

      uint32_t getDimSize(const DimensionID dim_id) {return DimSizes_[dim_id];}

      bool isTensorRW(TensorID tensor_id) {return TensorRW.at(tensor_id);}
      friend std::ostream& operator<<(std::ostream& out, const Workload& workload);
      
      //compound op related functions
      inline void set_name(const std::string & name){name_ = name;}
      void set_io(const std::vector<std::string>& ins, const std::vector<std::string>& outs);
      inline const std::vector<std::string>& get_ins() const { return ins_; }
      inline const std::string & get_out() const {return out_;}
      inline const std::string & get_name() const {return name_;} 

      friend class Workloads; //Workloads class can access all the private, public and protected functions and variables

    private:
      std::map<std::string, ShapeID> shapeNamesToID_; // ShapeID index into this vector 
      std::map<ShapeID, std::string> shapeNames_;
      ShapeID shape_count;
      const ShapeID newShape(const std::string& shape_name);
      std::map<ShapeID, std::vector<DimensionID>> shapeDimensions_;
      
      std::map<std::string, DimensionID> dimNamesToDimensionID_;
      std::map<DimensionID, std::string> dimensionIDToDimName_;
      DimensionID dim_count;
      void newDimension(const std::string& dim_name, std::map<std::string, DimensionID> global_dim_ids);
      void addDimensionToShape(const ShapeID shapeid, const DimensionID dimid);

      std::map<std::string, CoefficientID> coefficientNamesToID_;
      std::map<CoefficientID, std::string> coefficientIDToName_;
      CoefficientID coefficient_count;
      const CoefficientID newCoefficient(const std::string& coefficient_name);

      std::map<TensorID, std::string> tensorNames_;
      std::map<std::string, TensorID> tensorNamesToTensorID_;
      // std::vector<bool> TensorRW;
      std::map<TensorID, bool> TensorRW;

      TensorID tensor_count;
      TensorID common_tensor_count;
      const TensorID newTensor(const std::string& tensor_name, std::map<std::string, TensorID> global_tensor_ids);
      void SetRWForTensor(const TensorID tensor_id);

      std::map<TensorID, Projection> projection_map_; // shape to tensor to projection
      void AddProjection(TensorID tensor_id, Projection&& proj); 
      void parse(config::CompoundConfigNode problem_config);
      std::string parseProblem(config::CompoundConfigNode problem_config, ComputeTilingMatrix& dim_sizes, std::map<std::string, DimensionID> global_dim_ids, std::map<std::string, TensorID> global_tensor_ids);

      DimSizeExpression DimSizes_;
      void SetDimSizes(DimSizeExpression&& dimsizevec) {DimSizes_ = dimsizevec;}

  };

  class Workloads{ 

    
    public:
      std::unordered_map<std::string, std::shared_ptr<problem::Workload>> workloads_;
      std::vector<std::string> ins_;
      std::vector<std::string> outs_;
      
      std::vector<TensorID> intermediate_tensors;

      Workloads(){
        dim_count=0;
        tensor_count = 0; //number of tensors
      }

      bool add_workload(const std::string& name, std::shared_ptr<Workload>&workload);
      
      std::shared_ptr<problem::Workload> get_workload(const std::string& op_name) const {
        COMET_ASSERT(workloads_.count(op_name), op_name << "NOT FOUND");
        return workloads_.at(op_name);
      }

      std::map<std::string, DimensionID> dimNamesToDimensionID_;
      std::map<DimensionID, std::string> dimensionIDToDimName_;
      DimensionID dim_count;

      std::map<std::string, CoefficientID> coefficientNamesToID_;
      std::map<CoefficientID, std::string> coefficientIDToName_;
      CoefficientID coefficient_count;

      std::map<TensorID, std::string> tensorNames_;
      std::map<std::string, TensorID> tensorNamesToTensorID_;
      TensorID tensor_count;
      // std::vector<bool> TensorRW;
      std::map<TensorID, bool> TensorRW;

      int getDimensionCount() const {return dim_count;}
      std::string getDimName(DimensionID dim_id) {return dimensionIDToDimName_.at(dim_id);}
      TensorID getTensorID(std::string tensor_name) const;
      std::string getTensorName(TensorID tens_id) const {return tensorNames_.at(tens_id);}

      void set_io(const std::vector<std::string>& ins, const std::vector<std::string>& outs);
      void set_coeffs(const config::CompoundConfig& coeffs);
      DimensionID getDimID(std::string dimension_name) const;

      // std::map<DimensionID, std::int32_t> workloadDimSizes_;

      ComputeTilingMatrix workloadsDimSizes_;
      void set_dim_sizes(const std::string dim, uint32_t dim_size);

      // DimSizeExpression DimSizes_;
      // void SetDimSizes(DimSizeExpression&& dimsizevec) {DimSizes_ = dimsizevec;}


      void newDimension(const std::string& dim_name);
      
      std::vector<Projection> Projections;
      void ParseWorkloads(config::CompoundConfigNode config, config::CompoundConfigNode constants);


  };

}
