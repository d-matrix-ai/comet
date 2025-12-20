#pragma once

#include "analysis/nest_analysis.hpp"
#include "util/logger.hpp"

namespace analysis{

    // bool isStringInVector(const std::vector<std::string>& vec, const std::string& str){
    //     return std::find(vec.begin(), vec.end(), str) != vec.end();
    // }


    void DataMovementInfoConstructor::visitDataMovementTileNode(const DataMovementTileNode* node){
        
        // auto it = analysis_.workload_mapping_graph.find(node);

        // std::vector<LoopNode>& loop_node = analysis_.workload_mapping_graph[node]; //std::vector<loopNode>

        // auto loop_node= std::get<std::vector<LoopNode>>(analysis_.workload_mapping_graph[node]);
        auto& loop_nodes= std::get<LoopNode>(analysis_.workload_mapping_graph[node]);

        // int cnt=0;
        // for (auto gl_in_loopnode: loop_node){
        //     if (gl_in_loopnode.type == mapping_t::TEMPORAL){
                

        //         ComputeTemporalInfo(gl_in_loopnode, node);
                
        //         // for (auto& per_tensorNode: gl_in_loopnode.descriptor_){ //loop over all the tensors loopnode
        //         //     ComputeTemporalInfo(per_tensorNode.first, per_tensorNode.second);
        //         // }
        //         // ComputeTemporalInfo()
        //     } else {
                
        //         //check if the parent also has same target and child as this node
        //         auto parent_node = static_cast<const DataMovementTileNode*>(node->get_parent());
        //         auto parent_loop_node = std::get<std::vector<LoopNode>>(analysis_.workload_mapping_graph[parent_node]);

        //         ComputeSpatialInfo(gl_in_loopnode, node, parent_loop_node[cnt], cnt);
        //     }
        //     cnt++;
        // }

        // if (loop_node[0].type == mapping_t::TEMPORAL ){
        //     ComputeTemporalInfo(loop_node);
        // } else {
        //     ComputeSpatialInfo(loop_node);
        // }


        std::cout<<"!!! Node name::" <<node->get_name()<<"\n";

        if (loop_nodes.type == mapping_t::TEMPORAL){
            ComputeTemporalInfo(loop_nodes, node);
        } else{

            auto parent_node = static_cast<const DataMovementTileNode*>(node->get_parent());
            auto& parent_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[parent_node]);
            
            //find bindings for the children of the parent if it has multiple children
            // std::vector<InterTileBinding::type_t> binding;
            // if(parent_node->get_children().size()>1){
            //     for(auto&child:parent_node->get_children()){
            //         if(child->get_type()==Node::type_t::InterTileBinding){
            //             const auto& child_node=static_cast<const InterTileBinding*>(child);
            //             binding.emplace_back(child_node->get_inter_tile_binding_type());
            //         }                    
            //     }
            //     if(std::find(binding.begin(), binding.end(), InterTileBinding::type_t::pipeline)!=binding.end()){
            //         // if the binding is pipeline then fuse the loop nodes
            //         ComputeSpatialInfo_fused(loop_nodes,node,parent_loop_node);
            //     } else{
            //         ComputeSpatialInfo(loop_nodes,node, parent_loop_node);
            //     }
            // }
            // else{
            //     ComputeSpatialInfo_fused(loop_nodes, node, parent_loop_node);  
            // }              
            ComputeSpatialInfo_fused(loop_nodes, node, parent_loop_node);
        }
        
        for(auto child: node->get_children()){
            child->accept(this);
        }
        
    }

    // void DataMovementInfoConstructor::ComputeTemporalInfo(problem::TensorID tid, const LoopNestDescriptor& loop_node){
    void DataMovementInfoConstructor::ComputeTemporalInfo(LoopNode& loop_nodes, const DataMovementTileNode* node){

        //InitInfo
        // analysis_.datamovement_info[node].emplace_back();
        // analysis_.datamovement_info[node].back().info = &loop_nodes;
        // analysis_.datamovement_info[node].info = &loop_nodes;
        for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){
            auto tid       = loop_nodes.tensors[tens_cnt];
            auto loop_node = loop_nodes.descriptor_[tens_cnt];
            auto order     = loop_nodes.order[tens_cnt];
            auto tens_tag  = loop_nodes.tags[tens_cnt];


            std::shared_ptr<problem::Workload> workload=std::make_shared<problem::Workload>();

            if(analysis_.workloads_.workloads_.find(tens_tag)!=analysis_.workloads_.workloads_.end()){
                workload = analysis_.workloads_.workloads_[tens_tag];
            }
            else {
                // workload pointer is none when the tag is to collective operation which does not exist in problem.yaml file
                //therefore search manually

                //find the workload to which the tensor belongs to
                auto tensor_name = analysis_.workloads_.getTensorName(tid);
                for(auto& workload_pair: analysis_.workloads_.workloads_){
                    workload = workload_pair.second; //pointer to workload class
                    if(isStringInVector(workload_pair.second->ins_, tensor_name)||(workload_pair.second->out_ ==tensor_name)){
                        // if tensor belongs to workload1 set workload=workload1 
                        break;
                    } else{
                        continue;
                    }
                }
            }

            auto projection_expression = workload->getProjection(tid);

            auto& descriptor_map = loop_node.back(); //temporal has only one entry in LoopNest descriptor

            auto dim_cnt = descriptor_map.size();
            Point first_corner(dim_cnt);
            Point next_corner(dim_cnt);
            Point strides(dim_cnt);
            Point max_point(dim_cnt);
            
            uint32_t cnt=0;
            //fill the points in order
            for(auto& dim_id: order){
                strides[cnt] = descriptor_map.at(dim_id).stride;
                max_point[cnt] = descriptor_map.at(dim_id).end;
                next_corner[cnt] = first_corner[cnt] + strides[cnt];
                cnt++;
            }
            // for(auto& pair: descriptor_map){ //dimID, (end,stride) pair
            //     strides[cnt]     = pair.second.stride;
            //     max_point[cnt]   = pair.second.end; 
            //     next_corner[cnt] = first_corner[cnt] + strides[cnt];
            //     cnt++;
            // }

            HyperRectangleWalker space_walker(first_corner, max_point, strides.get(), order); 
            //walk the space walker in all the dimension by changing the cur_point 
            // space_walker.walk(true);
            HyperRectangle tile(first_corner, next_corner); //this HR has all the dimensions which are not related to this tensor
            auto proj_tile = ProjectHR(tile, projection_expression, order); //but the projection helps us to remove the unrelated dimensions   ---> make sure that the points in the tile have same order as the  
            DataMovementInfo cur_info;
            std::vector<DataMovementInfo> vec_data_movement; 

            cur_info.tile_access_size = proj_tile.getDelta();
            cur_info.num_unique_tiles = 1;
            cur_info.tile_count = 1; // this is equal to number of unique childrens
            cur_info.link_transfers = 0;
            cur_info.timesteps_for_tile = loop_nodes.relative_timestep_to_dependent[tens_cnt];
            vec_data_movement.emplace_back(cur_info);
            // analysis_.datamovement_info[node].m
            // add info
            // analysis_.datamovement_info[node].back().movement_info.at(tid).emplace_back(std::move(cur_info)); //.back to access global or local tensor location
            // analysis_.datamovement_info[node].movement_info.at(tid).emplace_back(std::move(cur_info)); //.back to access global or local tensor location
            
            auto timesteps_for_tile = loop_nodes.relative_timestep_to_dependent[tens_cnt];

            for(auto idx=1; idx<loop_nodes.iteration_count[tens_cnt]; idx++){
                
                COMET_ASSERT(space_walker.walk(), "Temporal Loop iteration count is incorrect as walker cant walk the descriptor but iteration count is still valid iter:" << idx ); 

                first_corner = space_walker.get();
                next_corner  = first_corner + strides; //doubt::snegi should we walk in all dimensions at once? 

                //after walking the space walker the cur_point in walker should be the next corner
                // next_corner = space_walker.get();

                HyperRectangle next_tile(first_corner, next_corner);
                auto next_proj_tile = ProjectHR(next_tile, projection_expression, order);
                DataMovementInfo dm_info;
                if(next_proj_tile == proj_tile) {dm_info.num_unique_tiles = 0;} //do not consider reuse for tensors associated with SIMD
                // if(next_proj_tile == proj_tile && tens_tag.find("GEMM")!=std::string::npos) {dm_info.num_unique_tiles = 0;} //do not consider reuse for tensors associated with SIMD
                else dm_info.num_unique_tiles =1;

                // if you are draining and there is reduction happening unique tiles can be 0 but that means you are doing a reduction op on the same tile, so total amount of data accessed in that iteration is 1 tile --> Rather than doing this in the arch file just adding this here
                // if(loop_nodes.tensor_is_rw[tens_cnt] && loop_nodes.rmw[tens_cnt] && dm_info.num_unique_tiles==0){
                // if(loop_nodes.tensor_is_rw[tens_cnt] && dm_info.num_unique_tiles==0){
                //     dm_info.num_unique_tiles = 1;
                // }   

                //default values
                dm_info.tile_access_size = next_proj_tile.getDelta();
                dm_info.tile_count = 1;
                dm_info.timesteps_for_tile = timesteps_for_tile;

                //add info
                // analysis_.datamovement_info[node].back().movement_info.at(tid).emplace_back(std::move(cur_info)); //.back to access global or local tensor location
                vec_data_movement.emplace_back(dm_info);
                // analysis_.datamovement_info[node].at(tid).emplace_back(std::move(cur_info)); //.back to access global or local tensor location

                timesteps_for_tile += loop_nodes.relative_timestep_to_dependent[tens_cnt];
                proj_tile = next_proj_tile;
            }
            analysis_.datamovement_info[node].emplace_back(std::move(vec_data_movement));

        }
    }

    void DataMovementInfoConstructor::ComputeSpatialInfo_fused(LoopNode& loop_nodes, const DataMovementTileNode* node, LoopNode& parent_loop_nodes){

        //InitInfo
        // analysis_.datamovement_info[node].emplace_back();
        // analysis_.datamovement_info[node].back().info = &loop_nodes;
        // analysis_.datamovement_info[node].info = &loop_nodes;

        auto parent_node = static_cast<const DataMovementTileNode*>(node->get_parent()); 

        for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){

            auto tid       = loop_nodes.tensors[tens_cnt];
            auto loop_node = loop_nodes.descriptor_[tens_cnt];
            auto order     = loop_nodes.order[tens_cnt];
            auto tens_tag  = loop_nodes.tags[tens_cnt];

            // need to find index of this tensor(tid) in parent
            auto idx = parent_loop_nodes.tid_idx[tid]; // now need the index in this vector which can be derived from the child index (index of this child out of all children)
            size_t child_idx;
            for(auto& i:idx){
                if(parent_loop_nodes.tags[i]==tens_tag){ //if the tag of the tensor in parent node matches the tag of the tensor in the child node
                    child_idx=i;
                    break;
                }
            }
            auto parent_loop_descriptor = parent_loop_nodes.descriptor_[child_idx];
            auto parent_datamovement_info = analysis_.datamovement_info[parent_node];
            auto& parent_datamovement_info_per_tensor = parent_datamovement_info[child_idx];

            auto target = loop_nodes.target[tens_cnt];
            auto target_parent = parent_loop_nodes.target[child_idx];

            if(target == target_parent){
                parent_loop_nodes.spatial_node_exist = true;
                //update the iteration count for spatial node to the iterations from temporal parent
                // loop_nodes.iteration_count = parent_loop_nodes.iteration_count; 

                auto tens_idx_in_parent = parent_loop_nodes.tid_idx[tid];
                loop_nodes.same_target_as_parent = true;

                for(auto tidx:tens_idx_in_parent){
                    if(tens_tag == parent_loop_nodes.tags[tidx]) loop_nodes.iteration_count[tens_cnt] = parent_loop_nodes.iteration_count[tidx];
                }



            }


            // auto workload = analysis_.workloads_.workloads_[tens_tag];

            std::shared_ptr<problem::Workload> workload=std::make_shared<problem::Workload>();

            if(analysis_.workloads_.workloads_.find(tens_tag)!=analysis_.workloads_.workloads_.end()){
                workload = analysis_.workloads_.workloads_[tens_tag];
            }
            else {
                // workload pointer is none when the tag is to collective operation which does not exist in problem.yaml file
                //therefore search manually

                //find the workload to which the tensor belongs to
                auto tensor_name = analysis_.workloads_.getTensorName(tid);
                for(auto& workload_pair: analysis_.workloads_.workloads_){
                    workload = workload_pair.second; //pointer to workload class
                    if(isStringInVector(workload_pair.second->ins_, tensor_name)||(workload_pair.second->out_ ==tensor_name)){
                        // if tensor belongs to workload1 set workload=workload1 
                        break;
                    } else{
                        continue;
                    }
                }
            }

            auto projection_expression = workload->getProjection(tid);

            auto& spatial_descriptor = loop_node;

            auto& temporal_descriptor = parent_loop_descriptor.back();

            auto dim_cnt = loop_node.back().size();
            Point first_corner(dim_cnt);
            Point next_corner(dim_cnt);
            Point strides(dim_cnt);
            Point max_point(dim_cnt);

            uint32_t cnt=0;
            for(auto& dim_id: order){
                strides[cnt] = temporal_descriptor.at(dim_id).stride;
                max_point[cnt] = temporal_descriptor.at(dim_id).end;
                next_corner[cnt] = first_corner[cnt] + strides[cnt];
                cnt++;
            }
            HyperRectangleWalker space_walker(first_corner, max_point, strides.get(), order);  
            HyperRectangle tile(first_corner, next_corner); //this HR has all the dimensions which are not related to this tensor

            std::vector<DataMovementInfo> vec_data_movement;
            analysis::Point prev_tile_access_size;
            
            auto num_tiles_in_parent = parent_loop_nodes.iteration_count[child_idx];
            for(auto idx=0; idx<num_tiles_in_parent; idx++){
                if(idx!=0 && parent_datamovement_info_per_tensor[idx].num_unique_tiles==0){
                   // if this condition is true then temporal parent has no unique tile then spatial child has no data movement to update
                   //add info
                   auto dm_info = parent_datamovement_info_per_tensor[idx];
                //    dm_info.tile_count = loop_nodes.child[tens_cnt]->getInstanceSize(); //change the dm_info to spatial dimension of child
                    dm_info.tile_count = loop_nodes.sp_iteration_count[tens_cnt]; //FIXME::snegi this should change to acutal number of spatial nodes given in mapping

                   dm_info.tile_access_size = prev_tile_access_size; 
                   dm_info.timesteps_for_tile = (idx+1)*loop_nodes.relative_timestep_to_dependent[tens_cnt];
                   vec_data_movement.emplace_back(dm_info);
                } else {
                    HyperRectangleSet new_set(first_corner, spatial_descriptor, order);
                    HyperRectangleSet new_proj_set = ProjectHRSet(std::move(new_set), projection_expression, order);

                    auto tile_access_size = new_proj_set[0].getDelta(); //new_proj_set is a set of HR's so taking the first one's size, all should have same size
                    auto unique_tile_count = new_proj_set.Count();
                    // auto tile_count = loop_nodes.child[tens_cnt]->getInstanceSize(); //FIXME::snegi this should change to acutal number of spatial nodes given in mapping
                    auto tile_count = loop_nodes.sp_iteration_count[tens_cnt];

                    uint32_t link_transfers=0;
                    if(idx!=0){
                        auto diff_set = DiffTemporalHRSets(new_proj_set, loop_nodes.prev_hr_set[tens_cnt]);
                        auto link_transfers = diff_set.Count();
                    }
                    if(!loop_nodes.child[tens_cnt]->isCompute()){
                        COMET_LOG(logger::DEBUG, "TensorMovementAnalysis::Spatiotemporal:: Between {} and {} tile_count {} num_unique_tiles:{}\n Walking the UnProj HR Set: \n {} \n Walking the Proj HR Set: \n {} \n", loop_nodes.target[tens_cnt]->getName(), loop_nodes.child[tens_cnt]->getName(), tile_count, unique_tile_count, new_set.print(), new_proj_set.print());
                    }
                    loop_nodes.prev_hr_set[tens_cnt] = std::move(new_proj_set);
                    DataMovementInfo dm_info;
                    dm_info.tile_access_size   = tile_access_size;
                    dm_info.num_unique_tiles   = unique_tile_count;
                    dm_info.link_transfers     = link_transfers;
                    dm_info.tile_count         = tile_count;
                    dm_info.timesteps_for_tile = (idx+1)*loop_nodes.relative_timestep_to_dependent[tens_cnt];

                    vec_data_movement.emplace_back(dm_info);
                    prev_tile_access_size = tile_access_size;
                }
                first_corner = space_walker.get();
            }             
            analysis_.datamovement_info[node].emplace_back(std::move(vec_data_movement));
        }
    }


    void DataMovementInfoConstructor::ComputeSpatialInfo(LoopNode& loop_nodes, const DataMovementTileNode* node, LoopNode& parent_loop_nodes){

        //InitInfo
        // analysis_.datamovement_info[node].emplace_back();
        // analysis_.datamovement_info[node].back().info = &loop_nodes;
        // analysis_.datamovement_info[node].info = &loop_nodes;

        auto parent_node = static_cast<const DataMovementTileNode*>(node->get_parent()); 

        for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){

            auto tid       = loop_nodes.tensors[tens_cnt];
            auto loop_node = loop_nodes.descriptor_[tens_cnt];
            auto order     = loop_nodes.order[tens_cnt];
            auto tens_tag  = loop_nodes.tags[tens_cnt];

            // need to find index of this tensor(tid) in parent
            auto idx = parent_loop_nodes.tid_idx[tid]; // now need the index in this vector which can be derived from the child index (index of this child out of all children)
            size_t child_idx;
            for(auto& i:idx){
                if(parent_loop_nodes.tags[i]==tens_tag){ //if the tag of the tensor in parent node matches the tag of the tensor in the child node
                    child_idx=i;
                    break;
                }
            }
            auto parent_loop_descriptor = parent_loop_nodes.descriptor_[child_idx];
            auto parent_datamovement_info = analysis_.datamovement_info[parent_node];
            auto& parent_datamovement_info_per_tensor = parent_datamovement_info[child_idx];

            auto target = loop_nodes.target[tens_cnt];
            auto target_parent = parent_loop_nodes.target[child_idx];

            if(target == target_parent){
                parent_loop_nodes.spatial_node_exist = true;
                //update the iteration count for spatial node to the iterations from temporal parent
                loop_nodes.iteration_count = parent_loop_nodes.iteration_count; 
            }


            auto workload = analysis_.workloads_.workloads_[tens_tag];
            auto projection_expression = workload->getProjection(tid);

            auto& spatial_descriptor = loop_node;

            auto& temporal_descriptor = parent_loop_descriptor.back();

            auto dim_cnt = loop_node.back().size();
            Point first_corner(dim_cnt);
            Point next_corner(dim_cnt);
            Point strides(dim_cnt);
            Point max_point(dim_cnt);

            uint32_t cnt=0;
            for(auto& dim_id: order){
                strides[cnt] = temporal_descriptor.at(dim_id).stride;
                max_point[cnt] = temporal_descriptor.at(dim_id).end;
                next_corner[cnt] = first_corner[cnt] + strides[cnt];
                cnt++;
            }
            HyperRectangleWalker space_walker(first_corner, max_point, strides.get(), order);  
            HyperRectangle tile(first_corner, next_corner); //this HR has all the dimensions which are not related to this tensor

            std::vector<DataMovementInfo> vec_data_movement;
            analysis::Point prev_tile_access_size;
            
            auto num_tiles_in_parent = parent_loop_nodes.iteration_count[child_idx];
            for(auto idx=0; idx<num_tiles_in_parent; idx++){
                if(idx!=0 && parent_datamovement_info_per_tensor[idx].num_unique_tiles==0){
                   // if this condition is true then temporal parent has no unique tile then spatial child has no data movement to update
                   //add info
                   auto dm_info = parent_datamovement_info_per_tensor[idx];
                   dm_info.tile_count = loop_nodes.child[tens_cnt]->getInstanceSize(); //change the dm_info to spatial dimension of child
                   dm_info.tile_access_size = prev_tile_access_size; 
                   dm_info.timesteps_for_tile = (idx+1)*loop_nodes.relative_timestep_to_dependent[tens_cnt];
                } else {
                    HyperRectangleSet new_set(first_corner, spatial_descriptor, order);
                    HyperRectangleSet new_proj_set = ProjectHRSet(std::move(new_set), projection_expression, order);

                    auto tile_access_size = new_proj_set[0].getDelta(); //new_proj_set is a set of HR's so taking the first one's size, all should have same size
                    auto unique_tile_count = new_proj_set.Count();
                    auto tile_count = loop_nodes.child[tens_cnt]->getInstanceSize();
                    uint32_t link_transfers=0;
                    if(idx!=0){
                        auto diff_set = DiffTemporalHRSets(new_proj_set, loop_nodes.prev_hr_set[tens_cnt]);
                        auto link_transfers = diff_set.Count();
                    }
                    if(!loop_nodes.child[tens_cnt]->isCompute()){
                        COMET_LOG(logger::DEBUG, "TensorMovementAnalysis::Spatiotemporal:: Between {} and {} tile_count {} num_unique_tiles:{}\n Walking the UnProj HR Set: \n {} \n Walking the Proj HR Set: \n {} \n", loop_nodes.target[tens_cnt]->getName(), loop_nodes.child[tens_cnt]->getName(), tile_count, unique_tile_count, new_set.print(), new_proj_set.print());
                    }
                    loop_nodes.prev_hr_set[tens_cnt] = std::move(new_proj_set);
                    DataMovementInfo dm_info;
                    dm_info.tile_access_size   = tile_access_size;
                    dm_info.num_unique_tiles   = unique_tile_count;
                    dm_info.link_transfers     = link_transfers;
                    dm_info.tile_count         = tile_count;
                    dm_info.timesteps_for_tile = (idx+1)*loop_nodes.relative_timestep_to_dependent[tens_cnt];

                    vec_data_movement.emplace_back(dm_info);
                    prev_tile_access_size = tile_access_size;
                }
                first_corner = space_walker.get();
            }             
            analysis_.datamovement_info[node].emplace_back(std::move(vec_data_movement));
        }
    }


// non-GEMM operation node data movement is considered in the data movement tile
    // void DataMovementInfoConstructor::visitOperationNode(const OperationNode* node){

    //     auto op_node = std::get<OpNode>(analysis_.workload_mapping_graph[node]);
        
    //     auto movement_info = createMovementInfo(op_node.descriptor_);
    //     int cnt=0;
    //     for(auto& [tid, descriptor]: descriptor_){
    //         analysis_.datamovement_info[node].at(tid).emplace_back(std::move(movement_info[cnt])); //.back to access global or local tensor location
    //         cnt++;
    //     }
    //     COMET_ASSERT(analysis_.datamovement_info[node].size()==2, "Non GEMM operation node should have two entries in data movement info for input and output tensor");
        

    // }


    void DataMovementInfoConstructor::visitCollectiveOperationNode(const CollectiveOperationNode* node){
        std::cout<<"!!! Node name::" <<node->get_name()<<"\n";
        
        auto col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[node]);

        auto movement_info1 = createMovementInfo(col_op_node.descriptor_input);
        // analysis_.datamovement_info[node][col_op_node.in_tensor]=std::move(movement_info1);
        analysis_.datamovement_info[node].emplace_back(std::move(movement_info1));

        // analysis_.datamovement_info[node].at(col_op_node.in_tensor).emplace_back(std::move(movement_info1[0])); // 0th indec of datamovement_info has input tensor data movement

        auto movement_info2 = createMovementInfo(col_op_node.descriptor_output);
        // analysis_.datamovement_info[node].at(col_op_node.out_tensor).emplace_back(std::move(movement_info2)); //1st index of datamovement_info has output tensor data movement

        // analysis_.datamovement_info[node][col_op_node.in_tensor]=std::move(movement_info2);        
        analysis_.datamovement_info[node].emplace_back(std::move(movement_info2));

        // COMET_ASSERT(analysis_.datamovement_info[node].at(col_op_node.out_tensor).size()==2, "collective operation should have 2 data movement info: 1st for input collection and 2nd for collected output storage");
    }


    // void DataMovementInfoConstructor::createMovementInfo(ColOpNode& col_op_descriptor){
    std::vector<DataMovementInfo> DataMovementInfoConstructor::createMovementInfo(std::map<problem::TensorID, LoopNestDescriptor>& loop_descriptor){
        std::vector<DataMovementInfo> movement_info_vec;
        for(auto&[tid, descriptor]: loop_descriptor){
            auto tensor_name = analysis_.workloads_.getTensorName(tid);
            // analysis_.workloads_.getTensorName(tid);
            std::shared_ptr<problem::Workload> workload=std::make_shared<problem::Workload>();
            // problem::Workload* workload=nullptr;
            
            //find to workload to which the tensor belongs to
            for(auto& workload_pair: analysis_.workloads_.workloads_){
                workload = workload_pair.second; //pointer to workload class
                if(isStringInVector(workload_pair.second->ins_, tensor_name)||(workload_pair.second->out_ ==tensor_name)){
                    // if tensor belongs to workload1 set workload=workload1 
                    break;
                } else{
                    continue;
                }
            }

            //once we know the workload, get the dimensions for the tensor and set the order vector
            problem::LoopOrder order;
            for (auto& [name, dimid]: analysis_.workloads_.dimNamesToDimensionID_){
                order.push_back(dimid);
            }

            auto projection_expression = workload->getProjection(tid);

            auto descriptor_ = descriptor.back(); // last descriptor in the descriptor vector is the outer most loop

            auto dim_cnt = descriptor_.size();
            
            Point first_corner(dim_cnt);
            Point next_corner(dim_cnt);
            Point strides(dim_cnt);
            Point max_point(dim_cnt);

            // uint32_t cnt=0;
            // for(auto&[dimID, loop_desc] : descriptor_){
            //     strides[cnt]     = loop_desc.stride;
            //     max_point[cnt]   = loop_desc.end;
            //     next_corner[cnt] = first_corner[cnt] + strides[cnt];
            //     cnt++;
            // } 

            uint32_t cnt=0;
            //fill the points in order
            for(auto& dim_id: order){
                strides[cnt] = descriptor_.at(dim_id).stride;
                max_point[cnt] = descriptor_.at(dim_id).end;
                next_corner[cnt] = first_corner[cnt] + strides[cnt];
                cnt++;
            }


            
            COMET_ASSERT(strides==max_point, "strides and max_point should be same for collective operations data movement info");

            HyperRectangle tile(first_corner, next_corner);

            auto proj_tile = ProjectHR(tile, projection_expression, order);

            DataMovementInfo cur_info;

            cur_info.tile_access_size = proj_tile.getDelta();
            cur_info.num_unique_tiles = 1;
            cur_info.tile_count = 1; // this is equal to number of unique children
            cur_info.link_transfers = 0;
            cur_info.timesteps_for_tile = 1; // FIXME::snegi check this later. loop_nodes.relative_timestep_to_dependent[tid];
            
            movement_info_vec.emplace_back(cur_info);
        }

        return movement_info_vec;
    }



}