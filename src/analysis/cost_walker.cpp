#include <cmath>
#include <algorithm>
#include <numeric>
#include <variant>
#include <queue>

#include "analysis/cost_walker.hpp"
#include "analysis/nest_analysis.hpp"
#include "arch/arch.hpp"
#include "arch/network.hpp"
#include "arch/inc_utils.hpp"
#include "util/logger.hpp"



namespace analysis{

    void elementWiseOperation(const std::vector<int>& vec1, std::vector<int>& vec2, const std::string& operation) {
        // Initialize vec2 if it is empty to match the size of vec1, with elements set to 0
        if (vec2.empty()) {
            vec2.resize(vec1.size(), 0);
        }

        // Check if vec1 and vec2 have the same size
        if (vec1.size() != vec2.size()) {
            std::cerr << "Error: Vectors must be of the same size for element-wise operation." << std::endl;
            return;
        }

        // Perform element-wise maximum or addition based on the value of the operation string
        if (operation == "max") {
            std::transform(vec1.begin(), vec1.end(), vec2.begin(), vec2.begin(),
                        [](uint64_t a, uint64_t b) { return std::max(a, b); });
        } else if (operation == "sum") {
            std::transform(vec1.begin(), vec1.end(), vec2.begin(), vec2.begin(),
                        [](uint64_t a, uint64_t b) { return a + b; });
        } else {
            std::cerr << "Error: Unknown operation. Use 'max' or 'sum'." << std::endl;
        }
    }


    // std::map<const Node*, uint32_t> CostWalker::calculateSteadyStateTime(){
    cost_struct CostWalker::calculateSteadyStateTime(){

    // void CostWalker::calculateSteadyStateTime(){
        SteadyStateWalker ssw(analysis_);
        ssw.tree_traversal_steady_state(analysis_.mapping_.root);


        cost_struct retval;

        retval.node_cost = ssw.get_mapping_cost();

        retval.arch_level_node_cost = ssw.get_archlevel_mapping_cost();

        // retval.arch_level_node_energy = ssw.get_archlevel_energy();


        //calculate energy if the flag is true
        //call estimate energy function with root node as the input
        ssw.estimate_energy(analysis_.mapping_.root);

        retval.arch_level_node_energy = ssw.get_archlevel_energy();

        retval.node_level_noc_energy = ssw.get_noc_energy();

        retval.access_count = ssw.get_access_count_map();
        retval.parent_iteration_count = ssw.get_parent_iteration_count();

        auto root_node = analysis_.mapping_.root;
        auto total_compute_time = retval.node_cost[root_node];

        IdealComputationWalker icw(analysis_, total_compute_time);
        icw.tree_traversal_ideal_computation_latency(analysis_.mapping_.root);
        icw.calculate_ideal_computation_time();

        retval.comp_time_map = icw.get_comp_time_map();
        retval.compute_utilization_map = icw.calculate_compute_utilization();
        
        return retval;

        // auto node_cost = ssw.get_node_level_cost();
        // auto write_port_stall = ssw.get_write_port_stall();

        // for(auto&[arch, total_cost]: node_cost[analysis_.mapping_.root]){
        //     std::cout<<"Total Cost at read port of root: "<< total_cost[0]<<std::endl;
        // }

        // std::cout<<"Total Cost at read port of root: "<< node_cost[analysis_.mapping_.root][0];
        // std::cout<<"Total Cost at write port of root: " << write_port_stall<<std::endl;


        // return node_cost;
    }

// ************************ Ideal Computation related functions ************************
    void IdealComputationWalker::visitDataMovementTileNode(const DataMovementTileNode* node){
        for (auto child:node->get_children()){
            child->accept(this);
        }
    }
    
    void IdealComputationWalker::visitOperationNode(const OperationNode* node){

        auto common_attr  = node->get_common_attributes();
        auto comp_mapping = node->get_compute_mapping();
        auto parent    = node->get_parent();
        auto child = std::get<LoopNode>(analysis_.workload_mapping_graph[parent]).child.front();

        op_type_map[common_attr.op_name_] = common_attr.type_;
        arch_level_map[common_attr.op_name_] = child;

        if(common_attr.type_==operation_t::GEMM){
            tileprimitive_map[common_attr.op_name_] = comp_mapping.tileprimitives;
            comp_attributes[common_attr.op_name_]  = comp_mapping.comp_attr;
        } else {
            //currently we only have GEMM and SIMD units
            //SIMD operation
            comp_attributes[common_attr.op_name_]  = comp_mapping.comp_attr;
            tileprimitive_map[common_attr.op_name_] = comp_mapping.tileprimitives;
        }
    }    

    void IdealComputationWalker::calculate_ideal_computation_time(){

        for(auto&[op_name, tileprimitive]:tileprimitive_map){
            auto comp_attr = comp_attributes[op_name];
            auto op_type   = op_type_map[op_name];
            auto child_id  = arch_level_map[op_name]->getLevelID();

            auto dim_cnt = tileprimitive.size();
            DimSizeExpression compute_tileprimitive(dim_cnt,1);

            int cnt=0;
            for(auto pair:tileprimitive){
                compute_tileprimitive[cnt]=pair.second;
                cnt++;
            }

            auto num_comp_nodes = analysis_.topology_.getTotalNodes(child_id);

            auto problem_size = analysis_.workloads_.get_workload(op_name)->getInstanceSize();
            auto num_iterations = problem_size.resolve()/compute_tileprimitive.resolve();

            auto comp_node_time = analysis_.topology_.getCompute(child_id)->getComputeLatency(compute_tileprimitive, comp_attr, op_type); //time for running 1 tile primitive

            comp_time_map[op_name] = (num_iterations*comp_node_time)/num_comp_nodes;
        }
    }

    std::map<std::string, float> IdealComputationWalker::calculate_compute_utilization(){

        std::map<std::string, float> retval;
        
        for(auto&[op_name, comp_time]:comp_time_map){

            auto arch_level = arch_level_map[op_name];

            if(retval.find(arch_level->getName())!=retval.end()){
                //compute unit exist
                retval[arch_level->getName()] +=(float)comp_time/total_comp_time_;

            } else{
                //this compute unit doesn't exist
                retval[arch_level->getName()] = (float)comp_time/total_comp_time_;
            }
        }
        return retval;
    }

// ************************ Steady state related functions ************************

    std::pair<bool, uint32_t> find_count(std::map<std::string, std::map<TensorID, uint32_t>> map, std::vector<TensorID> tensors){
        //function to find count irrespective of the tensor
        bool exist=false;
        uint32_t max = 1;

        for(auto&[tag, tid_count]: map){
            for(auto&[tid, count]: tid_count){
                if(std::find(tensors.begin(), tensors.end(), tid)!=tensors.end()){
                    //tensor from this node exist in the iteration_count_map 
                    max = std::max(max, count);
                    exist = true;
                }
            }
        }

        return {exist, max};

    }

    std::vector<TensorID> SteadyStateWalker::find_tens_on_left(const Node* node){

        auto parent_node = node->get_parent();
        std::vector<TensorID> retval;

        for(auto& child: parent_node->get_children()){
            if(child==node) break;
            if(child->get_type()==Node::type_t::DataMovementTileNode){
                auto loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);
                for(auto tid: loop_node.tensors){
                    if(std::find(retval.begin(), retval.end(), tid)==retval.end()){// if tid not present in the vector add it
                        //tid doesn't exist in retval
                        retval.push_back(tid);
                    }
                }
            }
        }
        return retval;
    }


    //function to estimate the energy at the arch level
    //start from root node and go to the leaf nodes, so that number of iterations of parent-node can be propogated to the leaf nodes
    // std::map<TensorID, uint32_t> iteration_count_map;
    void SteadyStateWalker::estimate_energy(const Node* node, std::map<std::string, std::map<TensorID, uint32_t>> num_sp_parents, std::map<std::string, std::map<TensorID, uint32_t>> iteration_count_map){

        //base conditions
        if(node->get_type()==mapping::Node::OperationNode) return;
        // std::map<TensorID, uint32_t> current_node_iteration_count;
        std::map<std::string, std::map<TensorID, uint32_t>> current_node_iteration_count;

        if(node->get_type()==mapping::Node::DataMovementTileNode){
            // auto datamov_node = static_cast<const DataMovementTileNode*>(node);
            // auto tile_info = datamov_node->get_mapping();
            auto loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);

            //energy calculation only done at the spatial node. Exception: the temporal leaf node
            if (loop_node.type==mapping_t::SPATIAL || node->get_children().front()->get_type()==mapping::Node::OperationNode){
                ArchLevelEnergy arch_energy;
                NOCEnergy noc_energy_;
                for(auto& access_count: access_count_map[node]){
                    // energy_struct energy;
                    double energy;
                    // energy.arch_level = access_count.arch_level;
                    auto tag = access_count.tag;

                    if(access_count.target_child && !access_count.is_compute) {//update access_count only once for target,child pair and no need to update it for compute node
                        // iteration_count_map[access_count.tid] = access_count.access_count;
                        current_node_iteration_count[tag][access_count.tid] = access_count.access_count;
                    }


                    // if(iteration_count_map.find(access_count.tid)!=iteration_count_map.end()){// tid exist in the iteration_count_map
                    //     access_count.access_count *= iteration_count_map[access_count.tid];
                        
                    //     // iteration_count_map[access_count.tid] *= access_count.access_count; //update for the child node of this parent

                    // } 
                    // if((iteration_count_map.find(tag)!=iteration_count_map.end())&&(iteration_count_map[tag].find(access_count.tid)!=iteration_count_map[tag].end())&&(!access_count.is_compute)){
                    bool root_node = node->get_parent()==analysis_.mapping_.root; // True if node is a root node
                    // if(iteration_count_map.count(tag) && (!access_count.is_compute)){ // rather than taking the iteration for that particular tensor just take maximum iteration from the associated tag
                    // // if(iteration_count_map[tag].find(access_count.tid)!=iteration_count_map[tag].end()){// tid exist in the iteration_count_map
                    //     // access_count.access_count *= iteration_count_map[tag][access_count.tid];// rather than taking the iteration for that particular tensor just take maximum iteration from the associated tag
                    //     auto maxElement = max_element(iteration_count_map[tag].begin(), iteration_count_map[tag].end(), [](const auto& a, const auto& b) {
                    //         return a.second < b.second;
                    //     });
                        
                    //     //if the tensor id is not present in the iteration_count_map update it with the max value
                    //     if(!iteration_count_map[tag].count(access_count.tid)){
                    //         iteration_count_map[tag][access_count.tid] = maxElement->second;
                    //     }

                    //     if(maxElement!=iteration_count_map[tag].end()){
                    //         access_count.access_count *= maxElement->second;
                    //     } else{
                    //         // COMET_ASSERT(false, "No TAG element found in iteration count map");
                    //         std::cout<<"No TAG element found in the PARENT node for "<<node->get_name()<<std::endl;
                    //     }

                    //     if(parent_iteration_count.find(node)!=parent_iteration_count.end()){
                    //         parent_iteration_count[node] = std::max(parent_iteration_count[node], (uint64_t)maxElement->second);
                    //     } else {
                    //         parent_iteration_count[node] = maxElement->second; //iteration count of current node with respect to the parent node
                    //     }

                    //     if(!access_count.target_child && !access_count.is_compute){// if child and not a compute
                    //         // auto noc_energy = access_count.noc_energy*iteration_count_map[tag][access_count.tid]; //multiply by parents iteration
                    //         auto noc_energy = access_count.noc_energy*maxElement->second; //multiply by parents iteration
                    //         std::string key=access_count.arch_level->getName()+"<->"+access_count.arch_level->getName();
                    //         noc_energy_[key] += noc_energy;
                    //     }
                    //     // }
                    // } else if(!iteration_count_map.count(tag) && !access_count.is_compute && !root_node){
                    if(!access_count.is_compute && !root_node){
                        //tag not present in teh iteration count map and it is not a compute node -----> Intermediate tensor
                        //tag is not present, find iterations irrespective of tag just match the tensor id
                        if(verbose_level) std::cout<< " Tag not present for iteration_count_map during energy calculation for " << node->get_name()<<std::endl;

                        bool exist = false;
                        auto left_tensors = find_tens_on_left(node);
                        left_tensors.insert(left_tensors.end(), loop_node.tensors.begin(), loop_node.tensors.end()); //for intermediate tensor only node in Flash Attention it was creating problem hence using this
                        
                        auto retval = find_count(iteration_count_map, left_tensors);
                        exist = retval.first;
                        auto max_iterations = retval.second;

                        // auto retval = find_count(iteration_count_map, loop_node.tensors);

                        // exist = retval.first;
                        // auto max_iterations = retval.second;

                        // //
                        // if(!exist){
                        //     //get tensors on the left and find their num_sp_parents
                        //     auto left_tensors = find_tens_on_left(node);
                        //     auto retval = find_count(iteration_count_map, left_tensors);
                            
                        //     exist = retval.first;
                        //     max_iterations= retval.second;
                        // }
                        COMET_ASSERT(exist, "Tensors on the left of this node also not found in iteration_count_map " + node->get_name());

                        access_count.access_count *= max_iterations;

                        //update the iteration_count_map with the intermediate tensors iteration
                        iteration_count_map[tag][access_count.tid] = max_iterations;

                        if(parent_iteration_count.count(node)){
                            parent_iteration_count[node] = std::max(parent_iteration_count[node], (uint64_t)max_iterations);
                        } else {
                            parent_iteration_count[node] = max_iterations;
                        }
                        
                        if(!access_count.target_child && !access_count.is_compute){// if child and not a compute
                            // auto noc_energy = access_count.noc_energy*iteration_count_map[tag][access_count.tid]; //multiply by parents iteration
                            auto noc_energy = access_count.noc_energy*max_iterations; //multiply by parents iteration
                            std::string key=access_count.arch_level->getName()+"<->"+access_count.arch_level->getName();
                            noc_energy_[key] += noc_energy;
                        }
                    } 
                    else if(!access_count.target_child && !access_count.is_compute && !root_node){
                        //for root node
                        //calculate NOC energy only for children
                        std::string key=access_count.arch_level->getName()+"<->"+access_count.arch_level->getName();
                        noc_energy_[key] += access_count.noc_energy;     //noc_energy already summed up over its own iterations                   
                    }
                    
                    // else {
                    //     access_count.access_count = current_node_iteration_count[access_count.tid];
                    // }


                    //update access_count.access_count to consider the # of iters of parent
                    // access_count.access_count *=parent_iterations;
                    // if(access_count.target_child && !access_count.is_compute){ // only update the access count once for target,child pair and no need to update it for compute nod
                    //     current_node_iteration_count[access_count.tid] = access_count.access_count;
                    // }
                    
                    // access_count.tid exist in num_spatial_parents then assign it to sp_nodes else assign 1 to it
                    uint32_t sp_nodes=1;
                    // if(num_sp_parents.find(access_count.tid)!=num_sp_parents.end()){
                    //     sp_nodes = num_sp_parents[access_count.tid];
                    // } else {
                    //     sp_nodes = 1;
                    // }

                    if(num_sp_parents.count(tag)){
                        if(num_sp_parents[tag].count(access_count.tid)){
                            sp_nodes = num_sp_parents[tag][access_count.tid];
                        } else {
                            //tag present but tid is not
                            for(auto tid: loop_node.tensors){
                                if(num_sp_parents[tag].count(tid)){ //since access_count.tid is not present
                                    sp_nodes = std::max(sp_nodes, num_sp_parents[tag][tid]); //FIXME::snegi assuming all the nodes at the tensor are spatially present
                                }
                            }
                            //since tid is not present hence it must be an intermediate tensor. Update num_sp_parents for this intermediate tensor
                            num_sp_parents[tag][access_count.tid]=sp_nodes;
                        } 
                    } 
                    else if (!root_node){
                        //tag is not present, find iterations irrespective of tag just match the tensor id
                        if(verbose_level) std::cout<< " Tag not present for num_sp_parents during energy calculation for " << node->get_name()<<std::endl;

                        bool exist = false;
                        auto left_tensors = find_tens_on_left(node);
                        left_tensors.insert(left_tensors.end(), loop_node.tensors.begin(), loop_node.tensors.end()); //for intermediate tensor only node in Flash Attention it was creating problem hence using this
                        
                        auto retval = find_count(num_sp_parents, left_tensors);
                        exist = retval.first;
                        sp_nodes = retval.second;


                        // auto retval = find_count(num_sp_parents, loop_node.tensors);

                        // exist = retval.first;
                        // sp_nodes = retval.second;
                        // //since tid is not present hence it must be an intermediate tensor. Update num_sp_parents for this intermediate tensor
                        // num_sp_parents[tag][access_count.tid]=sp_nodes;
                        // //
                        // if(!exist){
                        //     //get tensors on the left and find their num_sp_parents
                        //     auto left_tensors = find_tens_on_left(node);

                        //     auto retval = find_count(num_sp_parents, left_tensors);
                            
                        //     exist = retval.first;
                        //     sp_nodes = retval.second;
                        //     //since tid is not present hence it must be an intermediate tensor. Update num_sp_parents for this intermediate tensor
                        //     num_sp_parents[tag][access_count.tid]=sp_nodes;                            
                        // }
                        COMET_ASSERT(exist, "Tensors on the left of this node also not found in iteration_count_map " + node->get_name());
                    }
                    
                    float cost;
                    if(access_count.is_compute){
                        cost = std::get<arch::ComputeSpec>(access_count.arch_level->getSpec()).compute_energy_.Get();
                        //find maximum value in the iteration count map for the tensors that exist at this node
                        uint32_t max=1;
                        // COMET_ASSERT(iteration_count_map.find(tag)!=iteration_count_map.end(),"tag not found in iteration count map during energy estimation");
                        if(iteration_count_map.count(tag)){
                            //tag is present
                            for(auto tid: loop_node.tensors){
                                if(iteration_count_map[tag].count(tid)){ //find max out of the tensors that are present
                                    //tensor id is present
                                    max=std::max(max, iteration_count_map[tag][tid]);
                                }
                            }
                        } else {
                            if(verbose_level) std::cout<< " Tag not present for iteration_count_map during compute unit energy for " << node->get_name()<<std::endl;
                            //tag is not present, find max irrespective of tag just match the tensor id
                            bool exist = false;
                            auto retval = find_count(iteration_count_map, loop_node.tensors);

                            exist = retval.first;
                            max = retval.second;

                            if(!exist){
                                //none of the tensors from this node is present in iteration_count_map --> intermediate operation and intermediate tensors
                                //get all tensors from left of this node and send those tensors
                                auto left_tensors = find_tens_on_left(node);
                                
                                auto retval = find_count(iteration_count_map, left_tensors);
                                exist = retval.first;
                                max = retval.second;
                            }
                            COMET_ASSERT(exist, "Tensors on the left of this node also not found in iteration_count_map " + node->get_name());

                            // for(auto&[tag, tid_count]: iteration_count_map){
                            //     for(auto&[tid, count]: tid_count){
                            //         if(std::find(loop_node.tensors.begin(), loop_node.tensors.end(), tid)!=loop_node.tensors.end()){
                            //             //tensor from this node exist in the iteration_count_map 
                            //             max = std::max(max, count);
                            //             node_tensor_exist = true;
                            //         }
                            //     }
                            // }

                        

                        }

                        // for(auto tid: loop_node.tensors){
                        //     if(iteration_count_map[tag].find(tid)!=iteration_count_map[tag].end()){
                        //         max = std::max(max, iteration_count_map[tag][tid]);
                        //     }
                        // }
                        access_count.access_count *=max;
                        
                        // access_count
                        if(std::get<arch::ComputeSpec>(access_count.arch_level->getSpec()).name_=="SystolicArray"){
                            // energy = sp_nodes*access_count.compute_time*cost;
                            energy = sp_nodes*access_count.access_count*access_count.compute_energy;
                        } else if(std::get<arch::ComputeSpec>(access_count.arch_level->getSpec()).name_=="VSIMD"){

                            auto simd_width = std::get<arch::ComputeSpec>(access_count.arch_level->getSpec()).computeWidth_.Get();
                            if(access_count.op_type == mapping::operation_t::DIV){
                                cost = 0.8; //pJ
                            } else if(access_count.op_type == mapping::operation_t::EXP){
                                cost = 3.86;
                            } else if(access_count.op_type == mapping::operation_t::ADD){
                                cost = 0.11;
                            } else if(access_count.op_type == mapping::operation_t::MULT){
                                cost = 0.64;
                            } else if(access_count.op_type == mapping::operation_t::MAX){
                                cost = 0.0025;
                            } else if(access_count.op_type == mapping::operation_t::SQRT){
                                cost = 2.84;
                            } else if(access_count.op_type == mapping::operation_t::SOFTMAX){
                                cost=1;
                                std::cout<<"############# SOFTMAX opreation will be depricated later #############"<<std::endl;
                            } else if(access_count.op_type == mapping::operation_t::ROWSUM){
                                cost = 0.11*std::ceil(std::log2(simd_width));
                            } else if(access_count.op_type == mapping::operation_t::ROWMAX){
                                cost = 0.0025*std::ceil(std::log2(simd_width));
                            }
                            else {
                            COMET_ASSERT(false, "Operation not supported in VSIMDCompute");
                            }                            
                            energy = sp_nodes*access_count.access_count*cost*simd_width;
                        }
                        
                        else {
                            energy = sp_nodes*access_count.access_count*cost*access_count.arch_level->getInstanceSize(); // FIXME::snegi current assumption is that we are using all the spatial nodes
                        }
                    } else {    
                        if(access_count.tensor_is_rw){
                            // read-write tensor --> draining
                            if(access_count.target_child){
                                //target
                                //write-port
                                cost = std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).parent_write_energy_.Get();
                            } else {
                                //child
                                //read-port
                                cost = std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).child_read_energy_.Get();
                                auto idx = loop_node.tid_idx[access_count.tid][0]; // FIXME::snegi what if spatial node has multiple tensors with same id
                                // cost *= loop_node.sp_iteration_count[idx];

                                // if(!std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).) access_count.access_count *= loop_node.sp_iteration_count[idx];
                            }

                        } else {
                            // read-only tensor --> filling
                            if(access_count.target_child){
                                //target
                                //read-port
                                cost = std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).parent_read_energy_.Get();
                            } else {
                                //child
                                //write-port
                                cost = std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).child_write_energy_.Get();
                                auto idx = loop_node.tid_idx[access_count.tid][0]; // FIXME::snegi what if spatial node has multiple tensors with same id
                                // cost *= loop_node.sp_iteration_count[idx];
                                // access_count.access_count *= loop_node.sp_iteration_count[idx];    
                            }
                        }
                        
                        auto block_size = std::get<arch::MemorySpec>(access_count.arch_level->getSpec()).width_.Get();
                        energy = sp_nodes*access_count.access_count*access_count.tile_size*access_count.projected_spatial_count*access_count.precision*cost/(block_size*8.0); //multiplying by projected spatial count because tile size is calculated for single child node in datamovement.cpp 8.0 to convert precision bits to bytes

                        if(access_count.rmw){ energy*=2;} //FIXME::snegi rmw=true implies partial sum is accessed twice, changed from timeloop

                    }
                    // energy_vec.emplace_back(energy);
                    arch_energy[access_count.arch_level] += energy; 
                    
                }
                // node_level_access_energy[node] = energy_vec;
                node_level_access_energy[node] = arch_energy;
                node_level_noc_energy[node] = noc_energy_;

                //update number of spatial parents only after energy calculation for this node is completed
                size_t cnt=0;
                for(auto& access_count: access_count_map[node]){
                    // if(access_count.target_child && num_sp_parents.find(access_count.tid)!=num_sp_parents.end()){ // only update it once for a target, child pair
                    //     num_sp_parents[access_count.tid] *= loop_node.sp_iteration_count[loop_node.tid_idx[access_count.tid][0]];
                    // } else if(access_count.target_child){
                    //     num_sp_parents[access_count.tid] = 1;
                    // }
                    auto tag = access_count.tag;
                    auto tid = access_count.tid;
                    if(access_count.target_child && num_sp_parents.find(tag)!=num_sp_parents.end()){ //target and tag exist
                        if(num_sp_parents[tag].find(tid)!=num_sp_parents[tag].end()){//tid exist
                            // num_sp_parents[tag][tid] *= loop_node.sp_iteration_count[loop_node.tid_idx[access_count.tid][0]];
                            num_sp_parents[tag][tid] *= loop_node.sp_iteration_count[cnt];//FIXME::snegi assuming access_count_map is filled in the same order as the sp_iteration_count map (valid assumption but just check it again later)

                        } else {
                            num_sp_parents[tag][tid] = loop_node.sp_iteration_count[cnt];
                        }
                        cnt++;
                    } else if(access_count.target_child){ // tag doesn't exist --> tid also won't exist
                        num_sp_parents[tag][tid] =loop_node.sp_iteration_count[cnt]; //1;
                        cnt++;
                    }
                }
            }
            if(node->get_children().front()->get_type()==mapping::Node::OperationNode) return; //leaf node has no temporal or spatial children // no need to update the iteration count map
        
            // for(auto&[tid, count]: current_node_iteration_count){
            //     if(iteration_count_map.find(tid)!=iteration_count_map.end()){
            //         iteration_count_map[tid] *= count;
            //     } else {
            //         iteration_count_map[tid] = count;
            //     }
            // }

            for(auto&[tag, tid_count]: current_node_iteration_count){
                for(auto&[tid,count]:tid_count){
                    if(iteration_count_map.find(tag)!=iteration_count_map.end()){//tag exist in iteration count map
                        if(iteration_count_map[tag].find(tid)!=iteration_count_map[tag].end()){
                            iteration_count_map[tag][tid] *= count;
                        } else {
                            iteration_count_map[tag][tid] = count;
                        }
                    } else {
                        iteration_count_map[tag][tid] = count;
                    }
                }
            }
            
        }
        int child_cnt=0;
        for(auto child: node->get_children()){

            if(child->get_type()==mapping::Node::InterTileBinding) {
                child_cnt++;
                continue; 
            }
            else if(child->get_type()==mapping::Node::CollectiveOperationNode){
                // arch_level_access_energy[node] = 
                // continue; //FIXME::snegi add energy for collective operations once the collectives implementation is fixed
                //scale the numbers by the number of iterations of the parent node
                auto loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
                auto colOp_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child]);
                auto in_tid = colOp_node.in_tensor;
                auto op_on_left = find_ops_on_left(child, in_tid);                

                if(loop_node.type==mapping_t::TEMPORAL){ //collective operation is the child of a temporal node
                    //create duplicate of iteartion count map and update the iterations with the number of iterations of the parent node
                    const mapping::Node* left_node;
                    auto vec=node->get_children();
                    for(int i=child_cnt-1; i>=0; i--){
                        if(vec[i]->get_type()==mapping::Node::DataMovementTileNode){
                            left_node = vec[i];
                            break;
                        }
                    }
                    std::map<std::string, std::map<size_t, uint32_t>> dummy_iteration_count_map=iteration_count_map;//cannot change the iteration_count_map bcz it will be used by other children as well
                    for(auto& access_count: access_count_map[left_node]){
                        auto tag = access_count.tag;
                        if(access_count.target_child && !access_count.is_compute) {//update access_count only once for target,child pair and no need to update it for compute node
                            // iteration_count_map[access_count.tid] = access_count.access_count;
                            current_node_iteration_count[tag][access_count.tid] = access_count.access_count; // iterations of left node
                        }
                    }
                    // dummy_i
                    
                    auto iterations = current_node_iteration_count[op_on_left.back()][in_tid];
                    // auto parent_max_iterations = max_element(iteration_count_map[op_on_left.back()].begin(), iteration_count_map[op_on_left.back()].end(), [](const auto& a, const auto& b) {
                    //     return a.second < b.second;
                    // });
                

                    //account for parents iterations in the dummy_iteration_count map
                    // for(auto&[tag, tid_count]: current_node_iteration_count){
                    //     for(auto&[tid,count]:tid_count){
                    //         if(iteration_count_map.find(tag)!=iteration_count_map.end()){//tag exist in iteration count map

                    //             if(iteration_count_map[tag].find(tid)!=iteration_count_map[tag].end()){
                    //                 dummy_iteration_count_map[tag][tid] *= count;
                    //             } else {
                    //                 dummy_iteration_count_map[tag][tid] = count;
                    //             }
                    //         } else {
                    //             dummy_iteration_count_map[tag][tid] = count;
                    //         }
                    //     }
                    // }
                    // auto maxElement = max_element(iteration_count_map[tag].begin(), iteration_count_map[tag].end(), [](const auto& a, const auto& b) {
                    //     return a.second < b.second;
                    // });
                    // if(maxElement!=iteration_count_map[tag].end()){
                    //     access_count.access_count *= maxElement->second;
                    // } else{
                    //     // COMET_ASSERT(false, "No TAG element found in iteration count map");
                    //     std::cout<<"No TAG element found in the PARENT node for "<<node->get_name()<<std::endl;
                    // }


                    // uint32_t iterations=1;
                    // for(auto&[tag, tid_count]: dummy_iteration_count_map){
                    //     for(auto&[tid, count]:tid_count){
                    //         //check if the tag is of the operations on the left of the collective node and the tid matches with the in_tensor of the collective node
                    //         if(tid==in_tid && std::find(op_on_left.begin(), op_on_left.end(), tag)!=op_on_left.end()){
                    //             iterations = dummy_iteration_count_map[tag][tid];
                    //             break;
                    //         }
                    //     }
                    // }
                    // auto iterations = iterations_in_tensor*parent_max_iterations->second;
                    parent_iteration_count[child] = iterations;
                    for(auto&[arch,energy]: node_level_access_energy[child]){
                        energy*=iterations;
                    }
                    for(auto&[arch,energy]:node_level_noc_energy[child]){
                        energy*=iterations;
                    }
                    access_count_struct colop_access_count;
                    colop_access_count.access_count = iterations;
                    colop_access_count.arch_level = colOp_node.target;
                    colop_access_count.tid = in_tid;
                    std::vector<access_count_struct> colop_access_count_vec;
                    colop_access_count_vec.push_back(colop_access_count);
                    access_count_map[child] = colop_access_count_vec;
                } else {
                    uint32_t iterations=1;
                    for(auto&[tag, tid_count]: iteration_count_map){
                        for(auto&[tid, count]:tid_count){
                            //check if the tag is of the operations on the left of the collective node and the tid matches with the in_tensor of the collective node
                            if(tid==in_tid && std::find(op_on_left.begin(), op_on_left.end(), tag)!=op_on_left.end()){
                                iterations = iteration_count_map[tag][tid];
                                break;
                            } else if(tid!=in_tid){
                                //tid not present in iteration_count_map

                            }
                        }
                    }
                    parent_iteration_count[child] = iterations;
                    for(auto&[arch,energy]: node_level_access_energy[node]){
                        energy*=iterations;
                    }
                    for(auto&[arch,energy]:node_level_noc_energy[node]){
                        energy*=iterations;
                    }                    
                }


            }
            else if(child->get_type()==mapping::Node::OperationNode){
                continue;
            }
            else if(child->get_type()==mapping::Node::DataMovementTileNode){
                //datamovement tile node
                // auto datamov_node = static_cast<const DataMovementTileNode*>(node);//(child);
                // auto tile_info = datamov_node->get_mapping();
                
                estimate_energy(child, num_sp_parents, iteration_count_map);
                // auto loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[node]); //[child]);

                // auto child_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);
                
                // if(loop_node.type==mapping_t::TEMPORAL){
                //     //temporal
                //     // estimate_energy(child, parent_iterations);
                //     estimate_energy(child, current_node_iteration_count);

                // } else{
                //     //spatial
                //     // uint64_t num_iterations=1;
                //     // for(uint64_t i: loop_node.iteration_count){
                //     //     num_iterations = std::max(num_iterations,i);
                //     // }
                //     std::map<TensorID, uint32_t> child_node_iteration_count;
                //     for(size_t i=0; i<child_loop_node.iteration_count.size(); i++){
                //         auto tid = child_loop_node.tensors[i];
                //         if(iteration_count_map.find(tid)!=iteration_count_map.end()){
                //             child_node_iteration_count[tid] = current_node_iteration_count[tid]*iteration_count_map[tid];
                //         } 
                        
                //         // else {
                //         //     child_node_iteration_count[tid] = current_node_iteration_count[tid];
                //         // }

                //     }
                //     estimate_energy(child, child_node_iteration_count);
                // }     
                // if(loop_node.type==mapping_t::TEMPORAL && node->get_children().front()->get_type()!=mapping::Node::OperationNode) continue; //energy calculation only done at the spatial node. Exception: the temporal leaf node
            }
            child_cnt++;

        }
        

    }
    std::vector<std::string> SteadyStateWalker::find_ops_on_left(const Node* node, TensorID tens_id){

        auto parent_node = node->get_parent();
        std::vector<std::string> retval;

        for(auto& child: parent_node->get_children()){
            if(child==node) break;
            if(child->get_type()==Node::type_t::DataMovementTileNode){
                auto loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);
                
                auto tid_idx = loop_node.tid_idx[tens_id];
                for(auto i:tid_idx) retval.push_back(loop_node.tags[i]);
                // retval.insert(retval.end(), loop_node.tags.begin(), loop_node.tags.end());
            }
        }
        return retval;
    }

    //this function finds the binding associated with this node
    InterTileBinding::type_t SteadyStateWalker::find_binding(const Node* node){

        auto parent_node = node->get_parent();
        auto num_children = parent_node->get_children().size();

        //if the parent node has only single parent
        if(num_children==1) return InterTileBinding::type_t::none; //return none if the parent has only one children
        

        const Node* left_node      = nullptr;
        const Node* left_of_left_node = nullptr;
        size_t cnt=0;
        for(auto& child: parent_node->get_children()){

            if(child==node && cnt==num_children-1){
                // if child is equal to the node and this is the last child then return the type of left_left node because left node will be the data movement node
                auto binding_node = static_cast<const InterTileBinding*>(left_of_left_node);
                return binding_node->type;
            }

            if(child==node){
                auto binding_node = static_cast<const InterTileBinding*>(left_node);
                return binding_node->type;
            }

            left_of_left_node = left_node;
            left_node         = child;
            cnt++;
        }
    }

    //find next two data movement nodes
    std::vector<const Node*> find_next_two_nodes(const Node* node){
        
        std::vector<const Node*> retval;
        auto parent_node = node->get_parent();
        auto num_children = parent_node->get_children().size();

        auto binding_node_found=false;

        for(auto& child:parent_node->get_children()){
            if(child==node) binding_node_found=true;

            //once binding node is found add the next 2 data movement tile nodes to the retval vector
            if(binding_node_found && (child->get_type()==mapping::Node::DataMovementTileNode || child->get_type()==mapping::Node::CollectiveOperationNode)){
               retval.push_back(child); 
            }
            if(retval.size()==2) break;
        }

        return retval;
    }

    // function to find the iterations of a tensor if the tensor is not written back to the parent node
    // uint32_t find_tensors_iteration(problem::TensorID tid, std::vector<>)

// ************************ Steady state related functions ************************

    uint32_t get_iterations(std::vector<DataMovementInfo> tensor_movement){
        uint32_t cnt=0;
        for(auto&mov_info:tensor_movement){
            if(mov_info.num_unique_tiles!=0) cnt++;
        }
        return cnt;
    }


    std::vector<uint32_t> get_no_stall_time_vector(std::vector<DataMovementInfo> tensor_movement, uint32_t reuse_factor, uint32_t no_stall_time){
        std::vector<uint32_t> no_stall_time_vector;
        auto cnt=0;
        for(auto& mov_info: tensor_movement){
            if(cnt%reuse_factor==0){
                no_stall_time_vector.emplace_back(reuse_factor*no_stall_time); // no_stall vector should have same length as tensor_movement vector // serpentine iterative will not work, snake counting will not work
            } else {
                COMET_ASSERT(mov_info.num_unique_tiles==0, "number of unique tile is not zero, no_stall_time_vector might be incorrect");
                no_stall_time_vector.emplace_back(0); // just append 0 since for these iterations num_unique_tiles will be zero
            }
            cnt++;
            // if(mov_info.num_unique_tiles!=0) no_stall_time_vector.emplace_back(reuse_factor*no_stall_time); //relative_timestep_to_dependent* // every time the window seen by parent to fill the child will be reuse_factor multiplied by the compute time for the child for 1 iteration

        }
        return no_stall_time_vector;
    }


    void SteadyStateWalker::get_mem_win_vector(const Node* parent_node, ChildNodes child_nodes, std::vector<uint32_t>& mem_win_vec){


        if(std::holds_alternative<std::vector<const Node*>>(child_nodes)){

            for(auto& child_node:std::get<std::vector<const Node*>>(child_nodes)){
                for(auto&[key, reuseFactor]: reuse_factor[parent_node]){
                    if(reuseFactor!=1 && reuse_factor[child_node].count(key) && actual_iterations[child_node][key]==1 || perfect_pipeline){ //parent_node has some reuse, child node has the same tensor attached to same operation and the iterations in child node=1, if greater than 1 then it will have rampup
                        for(int i=0; i<mem_win_vec.size(); i++){
                            if(!perfect_pipeline && i%reuseFactor!=0){
                                mem_win_vec[i]-=rampupInfo[child_node][key]; //if the tensor corresponding to key is begin reused, remove it's time from the window
                            }
                            else if(perfect_pipeline && i!=0){ //always consider ramup in first cycle
                                mem_win_vec[i]-=rampupInfo[child_node][key];
                            }
                            if(perfect_pipeline && i!=mem_win_vec.size()-1){
                                mem_win_vec[i] -=rampdownInfo[child_node][key];
                            }
                        }
                    }
                }
            }
        }
        else if(std::holds_alternative<const Node*>(child_nodes)){
            auto child_node = std::get<const Node*>(child_nodes);

            if(child_node->get_type()==mapping::Node::DataMovementTileNode && child_node->get_children().front()->get_type()==Node::type_t::OperationNode){
                for(auto&[key, reuseFactor]: reuse_factor[parent_node]){
                    if(reuseFactor!=1 && reuse_factor[child_node].count(key) && actual_iterations[child_node][key]==1 || perfect_pipeline){
                        for(int i=0; i<mem_win_vec.size(); i++){
                            if(!perfect_pipeline &&i%reuseFactor!=0){
                                mem_win_vec[i]-=rampupInfo[child_node][key];
                            }
                            else if(perfect_pipeline && i!=0){ //always consider ramup in first cycle
                                mem_win_vec[i]-=rampupInfo[child_node][key];
                            }
                            if(perfect_pipeline && i!=mem_win_vec.size()-1){
                                mem_win_vec[i] -=rampdownInfo[child_node][key];
                            }                            
                        }
                    }
                }

            } else{
                //Sp node -> Tp node and Tp node has multiple children
                for(auto& child:child_node->get_children()){
                    for(auto&[key, reuseFactor]: reuse_factor[parent_node]){
                        if(reuseFactor!=1 && reuse_factor[child].count(key) && actual_iterations[child][key]==1 || perfect_pipeline){ //parent_node has some reuse, child node has the same tensor attached to same operation and the iterations in child node=1, if greater than 1 then it will have rampup
                            for(int i=0; i<mem_win_vec.size(); i++){
                                if(!perfect_pipeline &&i%reuseFactor!=0){
                                    mem_win_vec[i]-=rampupInfo[child][key];
                                }
                                else if(perfect_pipeline && i!=0){ //always consider ramup in first cycle
                                    mem_win_vec[i]-=rampupInfo[child_node][key];
                                }
                                if(perfect_pipeline && i!=mem_win_vec.size()-1){
                                    mem_win_vec[i] -=rampdownInfo[child_node][key];
                                }                                
                            }

                            //also subtract ramup from the lower level nodes
                            get_mem_win_vector(parent_node, child->get_children(), mem_win_vec);
                        }
                    }
                }
            }




        }


    }

    void SteadyStateWalker::visitDataMovementTileNode(const DataMovementTileNode* node){

        std::cout<<"!!! Node name in cost_walker:: "<<node->get_name()<<"\n";
        std::vector<InterTileBinding::type_t> binding;
        // record the binding at this level
        // LoopNode loop_nodes;
        auto& loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
        if(loop_nodes.type==mapping::mapping_t::SPATIAL || node->get_children().front()->get_type()==Node::type_t::OperationNode){

            for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){
                auto& tensor_mov = analysis_.datamovement_info[node][tens_cnt];
                auto actualIterations = get_iterations(tensor_mov);
                auto reuseFactor      = loop_nodes.iteration_count[tens_cnt]/actualIterations;
                if(reuseFactor==0) reuseFactor=1;
                // reuse_factor[node].push_back(reuseFactor);
                reuse_factor[node][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = reuseFactor;
                actual_iterations[node][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = actualIterations;

            }
        }

        for(auto child:node->get_children()){
            if(child->get_type()==Node::type_t::InterTileBinding){
                const auto& child_node=static_cast<const InterTileBinding*>(child);
                binding.emplace_back(child_node->get_inter_tile_binding_type());
            }

            //calculate and store the reuse factors
            // loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
            child->accept(this);
        }

        // std::vector<TargetChildCostVec> cost_vec; //this vector is needed 
        //base condition
        //DatamovementTileNode -> GEMM Node, SIMD Node
        if(node->get_children().front()->get_type()==Node::type_t::OperationNode){
            const auto& op_node = static_cast<const OperationNode*>(node->get_children().front());
            auto comp_mapping = op_node->get_compute_mapping();
            auto problem_size_map = comp_mapping.tileprimitives;
            // auto dim_cnt = op_node->get_workload()->getDimensionCount();
            auto dim_cnt = problem_size_map.size();

            DimSizeExpression compute_tileprimitive(dim_cnt,1);
            computation_attributes comp_attribute;
            auto op_type = op_node->get_common_attributes().type_;
            // if(op_type==operation_t::GEMM){
            //     // get compute primitives
            
            comp_attribute  = comp_mapping.comp_attr;
            int cnt=0;
            for(auto pair:problem_size_map){
                compute_tileprimitive[cnt]=pair.second;
                cnt++;
            }
            // }  
            // else if (op_node->get_common_attributes().type_==operation_t::SOFTMAX){
            //     // similar to GEMM calculate compute tileprimitive for SIMD
            //     // int a=10; //place holder
            //     // auto dim_cnt = op_node->get_workload()->getDimensionCount();
            //     for( auto&[tid, compute_tiling_matrix]:op_node->get_common_attributes().tilesizes){
            //         int cnt=0;
            //         for(auto&[dim_id, val]:compute_tiling_matrix){
            //             compute_tileprimitive[cnt]=val;
            //             cnt++;
            //         }

            //     }

            // }
            auto loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
            std::pair<const Node*, const Node*> key={node, node->get_children().front()};
            uint64_t max_accesses=0;
            uint64_t compute_time=1;

            std::map<arch::ArchLevel*, std::map<std::tuple<TensorID, std::string>, uint32_t>> rampupMap;

            uint32_t max_iterations=0;
            for(auto iterations:loop_nodes.iteration_count){
                max_iterations = std::max(iterations, max_iterations);
            }    

            for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){ // iterate the tensors in the same order in which the data movement tiles were inserted
                auto target    = loop_nodes.target[tens_cnt];
                auto child     = loop_nodes.child[tens_cnt];
                auto target_id = target->getLevelID();
                auto child_id  = child->getLevelID();

                std::pair<arch::ArchLevel*, arch::ArchLevel*> arch_key={target, child};

                // parent_child_arch_level_cost_map[key][arch_key]
                // parent_child_arch_level_cost_map[node][node->get_children().front()][arch_key].emplace_back(get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, op_type, 0, tens_cnt, true, false, compute_tileprimitive, comp_attribute));


                auto comp_node_time = analysis_.topology_.getCompute(child_id)->getComputeLatency(compute_tileprimitive, comp_attribute, op_type);
                std::vector<uint32_t> memWindow(max_iterations, comp_node_time);

                auto retval = get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, op_type, 0, tens_cnt, true, false, memWindow, compute_tileprimitive, comp_attribute);                
                
                access_count_struct temp_val;
                uint8_t tensor_type = loop_nodes.tensor_is_rw[tens_cnt]? 1 : 0;
                temp_val.arch_level = target;
                temp_val.access_count = retval[tensor_type].access_count; //FIXME::snegi either access count should take care of spatial nodes or should be considered in arch file
                temp_val.tile_size = retval[tensor_type].tile_size;
                temp_val.precision = loop_nodes.scale[tens_cnt];
                temp_val.tensor_is_rw = loop_nodes.tensor_is_rw[tens_cnt]? true: false;
                temp_val.target_child = true;
                temp_val.is_compute = false;
                temp_val.tid = loop_nodes.tensors[tens_cnt];
                temp_val.tag = loop_nodes.tags[tens_cnt];
                temp_val.rmw = loop_nodes.rmw[tens_cnt];
                
                auto sp_cnt = analysis_.datamovement_info[node][tens_cnt][0].num_unique_tiles;
                if(sp_cnt!=0){
                    temp_val.projected_spatial_count = sp_cnt;
                } else {
                    temp_val.projected_spatial_count = 1;
                }

                max_accesses = std::max(temp_val.access_count, max_accesses);

                compute_time = std::max(compute_time, retval[tensor_type].compute_time);

                access_count_map[node].push_back(temp_val);


                if(!loop_nodes.tensor_is_rw[tens_cnt]){
                    //ramup from only read tensors
                    rampupMap[target][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = retval[tensor_type].ramp_up;
                }
                if(loop_nodes.tensor_is_rw[tens_cnt]){
                    rampdownInfo[node][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = retval[tensor_type].ramp_down;
                }

                parent_child_arch_level_cost_map[node][node->get_children().front()][arch_key].emplace_back(retval);
            }
            //access count for the compute unit
            auto tens_cnt=0;
            access_count_struct temp_val;
            temp_val.arch_level = loop_nodes.child[tens_cnt];
            temp_val.access_count = max_accesses;
            temp_val.target_child = false;
            temp_val.is_compute = true;
            temp_val.tag = loop_nodes.tags[0];
            temp_val.compute_time = compute_time;
            temp_val.op_type = op_type;
            temp_val.rmw = loop_nodes.rmw[tens_cnt];
            temp_val.tid = loop_nodes.tensors[0]; //taking any of the tensor id to know the spatial count
            
            if(std::get<arch::ComputeSpec>(loop_nodes.child[0]->getSpec()).name_=="SystolicArray"){
                auto comp_energy = analysis_.topology_.getCompute(temp_val.arch_level->getLevelID())->getComputeEnergy(compute_tileprimitive, comp_attribute, op_type);
                temp_val.compute_energy = comp_energy;
            }
            
            access_count_map[node].push_back(temp_val);




            // get node_cost
            // node_cost[node] = get_node_cost(parent_child_arch_level_cost_map[node][node->get_children().front()], loop_nodes);
            std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>> child_arch_level_port_usage;

            parent_child_node_level_cost_map[node][node->get_children().front()] = get_node1_node2_cost(parent_child_arch_level_cost_map[node][node->get_children().front()], loop_nodes, child_arch_level_port_usage);


            parent_child_cost_map[node][node->get_children().front()] = get_node_cost(parent_child_node_level_cost_map[node][node->get_children().front()], loop_nodes, true); // compute node hence last argument true

            parent_child_total_cost[node][node->get_children().front()] = get_total_node_cost(parent_child_cost_map[node][node->get_children().front()]);
            node_cost[node] = parent_child_total_cost[node][node->get_children().front()]; // since compute nodes only have a single children these two data structures will hold the same values


            //find the arch level with max rampup cost
            std::unordered_map<arch::ArchLevel*, uint32_t> ramupArchLevel;

            for(auto&[key, tid_ramup]: rampupMap){
                for(auto&[optid, ramup]:tid_ramup){
                    ramupArchLevel[key] +=ramup;
                }
            }
            
            if (!ramupArchLevel.empty()) {
                auto max_it = std::max_element(ramupArchLevel.begin(), ramupArchLevel.end(), [](const auto& a, const auto& b){
                    return a.second<b.second;
                });                        
                auto maxValue = max_it->second;

                for(const auto&[arch, value]: ramupArchLevel){
                    if(value==maxValue){
                        for(const auto&[key, val]: rampupMap[arch]){
                            rampupInfo[node][key] = val;
                        }
                    }
                }  
            }  

            // for(auto&[key, value]: rampupMap[max_it->first]){
            //     rampupInfo[node][key] = value;
            // }
                
            return;
        }

        // auto loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
        //find window for this node
        if(binding.size()==0){ // implies this node has only one children
            
            auto child = node->get_children().back(); //should only have one children since NO bindings are present 
            if(loop_nodes.type==mapping::mapping_t::SPATIAL){
                //if spatial node has only one child then the memory window for this node is just the children total time, however if the parent temporal node has multiple children then this spatial node should do the ramp up every iteration if this branch is pipelined with other children of temporal node
                memory_windows[node][child] = node_cost[child]; // this will be used later to calculate the data movement
            } else {
                //temporal node with no children 
                node_cost[node] = node_cost[child];
                return;
            }
        }
        else {
            auto child_vec = node->get_children();
            // get the window for nodes using sequential binding first
            std::vector<const Node*> nodes_vec;
            std::set<const Node*> visited_nodes;
            //make a super node out of the sequential child nodes
            for(auto cnt=0; cnt<child_vec.size(); cnt++){
                if(child_vec[cnt]->get_type()==mapping::Node::InterTileBinding && cnt%2==0 && binding[cnt/2]==InterTileBinding::type_t::sequential){ //binding exist at the even indices
                    //if the child_node is a intertile binding, it is at the even index and it is sequential binding
                    // add next two non-binding node children to the nodes_vec vector
                    auto next_two_nodes = find_next_two_nodes(child_vec[cnt]);

                    for(auto val: next_two_nodes){
                        if(std::find(nodes_vec.begin(), nodes_vec.end(), val)==nodes_vec.end()){
                            //if val node doesn't exist in the next nodes vector
                            nodes_vec.push_back(val);
                        }
                        visited_nodes.insert(val);
                    }

                } else if(child_vec[cnt]->get_type()==mapping::Node::InterTileBinding && cnt%2==0 && binding[cnt/2]!=InterTileBinding::type_t::sequential){

                    if(nodes_vec.size()>0){ //nodes_vec size will be only higher if it was filled by sequential bindings before
                        auto total_mem_window=0;
                        for(auto child_nodes:nodes_vec){
                            total_mem_window +=node_cost[child_nodes];
                        }

                        memory_windows[node][nodes_vec] = total_mem_window;
                        nodes_vec.clear();// so that memory window is not calculated again outside this loop
                    }
                    
                }
            }

            //boundary case when there is only one sequential binding
            if(nodes_vec.size()>0){
                auto total_mem_window=0;
                for(auto child_nodes:nodes_vec){
                    total_mem_window +=node_cost[child_nodes];
                }

                memory_windows[node][nodes_vec] = total_mem_window;
            }

            for(auto cnt=0; cnt<child_vec.size(); cnt++){
                if(child_vec[cnt]->get_type()==mapping::Node::InterTileBinding && cnt%2==0 && binding[cnt/2]==InterTileBinding::type_t::pipeline ){
                    auto next_two_nodes = find_next_two_nodes(child_vec[cnt]);//find next two nodes that have pipeline binding
                    for(auto val: next_two_nodes){
                        // if(std::find(visited_nodes.begin(), visited_nodes.end(), val)==visited_nodes.end()){
                        if(visited_nodes.find(val) == visited_nodes.end()){
                            // if the child_vec doesn't exist in visited nodes add it
                            memory_windows[node][val] = node_cost[val];
                        }
                        visited_nodes.insert(val);
                    }
                }
            }
        }

        //process all the children of this node
        // NodeTypes child_loop_nodes;
        // const void* child_loop_nodes = nullptr;
        LoopNode child_loop_node;
        ColOpNode col_op_node;
        
        if(loop_nodes.type==mapping::mapping_t::SPATIAL){


            auto binding_at_parent = find_binding(node);
            bool hide_rw_latency=true;
            if(binding_at_parent==InterTileBinding::type_t::sequential){
                //if binding at parent is sequential then consider ramp-up in every iteration hence cannot hide RW latency in every cycle
                hide_rw_latency=false; //FIXME::snegi change this only for some specific tensors
            }            

            for(auto&[child_nodes, child_memory_window]:memory_windows[node]){
                
                bool data_mov_tile=false;

                
                if(std::holds_alternative<std::vector<const Node*>>(child_nodes)){
                    //calculate the memory window
                    uint32_t max_iterations=0;
                    for(auto iterations:loop_nodes.iteration_count){
                        max_iterations = std::max(iterations, max_iterations);
                    }                    
                    std::vector<uint32_t> memWindow(max_iterations, child_memory_window);

                    get_mem_win_vector(node, child_nodes, memWindow);
                
                    auto& child_vector = std::get<std::vector<const Node*>>(child_nodes);
                    std::map<arch::ArchLevel*, std::map<std::tuple<TensorID, std::string>, uint32_t>> rampupMap;
                    for(auto& child_node: child_vector){ // for each of the child see with which tensors in the parent node does it intersects with and process those tensors
                        // auto child_loop_nodes = analysis_.workload_mapping_graph[child];

                        if(child_node->get_type()==mapping::Node::DataMovementTileNode){
                            child_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child_node]);
                            data_mov_tile = true;
                            hide_rw_latency = true;
                        } else {
                            col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child_node]);
                            data_mov_tile = false;
                            hide_rw_latency = false; // if the child is a collective operation node then there is always a conflict between child-memory writing back the data and a collective operation between child-memories //FIXME::snegi add a flag in the architecture file so that this conflict can be considered on the basis of hardware architecture
                        }

                        std::pair<arch::ArchLevel*, arch::ArchLevel*> intersected_target_child_pair;
                        for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){
                            auto tag_parent = loop_nodes.tags[tens_cnt];
                            auto target    = loop_nodes.target[tens_cnt];
                            auto child     = loop_nodes.child[tens_cnt];
                            auto target_id = target->getLevelID();
                            auto child_id  = child->getLevelID();
                            std::pair<arch::ArchLevel*, arch::ArchLevel*> arch_key={target, child};

                            //cannot hide the rw latency if it is a dependent tensor and the parent has sequential binding
                            // if(data_mov_tile && (binding_at_parent==InterTileBinding::type_t::sequential)) hide_rw_latency = !loop_nodes.dependent_tensor[tens_cnt]; // if the tensor is a dependent then the read-write latency cannot be hidden
                            if(data_mov_tile){ //doesn't matter what the binding at parent is, if it is sequential here and tensor is dependent it will see a compulsory stall
                                hide_rw_latency = !loop_nodes.dependent_tensor[tens_cnt]; // if the tensor is a dependent then the read-write latency cannot be hidden
                            }
                            // hide_rw_latency=true; 
                            //tag check to know if this tensor is being moved from current parent to this child only. For instance we can move tensors A,B from ParentMemory to ChildMemory. But maybe for this child node we only need to move tensor A and for another child node we want to move tensor B. So this tag checking helps to intersect these tensors from parent to child
                            bool cond1 = data_mov_tile && (std::find(child_loop_node.tags.begin(), child_loop_node.tags.end(), tag_parent)!=child_loop_node.tags.end());
                            bool cond2 = !data_mov_tile && tag_parent==col_op_node.tag; //FIXME::snegi maybe add a tensor intersection check as well, just to be 100% sure

                            // if(binding_at_parent==InterTileBinding::type_t::sequential){
                            //     //if the temporal parent has sequential binding then need to process this node with a new window after combining time window from other sequential children
                            //     if(cond1||cond2){
                            //         parent_child_arch_level_cost_map[node][child_vector][arch_key].emplace_back(get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, mapping::operation_t::None, child_memory_window, tens_cnt, hide_rw_latency, true));
                            //     }

                            // }
                            // else if(cond1 || cond2){
                            if(cond1 || cond2){
                                // parent_child_arch_level_cost_map[node][child_vector][arch_key].emplace_back(get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, mapping::operation_t::None, child_memory_window, tens_cnt, hide_rw_latency, false));

                                auto retval = get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, mapping::operation_t::None, child_memory_window, tens_cnt, hide_rw_latency, false, memWindow);

                                //read energy from target memory
                                access_count_struct temp_val;
                                uint8_t tensor_type = loop_nodes.tensor_is_rw[tens_cnt]? 1 : 0;
                                temp_val.arch_level = target;
                                temp_val.access_count = retval[tensor_type].access_count;
                                temp_val.tile_size = retval[tensor_type].tile_size;
                                temp_val.precision = loop_nodes.scale[tens_cnt];
                                temp_val.tensor_is_rw = loop_nodes.tensor_is_rw[tens_cnt]? true: false;
                                temp_val.target_child = true;
                                temp_val.is_compute = false;
                                temp_val.tid = loop_nodes.tensors[tens_cnt];
                                temp_val.tag = loop_nodes.tags[tens_cnt];
                                temp_val.rmw = loop_nodes.rmw[tens_cnt];
                                
                                auto sp_cnt = analysis_.datamovement_info[node][tens_cnt][0].num_unique_tiles;
                                if(sp_cnt!=0){
                                    temp_val.projected_spatial_count = sp_cnt;
                                } else {
                                    temp_val.projected_spatial_count = 1;
                                }


                                access_count_map[node].push_back(temp_val);
                                
                                //write energy to the child memory
                                access_count_struct temp_val_child;
                                temp_val_child.arch_level = child;
                                temp_val_child.access_count = retval[tensor_type].access_count;
                                temp_val_child.tile_size = retval[tensor_type].tile_size;
                                temp_val_child.precision = loop_nodes.scale[tens_cnt];
                                temp_val_child.tensor_is_rw = loop_nodes.tensor_is_rw[tens_cnt]? true: false;
                                temp_val_child.target_child = false; //child memory
                                temp_val_child.is_compute = false;
                                temp_val_child.tag = loop_nodes.tags[tens_cnt];
                                temp_val_child.noc_energy = retval[tensor_type].noc_energy;
                                temp_val_child.rmw = loop_nodes.rmw[tens_cnt];

                                temp_val_child.tid = loop_nodes.tensors[tens_cnt];
                                if(sp_cnt!=0){
                                    temp_val_child.projected_spatial_count = sp_cnt;
                                } else {
                                    temp_val_child.projected_spatial_count = 1;
                                }


                                access_count_map[node].push_back(temp_val_child);

                                if(!loop_nodes.tensor_is_rw[tens_cnt]){
                                    //ramup from only read tensors
                                    rampupMap[target][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = retval[tensor_type].ramp_up;
                                }
                                parent_child_arch_level_cost_map[node][child_vector][arch_key].emplace_back(retval);
                                
                            }
                        }
                    }

                    //find the arch level with max rampup cost
                    std::unordered_map<arch::ArchLevel*, uint32_t> ramupArchLevel;

                    for(auto&[key, tid_ramup]: rampupMap){
                        for(auto&[optid, ramup]:tid_ramup){
                            ramupArchLevel[key] +=ramup;
                        }
                    }
                    
                    if (!ramupArchLevel.empty()) {
                        auto max_it = std::max_element(ramupArchLevel.begin(), ramupArchLevel.end(), [](const auto& a, const auto& b){
                            return a.second<b.second;
                        });                        
                        auto maxValue = max_it->second;

                        for(const auto&[arch, value]: ramupArchLevel){
                            if(value==maxValue){
                                for(const auto&[key, val]: rampupMap[arch]){
                                    rampupInfo[node][key] = val;
                                }
                            }
                        }  
                    }                    

                    //FIXME::snegi if there is no intersection between parent_node's children arch level and child_node's target arch level then add a dummy compute_cost value in the map
                    // if(parent_child_arch_level_cost_map[node].find(child_vector) == parent_child_arch_level_cost_map[node].end()){
                    //     // if the child_vector key does not exist in the map that means there is no data transfer from parent-node to child node but the parent should still see the compute cost of the child
                    //     for
                    //     //if there is no datamovement from parent_node to child_vec node then find the target_child arch pair so that a dummy cost can be added to reflect the child's compute_time to the parent node
                    //     if(data_mov_tile && std::find(child_loop_node.target.begin(), child_loop_node.target.end(), child)!=child_loop_node.target.end())

                        

                    // }


                } else if(std::holds_alternative<const Node*>(child_nodes)){
                    //children is just a single node
                    auto& child_node = std::get<const Node*>(child_nodes);

                    if(child_node->get_type() == mapping::Node::DataMovementTileNode){
                        child_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child_node]);
                        data_mov_tile = true;
                        hide_rw_latency = true; //by default we can hide the latency
                    } else {
                        col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child_node]);
                        data_mov_tile = false;
                        hide_rw_latency = false; // if the child is a collective operation node then there is always a conflict between child-memory writing back the data and a collective operation between child-memories //FIXME::snegi add a flag in the architecture file so that this conflict can be considered on the basis of hardware architecture

                    }

                    uint32_t max_iterations=0;
                    for(auto iterations:loop_nodes.iteration_count){
                        max_iterations = std::max(iterations, max_iterations);
                    }                    
                    std::vector<uint32_t> memWindow(max_iterations, child_memory_window);

                    get_mem_win_vector(node, child_nodes, memWindow);

                    std::pair<arch::ArchLevel*, arch::ArchLevel*> intersected_target_child_pair;
                    std::map<arch::ArchLevel*, std::map<std::tuple<TensorID, std::string>, uint32_t>> rampupMap;
                    for(auto tens_cnt=0; tens_cnt<loop_nodes.tensors.size(); tens_cnt++){
                        auto tag_parent = loop_nodes.tags[tens_cnt];
                        auto target     = loop_nodes.target[tens_cnt];
                        auto child      = loop_nodes.child[tens_cnt];
                        auto target_id  = target->getLevelID();
                        auto child_id   = child->getLevelID();
                        std::pair<arch::ArchLevel*, arch::ArchLevel*> arch_key = {target, child};

                        if(data_mov_tile && std::find(child_loop_node.target.begin(), child_loop_node.target.end(), child)!=child_loop_node.target.end()){
                            intersected_target_child_pair = arch_key; //store the taget_child pair that has same child as the target in child_node so that if there is NO data transfer from parent-memory to child-memory we can add a dummy cost in the cost map
                        } else if(!data_mov_tile && child==col_op_node.target){
                            intersected_target_child_pair = arch_key;
                        }
                        
                        if(data_mov_tile && (binding_at_parent==InterTileBinding::type_t::sequential)) hide_rw_latency = !loop_nodes.dependent_tensor[tens_cnt]; // if the tensor is a dependent then the read-write latency cannot be hidden
                        // hide_rw_latency=true; 

                        bool cond1 = data_mov_tile && (std::find(child_loop_node.tags.begin(), child_loop_node.tags.end(), tag_parent)!=child_loop_node.tags.end());

                        bool cond2 = !data_mov_tile && tag_parent == col_op_node.tag;

                        if(cond1 || cond2){
                            //if there is a tensor intersection between parent node and child node
                            // parent_child_arch_level_cost_map[node][child_node][arch_key].emplace_back(get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, mapping::operation_t::None, child_memory_window, tens_cnt, hide_rw_latency, false));

                            auto retval = get_target_child_cost(target_id, child_id, analysis_.datamovement_info[node][tens_cnt], loop_nodes, mapping::operation_t::None, child_memory_window, tens_cnt, hide_rw_latency, false, memWindow);

                            //read energy from target memory
                            access_count_struct temp_val;
                            uint8_t tensor_type = loop_nodes.tensor_is_rw[tens_cnt]? 1 : 0;
                            temp_val.arch_level = target;
                            temp_val.access_count = retval[tensor_type].access_count;
                            temp_val.tile_size = retval[tensor_type].tile_size;
                            temp_val.precision = loop_nodes.scale[tens_cnt];
                            temp_val.tensor_is_rw = loop_nodes.tensor_is_rw[tens_cnt]? true: false;
                            temp_val.target_child = true;
                            temp_val.is_compute = false;
                            temp_val.tid = loop_nodes.tensors[tens_cnt];
                            temp_val.tag = loop_nodes.tags[tens_cnt];
                            temp_val.rmw = loop_nodes.rmw[tens_cnt];

                            // temp_val.noc_energy = retval[tensor_type].noc_energy;

                            auto sp_cnt = analysis_.datamovement_info[node][tens_cnt][0].num_unique_tiles;
                            if(sp_cnt!=0){
                                temp_val.projected_spatial_count = sp_cnt;
                            } else {
                                temp_val.projected_spatial_count = 1;
                            }
                           
                            access_count_map[node].push_back(temp_val);
                            
                            //write energy to the child memory
                            access_count_struct temp_val_child;
                            temp_val_child.arch_level = child;
                            temp_val_child.access_count = retval[tensor_type].access_count;
                            temp_val_child.tile_size = retval[tensor_type].tile_size;
                            temp_val_child.precision = loop_nodes.scale[tens_cnt];
                            temp_val_child.tensor_is_rw = loop_nodes.tensor_is_rw[tens_cnt]? true: false;
                            temp_val_child.target_child = false;
                            temp_val_child.is_compute = false;
                            temp_val_child.tid = loop_nodes.tensors[tens_cnt];
                            temp_val_child.tag = loop_nodes.tags[tens_cnt];
                            temp_val_child.noc_energy = retval[tensor_type].noc_energy;
                            temp_val_child.rmw = loop_nodes.rmw[tens_cnt];

                            if(sp_cnt!=0){
                                temp_val_child.projected_spatial_count = sp_cnt;
                            } else {
                                temp_val_child.projected_spatial_count = 1;
                            }
                                        
                            access_count_map[node].push_back(temp_val_child);                            

                            if(!loop_nodes.tensor_is_rw[tens_cnt]){
                                //ramup from only read tensors
                                rampupMap[target][{loop_nodes.tensors[tens_cnt], loop_nodes.tags[tens_cnt]}] = retval[tensor_type].ramp_up;
                            }                            
                            parent_child_arch_level_cost_map[node][child_node][arch_key].emplace_back(retval);
                        }
                    }

                    //find the arch level with max rampup cost
                    std::unordered_map<arch::ArchLevel*, uint32_t> ramupArchLevel;

                    for(auto&[key, tid_ramup]: rampupMap){
                        for(auto&[optid, ramup]:tid_ramup){
                            ramupArchLevel[key] +=ramup;
                        }
                    }

                    // if more than 1 element exist at this maxValue
                    if (!ramupArchLevel.empty()) {
                        auto max_it = std::max_element(ramupArchLevel.begin(), ramupArchLevel.end(), [](const auto& a, const auto& b){
                            return a.second<b.second;
                        });                        
                        auto maxValue = max_it->second;

                        for(const auto&[arch, value]: ramupArchLevel){
                            if(value==maxValue){
                                for(const auto&[key, val]: rampupMap[arch]){
                                    rampupInfo[node][key] = val;
                                }
                            }
                        }  
                    }                      

                    //if child_node doesn't exist in the map implies there is no data movement between parent and child memory, but child will still execute as many times as the other children //this is for a case when a CollectiveOp node exist at the leaf Buffer<=>Buffer Collective Operation
                    if(parent_child_arch_level_cost_map[node].find(child_node) == parent_child_arch_level_cost_map[node].end()){
                        //find the iterations for this child_node
                        uint32_t max_iterations=0;
                        for(auto iterations:loop_nodes.iteration_count){
                            max_iterations = std::max(iterations, max_iterations); //FIXME::snegi maybe only consider tensors
                        }

                        TargetChildCost cost(max_iterations); //since it is just going to contain the compute cost so we can put it at any of the port

                        std::fill(cost.compute_time_vec.begin(), cost.compute_time_vec.begin()+max_iterations, node_cost[child_node]);

                        cost.compute_time = max_iterations*node_cost[child_node];
                        cost.reuse_factor = 1;

                        TargetChildCostVec cost_vec;
                        cost_vec.push_back(cost);
                        cost_vec.push_back(cost);

                        parent_child_arch_level_cost_map[node][child_node][intersected_target_child_pair].emplace_back(cost_vec); 


                        //FIXME::snegi later check all the tensors that are connected to the tensor in this node and then only check the maximum number of iterations of these related tensors
                        // if(data_mov_tile){
                        // } else {
                        //     //collective op tile
                        //     auto tid = col_op_node.in_tensor;
                        //     // std::vector<problem::TensorID> producing_tensors;
                        //     //find the operation that is producing this tensor
                        //     for(auto workload:analysis_.workloads_.workloads_){
                        //         auto op_name = workload.first;
                        //         auto tensor_map = workload.second->get_TensorNameMap();
                        //         if(tensor_map.find(tid) !=tensor_map.end()){
                        //             // this workload is producing the tensor get its input tensors
                        //             auto& producing_tensors = workload.second->ins_;
                        //         }
                        //     }
                        // }


                    }
                }

            }


        } else {
            //********TEMPORAL********
            // spatial children of this node are already processed
            //datamovement children are already processed so only assign values and process collective operation nodes 
            for(auto&[child_nodes, child_memory_window]:memory_windows[node]){

                if(std::holds_alternative<std::vector<const Node*>>(child_nodes)){
                    auto& child_vector = std::get<std::vector<const Node*>>(child_nodes);
                    for(auto& child_node: child_vector){ 
                        if(child_node->get_type()==mapping::Node::CollectiveOperationNode){
                            // if there is a collective operation node at a temporal node there is NO data write back from the collective operation node

                            col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child_node]);
                            uint32_t max_iterations=0;
                            for(auto iterations:loop_nodes.iteration_count){
                                max_iterations = std::max(iterations, max_iterations); //FIXME::snegi maybe only consider tensors
                            }
                            auto tid = col_op_node.in_tensor;

                            for (const auto& kv : actual_iterations[node]) {
                                const auto& key = kv.first;
                                const auto& val = kv.second;

                                if (std::get<0>(key) == tid) {
                                    max_iterations = val;
                                }
                            }
                            TargetChildCost cost(max_iterations); //since it is just going to contain the compute cost so we can put it at any of the port. initializing the stall vectors so that later this cost vector is not missed from the cost calculation

                            std::fill(cost.compute_time_vec.begin(), cost.compute_time_vec.begin()+max_iterations, node_cost[child_node]);

                            cost.compute_time = max_iterations*node_cost[child_node];
                            cost.cummulative_mem_net_cycles = max_iterations*node_cost[child_node]; 

                            TargetChildCostVec cost_vec;
                            cost_vec.push_back(cost);
                            cost_vec.push_back(cost); //FIXME::snegi change this so that cost is put on read and write port properly

                            std::pair<arch::ArchLevel*, arch::ArchLevel*> arch_pair={col_op_node.target, col_op_node.child}; //since this is a collective operation node at a temporal node the target and child for this node will be same as the target of the temporal node.

                            parent_child_arch_level_cost_map[node][child_node][arch_pair].emplace_back(cost_vec);
                            parent_child_node_level_cost_map[node][child_node][col_op_node.target] = cost_vec;

                            // node_cost[child_node] = 
                        }
                    }
                } else if(std::holds_alternative<const Node*>(child_nodes)){

                    auto& child_node = std::get<const Node*>(child_nodes);
                    if(child_node->get_type() == mapping::Node::CollectiveOperationNode){
                        col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child_node]);
                        uint32_t max_iterations=0;
                        for(auto iterations:loop_nodes.iteration_count){
                            max_iterations = std::max(iterations, max_iterations); //FIXME::snegi maybe only consider tensors
                        }
                        auto tid = col_op_node.in_tensor;
                        for (const auto& kv : actual_iterations[node]) {
                            const auto& key = kv.first;
                            const auto& val = kv.second;

                            if (std::get<0>(key) == tid) {
                                max_iterations = val;
                            }
                        }                        

                        TargetChildCost cost(max_iterations);

                        std::fill(cost.compute_time_vec.begin(), cost.compute_time_vec.begin()+max_iterations, node_cost[child_node]);

                        cost.compute_time = max_iterations*node_cost[child_node];
                        cost.cummulative_mem_net_cycles = max_iterations*node_cost[child_node]; 

                        TargetChildCostVec cost_vec;
                        cost_vec.push_back(cost);
                        cost_vec.push_back(cost);
                        
                        std::pair<arch::ArchLevel*, arch::ArchLevel*> arch_pair={col_op_node.target, col_op_node.child}; 

                        parent_child_arch_level_cost_map[node][child_node][arch_pair].emplace_back(cost_vec);
                        parent_child_node_level_cost_map[node][child_node][col_op_node.target] = cost_vec;

                        parent_child_total_cost[node][child_node] = cost.compute_time;


                    }

                }

                // fill in the cost of other children if the current binding at temporal node is sequential
                // FIXME::snegi remove this, not useful anymore
                if(std::find(binding.begin(), binding.end(), InterTileBinding::type_t::sequential)!=binding.end()){
                    //sequential binding exist
                    for(auto sp_child:node->get_children()){
                        if(sp_child->get_type()==Node::InterTileBinding) continue;
                        for(auto&[sp_child_child, arch_cost]: parent_child_arch_level_cost_map[sp_child]){
                            parent_child_arch_level_cost_map[node][sp_child] = arch_cost;
                        }                        
                    }
                }

            }
            // return; 
        }

        //calculate parent_child_node_level_cost map
        //basically have the parent_node-child_node map with the cost at the level of target memories

        if(loop_nodes.type==mapping::mapping_t::SPATIAL){
            for(auto&[child_nodes, child_memory_window]:memory_windows[node]){
                
                std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>> child_arch_level_port_usage;
                auto& child_arch_level = loop_nodes.child;
                auto shared_port=false;
                for(auto& c: child_arch_level){
                    auto level_spec = std::get<arch::MemorySpec>(c->getSpec());
                    auto given_val = level_spec.parent_child_ports_shared_.Get();
                    shared_port = shared_port || given_val;
                }

                if(shared_port){
                    child_arch_level_port_usage = get_port_level_cost(child_nodes);//arch-level:{port, value}
                }

                parent_child_node_level_cost_map[node][child_nodes] = get_node1_node2_cost(parent_child_arch_level_cost_map[node][child_nodes], loop_nodes, child_arch_level_port_usage);

                parent_child_cost_map[node][child_nodes] = get_node_cost(parent_child_node_level_cost_map[node][child_nodes], loop_nodes, false);

                parent_child_total_cost[node][child_nodes] = get_total_node_cost(parent_child_cost_map[node][child_nodes]);
            }

            //check if the bining only has sequential bindings
            bool only_sequential=true;
            for(auto b: binding){
                if(b==InterTileBinding::type_t::pipeline){
                    only_sequential=false;
                    break;
                } else if(b==InterTileBinding::type_t::sharing){
                    only_sequential=false;
                    break;
                } else if(b==InterTileBinding::type_t::parallel){
                    only_sequential=false;
                    break;
                }
            }


            // if only sequential pipelining exist parent_child_node_level_cost_map has all the numbers
            if(binding.size()==0){
                node_cost[node] = parent_child_total_cost[node][node->get_children().back()]; //since only single children
            } else if(binding.size()>0 && only_sequential){
                COMET_ASSERT(memory_windows[node].size()==1, "although there are only sequential bindings but memory window key has more than 1 entry for this level");
                auto it = memory_windows[node].begin();
                node_cost[node] = parent_child_total_cost[node][it->first]; //since only single children

            }
            else {
                // if some other binding exist call get_node_cost_with_binding function
                node_cost[node] = get_node_cost_with_binding(node, binding, loop_nodes);
            }
        } else if(loop_nodes.type==mapping::mapping_t::TEMPORAL){
            if(binding.size()>0 && (std::find(binding.begin(), binding.end(), InterTileBinding::type_t::sequential)!=binding.end())){

                node_cost[node] = get_temporal_node_cost1(node);
                // node_cost[node] = get_temporal_node_cost(parent_child_arch_level_cost_map[node], loop_nodes);
            } else if (binding.size()>0){
                //if not sequential it has pipeline binding
                node_cost[node] = get_node_cost_with_binding(node, binding, loop_nodes);
            }
        }

        return;
    }

    void SteadyStateWalker::visitCollectiveOperationNode(const CollectiveOperationNode* node){
        
        std::cout<<"!!! ColOpNode name in cost_walker:: "<<node->get_name()<<"\n";
        auto col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[node]);
        
        // node_arch_level_cost_map[node][]
        auto target_id = col_op_node.target->getLevelID();
        auto child_id  = col_op_node.child->getLevelID();
        auto& tensor_mov = analysis_.datamovement_info[node];

        auto cost = analysis_.topology_.EvaluateCollectiveOperation(target_id, child_id, tensor_mov, col_op_node, col_op_node.in_tensor); //memory_time and network_time

        auto key=std::make_pair(col_op_node.target, col_op_node.child);
        TargetChildCostVec cost_vec(2,0);
        
        //tensors considered read-write tensors for collective operations
        // same stalls on both read and write ports
        cost_vec[0].cummulative_mem_net_cycles = cost.memory_time + cost.network_time;
        cost_vec[0].cummulative_stall = cost.memory_time + cost.network_time;
        cost_vec[0].reuse_factor = 1;

        cost_vec[1].cummulative_mem_net_cycles = cost.memory_time + cost.network_time;
        cost_vec[1].cummulative_stall = cost.memory_time + cost.network_time;
        cost_vec[1].reuse_factor = 1;

        auto node_pair = std::make_pair(node, node);
        // parent_child_arch_level_cost_map[node_pair][key].emplace_back(cost_vec); //tensors from collective operation is always 

        parent_child_arch_level_cost_map[node][node][key].emplace_back(cost_vec);

        parent_child_node_level_cost_map[node][node][col_op_node.target] = cost_vec;

        node_cost[node] = cost.memory_time + cost.network_time;
        // no parent child cost for just the collective operation node(since it is a single node)
        // for(auto idx=0; idx<2; idx++){
        //     parent_child_arch_level_cost_map[node_pair][col_op_node.target].emplace_back(cost_vec[idx].compute_time + cost_vec[idx].cummulative_stall);
        // } 

        ArchLevelEnergy arch_energy;

        arch_energy[col_op_node.target]=cost.read_energy + cost.write_energy;

        node_level_access_energy[node] = arch_energy;

        NOCEnergy noc_energy;
        std::string noc_energy_str = col_op_node.target->getName() + "<->" + col_op_node.target->getName();
        noc_energy[noc_energy_str] = cost.network_energy;
        node_level_noc_energy[node] = noc_energy;
        return;
    }

    void SteadyStateWalker::visitOperationNode(const OperationNode* node){
        return;
    }

    uint32_t SteadyStateWalker::get_temporal_node_cost1(const Node* node){

        uint32_t time_wo_overlap=0;

        for(auto child: node->get_children()){
            if(child->get_type()==Node::InterTileBinding) continue;
                if(child->get_type()==Node::CollectiveOperationNode){
                    auto col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child]);
                    time_wo_overlap += parent_child_node_level_cost_map[node][child][col_op_node.target][0].compute_time; // this is multiplied by number of iteration of the parent
                } else{
                    time_wo_overlap +=node_cost[child];
                }
            
        }
        
        

        // for(auto child: node->g)

        // /*
        // std::map<const Node*, std::map<ArchLevel*, std::vector<uint32_t>>> arch_level_stalls; //only store the archlevel with max stall for that node
        std::map<const Node*, std::map<uint32_t, std::map<ArchLevel*, uint32_t>>>  arch_level_stalls; //2nd map keys are the port index

        std::map<const Node*, std::map<uint32_t, std::map<ArchLevel*,std::set<ArchLevel*>> >>  child_memories; //need to know with which archlevel in the child are we hidding the data movement energy. Can only hide if the child memory of the data movement does not intersect with the child memory of the slacking window, last map has target archlevel as the keys

        
        // std::map<const Node*, std::map<uint32_t, std::map<ArchLevel*, uint32_t>>>  compulsory_stalls;

        // std::map<const Node*, std::map<ArchLevel*, std::vector<uint32_t>>> compulsory_stalls; 

        // std::map<const Node*, std::map<ArchLevel*, std::vector<uint32_t>>> arch_level_slack; //archlevel here is of the child memory so basically at this memory it will not conflict
        std::map<const Node*, std::map<uint32_t, std::map<std::set<ArchLevel*>, uint32_t>>> slack_map;
        
        std::map<const Node*, std::map<uint32_t, std::map<std::set<ArchLevel*>, uint32_t>>> compulsory_slack;

        // for(auto&[child_node, node_level_cost]:parent_child_node_level_cost_map[node]){
        for(auto child: node->get_children()){
            if(child->get_type()==Node::InterTileBinding) continue;
            auto node_level_costs = parent_child_node_level_cost_map[child];
            for(auto& [childnodes, node_level_cost]: node_level_costs){ //childnodes is the child of spatial nodes
                for(auto idx=0; idx<2; idx++){

                    ArchLevel* max_arch_level;
                    int max_arch_level_stall=0;
                    for(auto&[arch_level, target_child_cost_vec]: node_level_cost){

                        auto stall = std::accumulate(target_child_cost_vec[idx].stall.begin(), target_child_cost_vec[idx].stall.end(),0);
                        auto compulsory_stall = std::accumulate(target_child_cost_vec[idx].compulsory_stall.begin(), target_child_cost_vec[idx].compulsory_stall.end(),0);
                        if(stall>=max_arch_level_stall){//even if stall is zero just get the max_arch_level key initialized
                            max_arch_level_stall = stall;
                            max_arch_level = arch_level;
                        }
                        // compulsory_stalls[child][arch_level].emplace_back(compulsory_stall);
                    }
                    // arch_level_stalls[child][max_arch_level].emplace_back(max_arch_level_stall);
                    if(max_arch_level_stall!=0){
                        arch_level_stalls[child][idx][max_arch_level] = max_arch_level_stall;
                    }
                    
                }
            }
        }
        
        uint32_t extra_stalls=0;
        std::map<const Node*, uint32_t> max_arch_level_stalls;
        //extra stalls added in time without overlap
        for(auto& [child, port_level_cost]: arch_level_stalls){
            
            uint32_t max=0;
            for(auto&[idx, cost_map]:port_level_cost){
                for(auto&[archlevel, val]:cost_map){
                    max=std::max(max,val);
                }
            }
            max_arch_level_stalls[child]=max;
            extra_stalls +=max;
        }


        //calculate slack
        for(auto child: node->get_children()){
            if(child->get_type()==Node::InterTileBinding) continue;
            auto node_level_costs = parent_child_node_level_cost_map[child];
            for(auto& [childnodes, node_level_cost]: node_level_costs){
                for(auto idx=0; idx<2; idx++){
                    for(auto&[arch_level, target_child_cost_vec]: node_level_cost){
                        
                        if(arch_level_stalls[child][idx].find(arch_level)!=arch_level_stalls[child][idx].end()){
                            //arch_level exists in the arch_level stalls map
                            auto compute_time = target_child_cost_vec[idx].compute_time;
                            auto slack = compute_time - target_child_cost_vec[idx].cummulative_mem_net_cycles;
                            // arch_level_stalls[child][idx][arch_level];

                            //if childnodes is a vector
                            bool no_col_op=true;
                            if(slack>0){
                                if(std::holds_alternative<std::vector<const Node*>>(childnodes)){
                                    auto& child_vector = std::get<std::vector<const Node*>>(childnodes);
                                    //if child vector has collective operation node
                                    for(auto c:child_vector){
                                        if(c->get_type()==Node::CollectiveOperationNode){
                                            auto col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[c]); 
                                            std::set<ArchLevel*> arch_set={col_op_node.target};
                                            slack_map[child][idx][arch_set]=slack; //FIXME::snegi put slacks individually for collective op and other nodes
                                            no_col_op=false;
                                        }
                                    }
                                    if(no_col_op){
                                        std::set<ArchLevel*> arch_set={nullptr}; //just keep nullptr as the key because the arch level doesn't matter for this
                                        // std::set<ArchLevel*> arch_set={analysis_.topology_.getCompute};
                                        if(slack_map[child][idx].find(arch_set)==slack_map[child][idx].end()){
                                            //nullptr key doesn't exist
                                            slack_map[child][idx][arch_set]=slack;
                                        } else {
                                            slack_map[child][idx][arch_set] +=slack;
                                        }
                                    }
                                } else{
                                    auto& child_vector = std::get<const Node*>(childnodes);
                                    if(child_vector->get_type()==Node::CollectiveOperationNode){
                                        auto col_op_node = std::get<ColOpNode>(analysis_.workload_mapping_graph[child_vector]); 
                                        std::set<ArchLevel*> arch_set={col_op_node.target};
                                        slack_map[child][idx][arch_set]=slack;
                                    } else{
                                        std::set<ArchLevel*> arch_set={nullptr};
                                        if(slack_map[child][idx].find(arch_set)==slack_map[child][idx].end()){
                                            //nullptr key doesn't exist
                                            slack_map[child][idx][arch_set]=slack;
                                        } else {
                                            slack_map[child][idx][arch_set] +=slack;
                                        }                                    
                                    }

                                }
                            }
                        }


                        //add compulsory stalls to the other port as a slack
                        auto slack = std::accumulate(target_child_cost_vec[idx].compulsory_stall.begin(), target_child_cost_vec[idx].compulsory_stall.end(),0);
                        std::set<ArchLevel*> arch_set={nullptr};
                        
                        if(slack>0 && slack_map[child][idx].find(arch_set)==slack_map[child][idx].end()){
                            //nullptr key doesn't exist
                            compulsory_slack[child][1-idx][arch_set]=slack;
                        } else if (slack>0) {
                            compulsory_slack[child][1-idx][arch_set] +=slack;
                        }
                    }
                }
            }
        }
        

        // */

        



        return time_wo_overlap; 
    }

    uint32_t SteadyStateWalker::get_temporal_node_cost(std::map<ChildNodes, ArchLevelCost> child_arch_level_cost, LoopNode& loop_nodes){

        uint32_t retval;
        //calculate the window which will have compute and write back time included in it

        // for(auto&[child, cost]: arch_level_cost){

        // }

        NodeLevelCost node_level_cost;
        std::map<arch::ArchLevel*, std::vector<std::vector<uint32_t>>> mem_win_per_tensor; //Outer std::vector because from one archlevel there can be multiple child-memories that can be fed. std::vector inside for number of iterations
        std::map<arch::ArchLevel*, uint32_t> tensor_cnt;
        size_t tens_cnt;

        uint32_t max_iterations=0;
        // for(auto&tens_cnt: loop_nodes.tensors){ //only go over the tensors which are between parent and child FIXME::snegi
        for(auto& iterations: loop_nodes.iteration_count){
            max_iterations = std::max(max_iterations, iterations);
            // loop_nodes.iteration_count[tens_cnt]);
        }

        uint32_t total_compute_time=0;
        // uint32_t total_col_op_time;
        for(auto&[childnode, arch_level_cost]: child_arch_level_cost){

            //if there is a collective operation children at temporal node do not add it in the compute time, add that time in the end ---> Can't do this because there can be other memories present and they might not conflict with the collective operation memory
            // if(std::holds_alternative<const Node*>(childnode)){
            //     //check if this node is a collective operation node
            //     auto& child_node = std::get<const Node*>(childnode);
            //     if(child_node->get_type() == mapping::Node::CollectiveOperationNode){
            //         total_col_op_time = node_cost[child_node];
            //         continue;
            //     }
            // }

            // uint32_t compute_time_per_child=0;
            for(auto&[arch_pair, cost_vec]: arch_level_cost){
                //all the children of childnode will have same compute time
                //sum up the compute_time of all the child nodes
                total_compute_time += std::max(cost_vec.front()[0].compute_time, cost_vec.front()[1].compute_time);
                // std::max(compute_time_per_child, ) to be safe take max of read and write port
                break;
            }
            

        }

        std::vector<uint32_t> compute_time_vec(max_iterations, 0); // not every tensor will have same number of iterations //FIXME::snegi
        std::fill(compute_time_vec.begin(), compute_time_vec.end(), total_compute_time/max_iterations);
        for(auto&[childnode, arch_level_cost]: child_arch_level_cost){
            if(std::holds_alternative<const Node*>(childnode)){
                auto child_node = std::get<const Node*>(childnode);
                std::cout<< "temporal node child name: "<< child_node->get_name()<<std::endl;
                if(child_node->get_type()==Node::CollectiveOperationNode) continue;
                // <<child_node->get_name()
            } else {
                auto child_node = std::get<std::vector<const Node*>>(childnode);
            }
            for(auto&[arch_pair, cost_vec]: arch_level_cost){
                if(arch_pair.first==arch_pair.second) continue; //this is collective operation node
                for(auto idx=0; idx<2; idx++){
                    for(auto& per_tensor_cost: cost_vec){
                        if(per_tensor_cost[idx].stall.size()>0){ // add a place holder for both read and read-write tensors so that index are correct when accessed
                            // uint32_t total_cost_window = 
                            // mem_win_per_tensor[arch_pair.first].emplace_back(per_tensor_cost[idx].compute_time_vec);

                            // std::vector<uint32_t> per_tensor_compute_time_vec(per_tensor_cost[idx].compute_time_vec.size(),0);
                            //by executing multiple operations sequentially the memory window for the tensor has increased;


                            mem_win_per_tensor[arch_pair.first].push_back(compute_time_vec);
                        }
                    }
                }
            }
        }


        // if(std::holds_alternative<const Node*>(childnode)){
        //     //check if this node is a collective operation node
        //     auto& child_node = std::get<const Node*>(childnode);
        //     if(child_node->get_type() == mapping::Node::CollectiveOperationNode){
        //         num_datamov_children-=1; 
        //         continue;
        //     }
        // }
        // auto num_datamov_children = child_arch_level_cost.size();

        for(auto&[childnode, arch_level_cost]: child_arch_level_cost){

            if(std::holds_alternative<const Node*>(childnode)){
                auto child_node = std::get<const Node*>(childnode);
                std::cout<< "temporal node child name: "<< child_node->get_name()<<std::endl;
                // <<child_node->get_name()
                if(child_node->get_type() == mapping::Node::CollectiveOperationNode) continue;
            } else {
                auto child_node = std::get<std::vector<const Node*>>(childnode);
            }

            for(auto&[arch_pair, cost_vec]: arch_level_cost){
                if(node_level_cost.find(arch_pair.first)==node_level_cost.end()){
                    TargetChildCost temp_cost(max_iterations);
                    TargetChildCostVec temp_cost_vec;
                    temp_cost_vec.emplace_back(temp_cost);
                    temp_cost_vec.emplace_back(temp_cost);

                    node_level_cost[arch_pair.first] = temp_cost_vec;
                    tensor_cnt[arch_pair.first] = 0;
                }

                
                for(auto idx=0; idx<2; idx++){   
                    for(auto& per_tensor_cost: cost_vec){
                        tens_cnt = tensor_cnt[arch_pair.first];
                        if(per_tensor_cost[idx].stall.size()>0){


                            node_level_cost[arch_pair.first][idx].ramp_up      += per_tensor_cost[idx].ramp_up;
                            node_level_cost[arch_pair.first][idx].ramp_down    += per_tensor_cost[idx].ramp_down;
                            node_level_cost[arch_pair.first][idx].reuse_factor = std::min(node_level_cost[arch_pair.first][idx].reuse_factor, per_tensor_cost[idx].reuse_factor);
                            node_level_cost[arch_pair.first][idx].compute_time_vec = compute_time_vec;
                            node_level_cost[arch_pair.first][idx].cummulative_mem_net_cycles += per_tensor_cost[idx].cummulative_mem_net_cycles;

                            uint32_t factor;

                            factor = per_tensor_cost[idx].reuse_factor;

                            for(auto i=idx; i<max_iterations; i++){
                                auto consider_stall = i%factor;
                                auto stall_idx = i/factor;

                                if(!per_tensor_cost[idx].hide_rw_latency && consider_stall==0){
                                    // node_level_cost[arch_pair.first][idx].stall[i] += per_tensor_cost[idx].stall[stall_idx];
                                    
                                    node_level_cost[arch_pair.first][idx].compulsory_stall[i] += per_tensor_cost[idx].stall[stall_idx];
                                    
                                    continue;

                                }
                                else if(consider_stall==0){
                                    int stall = per_tensor_cost[idx].mem_net_cycles[stall_idx] - mem_win_per_tensor[arch_pair.first][tens_cnt][stall_idx];
                                    if(stall>0){
                                        node_level_cost[arch_pair.first][idx].stall[i] +=stall;
                                    }
                                }


                                std::map<arch::ArchLevel*, uint32_t> kth_tensor_cnt;
                                size_t kth_tens_cnt;
                                if(kth_tensor_cnt.find(arch_pair.first)==kth_tensor_cnt.end()){
                                    kth_tensor_cnt[arch_pair.first] = 0;
                                }

                                for(auto&[childnode1, arch_level_cost1]: child_arch_level_cost){
                                    for(auto&[arch_pair1, cost_vec1]: arch_level_cost1){
                                        for(auto idx1=0; idx1<2; idx1++){
                                            for(auto& per_tensor_cost1: cost_vec1){
                                                if(per_tensor_cost1[idx1].stall.size()>0){ 
                                                    kth_tens_cnt = kth_tensor_cnt[arch_pair.first];
                                                    if(kth_tens_cnt==tens_cnt) continue; // skip window update for the same tensor
                                                    if(arch_pair.first == arch_pair1.first && idx==idx1){
                                                        auto kth_tens_idx = per_tensor_cost1[idx].tens_idx;

                                                        size_t kth_factor;
                                                        kth_factor = per_tensor_cost1[idx].reuse_factor;

                                                        auto kth_consider_stall = i%kth_factor;
                                                        auto kth_stall_idx = i/kth_factor;

                                                        if(kth_consider_stall==0){
                                                            auto temp = mem_win_per_tensor[arch_pair.first][kth_tens_cnt][kth_stall_idx] - per_tensor_cost[idx].mem_net_cycles[stall_idx];
                                                            mem_win_per_tensor[arch_pair.first][kth_tens_cnt][kth_stall_idx] = temp<0 ? 0 : temp;
                                                        }

                                                    }
                                                    kth_tensor_cnt[arch_pair.first]++;
                                                }
                                            }
                                        }
                                    }
                                }

                            }
                            // tens_cnt++;
                            tensor_cnt[arch_pair.first]++;
                        }
                    }


                }

            }
        }

        auto node_cost = get_node_cost(node_level_cost, loop_nodes, false);

        retval = get_total_node_cost(node_cost);

        return retval;

    }

    uint32_t SteadyStateWalker::get_node_cost_with_binding(const Node* parent_node, std::vector<InterTileBinding::type_t> bindings, LoopNode loop_nodes){

        //FIXME::snegi add an assert to make sure each children has same number of nodes for pipeline binding

        uint32_t retval;

        uint32_t time_without_conflict; // max_time + leftovers
        uint32_t pipeline_leftovers;

        uint32_t maxVal=0;
        ChildNodes maxNode;

        uint32_t max_iterations=0;
        for(auto iterations: loop_nodes.iteration_count){
            if(iterations>max_iterations){
                max_iterations = iterations;
            }
        }

        uint32_t total_iterations_with_pipeline;
        
        if(loop_nodes.type==mapping_t::TEMPORAL){
            uint32_t sp_colop_child=0;
            for(auto& child:parent_node->get_children()){
                if(child->get_type()==Node::InterTileBinding)continue;
                if(child->get_type()==Node::CollectiveOperationNode) sp_colop_child++;
                if(child->get_type()==Node::DataMovementTileNode){
                    //this should be the spatial node 
                    sp_colop_child += parent_child_node_level_cost_map[child].size();
                    // not running a for loop over child because some of the children might be the super node, so safe to get the size using parent_child_node_level_cost_map
                    // for(auto& c:child->get_children()){
                    //     if(c->get_type()==Node::InterTileBinding)continue;
                    //     sp_colop_child++;
                    // }
                }

            }
            total_iterations_with_pipeline = max_iterations + sp_colop_child - 1;
        } else{
            total_iterations_with_pipeline = max_iterations + parent_child_node_level_cost_map[parent_node].size()-1; // for 3 child pipelined with 4 max_iterations would require 2 more cycle
        }

        size_t max_child_idx;
        size_t cnt=0;
        
        if(loop_nodes.type==mapping_t::TEMPORAL){
            for(auto& child: parent_node->get_children()){
                if(child->get_type()==Node::InterTileBinding)continue;
                else if(child->get_type()==Node::CollectiveOperationNode){
                    auto cost = parent_child_total_cost[parent_node][child];
                    if(cost>maxVal){
                        maxVal = cost;
                        maxNode = child;
                        max_child_idx = cnt;
                    }
                    cnt++;
                }
                else if(child->get_type()==Node::DataMovementTileNode){
                    for(auto&[child_node, cost]:parent_child_total_cost[child]){
                        if(cost>maxVal){
                            maxVal = cost;
                            maxNode = child_node;
                            max_child_idx = cnt;
                        }
                        cnt++;
                    }
                }

            }
        } else {
            for(auto&[child_node, cost]: parent_child_total_cost[parent_node]){
                if(cost>maxVal) {
                    maxVal        = cost;
                    maxNode       = child_node;
                    max_child_idx = cnt;
                }
                cnt++;
            }
        }

        //calculate the leftover time
        std::map<ChildNodes, std::pair<size_t, size_t>> range_map; // child_index -> {start_cycle_index, end_cycle_index}
        cnt=0;

        if(loop_nodes.type==mapping_t::TEMPORAL){
            for(auto& child: parent_node->get_children()){
                if(child->get_type()==Node::InterTileBinding)continue;
                else if(child->get_type()==Node::CollectiveOperationNode){
                    std::pair<size_t, size_t> range={cnt, max_iterations-1+cnt};
                    range_map[child] = range;
                    cnt++;
                }
                else if(child->get_type()==Node::DataMovementTileNode){
                    for(auto&[child_node, cost]:parent_child_total_cost[child]){
                        std::pair<size_t, size_t> range={cnt, max_iterations-1+cnt};
                        range_map[child_node] = range;
                        cnt++;
                    }
                }

            }
        } else {
            for(auto&[child_node, cost]: parent_child_total_cost[parent_node]){
                std::pair<size_t, size_t> range = {cnt, max_iterations-1+cnt};
                range_map[child_node] = range;
                cnt++;
            }
        }

        std::vector<size_t> left_over_iterations;

        for(auto i=0; i<max_child_idx; i++){
            left_over_iterations.emplace_back(i);
        }

        for(auto i=max_child_idx+max_iterations; i<total_iterations_with_pipeline; i++){
            left_over_iterations.emplace_back(i);
        }

        uint32_t left_over_time=0;
        // std::map<size_t, std::vector<uint32_t>> max_time;

        for(auto idx:left_over_iterations){
            std::vector<uint32_t> max_time;

            if(loop_nodes.type==mapping_t::TEMPORAL){
                for(auto& child: parent_node->get_children()){
                    if(child->get_type()==Node::InterTileBinding)continue;
                    else if(child->get_type()==Node::CollectiveOperationNode){
                        auto range = range_map[child];
                        if((ChildNodes)child==maxNode) continue; //typecast child to ChildNodes

                        auto cost=parent_child_total_cost[parent_node][child];

                        if(idx>=range.first && idx<=range.second) {
                            // convert idx to the iterations of the individual nodes
                            size_t converted_idx = idx - range.first;
                            // if(idx>max_iterations-1){
                            //     converted_idx = idx - parent_child_arch_level_cost_map.size();
                            // }
                            max_time.push_back(cost/max_iterations);
                        }
                    }
                    else if(child->get_type()==Node::DataMovementTileNode){
                        for(auto&[child_node, cost]:parent_child_cost_map[child]){
                            auto range = range_map[child_node];
                            if(child_node==maxNode) continue;

                            if(idx>=range.first && idx<=range.second) {
                                // convert idx to the iterations of the individual nodes
                                size_t converted_idx = idx - range.first;
                                // if(idx>max_iterations-1){
                                //     converted_idx = idx - parent_child_arch_level_cost_map.size();
                                // }
                                if(converted_idx==0){
                                    //add ramp down cost as well
                                    max_time.push_back(cost.ramp_up + cost.compute_time/max_iterations + cost.stall[converted_idx]);
                                } else if(converted_idx==max_iterations-1){
                                    //add ramp up cost as well
                                    max_time.push_back(cost.ramp_down + cost.compute_time/max_iterations + cost.stall[converted_idx]);
                                } else{
                                    // just the total time
                                    max_time.push_back(cost.compute_time/max_iterations + cost.stall[converted_idx]);
                                }
                            }
                        }
                    }

                }
            } else {            
                // size_t cnt=0;
                for(auto&[child_node, cost]: parent_child_cost_map[parent_node]){
                    auto range = range_map[child_node];
                    if(child_node==maxNode) continue;

                    if(idx>=range.first && idx<=range.second) {
                        // convert idx to the iterations of the individual nodes
                        size_t converted_idx = idx - range.first;
                        // if(idx>max_iterations-1){
                        //     converted_idx = idx - parent_child_arch_level_cost_map.size();
                        // }
                        if(converted_idx==0){
                            //add ramp down cost as well
                            max_time.push_back(cost.ramp_up + cost.compute_time/max_iterations + cost.stall[converted_idx]);
                        } else if(converted_idx==max_iterations-1){
                            //add ramp up cost as well
                            max_time.push_back(cost.ramp_down + cost.compute_time/max_iterations + cost.stall[converted_idx]);
                        } else{
                            // just the total time
                            max_time.push_back(cost.compute_time/max_iterations + cost.stall[converted_idx]);
                        }
                    }
                    // cnt++;
                }

            }

            uint32_t max=0;
            for(auto m: max_time) max=std::max(max, m);

            left_over_time +=max;
        }

        time_without_conflict = maxVal + left_over_time;

        //calculate conflicts, use parent_child_node_level_cost map to find the conflicting memories and then calculate the total time
        std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>> level_conflict_cost_map; //std::vector inside for read and read-write port

        std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>> num_updates; //map to know the number of times an archlevel is updated, if it is updated just once that means there is no conflict        

        std::map<size_t, uint32_t> num_iterations;//required to scale up the cost of child level to the parent level //FIXME::snegi make this a fuunction of currentnode

        bfs_conflict_map_constructor(parent_node, level_conflict_cost_map, num_updates, num_iterations);

        std::map<size_t, uint32_t> memory_time;

        for(auto&[level, arch_level_cost_map]: level_conflict_cost_map){

            for(auto&[arch_level, cost_vec]: arch_level_cost_map){

                for(auto idx=0; idx<2; idx++){

                    if(num_updates[level][arch_level][idx]>1){
                        //if the number of updates is greater than 1 then there is a conflict add this to conflict time
                        //scaled value. If there is a deeper level then multiply the cost by the compound iteration of parent levels

                        auto scaled_value = cost_vec[idx];                    
                        for(auto i=level; i>0; i--){
                            scaled_value *=num_iterations[i-1];
                        }
                            
                        
                        if(memory_time.find(level)==memory_time.end()){
                            memory_time[level] = scaled_value; //cost_vec[idx];
                        } else {
                            memory_time[level] = std::max(memory_time[level], scaled_value); //taking maximum across different memories and ports at this level
                        }
                    }

                }
            }

        }

        // calculate conflict
        int conflict_time=0;
        uint32_t time_used_from_time_wo_conflict=0;
        for(auto[level, time]: memory_time){
            // conflict_time = time - (time_without_conflict-time_used_from_time_wo_conflict); //removing conflict_time from the time_without conflict because from level 0 used that much time so that time is not available for memories at level 1
            int diff = time - time_without_conflict; // different levels have different memories so we don't need "time_used_from_time_wo_conflict" variable

            // time_used_from_time_wo_conflict = time;
            if(diff>0) conflict_time +=diff;
        }
        
        retval = time_without_conflict + conflict_time; 
        
        return retval; //time_without_conflicts + memory_time;

    }


    void SteadyStateWalker::bfs_conflict_map_constructor(const Node* root, std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>>& level_map, std::map<size_t, std::map<arch::ArchLevel*, std::vector<uint32_t>>>& num_updates_map, std::map<size_t, uint32_t>& num_iterations){
        if(root->get_type()==mapping::Node::OperationNode) return;

        std::queue<std::pair<const Node*, size_t>> q;
        q.push({root, 0}); // add current node as the root node and process the children

        while(!q.empty()){
            auto [currentNode, level] = q.front();
            q.pop();

            // if(currentNode->get_type()==mapping::Node::DataMovementTileNode){
            auto loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[currentNode]);

            num_iterations[level] = *std::max_element(loop_nodes.iteration_count.begin(), loop_nodes.iteration_count.end()); //max_element returns an iterator pointint to the max element in the range. * retrives the value from the iterator
            // }


            for(auto&[child_node, node_level_cost]: parent_child_node_level_cost_map[currentNode]){
                for(auto&[arch_level, cost_vec]: node_level_cost){
                    for(auto idx=0; idx<cost_vec.size(); idx++){
                    // for(auto idx=0; idx<2; idx++){
                        if(level_map[level].find(arch_level)==level_map[level].end()){
                            // this arch level does not exist in the map
                            level_map[level][arch_level]={0,0};
                            num_updates_map[level][arch_level]={0,0};
                        } 
                        if(idx==0){
                            auto mem_time = cost_vec[idx].cummulative_mem_net_cycles + cost_vec[idx].ramp_up;
                            level_map[level][arch_level][idx] += mem_time;

                            if(mem_time>0) num_updates_map[level][arch_level][idx] +=1;

                        } else {
                            auto mem_time = cost_vec[idx].cummulative_mem_net_cycles + cost_vec[idx].ramp_down; 
                            level_map[level][arch_level][idx] += mem_time;
                            if(mem_time>0) num_updates_map[level][arch_level][idx] +=1;
                        }
                    }
                }

            }

            for(auto child: currentNode->get_children()){
                if(child->get_type()==Node::InterTileBinding) continue;
                if(child->get_type()==Node::OperationNode) continue;
                if(child->get_type()==Node::CollectiveOperationNode) continue;

                //datamovement tile node
                auto child_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);
                if(child_loop_node.same_target_as_parent){
                    //do not change the level because this child is the spatial node of the temporal parent
                    q.push({child, level});
                } else{
                    q.push({child, level+1});
                }

                // if(loop_nodes.type==mapping_t::TEMPORAL){
                //     //datamovement tile node
                //     auto child_loop_node = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);
                //     if()

                // } else {
                //     q.push({child, level+1});
                // }
            }

        }

    }
    //FIXME::snegi maybe store at the 1-idx port because the parent sees the reverse port compared to what child would see for the memory
    std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>> SteadyStateWalker::get_port_level_cost(ChildNodes child_nodes){

        std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>> retval; //internal map keys are the port indices
        if(std::holds_alternative<std::vector<const Node*>>(child_nodes)){
            //vector
            auto& child_vec = std::get<std::vector<const Node*>>(child_nodes);
            for(auto& child: child_vec){
                for(auto&[child_childnodes, node_cost] :parent_child_node_level_cost_map[child]){ //should only run once since child is a vector
                    for(auto&[arch_level, cost_vec]:node_cost){
                        for(auto idx=0; idx<2; idx++){
                            if(retval.find(arch_level)==retval.end()){
                                retval[arch_level][1-idx] = cost_vec[idx].cummulative_mem_net_cycles;
                            } else {
                                retval[arch_level][1-idx] += cost_vec[idx].cummulative_mem_net_cycles;
                            }
                            if(idx==0){
                                retval[arch_level][1-idx] +=cost_vec[idx].ramp_up;
                            } else{
                                retval[arch_level][1-idx] +=cost_vec[idx].ramp_down;
                            }
                            
                        }
                    }
                }

            }

        } else {
            // just a single node
            //this node can be temporal or spatial node
            auto& child = std::get<const Node*>(child_nodes);

            // auto loop_nodes = std::get<LoopNode>(analysis_.workload_mapping_graph[child]);

            // if(loop_nodes.type ==mapping_t::SPATIAL){ //will never happen since child of spatial is always a temporal node or a collective operation node
            //     COMET_ASSERT(false, "child of a spatial node cannot be spatial node");
            // } else {
            //TEMPORAL child

            if(child->get_type()==Node::CollectiveOperationNode){
                auto& node_cost = parent_child_node_level_cost_map[child][child];
                for(auto&[arch_level, cost_vec]:node_cost){
                    for(auto idx=0; idx<2; idx++){
                        if(retval.find(arch_level)==retval.end()){
                            retval[arch_level][1-idx] = cost_vec[idx].cummulative_mem_net_cycles;
                        } else {
                            retval[arch_level][1-idx] += cost_vec[idx].cummulative_mem_net_cycles;
                        }
                        if(idx==0){
                            retval[arch_level][1-idx] +=cost_vec[idx].ramp_up;
                        } else{
                            retval[arch_level][1-idx] +=cost_vec[idx].ramp_down;
                        }

                    }
                }
            } else {
                //else it is a datamovement node
                // this can have two cases 
                //one where the child is a temporal node of the Operation node
                auto child_child = child->get_children();
                // if(std::find(child_child.begin(), child_child.end(), Node::OperationNode)!=child_child.end()){
                if(child_child.size()==1 && child_child.back()->get_type()==Node::OperationNode){
                    //this is the temporal node of operation node
                    for(auto&[child_childnodes, node_cost] :parent_child_node_level_cost_map[child]){ //should only run once since child is a vector
                        for(auto&[arch_level, cost_vec]:node_cost){
                            for(auto idx=0; idx<2; idx++){
                                if(retval.find(arch_level)==retval.end()){
                                    retval[arch_level][1-idx] = cost_vec[idx].cummulative_mem_net_cycles;
                                } else {
                                    retval[arch_level][1-idx] += cost_vec[idx].cummulative_mem_net_cycles;
                                } 
                                if(idx==0){
                                    retval[arch_level][1-idx] +=cost_vec[idx].ramp_up;
                                } else{
                                    retval[arch_level][1-idx] +=cost_vec[idx].ramp_down;
                                }

                            }
                        }
                    }
                } else {
                    //other case is when the child is the temporal node with multiple or single spatial children
                    for(auto& c:child_child){
                        if(c->get_type()==Node::InterTileBinding) continue;
                        if(c->get_type()==Node::CollectiveOperationNode){
                            auto& node_cost = parent_child_node_level_cost_map[c][c];
                            for(auto&[arch_level, cost_vec]:node_cost){
                                for(auto idx=0; idx<2; idx++){
                                    if(retval.find(arch_level)==retval.end()){
                                        retval[arch_level][1-idx] = cost_vec[idx].cummulative_mem_net_cycles;
                                    } else {
                                        retval[arch_level][1-idx] += cost_vec[idx].cummulative_mem_net_cycles;
                                    }
                                    if(idx==0){
                                        retval[arch_level][1-idx] +=cost_vec[idx].ramp_up;
                                    } else{
                                        retval[arch_level][1-idx] +=cost_vec[idx].ramp_down;
                                    }

                                }
                            }
                        } else{
                            //data movement node which is spatial
                            for(auto&[child_childnodes, node_cost] :parent_child_node_level_cost_map[c]){ //should only run once since child is a vector
                                for(auto&[arch_level, cost_vec]:node_cost){
                                    for(auto idx=0; idx<2; idx++){
                                        if(retval.find(arch_level)==retval.end()){
                                            retval[arch_level][1-idx] = cost_vec[idx].cummulative_mem_net_cycles;
                                        } else {
                                            retval[arch_level][1-idx] += cost_vec[idx].cummulative_mem_net_cycles;
                                        }  
                                        if(idx==0){
                                            retval[arch_level][1-idx] +=cost_vec[idx].ramp_up;
                                        } else{
                                            retval[arch_level][1-idx] +=cost_vec[idx].ramp_down;
                                        }

                                    }
                                }
                            }
                        }
                    }

                }
            }




            // }
        }

        return retval;

    }





    //get cost between node1 and node2 in the form of memory at the node1 level
    NodeLevelCost SteadyStateWalker::get_node1_node2_cost(ArchLevelCost arch_level_cost, LoopNode& loop_nodes, std::map<arch::ArchLevel*, std::map<uint32_t, uint32_t>>& child_arch_level_port_usage){

        NodeLevelCost node_level_cost;

        //different window for every tensor because every tensor can have different number of iterations
        // different windows will be of different sizes, for instance for weight reuse the window might have 4 iterations but for activations it can have 8 iterations
        //making it a map of ports as well because 
        std::map<arch::ArchLevel*, std::vector<std::vector<uint32_t>>> mem_win_per_tensor; //Outer std::vector because from one archlevel there can be multiple child-memories that can be fed. std::vector inside for number of iterations
        // std::map<arch::ArchLevel*, std::map<uint32_t, std::vector<uint32_t>>> mem_win_per_tensor; // keys in internal map is the port index 
        std::map<arch::ArchLevel*, uint32_t> tensor_cnt;
        size_t tens_cnt=0;

        uint32_t max_iterations=0;
        // for(auto&tens_cnt: loop_nodes.tensors){ //only go over the tensors which are between parent and child FIXME::snegi
        for(auto& iterations: loop_nodes.iteration_count){
            max_iterations = std::max(max_iterations, iterations);
            // loop_nodes.iteration_count[tens_cnt]);
        }

        //code for removing the parent and child node conflicts. Current node might be writing data to the memory in child node but child node might also be using the same ports in that memory


        

        for(auto&[arch_pair, cost_vec]: arch_level_cost){
            for(auto idx=0; idx<2; idx++){
                for(auto& per_tensor_cost: cost_vec){
                    if(per_tensor_cost.size() > idx && per_tensor_cost[idx].stall.size()>0){ // add a place holder for both read and read-write tensors so that index are correct when accessed
                        // mem_win_per_tensor[arch_pair.first].push_back(per_tensor_cost[idx].compute_time_vec);
                        // mem_win_per_tensor[arch_pair.first][idx] = per_tensor_cost[idx].compute_time_vec;
                        auto& child_arch_level = arch_pair.second;
                        // Create vector of proper size first
                        std::vector<uint32_t> actual_vec;
                        actual_vec.resize(per_tensor_cost[idx].compute_time_vec.size());

                        if(!child_arch_level->isCompute() && !child_arch_level_port_usage.empty()){
                            auto level_spec = std::get<arch::MemorySpec>(child_arch_level->getSpec());
                            auto port_shared = level_spec.parent_child_ports_shared_.Get();
                            if(port_shared){
                                //remove the time the child uses the port from the window
                                uint32_t cost = 0;
                                if(child_arch_level_port_usage.count(arch_pair.second) > 0 && child_arch_level_port_usage[arch_pair.second].count(idx) > 0) {
                                    cost = child_arch_level_port_usage[arch_pair.second][idx];
                                }                                

                                // auto cost = child_arch_level_port_usage[arch_pair.second][idx];
                                // std::vector<uint32_t> actual_vec(per_tensor_cost[idx].compute_time_vec.size());

                                std::transform(per_tensor_cost[idx].compute_time_vec.begin(), per_tensor_cost[idx].compute_time_vec.end(), actual_vec.begin(), [cost](uint32_t element) { return (element>cost) ? (element-cost): 0;}); // window is always positive

                                // mem_win_per_tensor[arch_pair.first].emplace_back(actual_vec);
                                // actual_vec.push_back()
                            } else {
                                // mem_win_per_tensor[arch_pair.first].push_back(per_tensor_cost[idx].compute_time_vec);
                                actual_vec = per_tensor_cost[idx].compute_time_vec;
                            }
                        } else {
                            //no need to remove the child cost
                            // mem_win_per_tensor[arch_pair.first].push_back(per_tensor_cost[idx].compute_time_vec);
                            actual_vec = per_tensor_cost[idx].compute_time_vec;
                        }
                        if(mem_win_per_tensor.find(arch_pair.first) == mem_win_per_tensor.end()) {
                            mem_win_per_tensor[arch_pair.first] = {};
                        }
                        
                        mem_win_per_tensor[arch_pair.first].push_back(actual_vec); 
                    }
                }
            }
        }
        

        for(auto&[arch_pair, cost_vec]: arch_level_cost){

            if(node_level_cost.find(arch_pair.first)== node_level_cost.end()){ //if the key doesn't exist in the map initialize the cost vectors
                TargetChildCost temp_cost(max_iterations);
                TargetChildCostVec temp_cost_vec;
                temp_cost_vec.emplace_back(temp_cost);
                temp_cost_vec.emplace_back(temp_cost);

                node_level_cost[arch_pair.first] = temp_cost_vec;
                tensor_cnt[arch_pair.first] = 0;
            }
            // auto tens_idx_vector = loop_nodes.arch_idx[arch_pair.first];
            
            for(auto idx=0; idx<2; idx++){

                // size_t tens_cnt=0;

                for(auto& per_tensor_cost: cost_vec){ // per tensor cost is a TargetChildCostVec
                    tens_cnt=tensor_cnt[arch_pair.first]; //tensor count for that particular target memory
                    if(per_tensor_cost.size() > idx && per_tensor_cost[idx].stall.size()>0){
                        // auto tens_idx = tens_idx_vector[tens_cnt];

                        node_level_cost[arch_pair.first][idx].ramp_up      += per_tensor_cost[idx].ramp_up;
                        node_level_cost[arch_pair.first][idx].ramp_down    += per_tensor_cost[idx].ramp_down;
                        node_level_cost[arch_pair.first][idx].compute_time =  std::max(node_level_cost[arch_pair.first][idx].compute_time, per_tensor_cost[idx].compute_time);
                        if(node_level_cost[arch_pair.first][idx].reuse_factor==0) node_level_cost[arch_pair.first][idx].reuse_factor=1;
                        if(per_tensor_cost[idx].reuse_factor > 0) {
                            node_level_cost[arch_pair.first][idx].reuse_factor = std::min(node_level_cost[arch_pair.first][idx].reuse_factor, per_tensor_cost[idx].reuse_factor);
                        }
                        node_level_cost[arch_pair.first][idx].cummulative_mem_net_cycles += per_tensor_cost[idx].cummulative_mem_net_cycles;

                        uint32_t factor;

                        if(RTS_dependent && arch_pair.second->isCompute()){
                            // factor = loop_nodes.relative_timestep_to_dependent[tens_idx];//FIXME::snegi maybe change it to resue_factor*relative_timestep_dependent
                            auto tens_idx = per_tensor_cost[idx].tens_idx;
                            if(tens_idx < loop_nodes.relative_timestep_to_dependent.size()) {
                                factor = loop_nodes.relative_timestep_to_dependent[tens_idx];
                            } else {
                                factor = 1; // Default safe value
                            }                            
                            // factor = loop_nodes.relative_timestep_to_dependent[tens_idx];
                        } else{
                            factor = per_tensor_cost[idx].reuse_factor;
                        }
                        auto iterations = std::min((size_t)max_iterations, mem_win_per_tensor[arch_pair.first][tens_cnt].size());//FIXME::snegi check this should be correct

                        // for(auto i=idx; i<max_iterations; i++){ // this is always running for max_iterations because if GB->WB(which has reuse) comes first it should impact all the memory window of GB->IB. So if reuse is 2 and max_iteraiont=8. If WB comes first in the loop then stall won't be considered for i=1 cycle but it should take some time from the GB->IB memory window
                        for(auto i=idx; i<iterations; i++){ // this is always running for max_iterations because if GB->WB(which has reuse) comes first it should impact all the memory window of GB->IB. So if reuse is 2 and max_iteraiont=8. If WB comes first in the loop then stall won't be considered for i=1 cycle but it should take some time from the GB->IB memory window

                            for(auto j=0; j<factor; j++){
                                auto m = i*factor + j; // m is the index of bigger data structure and m<max_iterations
                                
                                
                                if(m<node_level_cost[arch_pair.first][idx].compute_time_vec.size() && i < per_tensor_cost[idx].compute_time_vec.size() && per_tensor_cost[idx].compute_time_vec[i]!=0){
                                    node_level_cost[arch_pair.first][idx].compute_time_vec[m] = std::max(node_level_cost[arch_pair.first][idx].compute_time_vec[m], per_tensor_cost[idx].compute_time_vec[i]/factor);
                                }
                                if(!per_tensor_cost[idx].hide_rw_latency && m<node_level_cost[arch_pair.first][idx].compulsory_stall.size() && i<per_tensor_cost[idx].compute_time_vec.size()){
                                    node_level_cost[arch_pair.first][idx].compulsory_stall[m] += per_tensor_cost[idx].stall[i]; //if the read-write latency can never be hidden, directly add the stall
                                }
                                // continue; // if this tensor always has a compulsory stall it will not utilize the mem_win_per_tensor
                                int stall = per_tensor_cost[idx].mem_net_cycles[i]/factor - mem_win_per_tensor[arch_pair.first][tens_cnt][i]/factor; //dividing the memory time and window time over reuse_factor number of cycles

                                if(stall>0 && m < node_level_cost[arch_pair.first][idx].stall.size()){
                                    node_level_cost[arch_pair.first][idx].stall[m] +=stall;
                                }

                            }

                            if(!per_tensor_cost[idx].hide_rw_latency) continue; //compulsory stall will not take time from the memory window

                            // since all tensors will see one compute window therefore remove the compute time that this tensor used to hide its data movement time from other tensors windows

                            std::map<arch::ArchLevel*, uint32_t> kth_tensor_cnt;
                            size_t kth_tens_cnt;
                            // if(kth_tensor_cnt.find(arch_pair.first)==kth_tensor_cnt.end()){ //moving inside because kth_tensor_cnt gets updated inside
                            //     kth_tensor_cnt[arch_pair.first] = 0;
                            // }
                            
                            for(auto&[arch_pair1, cost_vec1]: arch_level_cost){
                                if(kth_tensor_cnt.find(arch_pair1.first)==kth_tensor_cnt.end()){
                                    kth_tensor_cnt[arch_pair1.first] = 0;
                                }                                
                                for(auto idx1=0; idx1<2; idx1++){
                                    for(auto& per_tensor_cost1: cost_vec1){
                                        if(idx1 < per_tensor_cost1.size() && per_tensor_cost1[idx1].stall.size()>0){ 
                                            kth_tens_cnt = kth_tensor_cnt[arch_pair1.first];
                                            // if(kth_tens_cnt==tens_cnt) continue; // skip window update for the same tensor
                                            if(arch_pair.first == arch_pair1.first && idx==idx1 && kth_tens_cnt!=tens_cnt){ //if two tensors are coming from the same target and same port and different tensors then only they will share teh same memory window 
                                                auto kth_tens_idx = per_tensor_cost1[idx1].tens_idx;

                                                size_t kth_factor;
                                                if(RTS_dependent && arch_pair1.second->isCompute()){
                                                    kth_factor = loop_nodes.relative_timestep_to_dependent[kth_tens_idx];
                                                } else {
                                                    kth_factor = per_tensor_cost1[idx1].reuse_factor;
                                                }

                                                for(auto j=0; j<factor; j++){ // one iteration of 'i' will impact 'factor' iterations of other tensors
                                                    auto n = j + i*factor;
                                                    
                                                    n = n/kth_factor; // when slower iteration comes first(IB) and it updates the mem_win of other tensors(WB)
                                                    
                                                    if(n<mem_win_per_tensor[arch_pair1.first][kth_tens_cnt].size() && i<per_tensor_cost[idx].mem_net_cycles.size()){
                                                        int temp = mem_win_per_tensor[arch_pair1.first][kth_tens_cnt][n] - per_tensor_cost[idx].mem_net_cycles[i]/factor;
                                                        
                                                        mem_win_per_tensor[arch_pair1.first][kth_tens_cnt][n] = temp<0 ? 0 : temp;
                                                    }

                                                }
                                            }
                                            kth_tensor_cnt[arch_pair1.first]++;
                                        }
                                    }
                                }
                            }

                        }

                        // tens_cnt++;
                        tensor_cnt[arch_pair.first]++;
                    }

                    
                }


            }


        }

        return node_level_cost;


    }



    TargetChildCost SteadyStateWalker::get_node_cost(NodeLevelCost node_level_cost, LoopNode& loop_nodes, bool compute_child_node){

        //find max iterations
        uint32_t max_iterations=0;
        // for(auto&tens_cnt: loop_nodes.tensors){ //only go over the tensors which are between parent and child FIXME::snegi
        for(auto& iterations: loop_nodes.iteration_count){
            max_iterations = std::max(max_iterations, iterations);
            // loop_nodes.iteration_count[tens_cnt]);
        }

        TargetChildCost node_cost(max_iterations);


        auto arch_cnt=0;
        for(auto&[arch, cost]: node_level_cost){
            auto tens_idx = loop_nodes.arch_idx[arch];

            for(auto idx=0; idx<2; idx++){
                if(cost[idx].stall.size()>0 && cost[idx].compute_time!=0){
                    node_cost.ramp_up = std::max(node_cost.ramp_up, cost[idx].ramp_up);
                    node_cost.ramp_down = std::max(node_cost.ramp_down, cost[idx].ramp_down);
                    node_cost.compute_time = std::max(node_cost.compute_time, cost[idx].compute_time);

                    auto start = idx;     
                    uint32_t factor;

                    factor = cost[idx].reuse_factor;

                    // if(compute_child_node){
                    //     auto cnt=0;
                    //     for(auto tid: tens_idx){
                    //         if(cnt==0) {
                    //             factor = loop_nodes.relative_timestep_to_dependent[tid];     //FIXME::snegi maybe change it to resue_factor*relative_timestep_dependent
                    //         }
                    //         else {
                    //             factor = std::min(factor, loop_nodes.relative_timestep_to_dependent[tid]); // if multiple tensors are merged take minimum reuse_factor
                    //         }
                    //         cnt++;
                    //     }
                    // } else {
                    //     factor = cost[idx].reuse_factor;
                    // }

                    for(auto i=start; i<max_iterations; i++){
                        auto consider_stall = i%factor;
                        auto stall_idx = i/factor;

                        if(arch_cnt==0 && consider_stall==0){
                            node_cost.stall[i] = cost[idx].stall[stall_idx] + cost[idx].compulsory_stall[stall_idx];
                        } else if(arch_cnt!=0 && consider_stall==0){
                            node_cost.stall[i] = std::max(node_cost.stall[i], cost[idx].stall[stall_idx] + cost[idx].compulsory_stall[stall_idx]);
                        }

                    }
                    

                }
            }
            arch_cnt++;
        }

        return node_cost;
    }

    uint32_t SteadyStateWalker::get_total_node_cost(TargetChildCost node_cost){

        auto stalls = (uint32_t)std::accumulate(node_cost.stall.begin(), node_cost.stall.end(),static_cast<uint32_t>(0));
        uint32_t test_sum=0;
        size_t cnt=0;
        for(auto s: node_cost.stall){
            // if(s>0) std::cout<<"value greater than zero s="<< s<< " cnt= "<< cnt<<std::endl;
            test_sum +=s;
            cnt++;
        }
        uint32_t node_time = node_cost.compute_time + node_cost.ramp_up + node_cost.ramp_down + stalls;

        // auto node_time = std::max(node_cost.ramp_up+stalls+node_cost.compute_time, node_cost.ramp_down+stalls+node_cost.compute_time);
        // node_time = std::max(node_time, node_cost.compute_time);

        // auto node_time = std::max(node_cost.compute_time, node_cost.ramp_up + stalls);

        return node_time;
    }


    TargetChildCostVec SteadyStateWalker::get_target_child_cost(LevelID target_id, LevelID child_id, std::vector<DataMovementInfo> tensor_mov, LoopNode& loop_nodes, const mapping::operation_t& op_type, uint32_t child_cost, size_t tens_cnt, bool hide_rw_latency, bool run_single_iteration, std::vector<uint32_t>& memWindow, const DimSizeExpression& compute_tileprimitive, const computation_attributes& comp_attributes){
        // TargetChildCostVec retval;
        Time_t no_stall_time(2,0);
        TargetChildCostVec port_level_cost(2); // 0->read, 1->write

        DataMovementCostVec per_tensor_cost;
        // std::map<TensorID, PortLevelCost> per_tensor_cost;

        // TargetChildCost port_level_cost;
        
        size_t num_tiles;
        bool only_compute_cost=false; // for cases when the data is not moving from target memory to child memory only the compute cost of the child memory is the cost to the target memory
        uint8_t tensor_type = loop_nodes.tensor_is_rw[tens_cnt]? 1 : 0;

        if(op_type!=mapping::operation_t::None){
            auto comp_node_time = analysis_.topology_.getCompute(child_id)->getComputeLatency(compute_tileprimitive, comp_attributes, op_type);
            no_stall_time[tensor_type] = loop_nodes.relative_timestep_to_dependent[tens_cnt]*comp_node_time;
        } else {
            no_stall_time[tensor_type] = child_cost;
        }

        bool different_iterations = true;
        std::set<uint32_t> iter_set;
        for(auto itr: loop_nodes.iteration_count){
            iter_set.insert(itr);
        }
        if(iter_set.size()==1) different_iterations=false; // this was used bcz reuse for leafs were pushed to relative timestep to dependent and for otherr nodes they were in reuse_factor. But rn everything is in reuse_factor 


        auto actual_iterations = get_iterations(tensor_mov);
        auto reuse_factor      = loop_nodes.iteration_count[tens_cnt]/actual_iterations;

        if(reuse_factor==0) reuse_factor=1;

        // auto temp_vector = get_no_stall_time_vector(tensor_mov, reuse_factor, no_stall_time[tensor_type]);

        std::vector<uint32_t> no_stall_vector(memWindow.size(), 0);
        // if(op_type!=mapping::operation_t::None){
        //     std::fill(no_stall_vector.begin(), no_stall_vector.end(), no_stall_time[tensor_type]); 
        // }
        
        if(reuse_factor==1){
            no_stall_vector = memWindow;
        }else{
            //create no_stall_vector from memWindow
            for(int i=0; i<memWindow.size(); i++){
                if(i%reuse_factor==0){
                    no_stall_vector[i] = std::accumulate(memWindow.begin()+i, memWindow.begin()+i+reuse_factor,0);
                }
            }
        }

        // port_level_cost[tensor_type].compute_time_vec = no_stall_vector;
        port_level_cost[tensor_type].compute_time_vec.assign(actual_iterations, no_stall_time[tensor_type]*reuse_factor);

        // std::fill(port_level_cost[tensor_type].compute_time_vec.begin(), port_level_cost[tensor_type].compute_time_vec.begin()+actual_iterations, no_stall_time[tensor_type]);
        // if(different_iterations){
        //     port_level_cost[tensor_type].compute_time_vec = no_stall_vector;
        // } else {

        // }

        port_level_cost[tensor_type].reuse_factor = reuse_factor;
        port_level_cost[tensor_type].tens_idx = tens_cnt;
        port_level_cost[tensor_type].compute_time = (uint32_t)std::accumulate(no_stall_vector.begin(), no_stall_vector.end(),0);
        port_level_cost[tensor_type].hide_rw_latency = hide_rw_latency;

        if(!only_compute_cost){

            per_tensor_cost = std::move(analysis_.topology_.Evaluate(target_id, child_id, tensor_mov, no_stall_vector, loop_nodes, tens_cnt, hide_rw_latency, run_single_iteration, reuse_factor, different_iterations));

            port_level_cost[tensor_type].cummulative_mem_net_cycles = per_tensor_cost.cummulative_mem_net_cycles;
            port_level_cost[tensor_type].stall   = per_tensor_cost.stall;
            port_level_cost[tensor_type].mem_net_cycles = per_tensor_cost.mem_net_cycles;
            port_level_cost[tensor_type].ramp_up = per_tensor_cost.ramp_up;
            port_level_cost[tensor_type].ramp_down = per_tensor_cost.ramp_down;
            port_level_cost[tensor_type].tile_size = per_tensor_cost.tile_size;
            port_level_cost[tensor_type].access_count = per_tensor_cost.access_count;
            port_level_cost[tensor_type].noc_energy = per_tensor_cost.noc_energy; 
        }
        return port_level_cost;
    }


}