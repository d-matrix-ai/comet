#pragma once

#include "analysis/nest_analysis.hpp"
#include "analysis/cost_walker.hpp" //commented to test data movement only
#include <regex>
#include <iomanip>

namespace analysis{


    void loop_nest_frm_tensor_size(TensorSizeMap& tensor_size, std::map<problem::TensorID, LoopNestDescriptor>& desc){
        for(auto&[tid, spatial_values]: tensor_size){
            int sp_cnt=0;
            for(auto&sp_val: spatial_values){
                desc[tid].emplace_back();
                for(auto&[dim_id, dim_val]: sp_val){
                    // lnd.emplace_ba
                    desc[tid][sp_cnt][dim_id] ={.end= dim_val, .stride=dim_val};
                }
                sp_cnt++;
            }   
        }
    }

    void printArchLevelEnergy(const Node* node, std::map<const Node*, ArchLevelEnergy> node_energy, std::map<const Node*, NOCEnergy> noc_energy, const std::string& prefix, bool isLast, double& total_energy, bool calc_noc_energy, std::map<std::string, float>& energy_breakdown, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph ){
        // , std::unordered_map<const Node*, NodeTypes> workload_mapping_graph, const std::string& prefix, bool isLast){ 
      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      // Print node information (type and name)
      std::ostringstream oss;
      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");



      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if(node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);
        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;
      }
      
      else {
        std::cout << node->get_name() << " = ";
        const std::vector<const Node*>& children = node->get_children();
        oss << ": {";
        for(auto& [arch_level, energy]: node_energy[node]){
            oss << " "<< arch_level->getName() << ": " << energy;

            

            if(node->get_children().size()!=0 && node->get_children().front()->get_type() == Node::type_t::OperationNode && arch_level->isCompute()){ //w/o !=0 condition was breaking at colop node
                auto actual_node = std::get<OpNode>(workload_mapping_graph[node->get_children().front()]);
                auto key = arch_level->getName() + "-" + actual_node.op_name;
                energy_breakdown[key] +=energy;
            } else {
                energy_breakdown[arch_level->getName()] +=energy;
            }

            total_energy += energy;
        }

        if(calc_noc_energy && noc_energy.find(node)!=noc_energy.end()){
            oss << " |";
            for(auto&[arch_level, energy]:noc_energy[node]){
                oss<< " "<< arch_level << ": " <<energy;
                if(node->get_type()==Node::type_t::CollectiveOperationNode){
                    auto actual_node = std::get<ColOpNode>(workload_mapping_graph[node]);
                    auto key = arch_level + "-" + stype_to_string(actual_node.type_);    
                    energy_breakdown[key] +=energy;
                }else{
                    energy_breakdown[arch_level] +=energy;
                }

                total_energy +=energy;
            }
        }

        std::cout << oss.str() << std::endl; 
        for(auto i=0; i<children.size(); i++){
            printArchLevelEnergy(children[i], node_energy, noc_energy,childPrefix, i == children.size() - 1, total_energy, calc_noc_energy, energy_breakdown, workload_mapping_graph);
        }


      }

    }

    void printAccessCount(const Node* node, std::map<const Node*, std::vector<access_count_struct>> access_count_map, const std::string& prefix, bool isLast, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph){

      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      // Print node information (type and name)
      std::ostringstream oss;
      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");



      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if(node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);
        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;
      } 
      else {
        std::cout << node->get_name() << " = ";
        const std::vector<const Node*>& children = node->get_children();

        for(auto& access_struct: access_count_map[node]){
            // oss << " "<< arch_level->getName() << ": " << energy;
            auto port = access_struct.tensor_is_rw ? 1 : 0; // read-only tensor at port 0
            auto arch_level = access_struct.arch_level->getName();
            oss << "{ " << arch_level << ":[ " << port << ": " << access_struct.access_count<<" ] }";
        }
        std::cout << oss.str() << std::endl; 
        for(auto i=0; i<children.size(); i++){
            printAccessCount(children[i], access_count_map, childPrefix, i == children.size() - 1, workload_mapping_graph);
        }


      }


    }

    void printParentAccessCount(const Node* node, std::unordered_map<const Node*, uint64_t> access_count_map, const std::string& prefix, bool isLast, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph){

      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      // Print node information (type and name)
      std::ostringstream oss;
      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");



      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if(node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);
        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;
      }
      else {
        std::cout << node->get_name() << " = ";
        const std::vector<const Node*>& children = node->get_children();

        if(access_count_map.find(node)!=access_count_map.end()){
            oss << access_count_map[node];
        }
        
        // for(auto& access_struct: access_count_map[node]){
        //     // oss << " "<< arch_level->getName() << ": " << energy;
        //     auto port = access_struct.tensor_is_rw ? 1 : 0; // read-only tensor at port 0
        //     auto arch_level = access_struct.arch_level->getName();
        //     oss << "{ " << arch_level << ":[ " << port << ": " << access_struct.access_count<<" ] }";
        // }
        std::cout << oss.str() << std::endl; 
        for(auto i=0; i<children.size(); i++){
            printParentAccessCount(children[i], access_count_map, childPrefix, i == children.size() - 1, workload_mapping_graph);
        }


      }


    }

    void PrintArchLevelNodeCost(const Node* node, std::map<const mapping::Node *, std::map<ChildNodes, analysis::NodeLevelCost>> cost, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph, const std::string& prefix, bool isLast){

      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      std::ostringstream oss;

      if(node->get_type() !=Node::type_t::InterTileBinding && node->get_type() !=Node::type_t::OperationNode){
        std::vector<const Node*> child_vector;
        for(auto&c:node->get_children()){
            if(node->get_type()==Node::type_t::InterTileBinding || node->get_type()==Node::type_t::OperationNode) continue;
            child_vector.push_back(c);
        }

        for(auto&[child_node, node_level_cost]:cost[node]){
            if(std::holds_alternative<std::vector<const Node*>>(child_node)){
                auto child_node_vec = std::get<std::vector<const Node*>>(child_node);

                for(auto& cnv: child_node_vec){
                    if(std::find(child_vector.begin(), child_vector.end(), cnv)!=child_vector.end()){
                        auto node_name = cnv->get_name();

                        std::regex number_regex(R"(.*::(\d+)$)");
                        std::smatch match;
                        int number;

                        if (std::regex_match(node_name, match, number_regex)) {
                            // Extract the matched number
                            number = std::stoi(match[1].str());
                        } else {
                            std::cout << "No number found at the end of the string." << std::endl;
                        }

                        oss << number << ",";

                    }
                }
            } else {
                auto child_node_vec = std::get<const Node*>(child_node);

                auto node_name = child_node_vec->get_name();

                std::regex number_regex(R"(.*::(\d+)$)");
                std::smatch match;
                int number;

                if (std::regex_match(node_name, match, number_regex)) {
                    // Extract the matched number
                    number = std::stoi(match[1].str());
                } else {
                    std::cout << "No number found at the end of the string." << std::endl;
                }

                oss << number << ": {";
            }

            for(auto&[arch_level, cost_vec]:node_level_cost){
                std::vector<uint32_t> port_level_mem_time;
                oss<<" "<<arch_level->getName() << ":";
                for(auto idx=0; idx<2; idx++){
                    auto mem_time = cost_vec[idx].cummulative_mem_net_cycles + cost_vec[idx].ramp_up + cost_vec[idx].ramp_down;
                    port_level_mem_time.push_back(mem_time);  
                }
                oss<< "[" << "0: " <<port_level_mem_time[0] << "," << " 1: " <<port_level_mem_time[1] << " MW: " <<std::max(cost_vec[0].compute_time, cost_vec[1].compute_time) << "]";
            }
            oss << " }";
        }
      }
      
      
      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if (node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);

        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;

      }
      else {
        std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
      }

      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");

      // Recursively print each child node
      const std::vector<const Node*>& children = node->get_children();
      for (size_t i = 0; i < children.size(); ++i) {
          PrintArchLevelNodeCost(children[i], cost, workload_mapping_graph, childPrefix, i == children.size() - 1);
      }


    }



    void PrintCycleBreakDown(const Node* node, std::map<const mapping::Node *, std::map<ChildNodes, analysis::NodeLevelCost>> cost, std::unordered_map<const Node*, uint64_t> parent_iteration_count, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph, const std::string& prefix, bool isLast){

      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      std::ostringstream oss;

      if(node->get_type() !=Node::type_t::InterTileBinding && node->get_type() !=Node::type_t::OperationNode){
        std::vector<const Node*> child_vector;
        for(auto&c:node->get_children()){
            if(node->get_type()==Node::type_t::InterTileBinding || node->get_type()==Node::type_t::OperationNode) continue;
            child_vector.push_back(c);
        }

        for(auto&[child_node, node_level_cost]:cost[node]){
            if(std::holds_alternative<std::vector<const Node*>>(child_node)){
                auto child_node_vec = std::get<std::vector<const Node*>>(child_node);

                for(auto& cnv: child_node_vec){
                    if(std::find(child_vector.begin(), child_vector.end(), cnv)!=child_vector.end()){
                        auto node_name = cnv->get_name();

                        std::regex number_regex(R"(.*::(\d+)$)");
                        std::smatch match;
                        int number;

                        if (std::regex_match(node_name, match, number_regex)) {
                            // Extract the matched number
                            number = std::stoi(match[1].str());
                        } else {
                            std::cout << "No number found at the end of the string." << std::endl;
                        }

                        oss << number << ",";

                    }
                }
            } else {
                auto child_node_vec = std::get<const Node*>(child_node);

                auto node_name = child_node_vec->get_name();

                std::regex number_regex(R"(.*::(\d+)$)");
                std::smatch match;
                int number;

                if (std::regex_match(node_name, match, number_regex)) {
                    // Extract the matched number
                    number = std::stoi(match[1].str());
                } else {
                    std::cout << "No number found at the end of the string." << std::endl;
                }

                oss << number << ": {";
            }
            uint64_t compute_time=0;
            for(auto&[arch_level, cost_vec]:node_level_cost){
                std::vector<uint32_t> port_level_mem_time;
                std::vector<uint32_t> port_level_CS(2,0); //cumpulsory stall
                std::vector<int32_t> port_level_OS(2,0); //optional stall

                oss<<" "<<arch_level->getName() << ":";
                auto mw = std::max(cost_vec[0].compute_time, cost_vec[1].compute_time); //memory window
                uint32_t iterations=1;
                if(parent_iteration_count.find(node)!=parent_iteration_count.end()){
                    iterations= parent_iteration_count[node];
                }                
                for(auto idx=0; idx<2; idx++){
                    auto mem_time = cost_vec[idx].cummulative_mem_net_cycles + cost_vec[idx].ramp_up + cost_vec[idx].ramp_down;
                    port_level_mem_time.push_back(mem_time);  
                    port_level_CS[idx] += cost_vec[idx].ramp_up + cost_vec[idx].ramp_down;
                    port_level_OS[idx] += cost_vec[idx].cummulative_mem_net_cycles - mw;//FIXME::snegi incorrect way to calculate the Optional stall, because optional stall is only hidden by iteration-1 cycles of compute 

                    if(port_level_OS[idx]<0) port_level_OS[idx]=0;
                    // if(node->get_type()==Node::type_t::CollectiveOperationNode){
                    //    total_cycles += port_level_OS[idx]*iterations;
                    // } else{
                    //    total_cycles += port_level_CS[idx]*iterations;
                    // }
                }
                // if(node->get_type()!=Node::type_t::CollectiveOperationNode) total_cycles += std::max(port_level_OS[0], port_level_OS[1])*iterations;//optional stalls of read and write ports happen in parallel, already added this for collective op above


                oss<< "[" << "0: " << "(CS: "<< port_level_CS[0]*iterations <<" OS: " << port_level_OS[0]*iterations <<")" << "," << " 1: " << "(CS: "<< port_level_CS[1]*iterations <<" OS: " << port_level_OS[1]*iterations <<")" << "," << " TMW: " << mw*iterations << "]";
                
                compute_time = std::max(compute_time, mw*iterations);
            }
            // if(node->get_children().size()==1 && node->get_children().front()->get_type() == Node::type_t::OperationNode){
            //     total_cycles += compute_time;
            // }

            oss << " }";
        }
      }
      
      
      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if (node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);

        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;

      }
      else {//datamovement or colop node
        // if it is a temporal node and the children is not a compute node do not print any cost
        if(node->get_type()==Node::type_t::DataMovementTileNode){
            auto loop_node = std::get<LoopNode>(workload_mapping_graph[node]);
            auto loop_node_type = loop_node.type;
            if((loop_node_type == mapping_t::TEMPORAL && node->get_children().front()->get_type() == Node::type_t::OperationNode)|| loop_node_type==mapping_t::SPATIAL){
                //print output only if it is a temporal node of the compute node or spatial node
                std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
            } else {
                std::cout << node->get_name()<< " = "<<std::endl;
            }
        }else {
            std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
        }
      }

      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");

      // Recursively print each child node
      const std::vector<const Node*>& children = node->get_children();
      for (size_t i = 0; i < children.size(); ++i) {
          PrintCycleBreakDown(children[i], cost, parent_iteration_count, workload_mapping_graph, childPrefix, i == children.size() - 1);
      }


    }

    void PrintPreciseCycleBreakDown(const Node* node, std::map<const Node*, uint32_t> node_cost, std::map<const mapping::Node *, std::map<ChildNodes, analysis::NodeLevelCost>> cost, std::unordered_map<const Node*, uint64_t> parent_iteration_count, std::unordered_map<const Node*, NodeTypes> workload_mapping_graph, const std::string& prefix, bool isLast, uint64_t& total_cycles, std::map<std::string, std::map<std::string, uint64_t>>& cycle_breakdown){

      if (!node) return;

      // Print the current node with proper indentation and lines
      std::cout << prefix;

      // Print the current node connection
      if (isLast) {
          std::cout << "└─ ";
      } else {
          std::cout << "├─ ";
      }

      std::ostringstream oss;

      if(node->get_type() !=Node::type_t::InterTileBinding && node->get_type() !=Node::type_t::OperationNode){
        std::vector<const Node*> child_vector;
        for(auto&c:node->get_children()){
            if(node->get_type()==Node::type_t::InterTileBinding || node->get_type()==Node::type_t::OperationNode) continue;
            child_vector.push_back(c);
        }

        for(auto&[child_node, node_level_cost]:cost[node]){
            if(std::holds_alternative<std::vector<const Node*>>(child_node)){
                auto child_node_vec = std::get<std::vector<const Node*>>(child_node);

                for(auto& cnv: child_node_vec){
                    if(std::find(child_vector.begin(), child_vector.end(), cnv)!=child_vector.end()){
                        auto node_name = cnv->get_name();

                        std::regex number_regex(R"(.*::(\d+)$)");
                        std::smatch match;
                        int number;

                        if (std::regex_match(node_name, match, number_regex)) {
                            // Extract the matched number
                            number = std::stoi(match[1].str());
                        } else {
                            std::cout << "No number found at the end of the string." << std::endl;
                        }

                        oss << number << ",";

                    }
                }
            } else {
                auto child_node_vec = std::get<const Node*>(child_node);

                auto node_name = child_node_vec->get_name();

                std::regex number_regex(R"(.*::(\d+)$)");
                std::smatch match;
                int number;

                if (std::regex_match(node_name, match, number_regex)) {
                    // Extract the matched number
                    number = std::stoi(match[1].str());
                } else {
                    std::cout << "No number found at the end of the string." << std::endl;
                }

                oss << number << ": {";
            }
            uint64_t compute_time=0;
            std::vector<uint64_t> port_level_CS(2,0); //cumpulsory stall
            uint32_t iterations=1;
            int64_t optional_stall = 0;
            uint64_t compulsory_stall = 0;            
            if(parent_iteration_count.find(node)!=parent_iteration_count.end()){
                iterations= parent_iteration_count[node];
            }  
            std::string map_key;
            for(auto&[arch_level, cost_vec]:node_level_cost){
                
                // std::vector<int32_t> port_level_OS(2,0); //optional stall

                oss<<" "<<arch_level->getName() << ", ";
                map_key = map_key + arch_level->getName() + '-';
                auto mw = std::max(cost_vec[0].compute_time, cost_vec[1].compute_time); //memory window
              
                for(auto idx=0; idx<2; idx++){
                    auto mem_time = cost_vec[idx].cummulative_mem_net_cycles + cost_vec[idx].ramp_up + cost_vec[idx].ramp_down;
                    port_level_CS[idx] = std::max(port_level_CS[idx], cost_vec[idx].ramp_up + cost_vec[idx].ramp_down);
                    optional_stall = std::max(optional_stall, (int64_t)cost_vec[idx].cummulative_mem_net_cycles);
                    // port_level_OS[idx] += cost_vec[idx].cummulative_mem_net_cycles - mw;//incorrect way to calculate the Optional stall, because optional stall is only hidden by iteration-1 cycles of compute 
                }
                compute_time = std::max(compute_time, mw);
            }

            compulsory_stall = port_level_CS[0] + port_level_CS[1];
            

            if(node->get_type()!=Node::type_t::CollectiveOperationNode){
                optional_stall = node_cost[node] - port_level_CS[0]-port_level_CS[1] - compute_time;
                if (optional_stall<0) optional_stall=0;
            }
            


            if(node->get_type()==Node::type_t::DataMovementTileNode){
                auto loop_node = std::get<LoopNode>(workload_mapping_graph[node]);
                auto loop_node_type = loop_node.type;
                if((loop_node_type == mapping_t::TEMPORAL && node->get_children().front()->get_type() == Node::type_t::OperationNode)|| loop_node_type==mapping_t::SPATIAL){

                    oss << "[" << "CS: " << compulsory_stall*iterations<< " OS: " << optional_stall*iterations << " TMW: " << compute_time*iterations << "]";

                    total_cycles += (compulsory_stall + optional_stall)*iterations;
                    
                    cycle_breakdown[map_key]["CS"] +=compulsory_stall*iterations;
                    cycle_breakdown[map_key]["OS"] +=optional_stall;

                    if(node->get_children().size()==1 && node->get_children().front()->get_type() == Node::type_t::OperationNode){
                        total_cycles += compute_time*iterations;
                        auto actual_node = std::get<OpNode>(workload_mapping_graph[node->get_children().front()]);
                        cycle_breakdown[actual_node.op_name]["TMW"] = compute_time*iterations;
                    }

                    oss << " }";

                }
            } else if (node->get_type()==Node::type_t::CollectiveOperationNode){
                    oss << "[" << "CS: " << compulsory_stall*iterations<< " OS: " << optional_stall*iterations << " TMW: " << compute_time*iterations << "]";

                    total_cycles += (compulsory_stall + optional_stall)*iterations;


                    if(node->get_children().size()==1 && node->get_children().front()->get_type() == Node::type_t::OperationNode){
                        total_cycles += compute_time*iterations;
                    }
                    auto actual_node = std::get<ColOpNode>(workload_mapping_graph[node]);

                    auto node_name = node->get_name();

                    std::regex number_regex(R"(.*::(\d+)$)");
                    std::smatch match;
                    int number;

                    if (std::regex_match(node_name, match, number_regex)) {
                        // Extract the matched number
                        number = std::stoi(match[1].str());
                    } else {
                        std::cout << "No number found at the end of the string." << std::endl;
                    }                    
                    auto key = stype_to_string(actual_node.type_) + "-" + std::to_string(number);

                    cycle_breakdown[key]["OS"] += (compulsory_stall + optional_stall)*iterations;

                    oss << " }";
            }


        }
      }
      
      
      if(node->get_type()==Node::type_t::InterTileBinding){
        std::cout << node->get_name() << std::endl;
      } else if (node->get_type()==Node::type_t::OperationNode){
        auto actual_node = std::get<OpNode>(workload_mapping_graph[node]);

        std::cout << node->get_name() << "-"<< actual_node.op_name <<std::endl;

      }
      else {//datamovement or colop node
        // if it is a temporal node and the children is not a compute node do not print any cost
        if(node->get_type()==Node::type_t::DataMovementTileNode){
            auto loop_node = std::get<LoopNode>(workload_mapping_graph[node]);
            auto loop_node_type = loop_node.type;
            if((loop_node_type == mapping_t::TEMPORAL && node->get_children().front()->get_type() == Node::type_t::OperationNode)|| loop_node_type==mapping_t::SPATIAL){
                //print output only if it is a temporal node of the compute node or spatial node
                std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
            } else {
                std::cout << node->get_name()<< " = "<<std::endl;
            }
        }else {
            std::cout << node->get_name() << " = "<< oss.str()<<std::endl;
        }
      }

      // Prepare the prefix for the children
      std::string childPrefix = prefix + (isLast ? "   " : "│  ");

      // Recursively print each child node
      const std::vector<const Node*>& children = node->get_children();
      for (size_t i = 0; i < children.size(); ++i) {
          PrintPreciseCycleBreakDown(children[i], node_cost, cost, parent_iteration_count, workload_mapping_graph, childPrefix, i == children.size() - 1, total_cycles, cycle_breakdown);
      }


    }


    
    // NestAnalysis::NestAnalysis(problem::Workloads& workloads, const mapping::Mapping& mapping, const arch::Topology& topology): workloads_(workloads), mapping_(mapping), topology_(topology){}

    void NestAnalysis::get_loopnest(){
        LoopNestConstructor(*this).construct(mapping_.root);
    }


    void NestAnalysis::get_datamovement_info(){
        DataMovementInfoConstructor(*this).construct(mapping_.root);
    }

    // std::map<const Node*, TotalCost> NestAnalysis::get_cost(){ //commented to test data movement only
    // std::map<const Node*, uint32_t> NestAnalysis::get_cost(){ 
    cost_struct NestAnalysis::get_cost(){ 

    // void NestAnalysis::get_cost(){   
        CostWalker cw(*this);
        //// ramp latency
        //cw.calculateRampLatency();

        //// get ideal computation time
        //cw.calculateIdealComputationTime();

        // get computation time --> steady state
        auto cost = cw.calculateSteadyStateTime();
        // cw.calculateSteadyStateTime();

        // return node_level_cost;
        return cost;
    }

    using ComputeTimeMap = std::variant<std::map<std::string, uint64_t>, std::map<std::string, float>>;

    void printMap(const std::map<std::string, uint64_t>& m, size_t nameWidth) {
        uint64_t total = 0;
        for (const auto& entry : m) {
            std::cout << std::left << std::setw(nameWidth + 2) << entry.first 
                      << std::right << std::setw(15) << entry.second << std::endl;
            total += entry.second;
        }
        std::cout << std::left << std::setw(nameWidth + 2) << "Total"
                << std::right << std::setw(15) << total << std::endl;        
    }
    
    void printMap(const std::map<std::string, float>& m, size_t nameWidth) {
        float total = 0.0f;
        for (const auto& entry : m) {
            std::cout << std::left << std::setw(nameWidth + 2) << entry.first 
                      << std::right << std::setw(15) << std::fixed << std::setprecision(2) << entry.second << std::endl;
            total += entry.second;
        }
        std::cout << std::left << std::setw(nameWidth + 2) << "Total"
                << std::right << std::setw(15) << std::fixed << std::setprecision(2) << total << std::endl;

    }
    
    void printComputeTimeMap(const ComputeTimeMap& comp_time_map, const std::string& title) {
        size_t max_name_length = title.length(); 
    
        std::visit([&](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
    
            // Calculate the width of the first column
            for (const auto& entry : arg) {
                if (entry.first.length() > max_name_length) {
                    max_name_length = entry.first.length();
                }
            }
    
            // Print header
            std::cout << std::left << std::setw(max_name_length + 2) << title 
                      << std::right << std::setw(15) << "Value" << std::endl;
            std::cout << std::string(max_name_length + 17, '-') << std::endl;
    
            // Print the map
            if constexpr (std::is_same_v<T, std::map<std::string, uint64_t>>)
                printMap(arg, max_name_length);
            else if constexpr (std::is_same_v<T, std::map<std::string, float>>)
                printMap(arg, max_name_length);
            
        }, comp_time_map);
    }

    uint64_t printCycleBreakdown(const std::map<std::string, std::map<std::string, uint64_t>>& cycle_breakdown) {
        // Collect all unique column labels
        std::set<std::string> all_columns;
        for (const auto& row : cycle_breakdown) {
            for (const auto& col : row.second) {
                all_columns.insert(col.first);
            }
        }

        // Compute width for row labels
        size_t row_label_width = 4;
        for (const auto& [row_key, _] : cycle_breakdown) {
            row_label_width = std::max(row_label_width, row_key.length());
        }

        // Compute column widths
        std::map<std::string, size_t> col_widths;
        for (const auto& col : all_columns) {
            size_t max_width = col.length();
            for (const auto& row : cycle_breakdown) {
                auto it = row.second.find(col);
                if (it != row.second.end()) {
                    max_width = std::max(max_width, std::to_string(it->second).length());
                }
            }
            col_widths[col] = max_width;
        }

        // Determine width of the "Total" column
        std::string total_col_name = "Total";
        size_t total_col_width = total_col_name.length();
        for (const auto& [_, col_map] : cycle_breakdown) {
            uint64_t row_total = 0;
            for (const auto& [_, val] : col_map) {
                row_total += val;
            }
            total_col_width = std::max(total_col_width, std::to_string(row_total).length());
        }

        // Print header
        std::cout << std::left << std::setw(row_label_width + 2) << "Unit";
        for (const auto& col : all_columns) {
            std::cout << std::right << std::setw(col_widths[col] + 2) << col;
        }
        std::cout << std::right << std::setw(total_col_width + 2) << total_col_name << std::endl;

        // Print separator
        size_t total_width = row_label_width + 2 + total_col_width + 2;
        for (const auto& col : all_columns) {
            total_width += col_widths[col] + 2;
        }
        std::cout << std::string(total_width, '-') << std::endl;

        // Print rows and accumulate grand total
        uint64_t grand_total = 0;
        for (const auto& [row_key, col_map] : cycle_breakdown) {
            std::cout << std::left << std::setw(row_label_width + 2) << row_key;
            uint64_t row_total = 0;

            for (const auto& col : all_columns) {
                auto it = col_map.find(col);
                if (it != col_map.end()) {
                    std::cout << std::right << std::setw(col_widths[col] + 2) << it->second;
                    row_total += it->second;
                } else {
                    std::cout << std::right << std::setw(col_widths[col] + 2) << "-";
                }
            }

            std::cout << std::right << std::setw(total_col_width + 2) << row_total << std::endl;
            grand_total += row_total;
        }

        return grand_total;
    }

    void NestAnalysis::analyze(){
        std::cout<<"Begining the Loop Nest creation"<<std::endl;
        get_loopnest();
        std::cout<<"Completed Loop Nest creation"<<std::endl;
        
        std::cout<<"Begining DataMovement Info construction"<<std::endl;
        get_datamovement_info();
        std::cout<<"Completed DataMovement Info construction"<<std::endl;

        std::cout<<"Begining the Steady State cost calculation"<<std::endl;
        auto cost = get_cost();  //commented to test data movement only
        // get_cost();
        std::cout<<"Completed the Steady State cost calculation"<<std::endl;
        
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Cycle Hierarchy Overview *********"<<std::endl;
        std::cout<<"Notation --> └─ => last child"<< "    ├─ => node with further siblings" <<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;

        mapping_.root->printTree(mapping_.root, cost.node_cost);

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Cycles by Architecture Level *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        PrintArchLevelNodeCost(mapping_.root, cost.arch_level_node_cost, workload_mapping_graph,"",true);

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Cycle Breakdown by Archcitecture Level *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        PrintCycleBreakDown(mapping_.root, cost.arch_level_node_cost, cost.parent_iteration_count, workload_mapping_graph,"",true);

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Detailed Cycles Breakdown *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        uint64_t total_cycles=0;

        std::map<std::string, std::map<std::string, uint64_t>> cycle_breakdown; //first key is arch_name or collective op name or TMW

        PrintPreciseCycleBreakDown(mapping_.root, cost.node_cost, cost.arch_level_node_cost, cost.parent_iteration_count, workload_mapping_graph,"",true, total_cycles, cycle_breakdown);
        std::cout<<"********* Total Cycles = "<< total_cycles <<std::endl;

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Detailed Cycles Breakdown (Table) *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        auto check_total = printCycleBreakdown(cycle_breakdown);
        std::cout<<"  ➤ Total Cycles = "<< total_cycles <<std::endl;

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Ideal Compute Cycles *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        printComputeTimeMap(cost.comp_time_map, "Workload Name");
        // printMap(cost.comp_time_map, "Workload Name");

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Compute Unit Utilization *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        printComputeTimeMap(cost.compute_utilization_map, "Compute Unit");
        // printMap(cost.compute_utilization_map, "Compute Unit");

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* ArchLevel Energy (pJ) *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        double total_energy=0;
        std::map<std::string, float> energy_breakdown; //string is arch level name
        printArchLevelEnergy(mapping_.root, cost.arch_level_node_energy,cost.node_level_noc_energy,"",true, total_energy, calc_noc_energy_, energy_breakdown, workload_mapping_graph);
        
        std::cout<<"   ➤ Total Energy (pJ) = "<< std::scientific << std::setprecision(6)<<total_energy <<std::endl;

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Energy Breakdown by Arch Level (pJ) *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        printComputeTimeMap(energy_breakdown, "ArchLevel");

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Access Count per Spatial Instance *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        printAccessCount(mapping_.root, cost.access_count, "", true, workload_mapping_graph);

        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout<<"********* Access Count per Parent Memory *********"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        printParentAccessCount(mapping_.root, cost.parent_iteration_count, "", true, workload_mapping_graph);


        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;
        std::cout << "✅ Simulation Report Completed Successfully"<<std::endl;
        std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"<<std::endl;        
    }


    // ********************LoopNestConstructor related functions***************************

    // void visitDataMovementTileNode(const DataMovementTileNode* node) override {
    void LoopNestConstructor::visitDataMovementTileNode(const DataMovementTileNode* node) {
        std::vector<problem::TensorID> tensors_updated;
        for (auto child: node->get_children()){
            child->accept(this);
        }
        auto& stride_tilesteps = stride_tilesteps_;

        // analysis_.workload_mapping_graph[node].emplace_back();
        // NodeTypes temp=node->constructLoopNest(stride_tilesteps);
        // temp = static_cast<NodeTypes>temp;

        //  analysis_.workload_mapping_graph[node] = temp;

        // if(node->get_children().front()->get_type()==Node::type_t::OperationNode){
            // auto& comp_node_child = static_cast<const OperationNode*>(node->get_children().front())->get_common_attributes().type_;
        // const auto& comp_node_child = static_cast<const OperationNode*>(node->get_children().front());
        // if(comp_node_child->get_type()==Node::type_t::OperationNode&&comp_node_child->get_common_attributes().type_==mapping::operation_t::SOFTMAX){
        //     auto& tensor_size = std::get<OpNode>(analysis_.workload_mapping_graph[comp_node_child]).tensor_sizes; 
        //     analysis_.workload_mapping_graph[node] = node->constructLoopNest(stride_tilesteps, tensors_updated, analysis_.workloads_, tensor_size);
        // } else{
        //     analysis_.workload_mapping_graph[node] = node->constructLoopNest(stride_tilesteps, tensors_updated, analysis_.workloads_);
        // }
    
        analysis_.workload_mapping_graph[node] = node->constructLoopNest(stride_tilesteps, tensors_updated, false, analysis_.workloads_);

        auto& curr_node = std::get<LoopNode>(analysis_.workload_mapping_graph[node]);
        for (auto op_workload:analysis_.workloads_.workloads_){ //this gives Opname string, Worklad class data structure

            // get output tensor id for this workload
            auto operation_output_name = op_workload.second->get_out();
            auto output_tensor_id      = analysis_.workloads_.getTensorID(operation_output_name);

            //check if the current node is producing this output tensor
            // for (auto& curr_node_descriptor: curr_node.descriptor_){
            for(auto tens_cnt=0; tens_cnt<curr_node.tensors.size(); tens_cnt++){

                auto cur_node_tid = curr_node.tensors[tens_cnt];
                
                if(output_tensor_id==cur_node_tid){ //match the tensor id
                    auto arch_level = curr_node.target[tens_cnt];
                    int sp_id=0;
                    std::vector<std::unordered_map<problem::DimensionID, uint32_t>> vector_of_maps;
                    // vector_of_maps.resize(spatial_dims);
                    for (auto& sp_descriptor:curr_node.descriptor_[tens_cnt]){ //spatial id
                        std::unordered_map<problem::DimensionID, uint32_t> inner_map;
                        for(auto& dim: sp_descriptor){
                            
                            inner_map[dim.first] = dim.second.end;
                            // output_tensors_map_[curr_node.target[curr_node_descriptor.first]][curr_node_descriptor.first][sp_id][dim.first] = dim.second.end; //since we want whole tile to finish we take the .end value
                        }
                        vector_of_maps.emplace_back(std::move(inner_map));
                        sp_id++;
                    }
                    output_tensors_map_[arch_level][output_tensor_id] = vector_of_maps; 
                    //store the address and size of this tensor
                    // output_tensors_map_[curr_node.target[curr_node_descriptor.first]]   [curr_node_descriptor.first][]]
                }
            }
        }

        //access the tensor_read_write from the worklad parser and store the value to map in loop_nodes
        std::cout<<"!!! Node name:: "<<node->get_name()<<"\n";

    }

    void LoopNestConstructor::visitOperationNode(const OperationNode* node) {
        auto& stride_tilesteps = stride_tilesteps_;
        //stride and tilesteps update for 
        OpNode cur_node;
        auto comp_node = static_cast<const OperationNode*>(node);
        auto common_attributes = comp_node->get_common_attributes();            
        
        //common attributes for all operations
        cur_node.type    = common_attributes.type_;
        // cur_node.target  = common_attributes.target_;

        // cur_node.child   = common_attributes.child_;
        cur_node.op_name = common_attributes.op_name_;
        cur_node.tensors = common_attributes.tensors;

        auto parent = static_cast<const DataMovementTileNode*>(node->get_parent());

        auto workload = analysis_.workloads_.get_workload(cur_node.op_name);

        for(auto& ten_name:workload->ins_){
            cur_node.in_tensors.emplace_back(workload->getTensorID(ten_name));
        }    
        cur_node.out_tensor = workload->getTensorID(workload->out_);

        //we need this input tensor size for non GEMM operations, need to make sure that the tiling of shared tensors like C in GEMM-GEMM have same tile sizes. If we do this for GEMM operation also here than it will become like a loop DataMovement->GEMM
        // we don't need this in and out tensor size for GEMM operations because 

        analysis_.workload_mapping_graph[node] = cur_node;

        //update the stride_tilestep for the tensor that is produced by this operation 
        //GEMM
        if (common_attributes.type_ == operation_t::GEMM){
            //initialize stride_tilesteps with the operation nodes
            auto compute_map=comp_node->get_compute_mapping();
            //initialize global tensors strides and tilesteps
            // for (uint32_t gl_in_idx=0; gl_in_idx<compute_map.size(); gl_in_idx++){ //loop over global and intermediate tensors
            for (const auto& pair: compute_map.tilesizes){ // tensor id, tilesizes
                // stride_tilesteps[gl_in_idx].stride[pair.first] = pair.second;
                // stride_tilesteps[gl_in_idx].tilesteps[pair.first] = compute_map[gl_in_idx].relative_tilesteps[pair.first];
                stride_tilesteps.stride[pair.first][cur_node.op_name] = pair.second;
                stride_tilesteps.tilesteps[pair.first][cur_node.op_name] = compute_map.relative_tilesteps[pair.first];

            }
            // for (const auto& pair: compute_map[0].tilesizes){
            //     stride_tilesteps_global.stride[pair.first] = pair.second;
            //     stride_tilesteps_global.tilesteps[pair.first] = compute_map[0].relative_tilesteps[pair.first];
            // }

            // if (compute_map.size()>1){
            //     //initialize intermediate tensors stride and tilesteps
            //     for (const auto& pair: compute_map[1].tilesizes){
            //         stride_tilesteps_intermediate.stride[pair.first] = pair.second;
            //         stride_tilesteps_intermediate.tilesteps[pair.first] = compute_map[1].relative_tilesteps[pair.first];
                    
            //     }
            // }
        // } else if (common_attributes.type_ == operation_t::SOFTMAX){ 
        } else { 

            for (auto& [tid, ts]: common_attributes.tilesizes){
                stride_tilesteps.stride[tid][cur_node.op_name] = ts;
                stride_tilesteps.tilesteps[tid][cur_node.op_name] = 1;
            }
        }



            
    }



    void LoopNestConstructor::visitCollectiveOperationNode(const CollectiveOperationNode* node){
        auto& stride_tilesteps = stride_tilesteps_;
        ColOpNode cur_node;
        auto colOp_node = static_cast<const CollectiveOperationNode*>(node);

        cur_node.target = colOp_node->collective_op_description.target;
        cur_node.child  = colOp_node->collective_op_description.child;
        cur_node.type_  = colOp_node->collective_op_description.type_;
        cur_node.reduction_op  = colOp_node->collective_op_description.reduction_op;
        cur_node.dimension = colOp_node->collective_op_description.dimension;

        cur_node.src    = colOp_node->collective_op_description.src;
        cur_node.dest   = colOp_node->collective_op_description.dest;

        cur_node.in_tensor = colOp_node->collective_op_description.in_tensor;
        cur_node.out_tensor = colOp_node->collective_op_description.out_tensor;
        cur_node.scale      = colOp_node->collective_op_description.scale;
        cur_node.wb_output = {{cur_node.in_tensor, colOp_node->collective_op_description.wb_output}}; 
        cur_node.tensor_is_rw[cur_node.out_tensor] = true; // tensor is always read write for collective operation
        cur_node.tag = colOp_node->collective_op_description.tag;
        cur_node.spatial_factor = colOp_node->collective_op_description.spatial_factor;

        uint64_t num_src_devices=1;
        for (auto src: cur_node.src){
            num_src_devices*=src->getInstanceSize(); // number of source devices from which the data is collected from
        }
        uint64_t num_dest_devices=1;
        for (auto dest: cur_node.dest){
            num_dest_devices*=dest->getInstanceSize();
        }

        //get input tensor size
        for (auto& pair: output_tensors_map_){
            if(pair.first == cur_node.target){ //match if the data is present in same arch level where the CollectiveOperation node is present
                for(auto& pair_tid_dim: pair.second){

                    if(pair_tid_dim.first == cur_node.in_tensor){ //match if the input to this collective operation matches the output tensors in the output_tensor_map
                        cur_node.in_tensor_size[pair_tid_dim.first] = pair_tid_dim.second;

                    }

                }

            }
        }

        // calculate output tensor size which will depend on the type of collective operation we are doing

        //gather will make the tensor size to increase by #  of devices from which we are gathering the data from
        if (cur_node.type_ == stype_t::GATHER || cur_node.type_ == stype_t::ALLGATHER){
            cur_node.out_tensor_size[cur_node.out_tensor] = cur_node.in_tensor_size[cur_node.in_tensor]; // initialize output size same as input

            for(auto& pair_tid_dim:cur_node.out_tensor_size){
                int sp_cnt=0;
                for(auto& sp_id:pair_tid_dim.second){
                    for(auto& dim: sp_id){
                        if(dim.first==cur_node.dimension){
                            cur_node.out_tensor_size[pair_tid_dim.first][sp_cnt][dim.first] *=(num_src_devices); //FIXME::snegi think about spatial X and spatial Y here ---> specially when different dimensions are mapped to spatial X and spatial Y //FIXME::snegi this size should also depend on the source of collective operation node //num_devices-1 because we only need to read data from N-1 nodes data from one node is already present in that memory ----> Maybe don't need to worry about this because when col-op is present as the children of spatial node only then it writes back to the parent level and when it is the children of spatial node the other children will be temporal nodes
                        }
                    }
                    sp_cnt++;
                }
            }
        } 
        else if(cur_node.type_ == stype_t::BROADCAST){ // one to some, so writes are parallel
            cur_node.out_tensor_size = cur_node.in_tensor_size; // broadcast sends data of same size to all the other nodes
        }
        else if(cur_node.type_ == stype_t::REDUCTION || cur_node.type_ == stype_t::ALLREDUCE){ // some to one
            cur_node.out_tensor_size = cur_node.in_tensor_size; // reduction also reduces data from all the other nodes

            // for(auto&[tid, sp_tensor]:cur_node.out_tensor_size){
            //     int sp_cnt=0;
            //     for(auto& tens_size:sp_tensor){
            //         for(auto&[dimid,val]:tens_size){
            //             if(dimid==cur_node.dimension) val *=num_src_devices;
            //         }
            //     }

            // }
        }
        else if(cur_node.type_ == stype_t::SCATTER){
            // cur_node.out_tensor_size = cur_node.in_tensor_size; // FIXME::snegi check if this is correct. Do we scatter a single tensor or multiple tensors are scattered across nodes
            //read_size > write_size for sctter
            //example for scatter: GEMM1->Gather->SIMD->Scatter->GEMM2
            cur_node.out_tensor_size[cur_node.out_tensor] = cur_node.in_tensor_size[cur_node.in_tensor]; // initialize output size same as input

            for(auto& pair_tid_dim:cur_node.out_tensor_size){
                int sp_cnt=0;
                for(auto& sp_id:pair_tid_dim.second){
                    for(auto& dim: sp_id){
                        if(dim.first==cur_node.dimension){
                            cur_node.out_tensor_size[pair_tid_dim.first][sp_cnt][dim.first] /=(float)(num_dest_devices); //FIXME::snegi think about spatial X and spatial Y here ---> specially when different dimensions are mapped to spatial X and spatial Y //FIXME::snegi this size should also depend on the source of collective operation node //num_devices-1 because we only need to read data from N-1 nodes data from one node is already present in that memory
                        }
                    }
                    sp_cnt++;
                }
            }


        }
        else {
            COMET_ASSERT(false, stype_to_string(cur_node.type_) << " collective operation is not supported in COMET currently");
        }

        //update this tensor size in output_tensors_map_ as well --> this is updated so that if this size is used by next operation at this same level it can use this updated tensor size
        for (auto& pair: output_tensors_map_){
            if(pair.first == cur_node.target){
                for(auto& pair_tid_dim: pair.second){
                    if(pair_tid_dim.first == cur_node.out_tensor){ // tensor id of the output in output_map matches the output of collective op then update the output size in output_map
                        pair_tid_dim.second = cur_node.out_tensor_size[pair_tid_dim.first];
                    }
                }
            }
        }
    
        //update stride_tilestep for the tensor if the wb_output is true for this tensor
        // the tag for stride_tilestep from collective operation node is derived by the tag of this tensor in the parent node, same tensor will not be written back by multiple children
        for(auto&[tid, wb_output]: cur_node.wb_output){
            if(wb_output){
                stride_tilesteps.stride[tid][cur_node.tag] = cur_node.out_tensor_size[tid].back();// last item in the vector is the outer loop which will decide the stride for the parent
                stride_tilesteps.tilesteps[tid][cur_node.tag] = 1; //FIXME::snegi check if this is correct or not
            }
        }

        //create loop nest for the input and output
        loop_nest_frm_tensor_size(cur_node.in_tensor_size, cur_node.descriptor_input); // loop nest descriptor for input

        loop_nest_frm_tensor_size(cur_node.out_tensor_size, cur_node.descriptor_output); // loop nest descriptor for input

        analysis_.workload_mapping_graph[node] = cur_node;
    }



    


} // namespace analysis

