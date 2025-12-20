#include <cmath>
#include <vector>
#include <algorithm>

#include "arch/arch.hpp"
#include "arch/network.hpp"
#include "analysis/node.hpp"
#include "util/logger.hpp"
#include "analysis/data_movement.hpp"
#include "analysis/nest_analysis.hpp"


namespace arch{

    using analysis::NodeTypes;
    using analysis::ColOpNode;
    // using analysis::DataMovementCostVec;


    using mapping::stype_t;
    struct latency_energy_struct{
        double network_energy;
        int64_t network_time;
        int64_t memory_time;
    };

    latency_energy_struct getSpatioTemporalLatency(DataMovementInfo& mov_info, ArchLevel* target, ArchLevel* child, std::shared_ptr<NetworkBase> network, LoopNode& loop_nodes, size_t tens_cnt);


    DataMovementCostVec Topology::Evaluate(LevelID target_id, LevelID child_id, std::vector<DataMovementInfo>& tensor_movement, std::vector<uint32_t> no_stall_time, LoopNode& loop_nodes, size_t tens_cnt, bool hide_rw_latency, bool run_single_iteration, uint32_t reuse_factor, bool different_iterations){
        


        

        auto target = levelIDs_.at(target_id);
        auto child = levelIDs_.at(child_id);

        auto network = target->getNetwork(true, child_id);
        auto total_latency = 0;
        auto num_tiles = tensor_movement.size();

        size_t pseudo_tiles;

        pseudo_tiles = !different_iterations ? num_tiles/reuse_factor : num_tiles;

        if(run_single_iteration) num_tiles=1;
        
        DataMovementCostVec retval(pseudo_tiles);

        auto mem_spec = std::get<MemorySpec>(target->getSpec());
        bool remove_network_latency = mem_spec.dma_contiguous_.Get();

        int start;
        if(loop_nodes.tensor_is_rw[tens_cnt]){
            //if read write tensor then store stalls from index = 1
            start = 1;
        } else{
            // if read tensor then read stalls start from index=0 itself
            start = -1;
        }

        retval.tile_size = tensor_movement[0].tile_access_size.resolve();
        retval.access_count = pseudo_tiles;
        //FIXME::snegi assuming symmetric iterations
        // if(tensor_movement[0].num_unique_tiles!=0){
        //     retval.access_count*=tensor_movement[0].num_unique_tiles;
        // }
        latency_energy_struct tile_metrics;
        for(auto idx=0; idx<num_tiles; idx++){ //FIXME::snegi should this be pseduo_tiles or num_tiles?
            
            auto& mov_info = tensor_movement[idx]; 
            
            std::pair<int64_t, int64_t> tile_latency;

            tile_metrics= getSpatioTemporalLatency(mov_info, target, child, network, loop_nodes, tens_cnt); // returns a memory, network time pair

            // total_latency += tile_latency.first + tile_latency.second;
            total_latency += tile_metrics.memory_time + tile_metrics.network_time;
            int64_t effective_latency=0;
            int64_t stall=0;
            //doubt::snegi network cycle is removed twice here if between a target and a child we have multiple tensors
            if(!remove_network_latency){
                // effective_latency = tile_latency.first + tile_latency.second;
                effective_latency = tile_metrics.memory_time + tile_metrics.network_time;

            } else {
                effective_latency = (idx == 0 || (idx == num_tiles -1)) ? (tile_metrics.memory_time + tile_metrics.network_time) : tile_metrics.memory_time ; 
            }
            
            if(effective_latency==0) stall=0;
            else stall = effective_latency - (int64_t)no_stall_time[idx];

            // if(effective_latency!=0){
            //     // auto num_accesses = mov_info.
            //     retval.access_count +=1;
            // } 
            // retval.access_count+=1;

            if(!hide_rw_latency){
                stall = effective_latency; //for cases when Tp and Sp nodes are disjoint the Sp node always need to consider the stall, always does the reuse other than the case when there is a reuse. Considering the ramp-up like this is better because this considers the reuse of the tensor from parent memory to child memory as well
            }

            // TODO Do we need to take into account that if child is non compute node for reduction tensors we cant assume double buffering 
            COMET_LOG(logger::DEBUG, "IDX{} stall {} latency{} Network{}, Memory {} total_latency {} Energy {}", idx, stall,effective_latency, tile_metrics.network_time, tile_metrics.memory_time, total_latency, tile_metrics.network_energy);
            // FIXME:: MANIKS need to add a directive to mapping to enfore no pipelining between levels, current hack will be to force rmw tensors on non compute child to not assume pipelining

            // retval.mem_net_cycles[idx]=effective_latency;
            retval.noc_energy +=tile_metrics.network_energy;
            // if (loop_nodes.tensor_is_rw[tens_cnt] && (idx == num_tiles - 1)) { // for the first and last timestep in any level there is no notion of overlapping for the rw tensors
            if (loop_nodes.tensor_is_rw[tens_cnt] && (idx == (num_tiles - reuse_factor))) { // for the first and last timestep in any level there is no notion of overlapping for the rw tensors // num_tiles - reuse_factor because of output reuse last elements will be 0 data movement
                // stall = 0;//doubt::snegi --> last cycle is considered in ramp_down hence zero
                retval.ramp_down = effective_latency;
                continue;
            }
            if (idx == 0 && !loop_nodes.tensor_is_rw[tens_cnt]) { // read only tensors have already accounted for idx ==0 in the ramp up portion
                // stall = 0; // no Stall as ramp includes this portion of it
                retval.ramp_up = effective_latency;
                continue;
            }
            //TODO complete this
            if(different_iterations){
                retval.stall[idx+start]                  = stall; // put stalls and memory network time for read write tensors from 2nd cycle, //just indexed by idx because all the reuse is merged into the relative_timestep to dependent parameter
                retval.mem_net_cycles[idx+start]         = effective_latency;
            } else if(idx%reuse_factor==0){
                COMET_ASSERT((idx/reuse_factor+start)<retval.stall.size(), "OUT OF BOUND ACCESS for per_tensor_cost " + target->getName() + " " + child->getName());
                retval.stall[idx/reuse_factor+start]                  = stall; // put stalls and memory network time for read write tensors from 2nd cycle
                retval.mem_net_cycles[idx/reuse_factor+start]         = effective_latency;
            }

            if(different_iterations || idx%reuse_factor==0){
                retval.cummulative_mem_net_cycles  += effective_latency;
                retval.cummulative_stall           += stall;
            }
            // retval.network_cycles[idx] = tile_latency.second;

        }
        COMET_LOG(logger::INFO, "For Tensor:{} Moving between:{} and {} took {} cycles", loop_nodes.tensors[tens_cnt], target->getName(), child->getName(), total_latency);
        return retval;
    }

    
    latency_energy_struct getSpatioTemporalLatency(DataMovementInfo& mov_info, ArchLevel* target, ArchLevel* child, std::shared_ptr<NetworkBase> network, LoopNode& loop_nodes, size_t tens_cnt){

        auto size_of_data   = mov_info.tile_access_size.resolve();
        auto num_of_data    = mov_info.num_unique_tiles; //count of tiles in projection set, will be zero when there is reuse
        auto spatial_factor = mov_info.tile_count; //# of spatial nodes

        bool drain_or_fill = loop_nodes.tensor_is_rw[tens_cnt];
        float scale = 1;

        uint64_t child_bw  = 1;
        uint64_t target_bw = 1;

        auto mem_spec = std::get<MemorySpec>(target->getSpec());
        // is this fair unique tiles reside in unique banks FIXME:: check this
        auto num_ports_used = std::min(num_of_data, mem_spec.ports_.Get()); //doubt::snegi does every hardware supports having these multiple ports to write data parallely to the spatial children?//doubt:;snegi shouldn't we have this at dram as well? bcz dram also fill spatial children
        num_ports_used = std::max<int>(num_ports_used, 1);

        if (drain_or_fill){
            //drain
            target_bw = std::get<MemorySpec>(target->getSpec()).child_write_bw_.Get();
            if (num_of_data == 0) num_of_data = 1; // REDUCTION tiles need to be accounted for properly FIXME:: change this in the movement calculation  //FIXME::snegi if num_of_data is 1 in this case change the reuse_factor if there is any because there is no reuse now // but changing this in movement calculation is leading to considering extra energy when rmw is true, realized after comparing with timeloop, but maybe timeloop is missing it during latency calculation           
        } else{
            //fill
            target_bw = std::get<MemorySpec>(target->getSpec()).child_read_bw_.Get();
        }

        if (num_ports_used<mem_spec.ports_.Get()) target_bw = (target_bw/mem_spec.ports_.Get())*num_ports_used;

        scale = (float)loop_nodes.scale[tens_cnt]/8.0; //converting bits to bytes

        if (child->isCompute()){
            child_bw = network->getPortBW();
        } else {
            auto spec = std::get<MemorySpec>(child->getSpec());
            if (drain_or_fill) child_bw = spec.parent_read_bw_.Get();
            else child_bw = spec.parent_write_bw_.Get();
        }

        auto scatter_factor  = 1;
        bool network_support = false;

        auto network_bw = network->getPortBW();
        // for target its based on instance_array
        // FIXME IS this right? should we not take into account spatial_factor/num_of_data factor into the target_bw 
        // this would imply perfect hashing and no bank conflicts ???
        //doubt::snegi is it correct to scale network_bw by number of ports used? 
        if(target_bw > num_ports_used*network_bw) target_bw = num_ports_used*network_bw; // data transfer is limited by amount of data that the network can transfer
        if(child_bw > network_bw) child_bw = network_bw;

        //doubt::snegi is this between target and child? or across spatial units as well?
        //FIXME::snegi this network_support cost should depend on if we are even doing the reduction or not
        if(drain_or_fill){
            network_support = network->supportsReduction();
        } else{
            network_support = network->supportsBcast();
        }
        
        // doubt::snegi --> how to find the cost of these networks which does and doesn't support Bcast?
        // if the network does not support broadcast then we have to write the data many times manually, if the network does support broadcast then that cost will be considered in the config file in terms of area, latency and energy in the network 
        if(!network_support){
            if(num_of_data !=0) scatter_factor = (spatial_factor+num_of_data-1)/num_of_data; //doubt::snegi if network doesn't support broadcast then the cost from network should be removed? -> Might have to change this for pipeline binding
        }        

        float time_for_target = std::ceil((float)scatter_factor * (float)num_of_data * (float)size_of_data * scale / (float)target_bw); //already dividing by num of data here so no need to divide the target_bw like before
        float time_for_child = (num_of_data == 0) ? 0.0 : std::ceil((float)size_of_data * scale / (float)child_bw);


        // handle RMW for drain_or_fill / reduction (assuming always read even for first tile)
        //doubt::snegi shouldn't we add this read cost to the read port? 
        //doubt::snegi shouldn't this read cost be only considered when K(redcn dim)>1?
        if (drain_or_fill && loop_nodes.rmw[tens_cnt]) {
            auto target_read_bw = std::get<MemorySpec>(target->getSpec()).child_read_bw_.Get();
            // scale the BW
            if (num_ports_used < mem_spec.ports_.Get()) target_read_bw = (target_read_bw / mem_spec.ports_.Get()) * num_ports_used;

            auto time_for_read = std::ceil((float)(scatter_factor) * (float)num_of_data * (float)size_of_data * scale / (float)target_read_bw);
            time_for_target += time_for_read;
        } 

        //FIXME:: this needs attention, link transfers need to be taken into account
        int64_t memory_time = static_cast<int64_t>(std::max(time_for_target, time_for_child));
        
        // //doubt::snegi --> understand this getlatency function, shouldn't "reduction" parameter be based on if network can support reduction or not?
        // auto network_time = network->GetLatency(mov_info, drain_or_fill, scale); 
        // auto network_energy = network->GetEnergy(mov_info, drain_or_fill, scale);

        mapping::stype_t col_op_type;
        if(num_of_data==1){
            if(loop_nodes.tensor_is_rw[tens_cnt]){
                col_op_type = mapping::stype_t::GATHER;
            } else{
                //read only tensor
                col_op_type = mapping::stype_t::BROADCAST;
            }
        } else{
            if(loop_nodes.tensor_is_rw[tens_cnt]){
                col_op_type = mapping::stype_t::GATHER;
            } else {
                //read-only tensor
                col_op_type = mapping::stype_t::SCATTER;
            }
        }
        auto network_time = network->GetLinkLatency(col_op_type, mov_info, spatial_factor, scale, mapping::operation_t::None);
        auto network_energy = network->GetLinkEnergy(col_op_type, mov_info, spatial_factor, scale, mapping::operation_t::None);


        
        std::string type_of_move = drain_or_fill ? " Drain " : " Fill ";
        COMET_LOG(logger::DEBUG, "Trying to {} {} tiles of size {} from {} ({}) to {} ({}), spatial_factor {} and scatter_factor {}", type_of_move, num_of_data, size_of_data, target->getName(), target_bw, child->getName(), child_bw, spatial_factor,  scatter_factor);
        if (drain_or_fill) {
        COMET_LOG(logger::DEBUG, "REDUCTION:: rmw{} ", loop_nodes.rmw[tens_cnt]);
        }
        COMET_LOG(logger::DEBUG, "MemoryTime:{} Network_time:{}", memory_time, network_time);

        latency_energy_struct retval;
        retval.memory_time = memory_time;
        retval.network_time = network_time;
        retval.network_energy = network_energy;

        return retval; 
    }



    ColOp_struct Topology::EvaluateCollectiveOperation(LevelID target_id, LevelID child_id, std::vector<std::vector<DataMovementInfo>>& tensor_movement, ColOpNode& node, TensorID tid){

        COMET_ASSERT(target_id==child_id, "Target and Child id should be same for collective opeation");//collective op is done between same memories

        
        ColOp_struct retval;
        double read_energy;
        double write_energy;
        double network_energy=0;


        auto target = levelIDs_.at(target_id);
        auto child  = levelIDs_.at(child_id);     

        auto link_network    = target->getNetwork(true, child_id);

        auto read_tensor_info = tensor_movement[0][0];
        double read_size = static_cast<double>(read_tensor_info.tile_access_size.resolve());
        double read_num_of_data = static_cast<double>(read_tensor_info.num_unique_tiles);

        auto write_tensor_info = tensor_movement[1][0];
        double write_size = static_cast<double>(write_tensor_info.tile_access_size.resolve());
        double write_num_of_data = static_cast<double>(write_tensor_info.num_unique_tiles);


        auto scale = (float)node.scale/8.0; //divide by 8 to convert bits to bytes

        auto spatial_factor = node.spatial_factor;

        auto reduction_operation_type = node.reduction_op;

        uint64_t num_src_devices=1;
        uint64_t num_dest_devices=1;

        for(auto s:node.src){
            num_src_devices *=s->getInstanceSize(); // if src is [GB,buffer] total devices equals to #GB*#buffer
        }

        for(auto d:node.dest){
            num_dest_devices*=d->getInstanceSize(); // if des is [buffer] total devices equals to #buffer
        }

        if(spatial_factor==0) spatial_factor = num_src_devices;//num_src_devices;

        auto mem_spec        = std::get<MemorySpec>(target->getSpec());
        double link_read_bw  = static_cast<double>(mem_spec.link_read_bw_.Get());
        double link_write_bw = static_cast<double>(mem_spec.link_write_bw_.Get());
        double link_port_bw  = static_cast<double>(link_network->getPortBW());

        int64_t network_time = link_network->GetLinkLatency(node.type_, read_tensor_info, num_src_devices, scale, reduction_operation_type);

        network_energy = link_network->GetLinkEnergy(node.type_, read_tensor_info, num_src_devices, scale, reduction_operation_type);

        int64_t memory_time = 0;

        auto read_cost = std::get<arch::MemorySpec>(target->getSpec()).child_read_energy_.Get();
        auto write_cost = std::get<arch::MemorySpec>(target->getSpec()).child_write_energy_.Get();
        double block_size = std::get<arch::MemorySpec>(target->getSpec()).width_.Get();
        double mem_read_bw = std::get<arch::MemorySpec>(target->getSpec()).child_read_bw_.Get();
        double mem_write_bw = std::get<arch::MemorySpec>(target->getSpec()).child_write_bw_.Get();
        if(node.type_ == stype_t::GATHER){
            //reads are parallel from all the devices so only need to consider latency from one device ---> but we might not be able to read parallely from all the devices
            // if(num_dest_devices>1){
            //     num_dest_devices=1; //gather is MANY->ONE operation hence number of destination devices should be 1
            // }
            auto num_ports = std::min(link_network->getNumPorts(), static_cast<uint64_t>(num_src_devices-1)); // 
            link_read_bw = std::min(link_read_bw, link_port_bw);
            link_write_bw = std::min(link_write_bw, num_ports*link_port_bw);
            link_write_bw = std::min(link_write_bw,mem_write_bw);// eventually the write bw should be limited by the memory bw FIXME::snegi check with Manik
            double read_base = read_size*scale/(link_read_bw);//all reads done in parallel
            double write_base = (num_src_devices-1)*write_size*scale/(link_write_bw*num_src_devices);// one tile is already present in the destination device, so we need to write only (num_src_devices-1) tiles. Dividing by num_src_devices to get the size of 1 tile

            double time_for_read = std::ceil(read_base);
            double time_for_write = std::ceil(write_base);
            memory_time = static_cast<int64_t>(std::max<double>(time_for_write, time_for_read)); //doubt::snegi confirm this with Manik maybe for my case we have to add these costs //doubt::snegi which of the memory and network time can we hide?

            read_energy = (read_size*scale)/block_size*read_cost;
            write_energy = ((num_src_devices-1)*read_size*scale)/block_size*write_cost; 

        }
        else if(node.type_ == stype_t::BROADCAST){//one to some, so writes are parallel
            link_read_bw  = std::min(link_read_bw, link_port_bw);
            link_write_bw = std::min(link_write_bw, link_port_bw);

            double read_base  = read_size*scale/link_read_bw;
            double write_base = read_size*scale/link_write_bw; //writing is done in parallel

            double time_for_write = std::ceil(write_base);
            double time_for_read  = 0;
            
            write_energy = ((num_dest_devices-1)*read_size*scale/block_size)*write_cost;//we are have to perform #dest-1 # of writes

            if(link_network->supportsBcast()){ // if network supports bcast then the network propogates the data, so you read one set of data
                time_for_read = std::ceil(read_base);
                read_energy = (read_size*scale)/block_size*read_cost;//FIXME::snegi add broadcast cost
            } else { // if network cannot support Bcast then we have to read as many times we have to write the data to the destination devices
                time_for_read = std::ceil(read_base*num_dest_devices);
                read_energy = ((num_dest_devices*read_size*scale)/block_size)*read_cost;
            }
            memory_time = static_cast<int64_t>(std::max(time_for_read, time_for_write));
        }
        else if(node.type_ == stype_t::REDUCTION){ //many to one operation

            link_read_bw = std::min(link_read_bw, link_port_bw); // really should never be the case
            link_write_bw = std::min(link_write_bw, link_port_bw);

            double read_base  = read_size*scale/link_read_bw;//reads done in parallel
            double write_base = read_size*scale/link_write_bw;
            bool network_support = link_network->supportsReduction();
            
            read_energy = (read_size*scale*num_src_devices)/block_size*read_cost; //read from all src devices
            write_energy = (read_size*scale)/block_size*write_cost; //write the final reduced result to dest devices

            //FIXME::snegi add reduction cost on the basis of reduction operation. Talk to Manik

            auto reduction_cost = mem_spec.reduction_cost.Get(); // TODO
            if(network_support){
                auto time_at_reduction_point = read_base + write_base;
                time_at_reduction_point += write_base*reduction_cost;//doubt::snegi should be the cost of adding/multiplying two numbers? 
                memory_time = static_cast<int64_t>(std::ceil(time_at_reduction_point));
            } else {
                auto num_writes_at_reduction_point = num_src_devices - 1; // one tile already present
                auto num_reads_at_reduction_point  = num_src_devices - 1; //doubt::snegi if the network won't support reduction then we need to read from all the devices?

                double time_at_reduction_point = std::max<double>(read_base*num_reads_at_reduction_point, write_base*num_writes_at_reduction_point);
                time_at_reduction_point += write_base*reduction_cost;//doubt::snegi check with Manik
                memory_time = static_cast<int64_t>(std::ceil(time_at_reduction_point));

                read_energy *=num_reads_at_reduction_point;
                write_energy *=num_writes_at_reduction_point;
                //doubt::snegi reduction cost? 

            }

        }
        else if(node.type_ == stype_t::ALLGATHER){
            link_read_bw  = std::min(link_read_bw, link_port_bw); 
            link_write_bw = std::min(link_write_bw, link_port_bw);
            double read_base  = read_size*num_src_devices*scale/link_read_bw; //aren't reads done in parallel?
            double write_base = read_size*num_src_devices*scale/link_write_bw;
            
            read_energy = read_size*num_src_devices*scale/block_size*read_cost;
            write_energy = read_size*num_src_devices*scale/block_size*write_cost;

            bool network_suport = link_network->supportsBcast();
            if(network_suport){ //reads happen in parallel, writes are sequential at each drain point, network suports the bcast so write time > read time

                double time_for_write = std::ceil(write_base*spatial_factor);
                memory_time = static_cast<int64_t>(time_for_write);
                // assuming loaded latency is linear addition of spatial factor (FIXME::MANIKS:: does this need to be derated)

                write_energy *=spatial_factor;
            } else {
                auto time_for_write = std::ceil(write_base*spatial_factor);
                //since network doesn't support bcast need to read serially as well
                auto time_for_read = std::ceil(read_base*spatial_factor);
                memory_time = static_cast<int64_t>(std::ceil(std::max(time_for_write, time_for_read)));

                read_energy *= spatial_factor;
                write_energy *= spatial_factor;
            }
        }
        else if(node.type_ == stype_t::ALLREDUCE){ //reduction->bcast
            link_read_bw  = std::min(link_read_bw, link_port_bw); 
            link_write_bw = std::min(link_write_bw, link_port_bw);
            
            double read_base  = read_size*num_src_devices*scale/link_read_bw;
            double write_base = write_size*num_src_devices*scale/link_write_bw;

            bool bcast_support = link_network->supportsBcast(); // gather
            bool red_support = link_network->supportsReduction(); // reduction happens in network
            read_energy = read_size*num_src_devices*scale/block_size*read_cost;
            write_energy = write_size*num_src_devices*scale/block_size*write_cost;

            auto reduction_cost = mem_spec.reduction_cost.Get();

            if(bcast_support& red_support) { // reads happen in parallel, writes into reduction points also happen in parallel
                auto time_at_reduction_points = write_base + read_base; 
                time_at_reduction_points += write_base*reduction_cost;
                memory_time = static_cast<int64_t>(std::ceil(time_at_reduction_points));
            } else if(bcast_support & !red_support){
                auto time_at_reduction_points = (write_base + read_base)*spatial_factor; 
                time_at_reduction_points += write_base*reduction_cost;
                memory_time = static_cast<int64_t>(std::ceil(time_at_reduction_points));
                
                read_energy *= spatial_factor; 
                write_energy *= spatial_factor;//doubt::snegi add reduction cost

            } else if(!bcast_support & !red_support){
                // All data is read and sent from peers, at reduction_points you have all data read and written again
                auto time_at_reduction_points = (2*read_base + write_base)*spatial_factor;
                time_at_reduction_points += write_base*reduction_cost;
                memory_time = static_cast<int64_t>(std::ceil(time_at_reduction_points));

                read_energy *= spatial_factor*2;
                write_energy *= spatial_factor; 
            }
        } 
        else if(node.type_ == stype_t::SCATTER){
            spatial_factor = num_dest_devices; // spatial factor is num_dest devices for scatter
            // for scattering as we are reading different memory regions we should be able to use the number of network ports to scale our BW
            COMET_ASSERT(num_dest_devices>1, "SCATTER requires number of destination devices to be greater than 1");
            auto num_ports = std::min(link_network->getNumPorts(), static_cast<uint64_t>(num_dest_devices-1));
            link_read_bw = std::min(link_read_bw, num_ports*link_port_bw);
            link_write_bw = std::min(link_write_bw, link_port_bw);
            //for scatter read_size > write_size
            // you only read spatial_factor - 1/spatial_factor  portion of the data
            double read_base = (spatial_factor-1)*read_size*scale/(link_read_bw*spatial_factor); //dividing by spatial_factor because read_size has the size of whole tensor which has not be partitioned
            double write_base = write_size*scale/(spatial_factor*link_write_bw);

            double time_for_read = std::ceil(read_base);
            double time_for_write = std::ceil(write_base);

            memory_time = static_cast<int64_t>(time_for_read+time_for_write);

            read_energy = (((spatial_factor-1)*read_size*scale)/(spatial_factor*block_size))*read_cost;
            write_energy = (write_size*scale/block_size)*write_cost;
        }
        // return std::make_pair(memory_time, network_time);
        retval.memory_time = memory_time;
        retval.network_time = network_time;
        retval.network_energy = network_energy;
        retval.read_energy = read_energy;
        retval.write_energy = write_energy;

        return retval;

    }

}