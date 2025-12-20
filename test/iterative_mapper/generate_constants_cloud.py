from itertools import product
import yaml
import argparse
import pdb
from itertools import permutations
import random
import pickle as pkl
from functools import reduce
import multiprocessing
import time
import re
import csv
import os 
import pandas as pd 
import sys
import logging
from datetime import datetime
import math



n_proc = multiprocessing.cpu_count()
# n_proc = 1

# Add this near the top of your script
def setup_logging(log_file):
    """Set up logging to file and console"""
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger



ARCH_LEVELS = {"DRAM_temporal": 0, "DRAM_spatial_X": 1, "DRAM_spatial_Y": 2, "GB_temporal": 3, "GB_spatial_X": 4, "GB_spatial_Y": 5, "Core_temporal": 6, "Compute": 7}

def read_problem_instance(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        # Navigate to the 'problem' -> 'instance' and get all key-value pairs
        if 'problem' in data and 'instance' in data['problem']:
            instance_data = data['problem']['instance']
            if isinstance(instance_data, dict):
                return instance_data
            else:
                raise ValueError("The 'instance' attribute is not a dictionary.")
        else:
            raise KeyError("The 'problem' or 'instance' attribute is missing in the YAML file.")
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

def read_mapping_constraints(file_path: str) -> dict:

    constraint_dict={}# dim: {arch_level_index: value}
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        
        if 'mapping_constraints' in data:
            for constraints in data['mapping_constraints']:
                target = constraints['target']
                type = constraints['type']
                
                if(len(constraints['factors'])>1):
                    #spatial X and spatial Y
                    i=0
                    for factor_entry in constraints['factors']: #X and Y spatial dimensions
                        # dim, val = factor_entry.split("=")
                        factors = {}
                        for item in factor_entry.split():
                            key, value = item.split('=')
                            factors[key.strip()] = int(value.strip())

                        sp = "X" if i==0 else "Y"
                        i+=1
                        arch_level = target+"_"+type + "_"+sp
                        index = ARCH_LEVELS[arch_level]

                        for(dim, val) in factors.items():
                            if dim not in constraint_dict:
                                constraint_dict[dim] = {}
                                constraint_dict[dim][index] = int(val)
                            else:
                                #dim exists, append the value
                                if index not in constraint_dict[dim]:
                                    constraint_dict[dim][index] = int(val)
                                else:
                                    # error out since dim and index both exist
                                    raise ValueError(f"Duplicate entry for dimension {dim} at index {index}.")
                else:
                    # 1 dimension
                    arch_level = target+"_"+type
                    # dim, val = constraints['factors'][0].split("=")
                    index = ARCH_LEVELS[arch_level]

                    factors = {}
                    for item in constraints['factors'][0].split():
                        key, value = item.split('=')
                        factors[key.strip()] = int(value.strip())

                    for(dim, val) in factors.items():
                        if dim not in constraint_dict:
                            constraint_dict[dim] = {}
                            constraint_dict[dim][index] = int(val)
                        else:
                            #dim exists, append the value
                            if index not in constraint_dict[dim]:
                                constraint_dict[dim][index] = int(val)
                            else:
                                # error out since dim and index both exist
                                raise ValueError(f"Duplicate entry for dimension {dim} at index {index}.")
            
            return constraint_dict
        else:
            raise KeyError("The 'mapping_constraints' attribute is missing in the YAML file.")
        
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)

##################### CONSTRAINTS on MAPPING FILE ######################

def multiply_elements(factor_set):
    result = 1
    for factor in factor_set:
        result *= factor
    return result

# Condition: The product of all elements in the set must be equal to the target number
def constraint_multiply(factor_set, target_number):
    return multiply_elements(factor_set) == target_number

# Condition: The 2nd element of the set must have a particular value (e.g., 4)
def constraint_second_element(factor_set, value):
    return factor_set[1] == value

# Generic function to check multiple constraints
def check_constraints(factor_set, target_number, constraints):
    for constraint_func, constraint_args in constraints:
        if not constraint_func(factor_set, *constraint_args):
            return False
    return True
#########################################################################


def get_factors(num):
    return [i for i in range(1, num + 1) if num % i == 0]

def generate_factor_sets(num, N, fixed_indices):
    # Separate fixed and free indices
    fixed_values = [None] * N
    free_indices = []

    reduced_num = num

    # Assign fixed values and reduce the target number
    for index, value in fixed_indices.items():
        fixed_values[index] = value
        reduced_num //= value  # Divide the target by the fixed value

    # Identify free indices that need to be generated
    free_indices = [i for i in range(N) if i not in fixed_indices]

    if reduced_num <= 0:
        return []

    # Generate factors of the reduced number
    factors = get_factors(reduced_num)

    # Generate valid sets for free indices
    valid_sets = []

    # temp = list(product(factors, repeat=len(free_indices)))
    
    for factor_set in product(factors, repeat=len(free_indices)):
        if multiply_elements(factor_set) == reduced_num:
            # Merge fixed values and generated factors
            current_set = fixed_values[:]
            for idx, factor in zip(free_indices, factor_set):
                current_set[idx] = factor
            valid_sets.append(tuple(current_set))
    
    return valid_sets


def run(data, id, config, output, dimension_dict):
    filename = 'constants-' + str(id) + '.yaml'
    output_filename = 'output-' + str(id) + '.log'


    # Default values dictionary with all parameters
    #MNK-data['Mdata['N']data['K']
    default_values = {
        # DRAM parameters
        'M_A_DRAM_temporal': data['M'][0],  # 0
        'M_H_DRAM_temporal': data['M'][0],  

        'N_DRAM_spatial_X': data['N'][1],   # 1
        'N_DRAM_spatial_Y': data['N'][2],   # 2

        # Global Buffer parameters
        'K_A_GB_temporal': data['K'][3],    # 1

        'M_A_GB_temporal': data['M'][3],    # 3
        'M_D_GB_temporal': data['M'][3],    # 3
        'M_F_GB_temporal': data['M'][3],    # 3
        'M_H_GB_temporal': data['M'][3],    # 3
        'N_A_GB_temporal': data['N'][3],

        'N_GB_spatial_X': data['N'][4],     # 4
        'N_GB_spatial_Y': data['N'][5],     # 5

        # Core (MAC unit) temporal
        'M_Core_temporal': data['M'][6],    # 2
        'N_Core_temporal': data['N'][6],    # 3
        'K_Core_temporal': data['K'][6],    # 2

        # Compute tile
        'M_Compute': data['M'][7],          # 4
        'N_Compute': data['N'][7],          # 6
        'K_Compute': data['K'][7],          # 4

        # Simd Engine
        'M_SE': data['M'][6]*data['M'][7],               # 6
        'N_SE': data['N'][6], #math.ceil((data['N'][6]*data['N'][7])/16), #data['ND'][6],       data['N'][6]*data['N'][7]        # 6
        'K_SE': 1,               # 3

        # Shared Memory
        'M_SM': 1, #data['M'][7],               # 7
        'N_SM': data['N'][7], #data['N'][6] * data['N'][7] if (data['N'][6] * data['N'][7]) < 16 else 16, #data['N'][7],      simd width         # 7
        'K_SM': dimension_dict['K'], #data['K'][6],               # 6  # since all of K is at the core level just set to MAX for SOftmax
        "red_dim_simd": 'K',
        "red_dim_simd_val": dimension_dict['K'],

        # Reduction info (not indexed)
        'red_dim': 'M',
        'red_factor': data['M'][7],  # 7
    }



    # Wrap in top-level 'const' key
    merged = {**dimension_dict, **default_values}
    yaml_dict = {
        'const': merged
    }

    # Write updated constants to new file
    with open(f"{config}/{filename}", "w") as file:
        yaml.dump(yaml_dict, file, sort_keys=False)

    os.system(f'../../build/comet --constants_file {config}/{filename} --arch_file arch_cloud.yaml --problem_file problem.yaml --mapping_file mapping.yaml --calc_noc_energy 2>&1 | tee {output}/{output_filename}') #2>&1 redirects stderr (file descriptor 2) to stdout (file descriptor 1), so both are captured by tee.
    

    
def do_work(procnum, data, num_iterations, config_folder_path, output_folder_path, dimension_dict, counter, logger):
    stride = (num_iterations + n_proc -1)//n_proc
    start  = stride*procnum
    end    = min(num_iterations, start+stride)
    id=start
    # start, end = 0, 10
    # Calculate total work for this process
    process_total = end - start
    
    # Log the range this process will handle
    logger.info(f"Process {procnum}: Processing samples {start} to {end-1} (total: {process_total})")
    
    # Track local progress
    local_count = 0

    for d in data[start:end]:
        # d - (), (), (), () #M N K ND
        run(d, id, config_folder_path, output_folder_path, dimension_dict)
        
        id+=1
        
        # Update counters
        with counter.get_lock():
            counter.value += 1
            global_count = counter.value
        
        local_count += 1
        
        # Log progress periodically (every 5% of this process's work or at least every 10 iterations)
        log_interval = max(1, min(10, int(process_total * 0.05)))
        if local_count % log_interval == 0 or local_count == process_total:
            logger.info(f"Process {procnum}: Completed {local_count}/{process_total} samples " +
                       f"(Global progress: {global_count}/{num_iterations}, {global_count/num_iterations*100:.1f}%)")
         




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate possible mappings given the problem and constants file") 
    parser.add_argument('--problem_file', type=str, required=False, default='/PATH/prob.yaml', help='Path to the problem.yaml file')
    parser.add_argument('--constants_file', type=str, required=False, default='/PATH/const.yaml', help='Path to the constants.yaml file')
    parser.add_argument('--constraints_file', type=str, required=False, default='/PATH/constraints.yaml', help='Path to the constants.yaml file')

    parser.add_argument('--num_levels', type=int, required=False, default=8, help='Number of levels in the mapping')
    parser.add_argument('--num_iterations', type=int, required=False, default=1, help='Number of iterations to search for')
    
    parser.add_argument('--second_element_value', type=int, required=False, default=4, help='Required value for the second element in the set (for constraint testing)')


    parser.add_argument('--config_folder_name', type=str, required=True, help='Path to the constants file folder')
    parser.add_argument('--output_folder_name', type=str, required=True, help='Path to the output folder')
    parser.add_argument('--dims', type=str, required=True, help='Dimension key-value pairs like "M-128 N-256 K-1024"')

    # Add timestamp to log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"mapping_generator_{timestamp}.log"
    logger = setup_logging(log_file)
    logger.info(f"Number of processes used: {n_proc}")
    logger.info("Starting mapping generation process")


    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    
    # dimension_dict = read_problem_instance(args.problem_file)
    # dimension_dict = {'M':256, 'N':1024, 'K':4096}
    dimension_dict = dict(item.split("-") for item in args.dims.split())
    dimension_dict = {k: int(v) for k, v in dimension_dict.items()}

    dim_str=""
    for k,v in dimension_dict.items():
        dim_str += k + str(v);


    # dimension_dict["ND"] = dimension_dict['N']
    factor_set_dict = {}

    constraint_dict = read_mapping_constraints(args.constraints_file)

    key=""
    value=""
    cnt=0
    for(dim, dim_size) in dimension_dict.items(): #dimension, value
        #save the tensor size in the factor_set_dict
        if(cnt==0): 
            key += dim
            value += str(dim_size)
        else: 
            key += '-' + dim
            value += '-' + str(dim_size)
        cnt+=1

    # factor_set_dict[key] = value
    constraints=None
    for (dim, value) in dimension_dict.items(): #dimension, value

        fixed_indices = {}
        if dim in constraint_dict:
            fixed_indices = constraint_dict[dim]

        
        factor_set_dict[dim] = generate_factor_sets(value, args.num_levels, fixed_indices)

    # print(f'********Randomly shuffling all the dimensions')
    logger.info('********Randomly shuffling all the dimensions')
    for(k,v) in factor_set_dict.items():
        if(k==key): continue
        random.shuffle(factor_set_dict[k])


    for(k, v) in factor_set_dict.items():
        if(k==key): continue
        # print(f"Dimension {k} has {len(v)} possible factor sets:")
        logger.info(f"Dimension {k} has {len(v)} possible factor sets:")
        i=0
        for factor_set in v:
            # print(factor_set)
            logger.info(str(factor_set))
            i+=1
            if(i==10):
                break
        # print("\n")
        logger.info("\n")




    #permutations
    dim_permutations = [''.join(p) for p in permutations(dimension_dict.keys()) ]

    dims = list(dimension_dict.keys())

    num_dims = len(dims)
    i=0
    final_data=[]
    for combination in product(*[factor_set_dict[dim] for dim in dims]):
        # i+=1
        # final_data.append(combination)
        # for i in range(len(ARCH_LEVELS)):
        factordict={}
        cnt=0
        for d_string in dims:
            factordict[d_string]=combination[cnt]
            cnt+=1
        final_data.append(factordict)
        # print(i)

    # print('******** Total possible mapping for given constraints: {}'.format(len(final_data)))

            #print("done")
    # print(f'********* Done with generating data points')
    # filename = 'data' + key + ','+value + '.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(final_data, f)

    logger.info('******** Total possible mapping for given constraints: {}'.format(len(final_data)))
    logger.info(f'********* Done with generating data points')
    logger.info(f'********* Running COMET over all the generated data points')


    current_dir_name = os.path.dirname(os.path.abspath(__file__))
    config_folder_path = os.path.join(current_dir_name, "simulation_cloud" + "_" + dim_str, args.config_folder_name)
    output_folder_path = os.path.join(current_dir_name, "simulation_cloud" + "_" + dim_str, args.output_folder_name)

    if not os.path.exists(config_folder_path):
        folder_name=args.config_folder_name + dim_str
        os.makedirs(config_folder_path, exist_ok=True)
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path, exist_ok=True)

    
    manager = multiprocessing.Manager()
    counter = multiprocessing.Value('i', 0)
    # return_dict = manager.dict()

    num_iterations = min(args.num_iterations, len(final_data))
    logger.info(f"Starting processing of {num_iterations} samples using {n_proc} processes")

    # num_iterations = args.num_iterations
    # results = manager.list()
    jobs = []
    for i in range(n_proc):
        p = multiprocessing.Process(target = do_work, args = (i, final_data, num_iterations,config_folder_path, output_folder_path, dimension_dict, counter, logger))
        jobs.append(p)
        p.start()
    # for proc in jobs: proc.join()

    # Create a process to periodically log overall progress
    def monitor_progress():
        while any(p.is_alive() for p in jobs):
            with counter.get_lock():
                current = counter.value
            logger.info(f"Overall progress: {current}/{num_iterations} samples completed ({current/num_iterations*100:.1f}%)")
            time.sleep(30)  # Log every 30 seconds
    
    # Start the monitoring process
    monitor = multiprocessing.Process(target=monitor_progress)
    monitor.daemon = True  # This ensures the monitor will exit when the main program exits
    monitor.start()
            
    for proc in jobs: 
        proc.join()
    
    if monitor.is_alive():
        monitor.terminate()


    logger.info(f'********* Completed running all the files')

