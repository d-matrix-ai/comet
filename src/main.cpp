#include <iostream>
#include <csignal>
#include <cstring>
#include <fstream>

// logging help
#include <argparse/argparse.hpp>

#include "compound_config/compound_config.hpp"
#include "arch/arch.hpp"
#include "problem/workload.hpp"
#include "mapping/mapping.hpp"
#include "analysis/nest_analysis.hpp"

#include "util/logger.hpp"

bool gTerminateEval;

void handler(int s)
{
  if (!gTerminateEval)
  {
    std::cerr << "First " << strsignal(s) << " caught. Abandoning "
              << "ongoing evaluation and terminating immediately."
              << std::endl;
    gTerminateEval = true;
  }
  else
  {
    std::cerr << "Second " << strsignal(s) << " caught. Existing disgracefully."
              << std::endl;
    exit(0);
  }
}

void printFileContent(const std::string& filename) {
    std::ifstream file(filename);  // Open the file
    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    std::cout << "----- Content of " << filename << " -----" << std::endl;

    std::string line;
    while (std::getline(file, line)) {  // Read line by line
        std::cout << line << std::endl;
    }

    std::cout << "---------------------------------------" << std::endl;
}


//--------------------------------------------//
//                    MAIN                    //
//--------------------------------------------//
// TODO:: MAKE THIS USING ARGPARSE

int main(int argc, char* argv[])
{
  struct sigaction action;
  action.sa_handler = handler;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  sigaction(SIGINT, &action, NULL);

  std::vector<std::string> input_files;
  std::string output_dir = ".";

  argparse::ArgumentParser program("comet");
  program.add_argument("--arch_file")
    .required()
    .help("File Describing the Architecture")
    .nargs(1);

  program.add_argument("--mapping_file")
    .required()
    .help("File Describing the mapping (both memory and compute) in DAG form")
    .nargs(1);

  program.add_argument("--problem_file")
    .required()
    .help("File Describing the problem")
    .nargs(1);
  program.add_argument("--constants_file")
    .required()
    .help("File Describing the constants")
    .nargs(1);

  program.add_argument("--logging_verbosity")
    .help("Logging Severity Trace(0), DEBUG(1), INFO(2), WARN(3), ERROR(4), CRITICAL(5), OFF(6)")
    .scan<'i', int>()
    .default_value(2);

  //add bool flag for calc_noc_energy
  program.add_argument("--calc_noc_energy")
    .help("Calculate NOC energy")
    .default_value(false)
    .implicit_value(true);  

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << "argparse did not work" << std::endl;
    std::exit(1);
  }
  std::vector<std::string> all_files;
  all_files.emplace_back(program.get<std::string>("--arch_file"));
  all_files.emplace_back(program.get<std::string>("--problem_file"));
  all_files.emplace_back(program.get<std::string>("--mapping_file"));
  all_files.emplace_back(program.get<std::string>("--constants_file"));

  // Print the content of each file
  for (const auto& file : all_files) {
      printFileContent(file);
  }


  auto logging_verbosity = static_cast<logger::LogLevel>(program.get<int>("--logging_verbosity"));


  config::CompoundConfig full_config(all_files);
  auto root_node = full_config.getRoot();
  auto root_ynode = root_node.getYNode();
  for (const auto& name: root_ynode) { 
    std::cout << name.first.as<std::string>() << std::endl;
  }
  logger::Logger::InitLogger();
  logger::Logger::set_level(logging_verbosity);
  arch::Topology arch_topo(root_node);

  problem::Workloads workloads;

  std::cout<<"****** Parsing constants file *****"<<std::endl;
  auto constants = root_node.lookup("const");
  std::cout<<"***** Parsing constants file completed *****"<<std::endl;

  auto problem = root_node.lookup("problem");


  std::cout<<"****** Parsing problem file ******"<<std::endl;
  workloads.ParseWorkloads(problem, constants);
  std::cout<<"****** Parsing problem file completed ******"<<std::endl;

  std::cout<<"****** Parsing mapping file ******"<<std::endl;
  auto mapping = root_node.lookup("mapping");
  mapping::Mapping cur_mapping(mapping, arch_topo, workloads, constants);
  std::cout<<"****** Mapping process completed ******"<<std::endl;


  std::cout<<"****** Begining validation of mapping on the architecture ******"<<std::endl;
  mapping::Validate validate(cur_mapping.root, workloads, arch_topo);
  validate.calculate_tensor_size();
  std::cout<<"****** Tile sizes set correctly ******"<<std::endl;
  validate.fitsInMemory();
  std::cout<<"****** Tiles fit in the memories ******"<<std::endl;
  std::cout<<"****** Completed validation, the mapping is correct!! ******"<<std::endl;

/*
  // mark nodes that uses tensors produced by collective operation
  std::cout<<"****** Begining to mark nodes that use tensors produced by collective operation at same level as datamovement node ******"<<std::endl;
  cur_mapping.root->mark_nodes_with_colop_tensor(cur_mapping.root);
  std::cout<<"****** Completed node marking ******"<<std::endl;
  //problem::Workload cur_workload(root_node);
  //mapping::Mapping cur_mapping(root_node, arch_topo, cur_workload);
  
  //analysis::AnalysisEngine engine;
  //engine.Init(&cur_mapping, &arch_topo, &cur_workload);
  //engine.EvaluateCost();

  // 
*/
  analysis::NestAnalysis analyzer(workloads, cur_mapping, arch_topo, program.get<bool>("--calc_noc_energy"));

  analyzer.analyze();
  
  return 0;
}
