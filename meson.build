project('comet', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20'])

add_project_arguments('-g', '-O0', '-fno-inline', language: 'cpp')

inc_dir = include_directories('include')

subdir('src')

absl_libs = [
'absl_log'
]

absl_libs = []

deps = []

foreach lib : absl_libs
  deps += dependency(lib)
endforeach

deps += dependency('argparse')

deps += dependency('libconfig++')
deps += dependency('yaml-cpp')

spdlog = subproject('spdlog')
spdlog_dep = spdlog.get_variable('spdlog_dep')
#fmt = subproject('fmt')
#fmt_dep = fmt.get_variable('fmt_header_only_dep')

deps += spdlog_dep

executable('comet',
           sources,
           include_directories: inc_dir, dependencies: deps)

          #  , c_args: ['-g', '-O0', '-fno-inline']
