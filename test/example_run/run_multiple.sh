
N=100000
for ((i=1; i<=N; i++)); do
  ../../build/comet \
    --constants_file const.yaml \
    --arch_file arch.yaml \
    --problem_file problem.yaml \
    --mapping_file mapping.yaml
done

#--logging_verbosity 1
