# COMET

COMET is a framework for modeling and optimizing dataflow for compound operations on machine learning accelerators. COMET introduces a novel representation that explicitly models collective communication across spatial clusters, along with latency and energy cost models that account for GEMM and non-GEMM operation level dependencies within compound operations.


---

## 🛠️ Installation and Build

### Prerequisites
Make sure the following packages are installed:
```bash
sudo apt install meson ninja-build
pip3 install meson
```

### Setup Build Directory
From the repository root:
```bash
meson setup build --debug --warnlevel=2
```

### Compile
```bash
cd build
meson compile
```

---

## Usage

To run COMET, provide architecture, mapping, problem and config description files:
```bash
<path_to_comet>/build/comet --arch_file <arch_file> --mapping_file <mapping_file> --problem_file <prob_file> --constants_file <const_file>
```

### Example
```bash
cd test/example_run
./run.sh
```

---

## 📂 Input Files

- **arch_file** — describes the target hardware architecture (e.g., compute array, memory hierarchy).  
- **mapping_file** — defines how computations are mapped onto hardware resources.  
- **problem_file** — specifies the workload characteristics (e.g., GEMM dimensions, tensor shapes).
- **constants_file** - specifies the value of constants used in mapping or problem files
---

## 📊 Output

COMET produces detailed logs and cost reports summarizing:
- Latency and throughput  
- Energy and performance breakdowns  


