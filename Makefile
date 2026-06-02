# ============================================================================
#  Makefile  --  local (host C++) + Colab (nvcc) builds
#
#  Local targets (run on the Mac, no GPU):
#     make host_test    # compile & run the correctness harness
#
#  GPU targets (run on Colab / any CUDA box with nvcc):
#     make ntt          # NTT forward/inverse + convolution
#     make bench        # GPU vs CPU benchmark harness
#     make modexp_demo  # RSA-style batch modular exponentiation
#     make gpu          # all GPU targets
# ============================================================================

INC      := -Icuda/include
# Auto-detect: prefer clang++ (Mac) but fall back to g++ (Linux/Colab)
CXX      := $(shell command -v clang++ 2>/dev/null || command -v g++ 2>/dev/null || echo g++)
CXXFLAGS := -std=c++17 -O2 -Wall $(INC)

NVCC      := nvcc
NVCCFLAGS := -O3 -std=c++17 $(INC) -arch=sm_75   # sm_75 = Colab T4
BIN       := build

.PHONY: all host_test ntt_host_test local gpu ntt bench modexp_demo clean

all: local

local: host_test ntt_host_test

$(BIN):
	mkdir -p $(BIN)

# ---- Local correctness (no GPU) -------------------------------------------
host_test: $(BIN)
	$(CXX) $(CXXFLAGS) cuda/src/host_test.cpp -o $(BIN)/host_test
	./$(BIN)/host_test

ntt_host_test: $(BIN)
	$(CXX) $(CXXFLAGS) cuda/src/ntt_host_test.cpp -o $(BIN)/ntt_host_test
	./$(BIN)/ntt_host_test

# ---- GPU builds (require nvcc) --------------------------------------------
gpu: ntt bench modexp_demo

ntt: $(BIN)
	$(NVCC) $(NVCCFLAGS) cuda/src/ntt.cu -o $(BIN)/ntt

bench: $(BIN)
	$(NVCC) $(NVCCFLAGS) cuda/src/bench.cu -o $(BIN)/bench

modexp_demo: $(BIN)
	$(NVCC) $(NVCCFLAGS) cuda/src/modexp_demo.cu -o $(BIN)/modexp_demo

clean:
	rm -rf $(BIN)
