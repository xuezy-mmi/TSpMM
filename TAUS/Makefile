NVCC = /usr/local/cuda/bin/nvcc
SPUTNIK_PATH = /home/xuezeyu/sputnik
NVCC_FLAGS = -std=c++11 -arch=sm_89 -lineinfo -lcublas -lcusparse -I${SPUTNIK_PATH} --ptxas-options=-v

##################################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file directory
INC_DIR = include

# Library directory
LIB_DIR = /usr/local/cuda/lib64
SPUTNIK_DIR = ${SPUTNIK_PATH}/build/sputnik

#output directory
OUT_DIR = out
#csv output directory
RECORD_DIR = data
##################################################################

## Compile ##

# Compile main file to object file
$(OBJ_DIR)/%.o : ./%.cpp
	@$(NVCC) $(NVCC_FLAGS) -x c++ -c $< -o $@

# Compile CUDA source files to object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	@$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

tspmm_test: $(OBJ_DIR)/tspmm_test.o $(OBJ_DIR)/tspmm.o
	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

# tspmm_test: ./tspmm_test.cpp $(SRC_DIR)/tspmm.cu
# 	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

taus_spmm_test: $(OBJ_DIR)/taus_spmm_test.o $(OBJ_DIR)/taus_spmm.o
	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

# taus_spmm_test: ./taus_spmm_test.cpp $(SRC_DIR)/taus_spmm.cu
# 	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@


tspmm_test_run_volta:
##################################### [N] [v] [kernel] [sort] [verify][mixed][record][row merge][arch]
	$(OUT_DIR)/tspmm_test ./paths.txt 512  8     0        0      0      1      1          0     volta
tspmm_test_run_turing:
##################################### [N] [v] [kernel] [sort][verify][mixed][record][row merge][arch]
	$(OUT_DIR)/tspmm_test ./paths.txt 512  8     0        0     0       1      1          0     turing
tspmm_test_run_ampere:
##################################### [N] [v] [kernel] [sort][verify][mixed][record][row merge][arch]
	$(OUT_DIR)/tspmm_test ./paths.txt 512  8     0        0     0       1      1          0     ampere
tspmm_test_run_ada:
##################################### [N] [v] [kernel] [sort][verify][mixed][record][row merge][arch]
	$(OUT_DIR)/tspmm_test ./paths.txt 512  8     0        0     0       1      1          0     ada

taus_spmm_test_run_volta:
#####################################     [N] [v] [kernel][sort] [verify] [mixed][record][reorder][arch]
	$(OUT_DIR)/taus_spmm_test ./paths.txt 512  8   0       0        0       1      0         1    volta
taus_spmm_test_run_turing:
#####################################     [N] [v] [kernel] [sort][verify] [mixed][record][reorder][arch]
	$(OUT_DIR)/taus_spmm_test ./paths.txt 512  8   0        0       0       1      0         1    turing
taus_spmm_test_run_ampere:
#####################################     [N] [v] [kernel][sort] [verify] [mixed][record][reorder][arch]
	$(OUT_DIR)/taus_spmm_test ./paths.txt 512  8   0       0       0       1      0         1    ampere
taus_spmm_test_run_ada:
#####################################     [N] [v] [kernel][sort] [verify] [mixed][record][reorder][arch]
	$(OUT_DIR)/taus_spmm_test ./paths.txt 512  16   1        0        0       1      1         1    ada
clean:
	@rm -f $(OBJ_DIR)/*
	@rm -f $(OUT_DIR)/*
	@rm -f $(RECORD_DIR)/*

obj_clean:
	@rm -f $(OBJ_DIR)/*
out_clean:
	@rm -f $(OUT_DIR)/*
record_clean:
	@rm -f $(RECORD_DIR)/*
