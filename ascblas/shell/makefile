################################################################################
# SPMM (Sparse Matrix Multiplication) Makefile for Ascend 910B3
#
# This Makefile compiles the BSR-format SPMM kernel for Ascend 910B3
# Architecture: dav-c220 (Ascend 910B3 with 20 AI Cores)
################################################################################

CANN_DIR = $(ASCEND_HOME_PATH)
OP_NAME = spmm_kernel
INCLUDE_DIR = -I../include

# Target: Compile both AIC and AIV kernels, link them, and build host executable
# 使用.cce文件编译，参考ascblasHgemm的编译流程
all: prepare $(OP_NAME)_aic.o $(OP_NAME)_aiv.o $(OP_NAME).o main

# Create build directory
prepare:
	mkdir -p build

################################################################################
# Kernel Compilation (AIC - AI Core for computation)
# AIC kernel performs the actual BSR SPMM computation using Cube units
# 从.cce文件编译，而不是.cpp文件
################################################################################
$(OP_NAME)_aic.o: $(OP_NAME).cce
	ccec -std=c++17 -c -x cce -O2 $(OP_NAME).cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	-D__DAV_C220_CUBE__ \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o build/$(OP_NAME)_aic.o

################################################################################
# Kernel Compilation (AIV - AI Vector Core for data preprocessing)
# AIV kernel performs B matrix padding if needed
# 从.cce文件编译，而不是.cpp文件
################################################################################
$(OP_NAME)_aiv.o: $(OP_NAME).cce
	ccec -std=c++17 -c -x cce -O2 $(OP_NAME).cce --cce-aicore-arch=dav-c220-vec --cce-aicore-only \
	-D__DAV_C220_VEC__ \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o build/$(OP_NAME)_aiv.o

################################################################################
# Link AIC and AIV object files into a single kernel binary
# IMPORTANT: Output filename must be spmm.o (not spmm_kernel.o) to match host code
################################################################################
$(OP_NAME).o: $(OP_NAME)_aic.o $(OP_NAME)_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	build/$(OP_NAME)_aic.o build/$(OP_NAME)_aiv.o \
	--allow-multiple-definition -static \
	-o build/spmm.o

################################################################################
# Host Executable Compilation
# Compiles the host code that launches the SPMM kernel
################################################################################
main: main.cpp
	g++ -O2 -fpic -std=c++17 \
	handle.cc \
	main.cpp \
	-I$(CANN_DIR)/include \
	$(INCLUDE_DIR) \
	-L$(CANN_DIR)/lib64 \
	-o build/spmm \
	-lascendcl \
	-lruntime \
	-lstdc++ \
	-lpthread

################################################################################
# Cleanup
################################################################################
clean:
	rm -rf build/

.PHONY: all prepare clean ca main main_ca
