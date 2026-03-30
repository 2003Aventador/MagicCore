CANN_DIR = $(ASCEND_HOME_PATH)
OP_NAME = spmm_kernel
INCLUDE_DIR = -I../include -I../src/utils

all: prepare $(OP_NAME)_aic.o $(OP_NAME).o main

# Create build directory
prepare:
	mkdir -p build

$(OP_NAME)_aic.o: ../src/$(OP_NAME).cce
	ccec -std=c++17 -c -x cce -O2 ../src/$(OP_NAME).cce --cce-aicore-arch=dav-c220-cube --cce-aicore-only \
	-D__DAV_C220_CUBE__ \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/interface \
	-I$(CANN_DIR)/compiler/tikcpp/tikcfw/impl \
	$(INCLUDE_DIR) \
	-o build/$(OP_NAME)_aic.o

$(OP_NAME).o: $(OP_NAME)_aic.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	build/$(OP_NAME)_aic.o \
	--allow-multiple-definition -static \
	-o build/spmm.o

main: ../src/main.cpp
	g++ -O2 -fpic -std=c++17 \
	../include/handle.cc \
	../src/main.cpp \
	-I$(CANN_DIR)/include \
	$(INCLUDE_DIR) \
	-L$(CANN_DIR)/lib64 \
	-o build/spmm \
	-lascendcl \
	-lruntime \
	-lstdc++ \
	-lpthread

clean:
	rm -rf build/

.PHONY: all prepare clean