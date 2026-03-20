# SPMM Build Fix Documentation

## Problem Description

**Error**: `Cannot open file spmm.o`
**Root Cause**: Filename mismatch between makefile output and host code expectation

### Detailed Analysis

**Before Fix**:
- `makefile` generated: `build/spmm_kernel.o` (line 50: `-o build/$(OP_NAME).o`)
- `spmm.h` expected: `build/spmm.o` (line 34: `std::string bin_name = "spmm.o"`)
- Result: Host code tried to load `spmm.o` but file didn't exist

**Build Files Generated Before Fix**:
```
build/
├── spmm_kernel_aic.o    # AIC object
├── spmm_kernel_aiv.o    # AIV object
├── spmm_kernel.o        # Wrong filename!
└── spmm                 # Host executable
```

**What Host Code Expected**:
```cpp
// spmm.h:34
std::string kernel_aic_name = "spmm_kernel_aic";
std::string kernel_aiv_name = "spmm_kernel_aiv";
std::string bin_name = "spmm.o";  // ← Expected this file

RegisterBinaryKernel(kernel_aic_name.c_str(), bin_name.c_str());  // Loads spmm.o
RegisterBinaryKernel(kernel_aiv_name.c_str(), bin_name.c_str());  // Loads spmm.o
```

## Solution

### Fix Applied

**File**: `makefile`

**Changed** (line 47-51):
```makefile
# BEFORE:
$(OP_NAME).o: $(OP_NAME)_aic.o $(OP_NAME)_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	build/$(OP_NAME)_aic.o build/$(OP_NAME)_aiv.o \
	--allow-multiple-definition -static \
	-o build/$(OP_NAME).o  # ← Generated spmm_kernel.o

# AFTER:
$(OP_NAME).o: $(OP_NAME)_aic.o $(OP_NAME)_aiv.o
	ld.lld -m aicorelinux -Ttext=0 -z separate-code \
	build/$(OP_NAME)_aic.o build/$(OP_NAME)_aiv.o \
	--allow-multiple-definition -static \
	-o build/spmm.o  # ← Now generates spmm.o
```

**Changed** (line 92-96):
```makefile
# BEFORE:
$(OP_NAME)_ca.o: $(OP_NAME)_aic_ca.o $(OP_NAME)_aiv_ca.o
	ld.lld -Ttext=0 \
	build/$(OP_NAME)_aic_ca.o build/$(OP_NAME)_aiv_ca.o \
	-static \
	-o build/$(OP_NAME)_ca.o  # ← Generated spmm_kernel_ca.o

# AFTER:
$(OP_NAME)_ca.o: $(OP_NAME)_aic_ca.o $(OP_NAME)_aiv_ca.o
	ld.lld -Ttext=0 \
	build/$(OP_NAME)_aic_ca.o build/$(OP_NAME)_aiv_ca.o \
	-static \
	-o build/spmm_ca.o  # ← Now generates spmm_ca.o
```

## Verification Steps

### 1. Clean Build
```bash
cd ascblas/src
make clean
```

### 2. Build Kernel
```bash
make spmm_kernel.o

# Expected output:
# build/spmm_kernel_aic.o
# build/spmm_kernel_aiv.o
# build/spmm.o  ← Now correctly named!
```

### 3. Verify File
```bash
ls -lh build/spmm.o
# Should exist

# Verify kernel symbols
strings build/spmm.o | grep spmm_kernel
# Should show: spmm_kernel_aic and spmm_kernel_aiv
```

### 4. Continue Build
```bash
make main
ls -lh build/spmm
# Host executable should be created
```

### 5. Test Run
```bash
./run.sh 0 0 128 128 128 128 128 128 128
# Should now work without "Cannot open file spmm.o" error
```

## Build Chain

### Compiler Flow

```
1. Compile AIC kernel
   spmm_kernel.cpp → spmm_kernel_aic.o
   (dav-c220-cube, compute)

2. Compile AIV kernel
   spmm_kernel.cpp → spmm_kernel_aiv.o
   (dav-c220-vec, preprocessing)

3. Link kernels
   spmm_kernel_aic.o + spmm_kernel_aiv.o → spmm.o
   (combined kernel binary)

4. Compile host
   main.cpp + handle.cc → spmm
   (executable)

5. Run
   spmm loads spmm.o → extracts spmm_kernel_aic/spmm_kernel_aiv
   → launches kernels on NPU
```

## Correct Build Output

After the fix, you should see:

```bash
cd ascblas/src
make clean && make

# Output:
rm -rf build/
mkdir -p build
ccec ... -o build/spmm_kernel_aic.o  # ✓
ccec ... -o build/spmm_kernel_aiv.o  # ✓
ld.lld ... -o build/spmm.o           # ✓ FIXED! (was spmm_kernel.o)
g++ ... -o build/spmm               # ✓

build/
├── spmm_kernel_aic.o   (512KB)
├── spmm_kernel_aiv.o   (512KB)
├── spmm.o             (1.0MB)  ← Now present!
└── spmm               (856KB)  ← Host executable
```

## Related Files

### Unchanged Files (as requested)
- `spmm_kernel.cpp` - Kernel implementation
- `spmm.h` - Host interface
- `main.cpp` - Main program
- `bsr_utils.h` - BSR utilities

### Modified Files
- `makefile` - Fixed output filename (spmm.o)

## Testing

Run the test script:
```bash
cd ascblas/src
./test_build.sh
```

Expected output:
```
[1/4] Cleaning...
[2/4] Creating build directory...
[3/4] Building kernel objects...
  Compiling spmm_kernel_aic.o...
  ✓ SUCCESS: spmm_kernel_aic.o generated
  Compiling spmm_kernel_aiv.o...
  ✓ SUCCESS: spmm_kernel_aiv.o generated
  Linking to spmm.o...
  ✓ SUCCESS: spmm.o generated
[4/4] Verifying kernel symbols...
✓ SUCCESS: Found kernel symbols:
spmm_kernel_aic
spmm_kernel_aiv

Build test PASSED!
File: build/spmm.o (1.0M)
Ready for host compilation...
```

## Troubleshooting

### If error persists:

1. **Check file permissions**:
```bash
chmod +x run.sh build.sh check_env.sh
```

2. **Verify filename in host code**:
```bash
grep 'bin_name = "' ascblas/src/spmm.h
# Should show: bin_name = "spmm.o"
```

3. **Manually check build output**:
```bash
cd ascblas/src
make clean
make spmm_kernel.o
ls -lh build/
# Must show: spmm.o
```

4. **Rebuild everything**:
```bash
cd ascblas/src
make clean
make
```

## Summary

**Fix**: Changed makefile output from `spmm_kernel.o` to `spmm.o` to match host code expectation.

**Result**: Host code can now successfully load the kernel binary.

**Impact**: None on kernel functionality, only filename alignment.
