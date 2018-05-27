TEST_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
SM_TARGETS += -gencode=arch=compute_60,code=\"sm_60,compute_60\"
CLANG_SM_TARGETS += --cuda-gpu-arch=sm_60
BASE_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
NVCC := "$(shell which nvcc)"
NVCCFLAGS += -O3 -Xptxas -v
CLANG_CFLAGS += -O3 -Xcuda-ptxas -v
CC := g++
CUDART := "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"

#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# If defined, GEMMs only compiled with specified alignment restrictions on A and B
# matrices. Otherwise, kernels are compiled for all feasible alignment options, and
# the appropriate kernel is selected.
ifdef alignment
	DEFINES += -DGEMM_ALIGNMENT=$(alignment)
endif

# If defined as false, ragged handling can be disabled.
ifdef ragged
	DEFINES += -DGEMM_RAGGED=$(ragged)
endif

#-------------------------------------------------------------------------------
# Include and Library paths
#-------------------------------------------------------------------------------

INC += -I$(BASE_DIR)

LIBS += -lcublas
NVCCFLAGS += -std=c++11


#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

DEPS := $(BASE_DIR)*.h \
        $(BASE_DIR)*.h \
        $(BASE_DIR)common.mk \
        $(TEST_DIR)Makefile


#-------------------------------------------------------------------------------
# make default
#-------------------------------------------------------------------------------

default:


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean :
	rm -f *.i* *.cubin *.cu.c *.cudafe* *.fatbin.c *.ptx *.hash *.cu.cpp *.o *.out *.obj* *dlink.* *.res *.fatbin *.module_id

#-------------------------------------------------------------------------------
# make sgemm
#-------------------------------------------------------------------------------

sgemm : gemm.cu
	$(NVCC) -DTEST_SGEMM $(DEFINES) $(SM_TARGETS) gemm.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBINC) $(LIBS)
