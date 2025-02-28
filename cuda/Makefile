# CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

# NVCC compiler
NVCC          ?= $(CUDA_PATH)/bin/nvcc

# MPI complier with wrapper
MPICXX ?= $(shell which mpicxx)

# Libraries
LIBRARIES += -L$(CUDA_PATH)/lib64 -lcudart

# Target rules
all: stereo

stereo.o:
	$(NVCC) -c stereo.cu

stereo: stereo.o
	$(NVCC)  stereo.o -o stereo $(LIBRARIES)

clean:
	rm -f stereo *.o
