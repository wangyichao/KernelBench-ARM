CC=gcc
OBJ = armneon_dp raw_dp armneon_sp raw_sp blockingneon_sp blockingneon_dp
CFLAGS=-O3 -Wall -std=c99 -DGEMM_USE_ARM -fopenmp

all : $(OBJ)

armneon_dp : ../gemm.c calculateMatrix_arm.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -o $@

raw_dp : ../gemm.c ../calculateMatrix.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -fno-tree-vectorize -o $@

armneon_sp : ../gemm.c calculateMatrix_arm.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -o $@

raw_sp : ../gemm.c ../calculateMatrix.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -fno-tree-vectorize -DSP -o $@

blockingneon_sp : ../gemm.c calculateBlockingMatrix_arm.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -o $@

blockingneon_dp : ../gemm.c calculateBlockingMatrix_arm.c ../gemm.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -o $@

clean :
	rm -rf $(OBJ)
	
.PHONY : all clean
