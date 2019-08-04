CC=gcc
CFLAGS=-O3 -Wall -std=c99 -g -lm -DSUITESPARSE -fopenmp
CFLAGS_INIT=-Ofast -flto -Wall -std=c99

all: init test debug

init: initSpmv_fp32 initSpmv_fp64

test: spmv_fp32 spmv_fp64 spmv_fp32_neon spmv_fp64_neon

debug: spmv_errcheck_fp32 spmv_errcheck_fp64 spmv_errcheck_fp32_neon spmv_errcheck_fp64_neon

initSpmv_fp32: ../initSpmv.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) $(CFLAGS_INIT) -DSP -o $@

initSpmv_fp64: ../initSpmv.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) $(CFLAGS_INIT) -o $@

spmv_fp32: ../testSpmv.c ../spmv_general.c ../calculateSpmv.c ../spmv.h
	$(CC) $(filter %.c, $^) -DSP $(CFLAGS) -fno-tree-vectorize -o $@

spmv_fp64: ../testSpmv.c ../spmv_general.c ../calculateSpmv.c ../spmv.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -fno-tree-vectorize -o $@

spmv_errcheck_fp32: ../testSpmv.c ../calculateSpmv.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) -DSP -DDEBUG $(CFLAGS) -o $@	

spmv_errcheck_fp64: ../testSpmv.c ../calculateSpmv.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) -DDEBUG $(CFLAGS) -o $@	

spmv_fp32_neon: ../testSpmv.c ../spmv_general.c calculateSpmv_arm.c ../spmv.h
	$(CC) $(filter %.c, $^) -DSP $(CFLAGS) -o $@

spmv_fp64_neon: ../testSpmv.c ../spmv_general.c calculateSpmv_arm.c ../spmv.h
	$(CC) $(filter %.c, $^) $(CFLAGS) -o $@

spmv_errcheck_fp32_neon: ../testSpmv.c calculateSpmv_arm.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) -DSP -DDEBUG $(CFLAGS) -o $@	

spmv_errcheck_fp64_neon: ../testSpmv.c calculateSpmv_arm.c ../spmv_general.c ../spmv.h
	$(CC) $(filter %.c, $^) -DDEBUG $(CFLAGS) -o $@	

.PHONY : clean
clean :
	rm initSpmv* spmv*
