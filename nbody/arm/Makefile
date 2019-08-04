CC = gcc
OBJ = raw_dp raw_sp intrin_dp intrin_sp
CFLAGS = -O3 -Wall -std=c99 -fopenmp -DNOVALIDATE

all : $(OBJ)

raw_dp : ../nbody-soa.c ../nbody-soa.h ../byForce.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -fno-tree-vectorize -o $@ -I../ -lm

raw_sp : ../nbody-soa.c ../nbody-soa.h ../byForce.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -fno-tree-vectorize -o $@ -I../ -lm

intrin_dp : ../nbody-soa.c ../nbody-soa.h byForceIntrin.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -o $@ -I../ -lm

intrin_sp : ../nbody-soa.c ../nbody-soa.h byForceIntrin.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -o $@ -I../ -lm

clean :
	rm -rf $(OBJ)
	
.PHONY : all clean
