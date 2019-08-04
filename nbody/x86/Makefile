CC = icc
OBJ = raw_dp raw_sp intrin_dp intrin_sp
CFLAGS = -O3 -Wall -xHost -std=c99 -DNOVALIDATE -qopenmp

all : $(OBJ)

raw_dp : ../nbody-soa.c ../nbody-soa.h ../byForce.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -no-vec -o $@ -I../

raw_sp : ../nbody-soa.c ../nbody-soa.h ../byForce.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -no-vec -o $@ -I../

intrin_dp : ../nbody-soa.c ../nbody-soa.h byForceIntrin.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -o $@ -I../

intrin_sp : ../nbody-soa.c ../nbody-soa.h byForceIntrin.c
	$(CC) $(filter %.c, $^) $(CFLAGS) -DSP -o $@ -I../

clean :
	rm -rf $(OBJ)
	
.PHONY : all clean
