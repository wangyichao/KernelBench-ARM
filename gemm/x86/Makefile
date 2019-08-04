OBJ = intrin_dp raw_dp intrin_sp raw_sp unroll_dp unroll_sp

all : $(OBJ)

intrin_dp : ../gemm.c calculateMatrix_intrin.c
	icc $^ -O3 -Wall -xHost -std=c99 -o $@ -qopenmp

intrin_sp : ../gemm.c calculateMatrix_intrin.c
	icc $^ -O3 -Wall -xHost -std=c99 -DSP -o $@ -qopenmp

raw_dp : ../gemm.c ../calculateMatrix.c
	icc $^ -O3 -Wall -xHost -std=c99 -no-vec -o $@ -qopenmp

raw_sp : ../gemm.c ../calculateMatrix.c
	icc $^ -O3 -Wall -xHost -std=c99 -DSP -no-vec -o $@ -qopenmp

unroll_dp : ../gemm.c ../calculateMatrix.c
	icc $^ -O3 -Wall -xHost -std=c99 -no-vec -o $@ -qopenmp -unroll-aggressive

unroll_sp : ../gemm.c ../calculateMatrix.c
	icc $^ -O3 -Wall -xHost -std=c99 -DSP -no-vec -o $@ -qopenmp -unroll-aggressive

clean :
	rm -rf $(OBJ)
	
.PHONY : all clean
