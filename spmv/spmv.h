#ifndef SPMV_H
#define SPMV_H

#include <stdlib.h>

#ifdef SP
typedef float FLOAT;
#else
typedef double FLOAT;
#endif

#define MIN(x,y) (x < y ? x : y)

typedef enum {
    ALGO_CSR = 1,
    ALGO_BCSR = 2
} SPMV_ALGO;

typedef struct{
    FLOAT *val;
    int *col_ind, *row_ptr;
    size_t nrows, num_elements;
} CSR;

typedef struct{
    FLOAT *val;
    int *col_ind, *row_ptr;
    size_t nblockrows, numblocks;
    size_t nrows;
    int r, c;
} BCSR;

//general functions
void allocateCSR(CSR *A);

void freeCSR(CSR *A);

void allocateBCSR(BCSR *A_BLK, CSR *A);

void freeBCSR(BCSR *A);

// CSR calculate
void calculateSpmvCSR(FLOAT *y, CSR *a, FLOAT *x);

// BCSR calculate
void prepareSpmvBCSR(BCSR *a_blk, CSR *a);

void calculateSpmvBCSR(FLOAT *y, BCSR *a, FLOAT *x, FLOAT *y_local);
#endif