#include "spmv.h"

#include <string.h>

void calculateSpmvCSR(FLOAT *y, CSR *a, FLOAT *x)
{
    size_t nrows = a->nrows;

#pragma omp parallel for 
    for(size_t i = 0; i < nrows; ++i)
    {
        FLOAT sum = 0;
        for(int idx = a->row_ptr[i]; idx < a->row_ptr[i+1]; ++idx)
        {
            sum += a->val[idx] * x[a->col_ind[idx]];
        }
        y[i] += sum;
    }

}

void calculateSpmvBCSR(FLOAT *y, BCSR *a, FLOAT *x, FLOAT *y_local)
{
    size_t nblockrows = a->nblockrows;
    int r = a->r;
    int c = a->c;

    FLOAT *y_local_thrd;

#pragma omp parallel private(y_local_thrd)
{
    y_local_thrd = (FLOAT*)malloc(r * sizeof(FLOAT));

#pragma omp for
    for(size_t blockrow = 0; blockrow < nblockrows; ++blockrow)
    {
        int firstblock = a->row_ptr[blockrow];
        int lastblock = a->row_ptr[blockrow+1];

        memset(y_local_thrd, 0, r * sizeof(FLOAT));
        for(int block = firstblock; block < lastblock; ++block)
        {
            int col_s = a->col_ind[block];
            FLOAT *this_block = a->val + block * r * c;
            
            for(int colid = 0; colid < c; ++colid)
            {
                for(int rowid = 0; rowid < r; ++rowid)
                {
                    y_local_thrd[rowid] += x[colid + col_s] * this_block[colid * r + rowid];
                }
            }
        }

        size_t row_start = blockrow * a->r;
        size_t row_end = MIN((blockrow+1) * a->r, a->nrows);

        for(size_t row = row_start; row < row_end; ++row) y[row] += y_local_thrd[row - row_start];
    }
}
}