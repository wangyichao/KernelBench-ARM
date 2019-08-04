#include "../spmv.h"

#include <string.h>
#include <immintrin.h>

void calculateSpmvCSR(FLOAT *y, CSR *a, FLOAT *x)
{
    size_t nrows = a->nrows;

#pragma omp parallel for
    for(size_t i = 0; i < nrows; ++i)
    {
        FLOAT sum = 0;

        int idx;
#ifdef SP
        for(idx = a->row_ptr[i]; idx <= a->row_ptr[i+1] - 8; idx += 8)
        {
            __m256 vsum, va, vx;
            FLOAT xtmp[8];

            xtmp[0] = x[a->col_ind[idx]];
            xtmp[1] = x[a->col_ind[idx+1]];
            xtmp[2] = x[a->col_ind[idx+2]];
            xtmp[3] = x[a->col_ind[idx+3]];
            xtmp[4] = x[a->col_ind[idx+4]];
            xtmp[5] = x[a->col_ind[idx+5]];
            xtmp[6] = x[a->col_ind[idx+6]];
            xtmp[7] = x[a->col_ind[idx+7]];

            va = _mm256_load_ps(&(a->val[idx]));
            vx = _mm256_load_ps(xtmp);

            vsum = _mm256_mul_ps(va, vx);
            _mm256_store_ps(xtmp, vsum);
            sum += xtmp[0] + xtmp[1] + xtmp[2] + xtmp[3] + xtmp[4] + xtmp[5] + xtmp[6] + xtmp[7];
        }
        for(; idx < a->row_ptr[i+1]; ++idx)
        {
            sum += a->val[idx] * x[a->col_ind[idx]];
        }
#else
        for(idx = a->row_ptr[i]; idx <= a->row_ptr[i+1] - 4; idx += 4)
        {
            __m256d vsum, va, vx;
            FLOAT xtmp[4];

            xtmp[0] = x[a->col_ind[idx]];
            xtmp[1] = x[a->col_ind[idx+1]];
            xtmp[2] = x[a->col_ind[idx+2]];
            xtmp[3] = x[a->col_ind[idx+3]];

            va = _mm256_load_pd(a->val + idx);
            vx = _mm256_load_pd(xtmp);

            vsum = _mm256_mul_pd(va, vx);
            _mm256_store_pd(xtmp, vsum);
            sum += xtmp[0] + xtmp[1] + xtmp[2] + xtmp[3];
        }
        for(; idx < a->row_ptr[i+1]; ++idx)
        {
            sum += a->val[idx] * x[a->col_ind[idx]];
        }
#endif
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

#pragma omp for schedule(dynamic)
    for(size_t blockrow = 0; blockrow < nblockrows; ++blockrow)
    {
        int firstblock = a->row_ptr[blockrow];
        int lastblock = a->row_ptr[blockrow+1];

        memset(y_local_thrd, 0, r * sizeof(FLOAT));
        for(int block = firstblock; block < lastblock; ++block)
        {
            int col_s = a->col_ind[block];
            FLOAT *this_block = a->val + block * r * c;

#ifdef SP
            for(int colid = 0; colid < c; ++colid)
            {
                int rowid;
                for(rowid = 0; rowid <= r - 8; rowid += 8)
                {
                    __m256 vy_local, va;
                    vy_local = _mm256_load_ps(y_local_thrd + rowid);
                    va = _mm256_load_ps(this_block + colid * r + rowid);
                    __m256 tmp = _mm256_set1_ps(x[col_s + colid]);
                    vy_local = _mm256_fmadd_ps(va, tmp,vy_local);
                    _mm256_store_ps(y_local_thrd + rowid, vy_local);
                }
                for(; rowid < r; ++rowid)
                {
                    y_local_thrd[rowid] += x[colid + col_s] * this_block[colid * r + rowid]; 
                }
            }
#else
            for(int colid = 0; colid < c; ++colid)
            {
                int rowid;
                for(rowid = 0; rowid <= r - 4; rowid += 4)
                {
                    __m256d vy_local, va;
                    vy_local = _mm256_load_pd(y_local_thrd + rowid);
                    va = _mm256_load_pd(this_block + colid * r + rowid);
                    __m256d tmp=_mm256_set1_pd(x[col_s + colid]);
                    vy_local = _mm256_fmadd_pd(va, tmp, vy_local);
                    _mm256_store_pd(y_local_thrd + rowid, vy_local);
                }
                for(; rowid < r; ++rowid)
                {
                    y_local_thrd[rowid] += x[colid + col_s] * this_block[colid * r + rowid]; 
                }
            } 
#endif

        }

        size_t row_start = blockrow * a->r;
        size_t row_end = MIN((blockrow+1) * a->r, a->nrows);

        for(size_t row = row_start; row < row_end; ++row) y[row] += y_local_thrd[row - row_start];
    }

    free(y_local_thrd);
}

}