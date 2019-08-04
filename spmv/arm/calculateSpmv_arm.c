#include "../spmv.h"

#include <string.h>
#include <arm_neon.h>

void calculateSpmvCSR(FLOAT *y, CSR *a, FLOAT *x)
{
    size_t nrows = a->nrows;

#pragma omp parallel for
    for(size_t i = 0; i < nrows; ++i)
    {
        FLOAT sum = 0;

        int idx;
#ifdef SP
        for(idx = a->row_ptr[i]; idx <= a->row_ptr[i+1] - 4; idx += 4)
        {
            float32x4_t vsum, va, vx;
            FLOAT xtmp[4];

            xtmp[0] = x[a->col_ind[idx]];
            xtmp[1] = x[a->col_ind[idx+1]];
            xtmp[2] = x[a->col_ind[idx+2]];
            xtmp[3] = x[a->col_ind[idx+3]];

            va = vld1q_f32(&(a->val[idx]));
            vx = vld1q_f32(xtmp);

            vsum = vmulq_f32(va, vx);
            vst1q_f32(xtmp, vsum);
            sum += xtmp[0] + xtmp[1] + xtmp[2] + xtmp[3];
        }
        for(; idx < a->row_ptr[i+1]; ++idx)
        {
            sum += a->val[idx] * x[a->col_ind[idx]];
        }
#else
        for(idx = a->row_ptr[i]; idx <= a->row_ptr[i+1] - 2; idx += 2)
        {
            float64x2_t vsum, va, vx;
            FLOAT xtmp[2];

            xtmp[0] = x[a->col_ind[idx]];
            xtmp[1] = x[a->col_ind[idx+1]];

            va = vld1q_f64(a->val + idx);
            vx = vld1q_f64(xtmp);

            vsum = vmulq_f64(va, vx);
            vst1q_f64(xtmp, vsum);
            sum += xtmp[0] + xtmp[1];
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

#ifdef SP
            for(int colid = 0; colid < c; ++colid)
            {
                int rowid;
                for(rowid = 0; rowid <= r - 4; rowid += 4)
                {
                    float32x4_t vy_local, va;
                    vy_local = vld1q_f32(y_local_thrd + rowid);
                    va = vld1q_f32(this_block + colid * r + rowid);
                    vy_local = vfmaq_n_f32(vy_local, va, x[col_s + colid]);
                    vst1q_f32(y_local_thrd + rowid, vy_local);
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
                for(rowid = 0; rowid <= r - 2; rowid += 2)
                {
                    float64x2_t vy_local, va;
                    vy_local = vld1q_f64(y_local_thrd + rowid);
                    va = vld1q_f64(this_block + colid * r + rowid);
                    vy_local = vfmaq_n_f64(vy_local, va, x[col_s + colid]);
                    vst1q_f64(y_local_thrd + rowid, vy_local);
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