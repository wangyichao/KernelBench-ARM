#include "spmv.h"

#include <string.h>
#include <math.h>

void allocateCSR(CSR *A)
{
    size_t num_elements = A->num_elements;
    size_t nrows = A->nrows;

    A->val = (FLOAT*)malloc(num_elements*sizeof(FLOAT));
    A->col_ind = (int*)malloc(num_elements*sizeof(int));
    A->row_ptr = (int*)malloc((nrows+1)*sizeof(int));
}

void freeCSR(CSR *A)
{
    free(A->val);
    free(A->col_ind);
    free(A->row_ptr);
}

void allocateBCSR(BCSR *A_BLK, CSR *A)
{
    size_t nblockrows = ceil(A->nrows / (double)A_BLK->r);
    size_t numblocks = 0;

    size_t maxBlocksPerRow = (A->nrows + A_BLK->c - 1) / A_BLK->c;
    int *usedBlock = (int*)malloc(maxBlocksPerRow*sizeof(int));

    for(size_t blockrow = 0; blockrow < nblockrows; ++blockrow)
    {
        size_t row_start = blockrow * A_BLK->r;
        size_t row_end = MIN((blockrow+1) * A_BLK->r, A->nrows);

        memset(usedBlock, 0, maxBlocksPerRow*sizeof(int));

        for(size_t row = row_start; row < row_end; ++row)
        {
            for(int idx = A->row_ptr[row]; idx < A->row_ptr[row+1]; ++idx)
            {
                int colid = A->col_ind[idx];
                if(usedBlock[colid / A_BLK->c] == 0)
                {
                    usedBlock[colid / A_BLK->c] = 1;
                    ++numblocks;
                } 
            }
        }
    }

    A_BLK->nblockrows = nblockrows;
    A_BLK->numblocks = numblocks;
    free(usedBlock);

    A_BLK->val = (FLOAT*)malloc(numblocks*A_BLK->r*A_BLK->c*sizeof(FLOAT));
    A_BLK->col_ind = (int*)malloc(numblocks*sizeof(int));
    A_BLK->row_ptr = (int*)malloc((nblockrows+1)*sizeof(int));
}

void freeBCSR(BCSR *A_BLK)
{
    free(A_BLK->val);
    free(A_BLK->col_ind);
    free(A_BLK->row_ptr);
}

void prepareSpmvBCSR(BCSR *a_blk, CSR *a)
{
    size_t nblockrows = a_blk->nblockrows;
    size_t numblocks = a_blk->numblocks;
    size_t maxBlocksPerRow = (a->nrows + a_blk->c - 1) / a_blk->c;

    // int *eleinblock = (int*)malloc(r*c*maxBlocksPerRow*sizeof(int));
    int *blockid= (int*)malloc(maxBlocksPerRow*sizeof(int));
    int *usedBlock = (int*)malloc(maxBlocksPerRow*sizeof(int));

    memset(a_blk->val, 0, a_blk->r * a_blk->c * numblocks * sizeof(FLOAT));
    numblocks = 0;


    for(size_t blockrow = 0; blockrow < nblockrows; ++blockrow)
    {
        size_t row_start = blockrow * a_blk->r;
        size_t row_end = MIN((blockrow+1) * a_blk->r, a->nrows);

        memset(usedBlock, 0, maxBlocksPerRow*sizeof(int));
        for(size_t i = 0; i < maxBlocksPerRow; ++i) blockid[i] = -1;

        a_blk->row_ptr[blockrow] = numblocks;

        //first loop: fill the usedBlock array
        for(size_t row = row_start; row < row_end; ++row)
        {
            for(int idx = a->row_ptr[row]; idx < a->row_ptr[row+1]; ++idx)
            {
                int colid = a->col_ind[idx];
                if(usedBlock[colid / a_blk->c] == 0)
                {
                    usedBlock[colid / a_blk->c] = 1;
                    ++numblocks;
                }
            }
        }

        //compute block id in this row
        size_t idx = 0;
        for(size_t i = 0; i < maxBlocksPerRow; ++i)
        {
            if(usedBlock[i] == 1)
            {
                blockid[i] = idx;
                a_blk->col_ind[idx + a_blk->row_ptr[blockrow]] = i * a_blk->c;
                ++idx;
            }
        }

        //second loop: fill the val array in column major
        for(size_t row = row_start; row < row_end; ++row)
        {
            for(int idx = a->row_ptr[row]; idx < a->row_ptr[row+1]; ++idx)
            {
                int colid = a->col_ind[idx];
                int bid = blockid[colid / a_blk->c] + a_blk->row_ptr[blockrow];
                FLOAT *this_block = a_blk->val + bid * a_blk->r * a_blk->c;
                this_block[(colid % a_blk->c) * a_blk->r + (row % a_blk->r)] = a->val[idx];
            }
        }
    }

    a_blk->row_ptr[nblockrows] = numblocks;
}