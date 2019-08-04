#include <stdio.h>
#include <string.h>
#include <time.h>
#include "spmv.h"

#define RESOLUTION 100000000
#define PROB_AMP 1.01

int generate_number(double probability)
{
    int tmp = rand() % RESOLUTION;
    double tmpf = (double)tmp / RESOLUTION;

    if(tmpf < probability) return 1;
    else return 0;
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        printf("usage: %s n density continuity\n", argv[0]);
        return -1;
    }

    int n, density, continuity;
    size_t nrows, num_elements;
    size_t i,j,k,idx;
    double probability;

    n = atoi(argv[1]);
    density = atoi(argv[2]);
    continuity = atoi(argv[3]);

    nrows = n * n;
    num_elements = nrows * density;
    probability = (double)(nrows*density) / (nrows*nrows*continuity) * PROB_AMP;

    printf("nrows = %lu, num_elements = %lu, probability = %f\n", nrows, num_elements, probability);
    
    CSR *a;
    a = (CSR *)malloc(sizeof(CSR));
    a->nrows = nrows;
    a->num_elements = num_elements;
    allocateCSR(a);

    j = 0;
    idx = 0;

    srand(time(NULL));

    for(i = 0; i < nrows; ++i)
    {
        if(idx >= num_elements) break;

        int flag = 1;

        for(j = 0; j < nrows; ++j)
        {
            if(idx >= num_elements) break;

            if(!generate_number(probability)) continue;

            for(k = 0; k < continuity; ++k)
            {
                if(j+k >= nrows || idx >= num_elements) break;

                a->val[idx] = rand() / (double)RAND_MAX;
                a->col_ind[idx] = j+k;

                // printf("i = %lu, j = %lu, idx = %lu, val = %f\n", i, j+k, idx, a->val[idx]);

                if(flag)
                {
                    a->row_ptr[i] = idx;
                    flag = 0;
                }

                ++idx;
            }

            j = j + continuity - 1;
        }

        if(flag)
        {
            a->row_ptr[i] = idx;
        }
        
        if(i % 1000 == 0) printf("step %lu / %lu ... elements %lu / %lu\n", i / 1000, nrows / 1000, idx, num_elements);
    }
    for(; i <= nrows; ++i) a->row_ptr[i] = num_elements;

    // for(size_t row = 0; row < 4; ++row)
    // {
    //     for(int idx = a->row_ptr[row]; idx < a->row_ptr[row+1]; ++idx)
    //     {
    //         printf("row = %lu, idx = %lu, col = %lu, val = %f\n", row, idx, a->col_ind[idx], a->val[idx]);
    //     }
    // }

    if(idx < num_elements)
    {
        printf("the number of elements is not enough: %lu / %lu, please run it again\n", idx, num_elements);
        freeCSR(a); 
        return -1;
    }

    char filename[1024];

#ifdef SP
    sprintf(filename,"../data/sparseMatrix_fp32_%d_%d_%d.bin", n, density, continuity);
#else
    sprintf(filename,"../data/sparseMatrix_fp64_%d_%d_%d.bin", n, density, continuity);
#endif
    FILE *fp = fopen(filename,"wb");
    fwrite(&n, sizeof(int), 1, fp);
    fwrite(&num_elements, sizeof(size_t), 1, fp);
    fwrite(a->val, sizeof(FLOAT), num_elements, fp);
    fwrite(a->col_ind, sizeof(int), num_elements, fp);
    fwrite(a->row_ptr, sizeof(int), nrows + 1, fp);
    fclose(fp);

    printf("init sparse matrix done\n");

    freeCSR(a);
    free(a);

    return 0;
}