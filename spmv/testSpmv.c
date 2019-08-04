#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "spmv.h"

#ifdef DEBUG
#define LOOPTIME 1
#else
#define LOOPTIME 500
#endif

int main(int argc, char **argv)
{
    int n, r, c;
    size_t nrows, num_elements;
    FLOAT *x, *y;

    if(argc != 5)
    {
        printf("[spmv usage]: %s sparseMatrix algo r c\n", argv[0]);
        printf("algo explanation: 1 for CSR, 2 for BCSR (For CSR, set r = c = 1)\n");
        return -1;
    }

    SPMV_ALGO algo = atoi(argv[2]);

    if(algo == ALGO_BCSR)
    {
        r = atoi(argv[3]);
        c = atoi(argv[4]);
    }

    //read the sparse matrix
#ifdef SUITESPARSE
    FILE *fp = fopen(argv[1],"r");
    char tmp;
    char buff[2048];
    int ncols;
    while((tmp = fgetc(fp)) == '%')
    {
        ungetc(tmp, fp);
        fgets(buff, 2048, fp);
    }
    ungetc(tmp, fp);
    fscanf(fp, "%lu %d %lu\n", &nrows, &ncols, &num_elements);
    printf("nrows = %lu, ncols = %d, num_elements = %lu\n", nrows, ncols, num_elements);

    CSR *a = (CSR*)malloc(sizeof(CSR));
    a->nrows = nrows;
    a->num_elements = num_elements;
    allocateCSR(a);

    int rowid = 0;
    a->row_ptr[0] = 0;
    for(size_t idx = 0; idx < num_elements; ++idx)
    {
        int col, row;
        double value;
        fscanf(fp, "%d %d %lf\n", &col, &row, &value);

        if(rowid != row - 1)
        {
            ++rowid;
            a->row_ptr[rowid] = idx;
        }
        a->col_ind[idx] = col - 1;
        a->val[idx] = rand() / (FLOAT)RAND_MAX;
    }
    a->row_ptr[nrows] = num_elements;
    fclose(fp);
#else
    FILE *fp = fopen(argv[1],"rb");
    fread(&n, sizeof(int), 1, fp);
    fread(&num_elements, sizeof(size_t), 1, fp);
    nrows = n * n;

    CSR *a = (CSR*)malloc(sizeof(CSR));
    a->nrows = nrows;
    a->num_elements = num_elements;
    allocateCSR(a);

    fread(a->val, sizeof(FLOAT), num_elements, fp);
    fread(a->col_ind, sizeof(int), num_elements, fp);
    fread(a->row_ptr, sizeof(int), num_elements, fp);
    fclose(fp);
#endif

    printf("finish read sparse matrix\n");

    //init x and y
    x = (FLOAT*)malloc(nrows*sizeof(FLOAT));
    y = (FLOAT*)malloc(nrows*sizeof(FLOAT));

    for(size_t i = 0; i < nrows; ++i) x[i] = i;
    for(size_t i = 0; i < nrows; ++i) y[i] = i;

    //reconstruct the structure to BCSR
    BCSR *a_blk;
    FLOAT *y_local;
    if(algo == ALGO_BCSR)
    {
        a_blk = (BCSR*)malloc(sizeof(BCSR));
        a_blk->r = r;
        a_blk->c = c;
        a_blk->nrows = a->nrows;
        allocateBCSR(a_blk, a);
        prepareSpmvBCSR(a_blk, a);
        printf("[spmv bcsr]: finish transform\n");
        y_local = (FLOAT*)malloc(r*sizeof(FLOAT));
    }

    struct timeval beg, end;
    gettimeofday(&beg, NULL);
    if(algo == ALGO_CSR)
    {
        for(int loop = 0; loop < LOOPTIME; ++loop)
        {
            calculateSpmvCSR(y, a, x);
        }
    }
    else if(algo == ALGO_BCSR)
    {
        for(int loop = 0; loop < LOOPTIME; ++loop)
        {
            calculateSpmvBCSR(y, a_blk, x, y_local);
        }
    }
    else
    {
        printf("[spmv error]: unrecognized algorithm\n");
        return -1;
    }
    gettimeofday(&end, NULL);

    double t = end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
    printf("nrows = %lu, time = %fs\n", nrows, t);

#ifdef DEBUG
    //calculate correct answer
    FLOAT *y_standard = (FLOAT *)malloc(nrows*sizeof(FLOAT));
    for(size_t i = 0; i < nrows; ++i) y_standard[i] = i;

    for(size_t i = 0; i < nrows; ++i)
    {
        FLOAT sum = 0;
        for(int idx = a->row_ptr[i]; idx < a->row_ptr[i+1]; ++idx)
        {
            sum += a->val[idx] * x[a->col_ind[idx]];
        }
        y_standard[i] += sum;
    }

    //calculate error
    double error = 0;
    for(size_t i = 0; i < nrows; ++i)
    {
        error += abs(y_standard[i] - y[i]);
    }
    printf("error = %f\n",error);
    free(y_standard);
#endif

    freeCSR(a);
    free(a);
    free(x);
    free(y);

    if(algo == ALGO_BCSR)
    {
        freeBCSR(a_blk);
        free(a_blk);
        free(y_local);
    }

    return 0;
}