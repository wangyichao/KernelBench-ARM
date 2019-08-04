#include "gemm.h"

#ifndef GEMM_USE_ARM
#include <immintrin.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#ifdef SP
#define FLOAT float
#else
#define FLOAT double
#endif

#ifdef GEMM_USE_ARM
#define _mm_malloc(SIZE, ALIGN) (malloc(SIZE))
#endif

void initMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m);
void calculateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m, FLOAT *buffer);
void validateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m);

int main(int argc, char **argv) {
  double t;
  struct timeval beg, end;
  FLOAT *a, *b, *c;
  int n = atoi(argv[1]), k = atoi(argv[2]), m = atoi(argv[3]);
  a = _mm_malloc(sizeof(FLOAT) * n * k, 64);
  b = _mm_malloc(sizeof(FLOAT) * k * m, 64);
  c = _mm_malloc(sizeof(FLOAT) * n * m, 64);
  initMatrix(a, b, c, n, k, m);

  //caculate the sizeof tmp buffer;
  int m_pad = GEMM_PADDING(n, 8);
  int k_pad = GEMM_PADDING(k, 4);
#ifdef SP
  int n_pad = GEMM_PADDING(m, 12);
#else
  int n_pad = GEMM_PADDING(m, 6);
#endif

  int tmpbuffersize = m_pad * k_pad + k_pad * n_pad;
  printf("temp buffer size = %d\n", tmpbuffersize);
  FLOAT *buffer = (FLOAT*)malloc(tmpbuffersize * sizeof(FLOAT));

  gettimeofday(&beg, NULL);
  calculateMatrix(a, b, c, n, k, m, buffer);
  gettimeofday(&end, NULL);
  // validateMatrix(a, b, c, n, k, m);

  t = end.tv_sec - beg.tv_sec;
  t += (end.tv_usec - beg.tv_usec) / 1000000.0;

  free(buffer);

  printf("Time is %lf\n", t);
}

void initMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) a[i * k + j] = rand() / RAND_MAX;
  for (int i = 0; i < k; ++i)
    for (int j = 0; j < m; ++j) b[i * m + j] = rand() / RAND_MAX;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) c[i * m + j] = 0;
}

void validateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m) {
  FLOAT *d = malloc(sizeof(FLOAT) * n * m);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j)
      for (int l = 0; l < m; ++l) d[i * m + l] += a[i * k + j] * b[j * m + l];
  FLOAT error = 0.0;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j) error += c[i * m + j] - d[i * m + j];
  printf("Error is %g\n", error/(m*n));
}