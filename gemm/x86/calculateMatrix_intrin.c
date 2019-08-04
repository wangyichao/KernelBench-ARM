#include <immintrin.h>
#ifdef SP
#define FLOAT float
#else
#define FLOAT double
#endif
void calculateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m) {
#ifdef SP
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) {
      __m256 avxA;
      __m256 avxB;
      __m256 avxC;
      avxA = _mm256_set1_ps(a[i * k + j]);
      for (int l = 0; l < m; l += 8) {
        avxB = _mm256_load_ps(b + j * m + l);
        avxC = _mm256_load_ps(c + i * m + l);
        avxC = _mm256_fmadd_ps(avxA, avxB, avxC);
        _mm256_store_ps(c + i * m + l, avxC);
      }
    }
#else
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) {
      __m256d avxA;
      __m256d avxB;
      __m256d avxC;
      avxA = _mm256_set1_pd(a[i * k + j]);
      for (int l = 0; l < m; l += 4) {
        avxB = _mm256_load_pd(b + j * m + l);
        avxC = _mm256_load_pd(c + i * m + l);
        avxC = _mm256_fmadd_pd(avxA, avxB, avxC);
        _mm256_store_pd(c + i * m + l, avxC);
      }
    }
#endif
}