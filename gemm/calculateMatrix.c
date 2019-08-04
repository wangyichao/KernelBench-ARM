#ifdef SP
#define FLOAT float
#else
#define FLOAT double
#endif
void calculateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int n, int k, int m) {
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) {
      for (int l = 0; l < m; l++) {
        c[i * m + l] += a[i * k + j] * b[j * m + l];
      }
    }
}