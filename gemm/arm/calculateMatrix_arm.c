#include <arm_neon.h>
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
      float32x4_t va, vb, vc;
      int l;
      va = vdupq_n_f32(a[i*k+j]);
      for(l = 0; l <= m - 4; l += 4)
      {
        vb = vld1q_f32(b + j * m + l);
        vc = vld1q_f32(c + i * m + l);
        vc = vmlaq_f32(vc, va, vb);
        vst1q_f32(c + i * m + l, vc);
      }
      for(; l < m; ++l)
      {
        c[i * m + l] += a[i * k + j] * b[j * m + l];
      }
    }
#else
#pragma omp parallel for collapse(2)
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < k; ++j) {
      float64x2_t va, vb, vc;
      int l;
      va = vdupq_n_f64(a[i*k+j]);
      for(l = 0; l <= m - 2; l += 2)
      {
        vb = vld1q_f64(b + j * m + l);
        vc = vld1q_f64(c + i * m + l);
        vc = vmlaq_f64(vc, va, vb);
        vst1q_f64(c + i * m + l, vc);
      }
      for(; l < m; ++l)
      {
        c[i * m + l] += a[i * k + j] * b[j * m + l];
      } 
    }
#endif
}