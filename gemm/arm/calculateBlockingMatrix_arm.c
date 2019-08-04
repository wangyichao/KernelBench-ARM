#include <arm_neon.h>
#include <stdio.h>
#include <sys/time.h>
#include "../gemm.h"

#ifdef SP
#define FLOAT float
#else
#define FLOAT double
#endif

// Single: M % 8 == 0, K % 4 == 0, N % 12 == 0
// Double: M % 8 == 0, K % 4 == 0, N % 6 == 0

#ifdef SP
static void trans_a(float *a_trans, float *a, int K, int m_blk, int k_blk)
{
  int i, j;

  float *a_ptr;

  float32x4_t va[8];

  for(j = 0; j <= k_blk - 4; j += 4)
  {
    for(i = 0; i <= m_blk - 8; i += 8)
    {
      a_ptr = a + i * K + j;
      va[0] = vld1q_f32(a_ptr + 0 * K);
      va[1] = vld1q_f32(a_ptr + 1 * K);
      va[2] = vld1q_f32(a_ptr + 2 * K);
      va[3] = vld1q_f32(a_ptr + 3 * K);
      va[4] = vld1q_f32(a_ptr + 4 * K);
      va[5] = vld1q_f32(a_ptr + 5 * K);
      va[6] = vld1q_f32(a_ptr + 6 * K);
      va[7] = vld1q_f32(a_ptr + 7 * K);

      TRANSPOSE_FLOAT_4X4(va[0], va[1], va[2], va[3]);
      TRANSPOSE_FLOAT_4X4(va[4], va[5], va[6], va[7]);

      vst1q_f32(a_trans + 0, va[0]);
      vst1q_f32(a_trans + 4, va[4]);
      vst1q_f32(a_trans + 8, va[1]);
      vst1q_f32(a_trans + 12, va[5]);
      vst1q_f32(a_trans + 16, va[2]);
      vst1q_f32(a_trans + 20, va[6]);
      vst1q_f32(a_trans + 24, va[3]);
      vst1q_f32(a_trans + 28, va[7]);

      a_trans += 8 * 4;
    }
  }
}
#else
static void trans_a(double *a_trans, double *a, int K, int m_blk, int k_blk)
{
  int i, j;

  double *a_ptr;

  float64x2_t va[16];

  for(j = 0; j <= k_blk - 4; j += 4)
  {
    for(i = 0; i <= m_blk - 8; i += 8)
    {
      a_ptr = a + i * K + j;
      va[0] = vld1q_f64(a_ptr + 0 * K + 0);
      va[1] = vld1q_f64(a_ptr + 0 * K + 2);
      va[2] = vld1q_f64(a_ptr + 1 * K + 0);
      va[3] = vld1q_f64(a_ptr + 1 * K + 2);
      va[4] = vld1q_f64(a_ptr + 2 * K + 0);
      va[5] = vld1q_f64(a_ptr + 2 * K + 2);
      va[6] = vld1q_f64(a_ptr + 3 * K + 0);
      va[7] = vld1q_f64(a_ptr + 3 * K + 2);
      va[8] = vld1q_f64(a_ptr + 4 * K + 0);
      va[9] = vld1q_f64(a_ptr + 4 * K + 2);
      va[10] = vld1q_f64(a_ptr + 5 * K + 0);
      va[11] = vld1q_f64(a_ptr + 5 * K + 2);
      va[12] = vld1q_f64(a_ptr + 6 * K + 0);
      va[13] = vld1q_f64(a_ptr + 6 * K + 2);
      va[14] = vld1q_f64(a_ptr + 7 * K + 0);
      va[15] = vld1q_f64(a_ptr + 7 * K + 2);

      TRANSPOSE_DOUBLE_2x2(va[0], va[2]);
      TRANSPOSE_DOUBLE_2x2(va[1], va[3]);
      TRANSPOSE_DOUBLE_2x2(va[4], va[6]);
      TRANSPOSE_DOUBLE_2x2(va[5], va[7]);
      TRANSPOSE_DOUBLE_2x2(va[8], va[12]);
      TRANSPOSE_DOUBLE_2x2(va[9], va[13]);
      TRANSPOSE_DOUBLE_2x2(va[10], va[14]);
      TRANSPOSE_DOUBLE_2x2(va[11], va[15]);

      vst1q_f64(a_trans + 0, va[0]);
      vst1q_f64(a_trans + 2, va[4]);
      vst1q_f64(a_trans + 4, va[8]);
      vst1q_f64(a_trans + 6, va[12]);
      vst1q_f64(a_trans + 8, va[2]);
      vst1q_f64(a_trans + 10, va[6]);
      vst1q_f64(a_trans + 12, va[10]);
      vst1q_f64(a_trans + 14, va[14]);
      vst1q_f64(a_trans + 16, va[1]);
      vst1q_f64(a_trans + 18, va[5]);
      vst1q_f64(a_trans + 20, va[9]);
      vst1q_f64(a_trans + 22, va[13]);
      vst1q_f64(a_trans + 24, va[3]);
      vst1q_f64(a_trans + 26, va[7]);
      vst1q_f64(a_trans + 28, va[11]);
      vst1q_f64(a_trans + 30, va[15]);

      a_trans += 8 * 4;
    }
  }
}
#endif

void load_a_trans(FLOAT *a_trans, FLOAT *a, int M, int K)
{
  int m_pad = GEMM_PADDING(M, 8);
  int k_pad = GEMM_PADDING(K, 4);

  int i, j;

  for(j = 0; j <= k_pad - K_BLK; j += K_BLK)
  {
    for(i = 0; i <= m_pad - M_BLK; i += M_BLK)
    {
      trans_a(a_trans, a + i * K + j, K, M_BLK, K_BLK);
      a_trans += M_BLK * K_BLK;
    }
    if(i < m_pad)
    {
      trans_a(a_trans, a + i * K + j, K, m_pad - i, K_BLK);
      a_trans += (m_pad - i) * K_BLK;
    }
  }
  if(j < k_pad)
  {
    for(i = 0; i <= m_pad - M_BLK; i += M_BLK)
    {
      trans_a(a_trans, a + i * K + j, K, M_BLK, k_pad - j);
      a_trans += M_BLK * (k_pad - j);
    }
    if(i < m_pad)
    {
      trans_a(a_trans, a + i * K + j, K, m_pad - i, k_pad - j);
    } 
  }
}

#ifdef SP
static void trans_b(float *b_trans, float *b, int N, int k_blk, int n_blk)
{
    int i, j;

    float *b_ptr;
    float32x4_t vb[12];

    for(i = 0; i <= k_blk - 4; i += 4)
    {
      for(j = 0; j <= n_blk - 12; j += 12)
      {
        b_ptr = b + i * N + j;
        vb[0] = vld1q_f32(b_ptr + 0 * N + 0);
        vb[1] = vld1q_f32(b_ptr + 0 * N + 4);
        vb[2] = vld1q_f32(b_ptr + 0 * N + 8);
        vb[3] = vld1q_f32(b_ptr + 1 * N + 0);
        vb[4] = vld1q_f32(b_ptr + 1 * N + 4);
        vb[5] = vld1q_f32(b_ptr + 1 * N + 8);
        vb[6] = vld1q_f32(b_ptr + 2 * N + 0);
        vb[7] = vld1q_f32(b_ptr + 2 * N + 4);
        vb[8] = vld1q_f32(b_ptr + 2 * N + 8);
        vb[9] = vld1q_f32(b_ptr + 3 * N + 0);
        vb[10] = vld1q_f32(b_ptr + 3 * N + 4);
        vb[11] = vld1q_f32(b_ptr + 3 * N + 8);

        vst1q_f32(b_trans + 0, vb[0]);
        vst1q_f32(b_trans + 4, vb[1]);
        vst1q_f32(b_trans + 8, vb[2]);
        vst1q_f32(b_trans + 12, vb[3]);
        vst1q_f32(b_trans + 16, vb[4]);
        vst1q_f32(b_trans + 20, vb[5]);
        vst1q_f32(b_trans + 24, vb[6]);
        vst1q_f32(b_trans + 28, vb[7]);
        vst1q_f32(b_trans + 32, vb[8]);
        vst1q_f32(b_trans + 36, vb[9]);
        vst1q_f32(b_trans + 40, vb[10]);
        vst1q_f32(b_trans + 44, vb[11]);

        b_trans += 4 * 12;
      }
    }
}
#else
static void trans_b(double *b_trans, double *b, int N, int k_blk, int n_blk)
{
    int i, j;

    double *b_ptr;
    float64x2_t vb[12];

    for(i = 0; i <= k_blk - 4; i += 4)
    {
      for(j = 0; j <= n_blk - 6; j += 6)
      {
        b_ptr = b + i * N + j;
        vb[0] = vld1q_f64(b_ptr + 0 * N + 0);
        vb[1] = vld1q_f64(b_ptr + 0 * N + 2);
        vb[2] = vld1q_f64(b_ptr + 0 * N + 4);
        vb[3] = vld1q_f64(b_ptr + 1 * N + 0);
        vb[4] = vld1q_f64(b_ptr + 1 * N + 2);
        vb[5] = vld1q_f64(b_ptr + 1 * N + 4);
        vb[6] = vld1q_f64(b_ptr + 2 * N + 0);
        vb[7] = vld1q_f64(b_ptr + 2 * N + 2);
        vb[8] = vld1q_f64(b_ptr + 2 * N + 4);
        vb[9] = vld1q_f64(b_ptr + 3 * N + 0);
        vb[10] = vld1q_f64(b_ptr + 3 * N + 2);
        vb[11] = vld1q_f64(b_ptr + 3 * N + 4);

        vst1q_f64(b_trans + 0, vb[0]);
        vst1q_f64(b_trans + 2, vb[1]);
        vst1q_f64(b_trans + 4, vb[2]);
        vst1q_f64(b_trans + 6, vb[3]);
        vst1q_f64(b_trans + 8, vb[4]);
        vst1q_f64(b_trans + 10, vb[5]);
        vst1q_f64(b_trans + 12, vb[6]);
        vst1q_f64(b_trans + 14, vb[7]);
        vst1q_f64(b_trans + 16, vb[8]);
        vst1q_f64(b_trans + 18, vb[9]);
        vst1q_f64(b_trans + 20, vb[10]);
        vst1q_f64(b_trans + 22, vb[11]);

        b_trans += 4 * 6;
      }
    }
}
#endif

void load_b_trans(FLOAT *b_trans, FLOAT *b, int K, int N)
{
  int k_pad = GEMM_PADDING(K, 4);
#ifdef SP
  int n_pad = GEMM_PADDING(N, 12);
#else
  int n_pad = GEMM_PADDING(N, 6);
#endif

  int i, j;

  for(i = 0; i <= k_pad - K_BLK; i += K_BLK)
  {
    for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
    {
      trans_b(b_trans, b + i * N + j, N, K_BLK, N_BLK);
      b_trans += K_BLK * N_BLK;
    }
    if(j < n_pad)
    {
      trans_b(b_trans, b + i * N + j, N, K_BLK, n_pad - j);
      b_trans += K_BLK * (n_pad - j);
    }
  }
  if(i < k_pad)
  {
    for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
    {
      trans_b(b_trans, b + i * N + j, N, k_pad - i, N_BLK);
      b_trans += (k_pad - i) * N_BLK;
    }
    if(j < n_pad)
    {
      trans_b(b_trans, b + i * N + j, N, k_pad - i, n_pad - j);
    } 
  }
}

#ifdef SP
static void small_gemm_kernel(float *c, float *a, float *b, int m_blk, int k_blk, int n_blk, int N)
{
  int i, j, k;

  float *a_trans = a;
  float *b_trans_d = b;
  float *b_trans, *c_ptr;

  float32x4_t va[2], vb[6], vc[24];

  for(k = 0; k <= k_blk - 4; k += 4)
  {
    for(i = 0; i <= m_blk - 8; i += 8)
    {
      b_trans = b_trans_d;
      for(j = 0; j <= n_blk - 12; j += 12)
      {
        c_ptr = c + i * N + j;

        va[0] = vld1q_f32(a_trans + 0);
        va[1] = vld1q_f32(a_trans + 4);

        vb[0] = vld1q_f32(b_trans + 0);
        vb[1] = vld1q_f32(b_trans + 4);
        vb[2] = vld1q_f32(b_trans + 8);
        vb[3] = vld1q_f32(b_trans + 12);
        vb[4] = vld1q_f32(b_trans + 16);
        vb[5] = vld1q_f32(b_trans + 20);
        
        vc[0] = vld1q_f32(c_ptr + 0 * N + 0);
        vc[1] = vld1q_f32(c_ptr + 0 * N + 4);
        vc[2] = vld1q_f32(c_ptr + 0 * N + 8);
        vc[0] = vfmaq_laneq_f32(vc[0], vb[0], va[0], 0);
        vc[3] = vld1q_f32(c_ptr + 1 * N + 0);
        vc[1] = vfmaq_laneq_f32(vc[1], vb[1], va[0], 0);
        vc[4] = vld1q_f32(c_ptr + 1 * N + 4);
        vc[2] = vfmaq_laneq_f32(vc[2], vb[2], va[0], 0);
        vc[5] = vld1q_f32(c_ptr + 1 * N + 8);
        vc[3] = vfmaq_laneq_f32(vc[3], vb[0], va[0], 1);
        vc[6] = vld1q_f32(c_ptr + 2 * N + 0);
        vc[4] = vfmaq_laneq_f32(vc[4], vb[1], va[0], 1);
        vc[7] = vld1q_f32(c_ptr + 2 * N + 4);
        vc[5] = vfmaq_laneq_f32(vc[5], vb[2], va[0], 1);
        vc[8] = vld1q_f32(c_ptr + 2 * N + 8);
        vc[6] = vfmaq_laneq_f32(vc[6], vb[0], va[0], 2);
        vc[9] = vld1q_f32(c_ptr + 3 * N + 0);
        vc[7] = vfmaq_laneq_f32(vc[7], vb[1], va[0], 2);
        vc[10] = vld1q_f32(c_ptr + 3 * N + 4);
        vc[8] = vfmaq_laneq_f32(vc[8], vb[2], va[0], 2);
        vc[11] = vld1q_f32(c_ptr + 3 * N + 8);
        vc[9] = vfmaq_laneq_f32(vc[9], vb[0], va[0], 3);
        vc[12] = vld1q_f32(c_ptr + 4 * N + 0);
        vc[10] = vfmaq_laneq_f32(vc[10], vb[1], va[0], 3);
        vc[13] = vld1q_f32(c_ptr + 4 * N + 4);
        vc[11] = vfmaq_laneq_f32(vc[11], vb[2], va[0], 3);
        vc[14] = vld1q_f32(c_ptr + 4 * N + 8);
        vc[12] = vfmaq_laneq_f32(vc[12], vb[0], va[1], 0);
        vc[15] = vld1q_f32(c_ptr + 5 * N + 0);
        vc[13] = vfmaq_laneq_f32(vc[13], vb[1], va[1], 0);
        va[0] = vld1q_f32(a_trans + 8);
        vc[16] = vld1q_f32(c_ptr + 5 * N + 4);
        vc[14] = vfmaq_laneq_f32(vc[14], vb[2], va[1], 0);
        vc[17] = vld1q_f32(c_ptr + 5 * N + 8);
        vc[15] = vfmaq_laneq_f32(vc[15], vb[0], va[1], 1);
        vc[18] = vld1q_f32(c_ptr + 6 * N + 0);
        vc[16] = vfmaq_laneq_f32(vc[16], vb[1], va[1], 1);
        vc[19] = vld1q_f32(c_ptr + 6 * N + 4);
        vc[17] = vfmaq_laneq_f32(vc[17], vb[2], va[1], 1);
        vc[20] = vld1q_f32(c_ptr + 6 * N + 8);
        vc[18] = vfmaq_laneq_f32(vc[18], vb[0], va[1], 2);
        vc[21] = vld1q_f32(c_ptr + 7 * N + 0);
        vc[19] = vfmaq_laneq_f32(vc[19], vb[1], va[1], 2);
        vc[22] = vld1q_f32(c_ptr + 7 * N + 4);
        vc[20] = vfmaq_laneq_f32(vc[20], vb[2], va[1], 2);
        vc[23] = vld1q_f32(c_ptr + 7 * N + 8);
        vc[21] = vfmaq_laneq_f32(vc[21], vb[0], va[1], 3);
        vc[22] = vfmaq_laneq_f32(vc[22], vb[1], va[1], 3);
        vc[23] = vfmaq_laneq_f32(vc[23], vb[2], va[1], 3);

        vc[0] = vfmaq_laneq_f32(vc[0], vb[3], va[0], 0);
        vc[1] = vfmaq_laneq_f32(vc[1], vb[4], va[0], 0);
        vc[2] = vfmaq_laneq_f32(vc[2], vb[5], va[0], 0);
        va[1] = vld1q_f32(a_trans + 12);
        vc[3] = vfmaq_laneq_f32(vc[3], vb[3], va[0], 1);
        vc[4] = vfmaq_laneq_f32(vc[4], vb[4], va[0], 1);
        vc[5] = vfmaq_laneq_f32(vc[5], vb[5], va[0], 1);
        vb[0] = vld1q_f32(b_trans + 24);
        vc[6] = vfmaq_laneq_f32(vc[6], vb[3], va[0], 2);
        vc[7] = vfmaq_laneq_f32(vc[7], vb[4], va[0], 2);
        vc[8] = vfmaq_laneq_f32(vc[8], vb[5], va[0], 2);
        vb[1] = vld1q_f32(b_trans + 28);
        vc[9] = vfmaq_laneq_f32(vc[9], vb[3], va[0], 3);
        vc[10] = vfmaq_laneq_f32(vc[10], vb[4], va[0], 3);
        vc[11] = vfmaq_laneq_f32(vc[11], vb[5], va[0], 3);
        vb[2] = vld1q_f32(b_trans + 32);
        vc[12] = vfmaq_laneq_f32(vc[12], vb[3], va[1], 0);
        vc[13] = vfmaq_laneq_f32(vc[13], vb[4], va[1], 0);
        vc[14] = vfmaq_laneq_f32(vc[14], vb[5], va[1], 0);
        vb[3] = vld1q_f32(b_trans + 36);
        vc[15] = vfmaq_laneq_f32(vc[15], vb[3], va[1], 1);
        va[0] = vld1q_f32(a_trans + 16);
        vc[16] = vfmaq_laneq_f32(vc[16], vb[4], va[1], 1);
        vc[17] = vfmaq_laneq_f32(vc[17], vb[5], va[1], 1);
        vb[4] = vld1q_f32(b_trans + 40);
        vc[18] = vfmaq_laneq_f32(vc[18], vb[3], va[1], 2);
        vc[19] = vfmaq_laneq_f32(vc[19], vb[4], va[1], 2);
        vc[20] = vfmaq_laneq_f32(vc[20], vb[5], va[1], 2);
        vb[5] = vld1q_f32(b_trans + 44);
        vc[21] = vfmaq_laneq_f32(vc[21], vb[3], va[1], 3);
        vc[22] = vfmaq_laneq_f32(vc[22], vb[4], va[1], 3);
        vc[23] = vfmaq_laneq_f32(vc[23], vb[5], va[1], 3);

        
        vc[0] = vfmaq_laneq_f32(vc[0], vb[0], va[0], 0);
        vc[1] = vfmaq_laneq_f32(vc[1], vb[1], va[0], 0);
        vc[2] = vfmaq_laneq_f32(vc[2], vb[2], va[0], 0);
        va[1] = vld1q_f32(a_trans + 20);
        vc[3] = vfmaq_laneq_f32(vc[3], vb[0], va[0], 1);
        vc[4] = vfmaq_laneq_f32(vc[4], vb[1], va[0], 1);
        vc[5] = vfmaq_laneq_f32(vc[5], vb[2], va[0], 1);
        vc[6] = vfmaq_laneq_f32(vc[6], vb[0], va[0], 2);
        vc[7] = vfmaq_laneq_f32(vc[7], vb[1], va[0], 2);
        vc[8] = vfmaq_laneq_f32(vc[8], vb[2], va[0], 2);
        vc[9] = vfmaq_laneq_f32(vc[9], vb[0], va[0], 3);
        vc[10] = vfmaq_laneq_f32(vc[10], vb[1], va[0], 3);
        vc[11] = vfmaq_laneq_f32(vc[11], vb[2], va[0], 3);
        vc[12] = vfmaq_laneq_f32(vc[12], vb[0], va[1], 0);
        vc[13] = vfmaq_laneq_f32(vc[13], vb[1], va[1], 0);
        vc[14] = vfmaq_laneq_f32(vc[14], vb[2], va[1], 0);
        vc[15] = vfmaq_laneq_f32(vc[15], vb[0], va[1], 1);
        vc[16] = vfmaq_laneq_f32(vc[16], vb[1], va[1], 1);
        vc[17] = vfmaq_laneq_f32(vc[17], vb[2], va[1], 1);
        va[0] = vld1q_f32(a_trans + 24);
        vc[18] = vfmaq_laneq_f32(vc[18], vb[0], va[1], 2);
        vc[19] = vfmaq_laneq_f32(vc[19], vb[1], va[1], 2);
        vc[20] = vfmaq_laneq_f32(vc[20], vb[2], va[1], 2);
        vc[21] = vfmaq_laneq_f32(vc[21], vb[0], va[1], 3);
        vc[22] = vfmaq_laneq_f32(vc[22], vb[1], va[1], 3);
        vc[23] = vfmaq_laneq_f32(vc[23], vb[2], va[1], 3);
        
        vc[0] = vfmaq_laneq_f32(vc[0], vb[3], va[0], 0);
        vc[1] = vfmaq_laneq_f32(vc[1], vb[4], va[0], 0);
        vc[2] = vfmaq_laneq_f32(vc[2], vb[5], va[0], 0);
        va[1] = vld1q_f32(a_trans + 28);
        vc[3] = vfmaq_laneq_f32(vc[3], vb[3], va[0], 1);
        vst1q_f32(c_ptr + 0 * N + 0, vc[0]);
        vc[4] = vfmaq_laneq_f32(vc[4], vb[4], va[0], 1);
        vst1q_f32(c_ptr + 0 * N + 4, vc[1]);
        vc[5] = vfmaq_laneq_f32(vc[5], vb[5], va[0], 1);
        vst1q_f32(c_ptr + 0 * N + 8, vc[2]);
        vc[6] = vfmaq_laneq_f32(vc[6], vb[3], va[0], 2);
        vst1q_f32(c_ptr + 1 * N + 0, vc[3]);
        vc[7] = vfmaq_laneq_f32(vc[7], vb[4], va[0], 2);
        vst1q_f32(c_ptr + 1 * N + 4, vc[4]);
        vc[8] = vfmaq_laneq_f32(vc[8], vb[5], va[0], 2);
        vst1q_f32(c_ptr + 1 * N + 8, vc[5]);
        vc[9] = vfmaq_laneq_f32(vc[9], vb[3], va[0], 3);
        vst1q_f32(c_ptr + 2 * N + 0, vc[6]);
        vc[10] = vfmaq_laneq_f32(vc[10], vb[4], va[0], 3);
        vst1q_f32(c_ptr + 2 * N + 4, vc[7]);
        vc[11] = vfmaq_laneq_f32(vc[11], vb[5], va[0], 3);
        vst1q_f32(c_ptr + 2 * N + 8, vc[8]);
        vc[12] = vfmaq_laneq_f32(vc[12], vb[3], va[1], 0);
        vst1q_f32(c_ptr + 3 * N + 0, vc[9]);
        vc[13] = vfmaq_laneq_f32(vc[13], vb[4], va[1], 0);
        vst1q_f32(c_ptr + 3 * N + 4, vc[10]);
        vc[14] = vfmaq_laneq_f32(vc[14], vb[5], va[1], 0);
        vst1q_f32(c_ptr + 3 * N + 8, vc[11]);
        vc[15] = vfmaq_laneq_f32(vc[15], vb[3], va[1], 1);
        vst1q_f32(c_ptr + 4 * N + 0, vc[12]);
        vc[16] = vfmaq_laneq_f32(vc[16], vb[4], va[1], 1);
        vst1q_f32(c_ptr + 4 * N + 4, vc[13]);
        vc[17] = vfmaq_laneq_f32(vc[17], vb[5], va[1], 1);
        vst1q_f32(c_ptr + 4 * N + 8, vc[14]);
        vc[18] = vfmaq_laneq_f32(vc[18], vb[3], va[1], 2);
        vst1q_f32(c_ptr + 5 * N + 0, vc[15]);
        vc[19] = vfmaq_laneq_f32(vc[19], vb[4], va[1], 2);
        vst1q_f32(c_ptr + 5 * N + 4, vc[16]);
        vc[20] = vfmaq_laneq_f32(vc[20], vb[5], va[1], 2);
        vst1q_f32(c_ptr + 5 * N + 8, vc[17]);
        vc[21] = vfmaq_laneq_f32(vc[21], vb[3], va[1], 3);
        vst1q_f32(c_ptr + 6 * N + 0, vc[18]);
        vc[22] = vfmaq_laneq_f32(vc[22], vb[4], va[1], 3);
        vst1q_f32(c_ptr + 6 * N + 4, vc[19]);
        vc[23] = vfmaq_laneq_f32(vc[23], vb[5], va[1], 3);
        vst1q_f32(c_ptr + 6 * N + 8, vc[20]);
        vst1q_f32(c_ptr + 7 * N + 0, vc[21]);
        vst1q_f32(c_ptr + 7 * N + 4, vc[22]);
        vst1q_f32(c_ptr + 7 * N + 8, vc[23]);
        
        b_trans += 4 * 12;
      }
      a_trans += 8 * 4;
    }
    b_trans_d += 4 * n_blk;
  }
}
#else
static void small_gemm_kernel(double *c, double *a, double *b, int m_blk, int k_blk, int n_blk, int N)
{
  int i, j, k;

  double *a_trans = a;
  double *b_trans_d = b;
  double *b_trans, *c_ptr;

  float64x2_t va[2], vb[6], vc[24];

  for(k = 0; k <= k_blk - 4; k += 4)
  {
    for(i = 0; i <= m_blk - 8; i += 8)
    {
      b_trans = b_trans_d;
      for(j = 0; j <= n_blk - 6; j += 6)
      {
        c_ptr = c + i * N + j;

        va[0] = vld1q_f64(a_trans + 0);
        va[1] = vld1q_f64(a_trans + 2);

        vb[0] = vld1q_f64(b_trans + 0);
        vb[1] = vld1q_f64(b_trans + 2);
        vb[2] = vld1q_f64(b_trans + 4);
        vb[3] = vld1q_f64(b_trans + 6);
        vb[4] = vld1q_f64(b_trans + 8);
        vb[5] = vld1q_f64(b_trans + 10);
        
        vc[0] = vld1q_f64(c_ptr + 0 * N + 0);
        vc[1] = vld1q_f64(c_ptr + 0 * N + 2);
        vc[2] = vld1q_f64(c_ptr + 0 * N + 4);
        vc[0] = vfmaq_laneq_f64(vc[0], vb[0], va[0], 0);
        vc[3] = vld1q_f64(c_ptr + 1 * N + 0);
        vc[1] = vfmaq_laneq_f64(vc[1], vb[1], va[0], 0);
        vc[4] = vld1q_f64(c_ptr + 1 * N + 2);
        vc[2] = vfmaq_laneq_f64(vc[2], vb[2], va[0], 0);
        vc[5] = vld1q_f64(c_ptr + 1 * N + 4);
        vc[3] = vfmaq_laneq_f64(vc[3], vb[0], va[0], 1);
        vc[6] = vld1q_f64(c_ptr + 2 * N + 0);
        vc[4] = vfmaq_laneq_f64(vc[4], vb[1], va[0], 1);
        vc[7] = vld1q_f64(c_ptr + 2 * N + 2);
        vc[5] = vfmaq_laneq_f64(vc[5], vb[2], va[0], 1);
        vc[8] = vld1q_f64(c_ptr + 2 * N + 4);
        vc[6] = vfmaq_laneq_f64(vc[6], vb[0], va[1], 0);
        vc[9] = vld1q_f64(c_ptr + 3 * N + 0);
        vc[7] = vfmaq_laneq_f64(vc[7], vb[1], va[1], 0);
        vc[10] = vld1q_f64(c_ptr + 3 * N + 2);
        vc[8] = vfmaq_laneq_f64(vc[8], vb[2], va[1], 0);
        va[0] = vld1q_f64(a_trans + 4);
        vc[11] = vld1q_f64(c_ptr + 3 * N + 4);
        vc[9] = vfmaq_laneq_f64(vc[9], vb[0], va[1], 1);
        vc[12] = vld1q_f64(c_ptr + 4 * N + 0);
        vc[10] = vfmaq_laneq_f64(vc[10], vb[1], va[1], 1);
        vc[13] = vld1q_f64(c_ptr + 4 * N + 2);
        vc[11] = vfmaq_laneq_f64(vc[11], vb[2], va[1], 1);
        vc[14] = vld1q_f64(c_ptr + 4 * N + 4);
        vc[12] = vfmaq_laneq_f64(vc[12], vb[0], va[0], 0);
        vc[15] = vld1q_f64(c_ptr + 5 * N + 0);
        vc[13] = vfmaq_laneq_f64(vc[13], vb[1], va[0], 0);
        vc[16] = vld1q_f64(c_ptr + 5 * N + 2);
        vc[14] = vfmaq_laneq_f64(vc[14], vb[2], va[0], 0);
        vc[17] = vld1q_f64(c_ptr + 5 * N + 4);
        va[1] = vld1q_f64(a_trans + 6);
        vc[15] = vfmaq_laneq_f64(vc[15], vb[0], va[0], 1);
        vc[18] = vld1q_f64(c_ptr + 6 * N + 0);
        vc[16] = vfmaq_laneq_f64(vc[16], vb[1], va[0], 1);
        vc[19] = vld1q_f64(c_ptr + 6 * N + 2);
        vc[17] = vfmaq_laneq_f64(vc[17], vb[2], va[0], 1);
        vc[20] = vld1q_f64(c_ptr + 6 * N + 4);
        vc[18] = vfmaq_laneq_f64(vc[18], vb[0], va[1], 0);
        vc[21] = vld1q_f64(c_ptr + 7 * N + 0);
        vc[19] = vfmaq_laneq_f64(vc[19], vb[1], va[1], 0);
        vc[22] = vld1q_f64(c_ptr + 7 * N + 2);
        vc[20] = vfmaq_laneq_f64(vc[20], vb[2], va[1], 0);
        va[0] = vld1q_f64(a_trans + 8);
        vc[23] = vld1q_f64(c_ptr + 7 * N + 4);
        vc[21] = vfmaq_laneq_f64(vc[21], vb[0], va[1], 1);
        vc[22] = vfmaq_laneq_f64(vc[22], vb[1], va[1], 1);
        vc[23] = vfmaq_laneq_f64(vc[23], vb[2], va[1], 1);

        vc[0] = vfmaq_laneq_f64(vc[0], vb[3], va[0], 0);
        vc[1] = vfmaq_laneq_f64(vc[1], vb[4], va[0], 0);
        vc[2] = vfmaq_laneq_f64(vc[2], vb[5], va[0], 0);
        va[1] = vld1q_f64(a_trans + 10);
        vc[3] = vfmaq_laneq_f64(vc[3], vb[3], va[0], 1);
        vc[4] = vfmaq_laneq_f64(vc[4], vb[4], va[0], 1);
        vc[5] = vfmaq_laneq_f64(vc[5], vb[5], va[0], 1);
        vb[0] = vld1q_f64(b_trans + 12);
        vc[6] = vfmaq_laneq_f64(vc[6], vb[3], va[1], 0);
        vc[7] = vfmaq_laneq_f64(vc[7], vb[4], va[1], 0);
        vc[8] = vfmaq_laneq_f64(vc[8], vb[5], va[1], 0);
        va[0] = vld1q_f64(a_trans + 12);
        vb[1] = vld1q_f64(b_trans + 14);
        vc[9] = vfmaq_laneq_f64(vc[9], vb[3], va[1], 1);
        vc[10] = vfmaq_laneq_f64(vc[10], vb[4], va[1], 1);
        vc[11] = vfmaq_laneq_f64(vc[11], vb[5], va[1], 1);
        vb[2] = vld1q_f64(b_trans + 16);
        vc[12] = vfmaq_laneq_f64(vc[12], vb[3], va[0], 0);
        vc[13] = vfmaq_laneq_f64(vc[13], vb[4], va[0], 0);
        vc[14] = vfmaq_laneq_f64(vc[14], vb[5], va[0], 0);
        va[1] = vld1q_f64(a_trans + 14);
        vb[3] = vld1q_f64(b_trans + 18);
        vc[15] = vfmaq_laneq_f64(vc[15], vb[3], va[0], 1);
        vc[16] = vfmaq_laneq_f64(vc[16], vb[4], va[0], 1);
        vc[17] = vfmaq_laneq_f64(vc[17], vb[5], va[0], 1);
        vb[4] = vld1q_f64(b_trans + 20);
        vc[18] = vfmaq_laneq_f64(vc[18], vb[3], va[1], 0);
        vc[19] = vfmaq_laneq_f64(vc[19], vb[4], va[1], 0);
        vc[20] = vfmaq_laneq_f64(vc[20], vb[5], va[1], 0);
        va[0] = vld1q_f64(a_trans + 16);
        vb[5] = vld1q_f64(b_trans + 22);
        vc[21] = vfmaq_laneq_f64(vc[21], vb[3], va[1], 1);
        vc[22] = vfmaq_laneq_f64(vc[22], vb[4], va[1], 1);
        vc[23] = vfmaq_laneq_f64(vc[23], vb[5], va[1], 1);
        
        vc[0] = vfmaq_laneq_f64(vc[0], vb[0], va[0], 0);
        vc[1] = vfmaq_laneq_f64(vc[1], vb[1], va[0], 0);
        vc[2] = vfmaq_laneq_f64(vc[2], vb[2], va[0], 0);
        va[1] = vld1q_f64(a_trans + 18);
        vc[3] = vfmaq_laneq_f64(vc[3], vb[0], va[0], 1);
        vc[4] = vfmaq_laneq_f64(vc[4], vb[1], va[0], 1);
        vc[5] = vfmaq_laneq_f64(vc[5], vb[2], va[0], 1);
        vc[6] = vfmaq_laneq_f64(vc[6], vb[0], va[1], 0);
        vc[7] = vfmaq_laneq_f64(vc[7], vb[1], va[1], 0);
        vc[8] = vfmaq_laneq_f64(vc[8], vb[2], va[1], 0);
        va[0] = vld1q_f64(a_trans + 20);
        vc[9] = vfmaq_laneq_f64(vc[9], vb[0], va[1], 1);
        vc[10] = vfmaq_laneq_f64(vc[10], vb[1], va[1], 1);
        vc[11] = vfmaq_laneq_f64(vc[11], vb[2], va[1], 1);
        vc[12] = vfmaq_laneq_f64(vc[12], vb[0], va[0], 0);
        vc[13] = vfmaq_laneq_f64(vc[13], vb[1], va[0], 0);
        vc[14] = vfmaq_laneq_f64(vc[14], vb[2], va[0], 0);
        va[1] = vld1q_f64(a_trans + 22);
        vc[15] = vfmaq_laneq_f64(vc[15], vb[0], va[0], 1);
        vc[16] = vfmaq_laneq_f64(vc[16], vb[1], va[0], 1);
        vc[17] = vfmaq_laneq_f64(vc[17], vb[2], va[0], 1);
        vc[18] = vfmaq_laneq_f64(vc[18], vb[0], va[1], 0);
        vc[19] = vfmaq_laneq_f64(vc[19], vb[1], va[1], 0);
        vc[20] = vfmaq_laneq_f64(vc[20], vb[2], va[1], 0);
        va[0] = vld1q_f64(a_trans + 24);
        vc[21] = vfmaq_laneq_f64(vc[21], vb[0], va[1], 1);
        vc[22] = vfmaq_laneq_f64(vc[22], vb[1], va[1], 1);
        vc[23] = vfmaq_laneq_f64(vc[23], vb[2], va[1], 1);
        
        vc[0] = vfmaq_laneq_f64(vc[0], vb[3], va[0], 0);
        vc[1] = vfmaq_laneq_f64(vc[1], vb[4], va[0], 0);
        vc[2] = vfmaq_laneq_f64(vc[2], vb[5], va[0], 0);
        va[1] = vld1q_f64(a_trans + 26);
        vc[3] = vfmaq_laneq_f64(vc[3], vb[3], va[0], 1);
        vst1q_f64(c_ptr + 0 * N + 0, vc[0]);
        vc[4] = vfmaq_laneq_f64(vc[4], vb[4], va[0], 1);
        vst1q_f64(c_ptr + 0 * N + 2, vc[1]);
        vc[5] = vfmaq_laneq_f64(vc[5], vb[5], va[0], 1);
        vst1q_f64(c_ptr + 0 * N + 4, vc[2]);
        vc[6] = vfmaq_laneq_f64(vc[6], vb[3], va[1], 0);
        vst1q_f64(c_ptr + 1 * N + 0, vc[3]);
        vc[7] = vfmaq_laneq_f64(vc[7], vb[4], va[1], 0);
        vst1q_f64(c_ptr + 1 * N + 2, vc[4]);
        vc[8] = vfmaq_laneq_f64(vc[8], vb[5], va[1], 0);
        va[0] = vld1q_f64(a_trans + 28);
        vst1q_f64(c_ptr + 1 * N + 4, vc[5]);
        vc[9] = vfmaq_laneq_f64(vc[9], vb[3], va[1], 1);
        vst1q_f64(c_ptr + 2 * N + 0, vc[6]);
        vc[10] = vfmaq_laneq_f64(vc[10], vb[4], va[1], 1);
        vst1q_f64(c_ptr + 2 * N + 2, vc[7]);
        vc[11] = vfmaq_laneq_f64(vc[11], vb[5], va[1], 1);
        vst1q_f64(c_ptr + 2 * N + 4, vc[8]);
        vc[12] = vfmaq_laneq_f64(vc[12], vb[3], va[0], 0);
        vst1q_f64(c_ptr + 3 * N + 0, vc[9]);
        vc[13] = vfmaq_laneq_f64(vc[13], vb[4], va[0], 0);
        vst1q_f64(c_ptr + 3 * N + 2, vc[10]);
        vc[14] = vfmaq_laneq_f64(vc[14], vb[5], va[0], 0);
        va[1] = vld1q_f64(a_trans + 30);
        vst1q_f64(c_ptr + 3 * N + 4, vc[11]);
        vc[15] = vfmaq_laneq_f64(vc[15], vb[3], va[0], 1);
        vst1q_f64(c_ptr + 4 * N + 0, vc[12]);
        vc[16] = vfmaq_laneq_f64(vc[16], vb[4], va[0], 1);
        vst1q_f64(c_ptr + 4 * N + 2, vc[13]);
        vc[17] = vfmaq_laneq_f64(vc[17], vb[5], va[0], 1);
        vst1q_f64(c_ptr + 4 * N + 4, vc[14]);
        vc[18] = vfmaq_laneq_f64(vc[18], vb[3], va[1], 0);
        vst1q_f64(c_ptr + 5 * N + 0, vc[15]);
        vc[19] = vfmaq_laneq_f64(vc[19], vb[4], va[1], 0);
        vst1q_f64(c_ptr + 5 * N + 2, vc[16]);
        vc[20] = vfmaq_laneq_f64(vc[20], vb[5], va[1], 0);
        vst1q_f64(c_ptr + 5 * N + 4, vc[17]);
        vc[21] = vfmaq_laneq_f64(vc[21], vb[3], va[1], 1);
        vst1q_f64(c_ptr + 6 * N + 0, vc[18]);
        vc[22] = vfmaq_laneq_f64(vc[22], vb[4], va[1], 1);
        vst1q_f64(c_ptr + 6 * N + 2, vc[19]);
        vc[23] = vfmaq_laneq_f64(vc[23], vb[5], va[1], 1);
        vst1q_f64(c_ptr + 6 * N + 4, vc[20]);
        vst1q_f64(c_ptr + 7 * N + 0, vc[21]);
        vst1q_f64(c_ptr + 7 * N + 2, vc[22]);
        vst1q_f64(c_ptr + 7 * N + 4, vc[23]);
        
        b_trans += 4 * 6;
      }
      a_trans += 8 * 4;
    }
    b_trans_d += 4 * n_blk;
  }
}
#endif

void blocking_gemm_compute_opt(FLOAT *c, FLOAT *a_trans, FLOAT *b_trans, int M, int K ,int N)
{
  int m_pad = GEMM_PADDING(M, 8);
  int k_pad = GEMM_PADDING(K, 4);
#ifdef SP
  int n_pad = GEMM_PADDING(N, 12);
#else
  int n_pad = GEMM_PADDING(N, 6);
#endif

  int i, j, k;

#pragma omp parallel private(i, j, k)
{
  for(k = 0; k <= k_pad - K_BLK; k += K_BLK)
  {
    #pragma omp for collapse(2)
    for(i = 0; i <= m_pad - M_BLK; i += M_BLK)
    {
      for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * K_BLK, b_trans + k * n_pad + j * K_BLK, M_BLK, K_BLK, N_BLK, N);
      }
    }
  }
}
}

void blocking_gemm_compute(FLOAT *c, FLOAT *a_trans, FLOAT *b_trans, int M, int K ,int N)
{
  int m_pad = GEMM_PADDING(M, 8);
  int k_pad = GEMM_PADDING(K, 4);
#ifdef SP
  int n_pad = GEMM_PADDING(N, 12);
#else
  int n_pad = GEMM_PADDING(N, 6);
#endif

  int i, j, k;

#pragma omp parallel private(i, j, k)
{
  for(k = 0; k <= k_pad - K_BLK; k += K_BLK)
  {
    #pragma omp for
    for(i = 0; i <= m_pad - M_BLK; i += M_BLK)
    {
      for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * K_BLK, b_trans + k * n_pad + j * K_BLK, M_BLK, K_BLK, N_BLK, N);
      }
      if(j < n_pad)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * K_BLK, b_trans + k * n_pad + j * K_BLK, M_BLK, K_BLK, n_pad - j, N);
      }
    }
    if(i < m_pad)
    {
      #pragma omp for
      for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * K_BLK, b_trans + k * n_pad + j * K_BLK, m_pad - i, K_BLK, N_BLK, N);
      }
      #pragma omp single
      {
        if(j < n_pad)
        {
          small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * K_BLK, b_trans + k * n_pad + j * K_BLK, m_pad - i, K_BLK, n_pad - j, N);
        }
      }
    }
    #pragma omp barrier
  }
  if(k < k_pad)
  {
    #pragma omp for
    for(i = 0; i <= m_pad - M_BLK; i += M_BLK)
    {
      for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * (k_pad - k), b_trans + k * n_pad + j * (k_pad - k), M_BLK, k_pad - k, N_BLK, N);
      }
      if(j < n_pad)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * (k_pad - k), b_trans + k * n_pad + j * (k_pad - k), M_BLK, k_pad - k, n_pad - j, N);
      }
    }
    if(i < m_pad)
    {
      #pragma omp for
      for(j = 0; j <= n_pad - N_BLK; j += N_BLK)
      {
        small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * (k_pad - k), b_trans + k * n_pad + j * (k_pad - k), m_pad - i, k_pad - k, N_BLK, N);
      }
      #pragma omp single
      {
        if(j < n_pad)
        {
          small_gemm_kernel(c + i * N + j, a_trans + k * m_pad + i * (k_pad - k), b_trans + k * n_pad + j * (k_pad - k), m_pad - i, k_pad - k, n_pad - j, N);
        }
      }
    }
  }
}

}

void calculateMatrix(FLOAT *a, FLOAT *b, FLOAT *c, int m, int k, int n, FLOAT *tmpbuffer) 
{
  int m_pad = GEMM_PADDING(m, 8);
  int k_pad = GEMM_PADDING(k, 4);

  FLOAT *a_trans, *b_trans;

  a_trans = tmpbuffer;
  b_trans = a_trans + m_pad * k_pad;

  printf("m = %d, k = %d, n = %d\n", m, k, n);

  // load a trans
  load_a_trans(a_trans, a, m, k);

  // load b trans
  load_b_trans(b_trans, b, k, n);

  // compute c
  if((m % M_BLK == 0) && (n % N_BLK == 0) && (k % K_BLK == 0))
  {
    blocking_gemm_compute_opt(c, a_trans, b_trans, m, k, n); 
  }
  else
  {
    // TODO: optimize the OpenMP performance
    blocking_gemm_compute(c, a_trans, b_trans, m, k, n);
  }
}