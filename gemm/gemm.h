#ifndef GEMM_H
#define GEMM_H

#ifdef SP
#define M_BLK 120
#define K_BLK 80
#define N_BLK 120
#else
#define M_BLK 120
#define K_BLK 80
#define N_BLK 60
#endif

#define GEMM_PADDING(N, ALIGN) ((N + ALIGN - 1) / ALIGN * ALIGN)

#define TRANSPOSE_FLOAT_4X4(q0, q1, q2, q3)\
{\
    float32x4x2_t __Q0, __Q1; \
    __Q0 = vtrnq_f32(q0, q1); \
    __Q1 = vtrnq_f32(q2, q3); \
    q0 = vcombine_f32(vget_low_f32(__Q0.val[0]), vget_low_f32(__Q1.val[0]));\
    q1 = vcombine_f32(vget_low_f32(__Q0.val[1]), vget_low_f32(__Q1.val[1]));\
    q2 = vcombine_f32(vget_high_f32(__Q0.val[0]), vget_high_f32(__Q1.val[0]));\
    q3 = vcombine_f32(vget_high_f32(__Q0.val[1]), vget_high_f32(__Q1.val[1]));\
}

#define TRANSPOSE_DOUBLE_2x2(q0, q1)\
{\
    float64x2_t vtmp[2];\
    vtmp[0] = vtrn1q_f64(q0, q1);\
    vtmp[1] = vtrn2q_f64(q0, q1);\
    q0 = vtmp[0];\
    q1 = vtmp[1];\
}

#endif