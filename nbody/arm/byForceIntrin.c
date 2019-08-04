#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#include "nbody-soa.h"

void bodyForce(BodySystem p, FLOAT dt, int n) {
  // for (int i = 0; i < n; i++) {
  //   FLOAT Fx = 0.0;
  //   FLOAT Fy = 0.0;
  //   FLOAT Fz = 0.0;
  //   for (int j = 0; j < n; j += 1) {
  //     FLOAT dy = p.y[j] - p.y[i];
  //     FLOAT dz = p.z[j] - p.z[i];
  //     FLOAT dx = p.x[j] - p.x[i];
  //     FLOAT distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
  //     FLOAT invDist = 1.0 / sqrt(distSqr);
  //     FLOAT invDist3 = invDist * invDist * invDist;
  //     Fx += dx * invDist3;
  //     Fy += dy * invDist3;
  //     Fz += dz * invDist3;
  //   }
  //   p.vx[i] += dt * Fx;
  //   p.vy[i] += dt * Fy;
  //   p.vz[i] += dt * Fz;
  // }
  FLOAT *buf, *buf2, *buf3;
#pragma omp parallel private(buf, buf2, buf3)
{
#ifdef SP
  buf = malloc(4 * sizeof(FLOAT));
  buf2 = malloc(4 * sizeof(FLOAT));
  buf3 = malloc(4 * sizeof(FLOAT));
#pragma omp for
  for (int i = 0; i < n; i++) {
    FLOAT Fx = 0.0;
    FLOAT Fy = 0.0;
    FLOAT Fz = 0.0;

    float32x4_t pyi, pzi, pxi, soft;
    pyi = vdupq_n_f32(p.y[i]);
    pzi = vdupq_n_f32(p.z[i]);
    pxi = vdupq_n_f32(p.x[i]);
    soft = vdupq_n_f32(SOFTENING);

    for (int j = 0; j < n; j += 4) {
      float32x4_t dy, dz, dx;
      float32x4_t dy2, dz2, dx2;
      float32x4_t distSqr;
      
      dy = vld1q_f32(p.y + j);
      dz = vld1q_f32(p.z + j);
      dx = vld1q_f32(p.x + j);
      
      dy = vsubq_f32(dy, pyi);
      dz = vsubq_f32(dz, pzi);
      dx = vsubq_f32(dx, pxi);
      
      dy2 = vmulq_f32(dy, dy);
      dz2 = vmulq_f32(dz, dz);
      dx2 = vmulq_f32(dx, dx);

      distSqr = vaddq_f32(vaddq_f32(dx2, dy2), vaddq_f32(dz2, soft));
      distSqr = vrsqrteq_f32(distSqr);
      distSqr = vmulq_f32(distSqr, vmulq_f32(distSqr, distSqr));
      dx = vmulq_f32(dx, distSqr);
      dy = vmulq_f32(dy, distSqr);
      dz = vmulq_f32(dz, distSqr);
      vst1q_f32(buf, dx);
      vst1q_f32(buf2, dy);
      vst1q_f32(buf3, dz);
      for (int l = 0; l < 4; ++l) {
        Fx += buf[l];
        Fy += buf2[l];
        Fz += buf3[l];
      }
    }

    p.vx[i] += dt * Fx;
    p.vy[i] += dt * Fy;
    p.vz[i] += dt * Fz;
  }

  free(buf);
  free(buf2);
  free(buf3);

#else
  buf = malloc(2 * sizeof(FLOAT));
  buf2 = malloc(2 * sizeof(FLOAT));
  buf3 = malloc(2 * sizeof(FLOAT));
#pragma omp for
  for (int i = 0; i < n; i++) {
    FLOAT Fx = 0.0;
    FLOAT Fy = 0.0;
    FLOAT Fz = 0.0;

    float64x2_t pyi, pzi, pxi, soft;
    pyi = vdupq_n_f64(p.y[i]);
    pzi = vdupq_n_f64(p.z[i]);
    pxi = vdupq_n_f64(p.x[i]);
    soft = vdupq_n_f64(SOFTENING);

    for (int j = 0; j < n; j += 2) {
      float64x2_t dy, dz, dx;
      float64x2_t dy2, dz2, dx2;
      float64x2_t distSqr;
      
      dy = vld1q_f64(p.y + j);
      dz = vld1q_f64(p.z + j);
      dx = vld1q_f64(p.x + j);
      
      dy = vsubq_f64(dy, pyi);
      dz = vsubq_f64(dz, pzi);
      dx = vsubq_f64(dx, pxi);
      
      dy2 = vmulq_f64(dy, dy);
      dz2 = vmulq_f64(dz, dz);
      dx2 = vmulq_f64(dx, dx);

      distSqr = vaddq_f64(vaddq_f64(dx2, dy2), vaddq_f64(dz2, soft));
      distSqr = vrsqrteq_f64(distSqr);
      distSqr = vmulq_f64(distSqr, vmulq_f64(distSqr, distSqr));
      dx = vmulq_f64(dx, distSqr);
      dy = vmulq_f64(dy, distSqr);
      dz = vmulq_f64(dz, distSqr);
      vst1q_f64(buf, dx);
      vst1q_f64(buf2, dy);
      vst1q_f64(buf3, dz);
      for (int l = 0; l < 2; ++l) {
        Fx += buf[l];
        Fy += buf2[l];
        Fz += buf3[l];
      }
    }

    p.vx[i] += dt * Fx;
    p.vy[i] += dt * Fy;
    p.vz[i] += dt * Fz;
  }

  free(buf);
  free(buf2);
  free(buf3);
#endif

#pragma omp for
  for (int i = 0; i < n; i++) {  // integrate position
    p.x[i] += p.vx[i] * dt;
    p.y[i] += p.vy[i] * dt;
    p.z[i] += p.vz[i] * dt;
  }
}
}