#include <immintrin.h>
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
#ifdef SP
#pragma omp parallel private(buf, buf2, buf3)
  {
    buf = malloc(8 * sizeof(FLOAT));
    buf2 = malloc(8 * sizeof(FLOAT));
    buf3 = malloc(8 * sizeof(FLOAT));
#pragma omp for
    for (int i = 0; i < n; i++) {
      FLOAT Fx = 0.0;
      FLOAT Fy = 0.0;
      FLOAT Fz = 0.0;
      __m256 vbuf = _mm256_set1_ps(0.0);
      __m256 vbuf2 = _mm256_set1_ps(0.0);
      __m256 vbuf3 = _mm256_set1_ps(0.0);
      __m256 pyi = _mm256_set1_ps(p.y[i]);
      __m256 pzi = _mm256_set1_ps(p.z[i]);
      __m256 pxi = _mm256_set1_ps(p.x[i]);
      __m256 soft = _mm256_set1_ps(SOFTENING);

      for (int j = 0; j < n; j += 8) {
        __m256 dy = _mm256_load_ps(p.y + j);
        __m256 dz = _mm256_load_ps(p.z + j);
        __m256 dx = _mm256_load_ps(p.x + j);
        dy = _mm256_sub_ps(dy, pyi);
        dz = _mm256_sub_ps(dz, pzi);
        dx = _mm256_sub_ps(dx, pxi);
        __m256 dy2 = _mm256_mul_ps(dy, dy);
        __m256 dz2 = _mm256_mul_ps(dz, dz);
        __m256 dx2 = _mm256_mul_ps(dx, dx);
        __m256 distSqr =
            _mm256_add_ps(_mm256_add_ps(dx2, dy2), _mm256_add_ps(dz2, soft));
        distSqr = _mm256_rsqrt_ps(distSqr);
        distSqr = _mm256_mul_ps(distSqr, _mm256_mul_ps(distSqr, distSqr));
        dx = _mm256_mul_ps(dx, distSqr);
        dy = _mm256_mul_ps(dy, distSqr);
        dz = _mm256_mul_ps(dz, distSqr);
        vbuf = _mm256_add_ps(vbuf, dx);
        vbuf2 = _mm256_add_ps(vbuf2, dy);
        vbuf3 = _mm256_add_ps(vbuf3, dz);
      }
      _mm256_store_ps(buf, vbuf);
      _mm256_store_ps(buf2, vbuf2);
      _mm256_store_ps(buf3, vbuf3);
      for (int l = 0; l < 8; ++l) {
        Fx += buf[l];
        Fy += buf2[l];
        Fz += buf3[l];
      }

      p.vx[i] += dt * Fx;
      p.vy[i] += dt * Fy;
      p.vz[i] += dt * Fz;
    }

    free(buf);
    free(buf2);
    free(buf3);
  }
#else
#pragma omp parallel private(buf, buf2, buf3)
  {
    buf = malloc(4 * sizeof(FLOAT));
    buf2 = malloc(4 * sizeof(FLOAT));
    buf3 = malloc(4 * sizeof(FLOAT));
#pragma omp for
    for (int i = 0; i < n; i++) {
      FLOAT Fx = 0.0;
      FLOAT Fy = 0.0;
      FLOAT Fz = 0.0;
      __m256d vbuf = _mm256_set1_pd(0.0);
      __m256d vbuf2 = _mm256_set1_pd(0.0);
      __m256d vbuf3 = _mm256_set1_pd(0.0);
      __m256d pyi = _mm256_set1_pd(p.y[i]);
      __m256d pzi = _mm256_set1_pd(p.z[i]);
      __m256d pxi = _mm256_set1_pd(p.x[i]);
      __m256d soft = _mm256_set1_pd(SOFTENING);

      for (int j = 0; j < n; j += 4) {
        __m256d dy = _mm256_load_pd(p.y + j);
        __m256d dz = _mm256_load_pd(p.z + j);
        __m256d dx = _mm256_load_pd(p.x + j);
        dy = _mm256_sub_pd(dy, pyi);
        dz = _mm256_sub_pd(dz, pzi);
        dx = _mm256_sub_pd(dx, pxi);
        __m256d dy2 = _mm256_mul_pd(dy, dy);
        __m256d dz2 = _mm256_mul_pd(dz, dz);
        __m256d dx2 = _mm256_mul_pd(dx, dx);
        __m256d distSqr =
            _mm256_add_pd(_mm256_add_pd(dx2, dy2), _mm256_add_pd(dz2, soft));
        distSqr = _mm256_invsqrt_pd(distSqr);
        distSqr = _mm256_mul_pd(distSqr, _mm256_mul_pd(distSqr, distSqr));
        dx = _mm256_mul_pd(dx, distSqr);
        dy = _mm256_mul_pd(dy, distSqr);
        dz = _mm256_mul_pd(dz, distSqr);
        vbuf = _mm256_add_pd(vbuf, dx);
        vbuf2 = _mm256_add_pd(vbuf2, dy);
        vbuf3 = _mm256_add_pd(vbuf3, dz);
      }
      _mm256_store_pd(buf, vbuf);
      _mm256_store_pd(buf2, vbuf2);
      _mm256_store_pd(buf3, vbuf3);
      for (int l = 0; l < 4; ++l) {
        Fx += buf[l];
        Fy += buf2[l];
        Fz += buf3[l];
      }

      p.vx[i] += dt * Fx;
      p.vy[i] += dt * Fy;
      p.vz[i] += dt * Fz;
    }
    free(buf);
    free(buf2);
    free(buf3);
  }
#endif
  for (int i = 0; i < n; i++) {  // integrate position
    p.x[i] += p.vx[i] * dt;
    p.y[i] += p.vy[i] * dt;
    p.z[i] += p.vz[i] * dt;
  }
}