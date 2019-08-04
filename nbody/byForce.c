#include <math.h>
#include "nbody-soa.h"

void bodyForce(BodySystem p, FLOAT dt, int n)
{
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    FLOAT Fx = 0.0;
    FLOAT Fy = 0.0;
    FLOAT Fz = 0.0;
    for (int j = 0; j < n; j += 1) {
      FLOAT dy = p.y[j] - p.y[i];
      FLOAT dz = p.z[j] - p.z[i];
      FLOAT dx = p.x[j] - p.x[i];
      FLOAT distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
      FLOAT invDist = 1.0 / sqrt(distSqr);
      FLOAT invDist3 = invDist * invDist * invDist;
      Fx += dx * invDist3;
      Fy += dy * invDist3;
      Fz += dz * invDist3;
    }
    p.vx[i] += dt * Fx;
    p.vy[i] += dt * Fy;
    p.vz[i] += dt * Fz;
  }
}

// void bodyForce(BodySystem p, FLOAT dt, int n) {
//   int step = 1024;
// #pragma omp parallel for collapse(2)
//   for (int k = 0; k < n; k += step) {
//     for (int l = 0; l < n; l += step)
//       for (int i = k; i < k + step; i++) {
//         FLOAT Fx = 0.0;
//         FLOAT Fy = 0.0;
//         FLOAT Fz = 0.0;

//         for (int j = l; j < l + step; j++) {
//           FLOAT dy = p.y[j] - p.y[i];
//           FLOAT dz = p.z[j] - p.z[i];
//           FLOAT dx = p.x[j] - p.x[i];
//           FLOAT distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
//           FLOAT invDist = 1.0 / sqrt(distSqr);
//           FLOAT invDist3 = invDist * invDist * invDist;

//           Fx += dx * invDist3;
//           Fy += dy * invDist3;
//           Fz += dz * invDist3;
//         }

//         p.vx[i] += dt * Fx;
//         p.vy[i] += dt * Fy;
//         p.vz[i] += dt * Fz;
//       }
//   }
//   for (int i = 0; i < n; i++) {  // integrate position
//     p.x[i] += p.vx[i] * dt;
//     p.y[i] += p.vy[i] * dt;
//     p.z[i] += p.vz[i] * dt;
//   }
// }