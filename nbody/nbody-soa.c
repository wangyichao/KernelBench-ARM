#include "nbody-soa.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int main(const int argc, const char **argv) {
  int nBodies = 16384;
  if (argc > 1) nBodies = atoi(argv[1]);

  const FLOAT dt = 0.01;  // time step
  const int nIters = 10;  // simulation iterations

  FLOAT *buf = (FLOAT *)malloc(6 * nBodies * sizeof(FLOAT));
  BodySystem p, v;
  p.x = buf + 0 * nBodies;
  p.y = buf + 1 * nBodies;
  p.z = buf + 2 * nBodies;
  p.vx = buf + 3 * nBodies;
  p.vy = buf + 4 * nBodies;
  p.vz = buf + 5 * nBodies;
  randomizeBodies(buf, 6 * nBodies);  // Init pos / vel data
  #ifndef NOVALIDATE
  FLOAT *buf2 = (FLOAT *)malloc(6 * nBodies * sizeof(FLOAT));
  v.x = buf2 + 0 * nBodies;
  v.y = buf2 + 1 * nBodies;
  v.z = buf2 + 2 * nBodies;
  v.vx = buf2 + 3 * nBodies;
  v.vy = buf2 + 4 * nBodies;
  v.vz = buf2 + 5 * nBodies;
  memcpy(buf2, buf, 6 * nBodies * sizeof(FLOAT));
  for (int iter = 1; iter <= nIters; iter++) {
    bodyForceRaw(v, dt, nBodies);
  }
  #endif

  double totalTime = 0.0;
  struct timeval beg, end;
  for (int iter = 1; iter <= nIters; iter++) {
    gettimeofday(&beg, NULL);

    bodyForce(p, dt, nBodies);  // compute interbody forces

    gettimeofday(&end, NULL);
    const double tElapsed =
        end.tv_sec - beg.tv_sec + (end.tv_usec - beg.tv_usec) / 1000000.0;
    if (iter > 1) {  // First iter is warm up
      totalTime += tElapsed;
    }
    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters - 1);

#ifndef NOVALIDATE
  double error = 0.0;
  double sum = 0.0;
  for (int i = 0; i < nBodies; ++i) {
    error += fabs(v.x[i] - p.x[i]);
    sum += fabs(v.x[i]);
    // if (fabs(v.x[i] - p.x[i]) != 0.0) printf("%lf %lf\n", v.x[i], p.x[i]);
  }
  printf("Error is %lf%% in %lf\n", error/sum*100, sum);
  free(buf2);
#endif
  printf("Average rate for iterations 2 through %d: %.3f seconds per step.\n",
         nIters, avgTime);
  printf("%d Bodies: average %0.3f Billion Interactions / second.\n", nBodies,
         1e-9 * nBodies * nBodies / avgTime);
  free(buf);
}

void bodyForceRaw(BodySystem p, FLOAT dt, int n) {
  for (int i = 0; i < n; i++) {
    FLOAT Fx = 0.0;
    FLOAT Fy = 0.0;
    FLOAT Fz = 0.0;

    for (int j = 0; j < n; j++) {
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
  for (int i = 0; i < n; i++) {  // integrate position
    p.x[i] += p.vx[i] * dt;
    p.y[i] += p.vy[i] * dt;
    p.z[i] += p.vz[i] * dt;
  }
}

void randomizeBodies(FLOAT *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0 * ((FLOAT)rand() / RAND_MAX) - 1.0;
  }
}