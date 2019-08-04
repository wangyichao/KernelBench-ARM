#ifdef SP
#define FLOAT float
#else
#define FLOAT double
#endif

#define SOFTENING 1e-9f

typedef struct {
  FLOAT *x, *y, *z, *vx, *vy, *vz;
} BodySystem;

void bodyForce(BodySystem p, FLOAT dt, int n);
void bodyForceRaw(BodySystem p, FLOAT dt, int n);
void randomizeBodies(FLOAT *data, int n);