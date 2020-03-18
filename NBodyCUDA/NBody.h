//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

#define G			1.0f		//gravitational constant (not the actual value of G, a value of G used to avoid issues with numeric precision)
#define dt			0.01f		//time step
#define SOFTENING	2.0f		//softening parameter to help with numerical instability

struct nbody{
	float x, y, vx, vy, m;
};

struct nbody_soa {
	float* x, * y, * vx, * vy, * m;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;
typedef struct nbody_soa nbody_soa;

#endif	//__NBODY_HEADER__