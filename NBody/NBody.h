//Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_HEADER__
#define __NBODY_HEADER__

#define G			9.8f		//gravitational constant
#define dt			0.01f		//time step
#define SOFTENING	2.0f		//softening parameter to help with numerical instability

struct nbody{
	float x, y, vx, vy, m;
};

typedef enum MODE { CPU, OPENMP, CUDA } MODE;
typedef struct nbody nbody;

#endif	//__NBODY_HEADER__