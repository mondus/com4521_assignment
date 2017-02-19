#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "NBody.h"
#include "NBodyVisualiser.h"

#define USER_NAME "test"		//replace with your username

void print_help();
void step(void);


int main(int argc, char *argv[]) {

	//TODO: Processes the command line arguments
		//argc in the count of the command arguments
		//argv is an array (of length argc) of the arguments. The first argument is always the executable name (including path)

	//TODO: Allocate any heap memory
	
	//TODO: Depending on program arguments, either read initial data from file or generate random data.

	//TODO: Depending on program arguments, either configure and start the visualiser or perform a fixed number of simulation steps (then output the timing results).
	

	return 0;
}

void step(void)
{
	//TODO: Perform the main simulation of the NBody system
	
}



void print_help(){
	printf("nbody_%s N D M [-i I] [-i input_file]\n", USER_NAME);

	printf("where:\n");
	printf("\tN                Is the number of bodies to simulate.\n");
	printf("\tD                Is the integer dimension of the activity grid. The Grid has D*D locations.\n");
	printf("\tM                Is the operation mode, either  'CPU' or 'OPENMP'\n");
	printf("\t[-i I]           Optionally specifies the number of simulation iterations 'I' to perform. Specifying no value will use visualisation mode. \n");
	printf("\t[-f input_file]  Optionally specifies an input file with an initial N bodies of data. If not specified random data will be created.\n");
}
