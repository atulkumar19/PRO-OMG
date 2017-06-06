#ifndef H_RANDOM_START
#define H_RANDOM_START

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

#include <armadillo>

#include "structures.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

class RANDOMSTART{


public:

    RANDOMSTART(){};

	void beamVelocityDistribution(const inputParameters * params, ionSpecies * ions, const string parDirection);

	void ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions, const string parDirection);

	void maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions, const string parDirection);

	void shellVelocityDistribution(const inputParameters * params, ionSpecies * ions);
};

#endif
