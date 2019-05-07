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

    // Cartesian  unitary vectors
	arma::vec x = {1.0, 0.0, 0.0};
	arma::vec y = {0.0, 1.0, 0.0};
	arma::vec z = {0.0, 0.0, 1.0};


	arma::vec b1; // Unitary vector along B field
	arma::vec b2; // Unitary vector perpendicular to b1
	arma::vec b3; // Unitary vector perpendicular to b1 and b2

public:

    RANDOMSTART(const inputParameters * params);

	void ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions);

	void maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions);
};

#endif
