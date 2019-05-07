#ifndef H_QUIET_START
#define H_QUIET_START

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

class QUIETSTART{


    // Cartesian  unitary vectors
	arma::vec x = {1.0, 0.0, 0.0};
	arma::vec y = {0.0, 1.0, 0.0};
	arma::vec z = {0.0, 0.0, 1.0};


	arma::vec b1; // Unitary vector along B field
	arma::vec b2; // Unitary vector perpendicular to b1
	arma::vec b3; // Unitary vector perpendicular to b1 and b2

    uvec dec;

    double recalculateNumberSuperParticles(const inputParameters * params, ionSpecies *ions);

    vector<int> dec2bin(int dec);

    vector<int> dec2b3(int dec);

    void bit_reversedFractions_base2(const inputParameters * params, ionSpecies * ions, vec * b2fr);

    void bit_reversedFractions_base3(const inputParameters * params, ionSpecies * ions, vec * b3fr);

public:

    QUIETSTART(){};

    QUIETSTART(const inputParameters * params, ionSpecies * ions);

    void maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions);

    void ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions);

};



#endif
