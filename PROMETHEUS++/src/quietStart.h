// COPYRIGHT 2015-2019 LEOPOLDO CARBAJAL

/*	This file is part of PROMETHEUS++.

    PROMETHEUS++ is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    PROMETHEUS++ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/>.
*/

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

template <class IT> class QUIETSTART{
    // Cartesian  unitary vectors
	arma::vec x = {1.0, 0.0, 0.0};
	arma::vec y = {0.0, 1.0, 0.0};
	arma::vec z = {0.0, 0.0, 1.0};


	arma::vec b1; // Unitary vector along B field
	arma::vec b2; // Unitary vector perpendicular to b1
	arma::vec b3; // Unitary vector perpendicular to b1 and b2

    uvec dec; // Sequence of decimal numbers

    void recalculateNumberSuperParticles(const simulationParameters * params, IT *ions);

    vector<int> dec2bin(int dec);

    vector<int> dec2b3(int dec);

    void bit_reversedFractions_base2(const simulationParameters * params, unsigned int NSP, vec * b2fr);

    void bit_reversedFractions_base3(const simulationParameters * params, unsigned int NSP, vec * b3fr);

public:

    QUIETSTART(){};

    QUIETSTART(const simulationParameters * params, IT * ions);

/*
	void maxwellianVelocityDistribution(const simulationParameters * params, oneDimensional::ionSpecies * ions);

	void maxwellianVelocityDistribution(const simulationParameters * params, twoDimensional::ionSpecies * ions);
*/
	void maxwellianVelocityDistribution(const simulationParameters * params, IT * ions);

	void ringLikeVelocityDistribution(const simulationParameters * params, IT * ions);

/*
    void ringLikeVelocityDistribution(const simulationParameters * params, oneDimensional::ionSpecies * ions);

	void ringLikeVelocityDistribution(const simulationParameters * params, twoDimensional::ionSpecies * ions);
*/

};



#endif
