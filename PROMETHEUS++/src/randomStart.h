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

#ifndef H_RANDOM_START
#define H_RANDOM_START

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "structures.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

template <class IT> class RANDOMSTART
{
    // Cartesian  unitary vectors
    arma::vec x = {1.0, 0.0, 0.0};
    arma::vec y = {0.0, 1.0, 0.0};
    arma::vec z = {0.0, 0.0, 1.0};


    arma::vec b1; // Unitary vector along B field
    arma::vec b2; // Unitary vector perpendicular to b1
    arma::vec b3; // Unitary vector perpendicular to b1 and b2

    double target(const simulationParameters * params, IT * ions, double X, double V3, double V2, double V1);

    public:

    RANDOMSTART(const simulationParameters * params);

    void ringLikeVelocityDistribution(const simulationParameters * params, IT * ions);

    void maxwellianVelocityDistribution(const simulationParameters * params, IT * ions);

    void maxwellianVelocityDistribution_nonhomogeneous(const simulationParameters * params, IT * ions);
};

#endif
