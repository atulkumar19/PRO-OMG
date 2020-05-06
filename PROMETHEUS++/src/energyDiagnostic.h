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

#ifndef H_ENERGY_DIAGNOSTIC
#define H_ENERGY_DIAGNOSTIC

#include <iostream>
#include <cmath>
#include <vector>

#include "armadillo"
#include "structures.h"
#include "types.h"

#include "mpi_main.h"


template <class IT, class FT> class ENERGY_DIAGNOSTIC{

    arma::vec kineticEnergyDensity;

    arma::vec magneticEnergyDensity;

    arma::vec electricEnergyDensity;


    void computeKineticEnergyDensity(const simulationParameters * params, const vector<IT> * IONS);


    void computeElectromagneticEnergyDensity(const simulationParameters * params, const oneDimensional::fields * EB);

    void computeElectromagneticEnergyDensity(const simulationParameters * params, const twoDimensional::fields * EB);

public:

    ENERGY_DIAGNOSTIC(const simulationParameters * params, const FT * EB, const vector<IT> * IONS);

    arma::vec getKineticEnergyDensity();

    arma::vec getMagneticEnergyDensity();

    arma::vec getElectricEnergyDensity();

};



#endif
