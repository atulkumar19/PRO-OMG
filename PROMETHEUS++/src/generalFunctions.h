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

#ifndef H_GENERAL_FUNCTIONS
#define H_GENERAL_FUNCTIONS

#include <iostream>
#include <string>
#include <vector>
#include <cmath>

#include <omp.h>

#include "armadillo"
#include "structures.h"
#include "boundaryConditions.h"

#include "mpi_main.h"


using namespace std;
using namespace arma;

class GENERAL_FUNCTIONS{

	#ifdef ONED
	#define Cmax 1.0f
	#endif

	#ifdef TWOD
	#define Cmax 1.0f/sqrt(2.0f)
	#endif

	#ifdef THREED
	#define Cmax 1.0f/sqrt(3.0f)
	#endif

	int logicVariable;

	double initialIonsEnergy;
	double initialFieldsEnergy;

	int index;
	vec EMFE;//electromagnetic field energy
	vec electricFieldEnergy;
	vec magneticFieldEnergy;
	vec IE;//ions energy

	vector<int> appliedFilters;
	vector<double> smoothingParameter;

	void bCastTimestep(simulationParameters * params, int logicVariable);

	public:

	GENERAL_FUNCTIONS(){};

	void checkStability(simulationParameters * params, const meshParams *mesh, const characteristicScales * CS, const vector<oneDimensional::ionSpecies> * IONS);

	void checkEnergy(simulationParameters * params, meshParams *mesh, characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS, oneDimensional::fields * EB, int IT);

	void saveDiagnosticsVariables(simulationParameters * params);

};

#endif
