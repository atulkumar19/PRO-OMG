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

#ifndef H_UNITS
#define H_UNITS

#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>

#include "structures.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

class UNITS{

	#ifdef ONED
	#define Cmax 1.0f
	#endif

	#ifdef TWOD
	#define Cmax 1.0f/sqrt(2.0f)
	#endif

	#ifdef THREED
	#define Cmax 1.0f/sqrt(3.0f)
	#endif

	void dimensionlessForm(inputParameters * params, meshGeometry * mesh, vector<ionSpecies> * IONS, fields * EB, const characteristicScales * CS);

	void broadcastCharacteristicScales(inputParameters * params, characteristicScales * CS);

public:

	UNITS(){};

	void defineTimeStep(inputParameters * params, meshGeometry * mesh, vector<ionSpecies> * IONS, fields * EB);

	void calculateFundamentalScales(inputParameters * params, vector<ionSpecies> * IONS, fundamentalScales * FS, meshGeometry * mesh);

	void defineCharacteristicScales(inputParameters * params, vector<ionSpecies> * IONS, characteristicScales * CS);

	void normalizeVariables(inputParameters * params, meshGeometry * mesh, vector<ionSpecies> * IONS, fields * EB, const characteristicScales * CS);

	void defineCharacteristicScalesAndBcast(inputParameters * params, vector<ionSpecies> * IONS, characteristicScales * CS);

	void calculateFundamentalScalesAndBcast(inputParameters * params, vector<ionSpecies> * IONS, fundamentalScales * FS, meshGeometry * mesh);

};

#endif
