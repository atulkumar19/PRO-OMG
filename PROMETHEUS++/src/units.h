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

template <class IT, class FT> class UNITS{

	void dimensionlessForm(simulationParameters * params, vector<IT> * IONS, FT * EB, const characteristicScales * CS);

	void broadcastCharacteristicScales(simulationParameters * params, characteristicScales * CS);

	void broadcastFundamentalScales(simulationParameters * params, fundamentalScales * FS);

public:

	UNITS(){};

	void spatialScalesSanityCheck(simulationParameters * params, fundamentalScales * FS);

	void defineTimeStep(simulationParameters * params, vector<IT> * IONS);

	void calculateFundamentalScales(simulationParameters * params, vector<IT> * IONS, fundamentalScales * FS);

	void defineCharacteristicScales(simulationParameters * params, vector<IT> * IONS, characteristicScales * CS);

	void normalizeVariables(simulationParameters * params, vector<IT> * IONS, FT * EB, const characteristicScales * CS);

	void defineCharacteristicScalesAndBcast(simulationParameters * params, vector<IT> * IONS, characteristicScales * CS);

	void calculateFundamentalScalesAndBcast(simulationParameters * params, vector<IT> * IONS, fundamentalScales * FS);

};

#endif
