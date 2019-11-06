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

#ifndef H_OUTPUTHDF5
#define H_OUTPUTHDF5

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif

#include <string>
#include <cmath>

#include <armadillo>
#include "structures.h"
#include "boundaryConditions.h"
#include "types.h"

#include "H5Cpp.h"

using namespace H5;
using namespace std;
using namespace arma;

class HDF{

#ifdef HDF5_DOUBLE
	#define HDF_TYPE PredType::NATIVE_DOUBLE
	#define CPP_TYPE double
#elif defined HDF5_FLOAT
	#define HDF_TYPE PredType::NATIVE_FLOAT
	#define CPP_TYPE float
#endif

void siv_1D(const simulationParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT);

void siv_2D(const simulationParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT);

void siv_3D(const simulationParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT);

void saveIonsVariables(const simulationParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT);



void saveFieldsVariables(const simulationParameters * params, oneDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void saveFieldsVariables(const simulationParameters * params, twoDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void saveFieldsVariables(const simulationParameters * params, threeDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void armaCastDoubleToFloat(vec * doubleVector, fvec * floatVector);

public:

HDF(simulationParameters *params,meshGeometry *mesh,vector<ionSpecies> *IONS);

void saveOutputs(const simulationParameters * params, const vector<ionSpecies> * IONS_OUT, fields * EB, const characteristicScales * CS, const int IT, double totalTime);

};


#endif
