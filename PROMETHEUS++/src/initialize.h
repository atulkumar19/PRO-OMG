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

#ifndef H_INITIALIZE
#define H_INITIALIZE

// Intrinsic header files:
// =======================
#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <typeinfo>

// Armadillo header:
// =================
#include <armadillo>

// User-defined headers:
// =====================
#include "structures.h"
#include "randomStart.h"
#include "quietStart.h"
#include "PIC.h"
#include "mpi_main.h"

using namespace std;
using namespace arma;

// using namespace oneDimensional;

template <class IT, class FT> class INITIALIZE
{
	vector<string> split(const string& str, const string& delim);

	map<string, string> ReadAndloadInputFile(string *  inputFile);

	void initializeParticlesArrays(const simulationParameters * params, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);

	void initializeParticlesArrays(const simulationParameters * params, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);

	void initializeBulkVariablesArrays(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void initializeBulkVariablesArrays(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

public:

	INITIALIZE(simulationParameters * params, int argc, char* argv[]);

	void loadInputParameters(simulationParameters * params, int argc, char* argv[]);

	void loadMeshGeometry(simulationParameters * params, fundamentalScales * FS);

  void loadIonParameters(simulationParameters * params, vector<IT> * IONS);

  void loadPlasmaProfiles(simulationParameters * params, vector<IT> * IONS);

	void setupIonsInitialCondition(const simulationParameters * params, const characteristicScales * CS, FT * EB, vector<IT> * IONS);

	void initializeFieldsSizeAndValue(const simulationParameters * params, oneDimensional::fields * EB);

	void initializeFieldsSizeAndValue(const simulationParameters * params, twoDimensional::fields * EB);

	void initializeFields(const simulationParameters * params, FT * EB);

};

#endif
