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

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <typeinfo>

#include <armadillo>

#include "structures.h"

#include "randomStart.h"
#include "quietStart.h"
#include "PIC.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

using namespace oneDimensional;

template <class T> class INITIALIZE{

	double ionSkinDepth;
	double LarmorRadius;

	vector<string> split(const string& str, const string& delim);

	map<string, float> loadParameters(string *  inputFile);

	map<string, string> loadParametersString(string *  inputFile);

public:

	INITIALIZE(simulationParameters * params, int argc, char* argv[]);

	void loadInputParameters(simulationParameters * params, int argc, char* argv[]);

	void loadMeshGeometry(const simulationParameters * params, fundamentalScales * FS, meshParams * mesh);

	void loadIonParameters(simulationParameters * params, vector<T> * IONS,  vector<GCSpecies> * GCP);

	void setupIonsInitialCondition(const simulationParameters * params, const characteristicScales * CS, const meshParams * mesh, vector<ionSpecies> * IONS);

	void initializeFields(const simulationParameters * params,  const meshParams * mesh,  fields * EB);

};

#endif
