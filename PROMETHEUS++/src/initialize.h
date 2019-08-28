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

#include <armadillo>

#include "structures.h"

#include "randomStart.h"
#include "quietStart.h"
#include "PIC.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

using namespace oneDimensional;

class INITIALIZE{

	double ionSkinDepth, LarmorRadius;

	vector<string> split(const string& str, const string& delim);

	map<string,float> loadParameters(string *  inputFile);

	map<string,string> loadParametersString(string *  inputFile);

public:

	INITIALIZE(inputParameters * params,int argc,char* argv[]);

	void loadInputParameters(inputParameters * params,int argc,char* argv[]);

	void loadMeshGeometry(const inputParameters * params,characteristicScales * CS,meshGeometry * mesh);

	void loadIonParameters(inputParameters * params,vector<ionSpecies> * IONS);

	void setupIonsInitialCondition(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,vector<ionSpecies> * IONS);

	void initializeFields(const inputParameters * params, const meshGeometry * mesh, fields * EB);

};

#endif
