#ifndef H_INITIALIZE
#define H_INITIALIZE

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>

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
