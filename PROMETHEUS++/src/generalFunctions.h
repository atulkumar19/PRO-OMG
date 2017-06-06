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

	void bCastTimestep(inputParameters * params, int logicVariable);

	public:

	GENERAL_FUNCTIONS(){};

	void checkStability(inputParameters * params, const meshGeometry *mesh, const characteristicScales * CS, const vector<ionSpecies> * IONS);

	void checkEnergy(inputParameters * params, meshGeometry *mesh, characteristicScales * CS, vector<ionSpecies> * IONS, emf * EB, int IT);

	void saveDiagnosticsVariables(inputParameters * params);

};

#endif
