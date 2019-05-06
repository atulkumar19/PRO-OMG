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

	void dimensionlessForm(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB,const characteristicScales * CS);

public:

	UNITS(){};

	void defineTimeStep(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB);

	void defineCharacteristicScales(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS);

	void normalizeVariables(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB,const characteristicScales * CS);

	void defineCharacteristicScalesAndBcast(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS);

};

#endif
