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

void siv_1D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT);

void siv_2D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT);

void siv_3D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT);

void saveIonsVariables(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT);



void saveFieldsVariables(const inputParameters * params, oneDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void saveFieldsVariables(const inputParameters * params, twoDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void saveFieldsVariables(const inputParameters * params, threeDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT);

void armaCastDoubleToFloat(vec * doubleVector, fvec * floatVector);

public:

HDF(inputParameters *params,meshGeometry *mesh,vector<ionSpecies> *IONS);

void saveOutputs(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, emf * EB, const characteristicScales * CS, const int IT, double totalTime);

};


#endif
