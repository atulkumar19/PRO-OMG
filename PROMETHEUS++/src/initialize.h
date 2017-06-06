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

#include "mpi_main.h"

using namespace std;
using namespace arma;

using namespace oneDimensional;

class INITIALIZE{

	double ionSkinDepth, LarmorRadius;

	map<string,float> loadParameters(const char *  inputFile);
	map<string,float> loadParameters(string *  inputFile);

	void ringLikeVelocityDistribution(const inputParameters * params,ionSpecies * ions,const string parDirection);

	void beamVelocityDistribution(const inputParameters * params,ionSpecies * ions,const string parDirection);

	void maxwellianVelocityDistribution(const inputParameters * params,ionSpecies * ions,const string parDirection);

public:

	INITIALIZE(inputParameters * params,int argc,char* argv[]);

	void loadInputParameters(inputParameters * params,int argc,char* argv[]);

	void loadMeshGeometry(const inputParameters * params,characteristicScales * CS,meshGeometry * mesh);

	void calculateSuperParticleNumberDensity(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,vector<ionSpecies> * IONS);

	void loadIons(inputParameters * params,vector<ionSpecies> * IONS);

	void initializeFields(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS);

};

#endif
