#ifndef H_PIC
#define H_PIC

#include <iostream>
#include <cmath>
#include <vector>

#ifdef __linux__
#include <chrono>//C++11 standard
#include <random>//C++11 standard

#elif __APPLE__
//define something else for apple compiler
#endif

#include <omp.h>

#include "armadillo"
#include "structures.h"
#include "boundaryConditions.h"
#include "generalFunctions.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

using namespace oneDimensional;

class PIC{

	int NX, NY, NZ;
	int ix, iy, iz;
	vec x;
	uvec logic;


	void MPI_BcastDensity(const inputParameters * params,ionSpecies * ions);

	void MPI_BcastBulkVelocity(const inputParameters * params,ionSpecies * ions);

	void MPI_AllgatherField(const inputParameters * params,vfield_vec * field);


	void smooth_TOS(vec * v, double as);

	void smooth_TOS(vfield_vec * vf,double as);

	void smooth_TOS(vfield_mat * vf,double as);

	void smooth_TSC(vec * v,double as);

	void smooth_TSC(vfield_vec * vf,double as);

	void smooth_TSC(vfield_mat * vf,double as);

	void smooth(vec * v,double as);

	void smooth(vfield_vec * vf,double as);

	void smooth(vfield_mat * vf,double as);

	void crossProduct(const mat * A,const mat * B,mat * AxB);


	void assingCell_TOS(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions,int dim);

	void assingCell_TSC(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions,int dim);

	void assingCell(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions,int dim);


	void eivTOS_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eivTSC_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eivTSC_2D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eivTSC_3D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void extrapolateIonVelocity(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);



	void eidTOS_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eidTSC_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eidTSC_2D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void eidTSC_3D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);

	void extrapolateIonDensity(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions);


	void EMF_TOS_1D(const inputParameters * params,const ionSpecies * ions,vfield_vec * EMF,mat * F);

	void EMF_TSC_1D(const inputParameters * params,const ionSpecies * ions,vfield_vec * emf,mat * F);

	void EMF_TSC_2D(const meshGeometry * mesh,const ionSpecies * ions,vfield_cube * emf,mat * F);

	void EMF_TSC_3D(const meshGeometry * mesh,const ionSpecies * ions,vfield_cube * emf,mat * F);


	void aiv_1D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT);

	void aiv_2D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT);

	void aiv_3D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT);


	void aip_1D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT);

	void aip_2D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT);

	void aip_3D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT);

  public:

	PIC(){};

	void ionVariables(vector<ionSpecies> * IONS,vector<ionSpecies> * copyIONS,const int flag);

	void advanceIonsVelocity(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT);

	void advanceIonsPosition(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT);
};

#endif
