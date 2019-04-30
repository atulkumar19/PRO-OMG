#ifndef H_EMF_SOLVER
#define H_EMF_SOLVER

#include <iostream>
#include <cmath>
#include <vector>

#include "armadillo"
#include "structures.h"
#include "boundaryConditions.h"
#include "generalFunctions.h"
#include "types.h"

#include "mpi_main.h"

using namespace std;
using namespace arma;

class EMF_SOLVER{

	void MPI_passGhosts(const inputParameters * params,vfield_vec * field);

	double dt;//Time step for the RK4 function

	emf AUX, K1, K2, K3, K4;

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB);

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB);

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB);


	void aef_1D(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS);

	void aef_2D(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS);

	void aef_3D(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS);

  public:

	EMF_SOLVER(){};

	void smooth_TOS(const inputParameters * params,vfield_vec * vf,double as);

	void smooth_TOS(const inputParameters * params,vfield_mat * vf,double as);

	void smooth_TSC(const inputParameters * params,vfield_vec * vf,double as);

	void smooth_TSC(const inputParameters * params,vfield_mat * vf,double as);

	void smooth(const inputParameters * params,vfield_vec * vf,double as);

	void smooth(const inputParameters * params,vfield_mat * vf,double as);

	void equilibrium(const inputParameters * params,vector<ionSpecies> * IONS,emf * EB,characteristicScales * CS);

	void advanceBField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,characteristicScales * CS);

	void advanceEField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,characteristicScales * CS);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE);

};

#endif
