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

	int dim_x;

	double n_cs;

	vec ne;			// Electron plasma density at time level "l + 1"

	vec n; 			// Total plasma density at time level "l + 1"

	vec n_; 		// Total plasma density at time level "l - 1/2"

	vec n__; 		// Total plasma density at time level "l - 3/2"

	vfield_vec V; 	// Extrapolated ions' bulk velocity at time level "l + 1"

	vfield_vec U; 	// Ions' bulk velocity at time level "l + 1/2"

	vfield_vec U_; 	// Ions' bulk velocity at time level "l - 1/2"

	vfield_vec U__; // Ions' bulk velocity at time level "l - 3/2"

	vfield_vec Ui; // Ions' bulk velocity at time level "l - 3/2"

	vfield_vec Ui_; // Ions' bulk velocity at time level "l - 3/2"

	vfield_vec Ui__; // Ions' bulk velocity at time level "l - 3/2"

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB);

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB);

	void FaradaysLaw(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB);


	void aef_1D(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS);

	void aef_2D(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS);

	void aef_3D(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS);

  public:

	EMF_SOLVER(){};

	EMF_SOLVER(const inputParameters * params, characteristicScales * CS);

	void smooth_TOS(const inputParameters * params,vfield_vec * vf,double as);

	void smooth_TOS(const inputParameters * params,vfield_mat * vf,double as);

	void smooth_TSC(const inputParameters * params,vfield_vec * vf,double as);

	void smooth_TSC(const inputParameters * params,vfield_mat * vf,double as);

	void smooth(const inputParameters * params,vfield_vec * vf,double as);

	void smooth(const inputParameters * params,vfield_mat * vf,double as);

	void equilibrium(const inputParameters * params,vector<ionSpecies> * IONS,emf * EB);

	void advanceBField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS);

	void advanceEField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,const int BAE);

};

#endif
