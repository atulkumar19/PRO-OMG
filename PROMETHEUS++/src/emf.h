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

	void MPI_passGhosts(const simulationParameters * params, vfield_vec * field);

	void MPI_passGhosts(const simulationParameters * params, arma::vec * field);

	double dt;//Time step for the RK4 function

	fields AUX;
	fields  K1, K2, K3, K4; // Temporary fields of the four stages of the RK calculation of the magnetic field.

	int dim_x; // Number of grid cells per subdomain including ghost cells.

	double n_cs;

	arma::vec ne;			// Electron plasma density at time level "l + 1"
	arma::vec n; 			// Total plasma density at time level "l + 1"
	arma::vec n_; 		// Total plasma density at time level "l - 1/2"
	arma::vec n__; 		// Total plasma density at time level "l - 3/2"

	vfield_vec V; 	// Extrapolated ions' bulk velocity at time level "l + 1"
	vfield_vec U; 	// Ions' bulk velocity at time level "l + 1/2"
	vfield_vec U_; 	// Ions' bulk velocity at time level "l - 1/2"
	vfield_vec U__; // Ions' bulk velocity at time level "l - 3/2"
	vfield_vec Ui; // Ions' bulk velocity at time level "l - 3/2"
	vfield_vec Ui_; // Ions' bulk velocity at time level "l - 3/2"
	vfield_vec Ui__; // Ions' bulk velocity at time level "l - 3/2"

	vfield_vec curlB; // Vector field keeping the values of curl(B) used in the electric field calculation.

	void FaradaysLaw(const simulationParameters * params, const meshGeometry * mesh, oneDimensional::electromagneticFields * EB);

	void FaradaysLaw(const simulationParameters * params, const meshGeometry * mesh, twoDimensional::electromagneticFields * EB);

	void FaradaysLaw(const simulationParameters * params, const meshGeometry * mesh, threeDimensional::electromagneticFields * EB);


	void aef_1D(const simulationParameters * params, const meshGeometry * mesh, oneDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS);

	void aef_2D(const simulationParameters * params, const meshGeometry * mesh, twoDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS);

	void aef_3D(const simulationParameters * params, const meshGeometry * mesh, threeDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS);

  public:

	EMF_SOLVER(){};

	EMF_SOLVER(const simulationParameters * params, characteristicScales * CS);

	void smooth_TOS(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth_TOS(const simulationParameters * params, vfield_mat * vf, double as);

	void smooth_TSC(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth_TSC(const simulationParameters * params, vfield_mat * vf, double as);

	void smooth(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth(const simulationParameters * params, vfield_mat * vf, double as);

	void equilibrium(const simulationParameters * params, vector<ionSpecies> * IONS, fields * EB);

	void advanceBField(const simulationParameters * params, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS);

	void advanceEField(const simulationParameters * params, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS);

	void advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, const meshGeometry * mesh, oneDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS, const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, const meshGeometry * mesh, twoDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS, const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, const meshGeometry * mesh, threeDimensional::electromagneticFields * EB, vector<ionSpecies> * IONS, const int BAE);

};

#endif
