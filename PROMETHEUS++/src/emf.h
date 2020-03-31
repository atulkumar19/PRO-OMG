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

	double dt;//Time step for the RK4 function

	oneDimensional::fields AUX; // Temporary fields of the four stages of the RK calculation of the magnetic field.
	oneDimensional::fields  K1; // Temporary fields of the four stages of the RK calculation of the magnetic field.
	oneDimensional::fields  K2; // Temporary fields of the four stages of the RK calculation of the magnetic field.
	oneDimensional::fields  K3; // Temporary fields of the four stages of the RK calculation of the magnetic field.
	oneDimensional::fields  K4; // Temporary fields of the four stages of the RK calculation of the magnetic field.

	int NX_S; // Number of grid cells per subdomain, including 2 ghost cells.
	int NX_T; // Number of grid cells in entire simulation domain, including 2 ghost cells.
	int NX_R; // Number of grid cells in entire simulation domain, not including ghost cells.

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


	void MPI_AllgatherField(const simulationParameters * params, arma::vec * field);

	void MPI_AllgatherField(const simulationParameters * params, vfield_vec * field);

	void MPI_passGhosts(const simulationParameters * params, vfield_vec * field);

	void MPI_passGhosts(const simulationParameters * params, arma::vec * field);


	void FaradaysLaw(const simulationParameters * params, oneDimensional::fields * EB);

	void FaradaysLaw(const simulationParameters * params, twoDimensional::fields * EB);


	void aef_1D(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);

	void aef_2D(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

  public:

	EMF_SOLVER(){};

	EMF_SOLVER(const simulationParameters * params, characteristicScales * CS);

	void smooth_TOS(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth_TOS(const simulationParameters * params, vfield_mat * vf, double as);

	void smooth_TSC(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth_TSC(const simulationParameters * params, vfield_mat * vf, double as);

	void smooth(const simulationParameters * params, vfield_vec * vf, double as);

	void smooth(const simulationParameters * params, vfield_mat * vf, double as);

	void equilibrium(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, oneDimensional::fields * EB);

	void advanceBField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);

	void advanceEField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);

	void advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const int BAE);

	void advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const int BAE);


};

#endif
