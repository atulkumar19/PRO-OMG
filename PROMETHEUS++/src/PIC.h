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
#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"
#include "structures.h"
#include "boundaryConditions.h"

#include "mpi_main.h"

//#include <cfenv>
//#pragma STDC FENV_ACCESS ON

using namespace std;
using namespace arma;

class PIC{

	int NX;
	int NY;
	int NZ;
	int ix;
	int iy;
	int iz;
	arma::vec x;
	uvec logic;

protected:

	void MPI_AllreduceVec(const simulationParameters * params, arma::vec * v);

	void MPI_AllreduceMat(const simulationParameters * params, arma::mat * m);

	void MPI_SendVec(const simulationParameters * params, arma::vec * v);

	void MPI_ReduceVec(const simulationParameters * params, arma::vec * v);

	void MPI_SendMat(const simulationParameters * params, arma::mat * m);

	void MPI_ReduceMat(const simulationParameters * params, arma::mat * m);

	void MPI_Allgathervec(const simulationParameters * params, arma::vec * field);

	void MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield);

	void MPI_Allgathermat(const simulationParameters * params, arma::mat * field);

	void MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield);

	void MPI_Recvvec(const simulationParameters * params, arma::vec * field);

	void MPI_Recvvfield_vec(const simulationParameters * params, vfield_vec * vfield);

	void MPI_Recvmat(const simulationParameters * params, arma::mat * field);

	void MPI_Recvvfield_mat(const simulationParameters * params, vfield_mat * vfield);

	void include4GhostsContributions(arma::vec * v);

	//void include4GhostsContributions(arma::mat * m);

	//void fill4Ghosts(arma::vec * v);

	//void fill4Ghosts(arma::mat * m);


	void smooth(arma::vec * v, double as);

	void smooth(arma::mat * m, double as);

	void smooth(vfield_vec * vf, double as);

	void smooth(vfield_mat * vf, double as);


	void crossProduct(const arma::vec A, const arma::vec B, arma::vec * AxB);

	void crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB);


	void interpolateVectorField(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, vfield_vec * field, arma::mat * F);

	void interpolateElectromagneticFields(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, oneDimensional::fields * EB, arma::mat * E, arma::mat * B);

	void interpolateVectorField(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, vfield_mat * field, arma::mat * F);

	void interpolateElectromagneticFields(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, twoDimensional::fields * EB, arma::mat * E, arma::mat * B);


	// PRO++ OMG:
	void eim(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);

	void eim(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);

	void calculateIonMoments(const simulationParameters * params, const characteristicScales * CS,oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);

	void calculateIonMoments(const simulationParameters * params, const characteristicScales * CS,twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);

	void calculateDerivedIonMoments(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void calculateDerivedIonMoments(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec field, arma::vec * F);

	void interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat field, arma::mat * F);

  public:

	PIC();

	void assignCell(const simulationParameters * params, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);

	void assignCell(const simulationParameters * params, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);


	void advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT);

	void advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT);


	void advanceIonsPosition(const simulationParameters * params,  oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT);

	void advanceIonsPosition(const simulationParameters * params,  twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT);


	// PRO++ OMG additions:

	void extrapolateIonsMoments(const simulationParameters * params, const characteristicScales * CS,oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);

	void extrapolateIonsMoments(const simulationParameters * params, const characteristicScales * CS,twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

};

#endif
