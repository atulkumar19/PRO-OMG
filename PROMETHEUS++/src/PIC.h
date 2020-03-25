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

#include "armadillo"
#include "structures.h"
#include "boundaryConditions.h"
#include "generalFunctions.h"

#include "mpi_main.h"

//#include <cfenv>
//#pragma STDC FENV_ACCESS ON

using namespace std;
using namespace arma;

template <class I, class F> class PIC{

	int NX;
	int NY;
	int NZ;
	int ix;
	int iy;
	int iz;
	arma::vec x;
	uvec logic;

protected:

	void MPI_BcastDensity(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void MPI_BcastDensity(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void MPI_BcastBulkVelocity(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void MPI_BcastBulkVelocity(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void MPI_AllgatherField(const simulationParameters * params, vfield_vec * field);

	void MPI_AllgatherField(const simulationParameters * params, arma::vec * field);


	void smooth_TOS(arma::vec * v, double as);

	void smooth_TOS(arma::mat * m, double as);

	void smooth_TOS(vfield_vec * vf, double as);

	void smooth_TOS(vfield_mat * vf, double as);

	void smooth_TSC(arma::vec * v, double as);

	void smooth_TSC(arma::mat * m, double as);

	void smooth_TSC(vfield_vec * vf, double as);

	void smooth_TSC(vfield_mat * vf, double as);

	void smooth(arma::vec * v, double as);

	void smooth(arma::mat * m, double as);

	void smooth(vfield_vec * vf, double as);

	void smooth(vfield_mat * vf, double as);

	void crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB);


	void eivTOS(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void eivTSC(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void extrapolateIonVelocity(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void eivTOS(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void eivTSC(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void extrapolateIonVelocity(const simulationParameters * params, twoDimensional::ionSpecies * IONS);


	void eidTOS(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void eidTSC(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void extrapolateIonDensity(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void eidTOS(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void eidTSC(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void extrapolateIonDensity(const simulationParameters * params, twoDimensional::ionSpecies * IONS);


	void EMF_TOS_1D(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, vfield_vec * fields, arma::mat * F);

	void EMF_TSC_1D(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, vfield_vec * fields, arma::mat * F);

	void interpolateElectromagneticFields_1D(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, fields * EB, arma::mat * E, arma::mat * B);


	void EMF_TSC_2D(const ionSpecies * IONS, vfield_cube * fields, arma::mat * F);

	void EMF_TSC_3D(const ionSpecies * IONS, vfield_cube * fields, arma::mat * F);



	void aiv_Vay_1D(const simulationParameters * params, const characteristicScales * CS, fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT);


	void aiv_Boris_1D(const simulationParameters * params, const characteristicScales * CS, fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT);


	void aip(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, const double DT);

	void aip(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, const double DT);

  public:

	PIC();

	void assignCell(const simulationParameters * params, oneDimensional::ionSpecies * IONS);

	void assignCell(const simulationParameters * params, twoDimensional::ionSpecies * IONS);

	void advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, Y * EB, vector<T> * IONS, const double DT);

	void advanceIonsPosition(const simulationParameters * params, vector<T> * IONS, const double DT);
};

/*
class PIC_GC : public PIC{
private:

	int NX;
	double LX;

	struct GC_VARS{
		double Q;
		double M;

		arma::vec wx;
		arma::vec B;
		arma::vec Bs;
		arma::vec E;
		arma::vec Es;
		arma::vec b;

		int mn;
		double mu;

		double Xo;
		double Pparo;
		double go;

		double Xo_;
		double Pparo_;
		double go_;

		double X;
		double Ppar;
		double g;
	};

protected:

	// double Tol;
	#define Tol 1E-12

	// Runge-Kutta 45 (Dorman-Prince) methd
	arma::mat::fixed<7,7> A;
	arma::vec::fixed<7> B4;
	arma::vec::fixed<7> B5;

	arma::vec K1;
	arma::vec K2;
	arma::vec K3;
	arma::vec K4;
	arma::vec K5;
	arma::vec K6;
	arma::vec K7;

	arma::vec S4;
	arma::vec S5;

	// GC_VARS gcv;

	void set_to_zero_RK45_variables();


	void set_to_zero_GC_vars(PIC_GC::GC_VARS * gcv);

	void set_GC_vars(ionSpecies * IONS, PIC_GC::GC_VARS * gcv, int pp);

	void reset_GC_vars(PIC_GC::GC_VARS * gcv);


	void depositIonDensityAndBulkVelocity(const simulationParameters * params, const meshParams * mesh, ionSpecies * IONS);


	void EFF_EMF_TSC_1D(const double DT, const double DX, GC_VARS * gcv, const fields * EB);


	void assignCell_TSC(const simulationParameters * params, const meshParams * mesh, GC_VARS * gcv, int dim);


	void computeFullOrbitVelocity(const simulationParameters * params, const meshParams * mesh, const fields * EB, GC_VARS * gcv, arma::rowvec * V, int dim);


	void advanceRungeKutta45Stages_1D(const simulationParameters * params, const meshParams * mesh, double * DT_RK, GC_VARS * gcv, const fields * EB, int STG);


	void ai_GC_1D(const simulationParameters * params, const characteristicScales * CS, const meshParams * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT);

	void ai_GC_2D(const simulationParameters * params, const characteristicScales * CS, const meshParams * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT);

	void ai_GC_3D(const simulationParameters * params, const characteristicScales * CS, const meshParams * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT);

public:

	PIC_GC(const simulationParameters * params, const meshParams * mesh);

	void assignCell(const simulationParameters * params, const meshParams * mesh, GC_VARS * gcv, int dim);

	void advanceGCIons(const simulationParameters * params, const characteristicScales * CS, const meshParams * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT);
};
*/
#endif
