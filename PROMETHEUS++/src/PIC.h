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

	int NX;
	int NY;
	int NZ;
	int ix;
	int iy;
	int iz;
	arma::vec x;
	uvec logic;

	// Runge-Kutta 45 (Dorman-Prince) methd
	arma::mat::fixed<7,7> A;
	arma::vec::fixed<7> B4;
	arma::vec::fixed<7> B5;

	void MPI_BcastDensity(const inputParameters * params, ionSpecies * ions);

	void MPI_BcastBulkVelocity(const inputParameters * params, ionSpecies * ions);

	void MPI_AllgatherField(const inputParameters * params, vfield_vec * field);


	void smooth_TOS(arma::vec * v, double as);

	void smooth_TOS(vfield_vec * vf, double as);

	void smooth_TOS(vfield_mat * vf, double as);

	void smooth_TSC(arma::vec * v, double as);

	void smooth_TSC(vfield_vec * vf, double as);

	void smooth_TSC(vfield_mat * vf, double as);

	void smooth(arma::vec * v, double as);

	void smooth(vfield_vec * vf, double as);

	void smooth(vfield_mat * vf, double as);

	void crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB);


	void assingCell_TOS(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions, int dim);

	void assingCell_TSC(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions, int dim);

	void assingCell(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions, int dim);


	void eivTOS_1D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eivTSC_1D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eivTSC_2D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eivTSC_3D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void extrapolateIonVelocity(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);


	void eidTOS_1D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eidTSC_1D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eidTSC_2D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void eidTSC_3D(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);

	void extrapolateIonDensity(const inputParameters * params, const meshGeometry * mesh, ionSpecies * ions);


	void EMF_TOS_1D(const inputParameters * params, const ionSpecies * ions, vfield_vec * EMF, arma::mat * F);

	void EMF_TSC_1D(const inputParameters * params, const ionSpecies * ions, vfield_vec * emf, arma::mat * F);

	void EMF_TSC_2D(const meshGeometry * mesh, const ionSpecies * ions, vfield_cube * emf, arma::mat * F);

	void EMF_TSC_3D(const meshGeometry * mesh, const ionSpecies * ions, vfield_cube * emf, arma::mat * F);


	void aiv_Vay_1D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void aiv_Vay_2D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void aiv_Vay_3D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);


	void aiv_Boris_1D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void aiv_Boris_2D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void aiv_Boris_3D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);


	void aip_1D(const inputParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT);

	void aip_2D(const inputParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT);

	void aip_3D(const inputParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT);


	void getEffectiveFields();

	void ai_GC_1D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void ai_GC_2D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void ai_GC_3D(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

  public:

	PIC();

	void advanceIonsVelocity(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);

	void advanceIonsPosition(const inputParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT);

	void advanceGCIons(const inputParameters * params, const characteristicScales * CS, const meshGeometry * mesh, emf * EB, vector<ionSpecies> * IONS, const double DT);
};

#endif
