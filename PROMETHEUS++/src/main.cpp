#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <ctime>

#include "structures.h"
#include "initialize.h"
#include "PIC.h"
#include "emf.h"
#include "units.h"
#include "generalFunctions.h"
#include "outputHDF5.h"
#include "alfvenic.h"
#include "timeSteppingMethods.h"

#include <omp.h>
#include "mpi_main.h"

using namespace std;
using namespace arma;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	MPI_MAIN mpi_main;

	inputParameters params; 		// Input parameters for the simulation.
	vector<ionSpecies> IONS; 		// Vector of ionsSpecies structures each of them storing the properties of each ion species.
	characteristicScales CS;		// Derived type for keeping info about characteristic scales.
	meshGeometry mesh; 				// Derived type with info of geometry of the simulation mesh (initially with units).
	fields EB; 						// Derived type with variables of electromagnetic fields.

	INITIALIZE init(&params, argc, argv);

	mpi_main.createMPITopology(&params);

	init.loadIonParameters(&params, &IONS);

	UNITS units;
	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);

	init.loadMeshGeometry(&params, &CS, &mesh);

	init.initializeFields(&params, &mesh, &EB);

	init.setupIonsInitialCondition(&params, &CS, &mesh, &IONS); // Calculation of IONS[ii].NCP for each species

	HDF hdfObj(&params, &mesh, &IONS); // Outputs in HDF5 format

	ALFVENIC alfvenPerturbations(&params, &mesh, &EB, &IONS); // Include Alfvenic perturbations in the initial condition

	units.defineTimeStep(&params, &mesh, &IONS, &EB);

	/*By calling this function we set up some of the simulation parameters and normalize the variables*/
	units.normalizeVariables(&params, &mesh, &IONS, &EB, &CS);

	alfvenPerturbations.normalize(&CS);

	/**************** All the quantities below are dimensionless ****************/

	alfvenPerturbations.addPerturbations(&params, &IONS, &EB);

	TIME_STEPPING_METHODS timeStepping(&params);

	switch (params.particleIntegrator){
		case(1):{
				timeStepping.advanceFullOrbitIonsAndMasslessElectrons(&params, &mesh, &CS, &hdfObj, &IONS, &EB);
				break;
				}
		case(2):{
				timeStepping.advanceFullOrbitIonsAndMasslessElectrons(&params, &mesh, &CS, &hdfObj, &IONS, &EB);
				break;
				}
		case(3):{
				timeStepping.advanceGCIonsAndMasslessElectrons(&params, &mesh, &CS, &hdfObj, &IONS, &EB);
				break;
				}
		default:{
				timeStepping.advanceFullOrbitIonsAndMasslessElectrons(&params, &mesh, &CS, &hdfObj, &IONS, &EB);
				}
	}

	MPI_Finalize();

	return(0);
}
