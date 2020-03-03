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

	simulationParameters params; 	// Input parameters for the simulation.
	vector<oneDimensional::ionSpecies> IONS; 		// Vector of ionsSpecies structures each of them storing the properties of each ion species.
	vector<oneDimensional::GCSpecies> GCP;
	characteristicScales CS;		// Derived type for keeping info about characteristic scales.
	meshParams mesh; 				// Derived type with info of geometry of the simulation mesh (initially with units).
	oneDimensional::fields EB; 						// Derived type with variables of electromagnetic fields.

	vector<twoDimensional::ionSpecies> IONS_2D; //*** @todelete
	twoDimensional::fields EB_2D; //*** @todelete

	INITIALIZE init(&params, argc, argv);

	mpi_main.createMPITopology(&params);

	init.loadIonParameters(&params, &IONS, &GCP); //*** @tomodify

	UNITS units;

	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);

	fundamentalScales FS(&params);

	units.calculateFundamentalScalesAndBcast(&params, &IONS, &FS);

	init.loadMeshGeometry(&params, &FS, &mesh);

	units.spatialScalesSanityCheck(&params, &FS, &mesh);

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

	mpi_main.finalizeCommunications(&params);

	return(0);
}
