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
#include <typeinfo>

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
	namespace nDimensional = twoDimensional;

	MPI_Init(&argc, &argv);
	MPI_MAIN mpi_main;

	simulationParameters params; 				// Input parameters for the simulation.
	vector<nDimensional::ionSpecies> IONS; 	// Vector of ionsSpecies structures each of them storing the properties of each ion species.
	characteristicScales CS;					// Derived type for keeping info about characteristic scales.
	nDimensional::fields EB; 					// Derived type with variables of electromagnetic fields.

	INITIALIZE<nDimensional::ionSpecies, nDimensional::fields> init(&params, argc, argv);

	mpi_main.createMPITopology(&params);

	init.loadIonParameters(&params, &IONS); //*** @tomodify

	UNITS<nDimensional::ionSpecies, nDimensional::fields> units;

	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);

	fundamentalScales FS(&params);

	units.calculateFundamentalScalesAndBcast(&params, &IONS, &FS);

	init.loadMeshGeometry(&params, &FS);

	units.spatialScalesSanityCheck(&params, &FS);

	init.initializeFields(&params, &EB);

	init.setupIonsInitialCondition(&params, &CS, &IONS); // Calculation of IONS[ii].NCP for each species

	HDF<nDimensional::ionSpecies, nDimensional::fields> hdfObj(&params, &FS, &IONS); // Outputs in HDF5 format

	//*** @tomodify
	// ALFVENIC alfvenPerturbations(&params, &mesh, &EB, &IONS); // Include Alfvenic perturbations in the initial condition

	units.defineTimeStep(&params, &IONS);

	/*By calling this function we set up some of the simulation parameters and normalize the variables*/
	units.normalizeVariables(&params, &IONS, &EB, &CS);

	//*** @tomodify
	// alfvenPerturbations.normalize(&CS);

	/**************** All the quantities below are dimensionless ****************/

	//*** @tomodify
	// alfvenPerturbations.addPerturbations(&params, &IONS, &EB);

	TIME_STEPPING_METHODS<nDimensional::ionSpecies, nDimensional::fields> timeStepping(&params);

	switch (params.particleIntegrator){
		case(1):{
				timeStepping.advanceFullOrbitIonsAndMasslessElectrons(&params, &CS, &hdfObj, &IONS, &EB);
				break;
				}
		case(2):{
				timeStepping.advanceGCIonsAndMasslessElectrons(&params, &CS, &hdfObj, &IONS, &EB);
				break;
				}
		default:{
				timeStepping.advanceFullOrbitIonsAndMasslessElectrons(&params, &CS, &hdfObj, &IONS, &EB);
				}
	}

	mpi_main.finalizeCommunications(&params);

	return(0);
}
