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
#include "fields.h"
#include "units.h"
#include "generalFunctions.h"
#include "outputHDF5.h"

#include <omp.h>
#include "mpi_main.h"

using namespace std;
using namespace arma;

int main(int argc, char* argv[]){
	// namespace nDimensional = oneDimensional;
	namespace nDimensional = twoDimensional;

	MPI_Init(&argc, &argv);
	MPI_MAIN mpi_main;

	simulationParameters params; 				// Input parameters for the simulation.
	vector<nDimensional::ionSpecies> IONS; 		// Vector of ionsSpecies structures each of them storing the properties of each ion species.
	characteristicScales CS;					// Derived type for keeping info about characteristic scales.
	nDimensional::fields EB; 					// Derived type with variables of electromagnetic fields.

	INITIALIZE<nDimensional::ionSpecies, nDimensional::fields> init(&params, argc, argv);

	mpi_main.createMPITopology(&params);

	init.loadIonParameters(&params, &IONS);

	UNITS<nDimensional::ionSpecies, nDimensional::fields> units;

	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);

	fundamentalScales FS(&params);

	units.calculateFundamentalScalesAndBcast(&params, &IONS, &FS);

	init.loadMeshGeometry(&params, &FS);

	units.spatialScalesSanityCheck(&params, &FS);

	init.initializeFields(&params, &EB);

	init.setupIonsInitialCondition(&params, &CS, &IONS); // Calculation of IONS[ii].NCP for each species

	HDF<nDimensional::ionSpecies, nDimensional::fields> hdfObj(&params, &FS, &IONS); // Outputs in HDF5 format

	units.defineTimeStep(&params, &IONS);

	/*By calling this function we set up some of the simulation parameters and normalize the variables*/
	units.normalizeVariables(&params, &IONS, &EB, &CS);

	/**************** All the quantities below are dimensionless ****************/

	// Definition of variables for advancing in time particles and fields.
	double t1 = 0.0;
    double t2 = 0.0;
    double currentTime = 0.0;
    int outputIterator = 0;
	int numberOfIterationsForEstimator = 1000;

	EMF_SOLVER fields_solver(&params, &CS); // Initializing the EMF_SOLVER class object.
	PIC ionsDynamics; // Initializing the PIC class object.

	//*** @tomodiify
	// GENERAL_FUNCTIONS genFun;

    // Repeat 3 times
    for(int tt=0; tt<3; tt++){
        ionsDynamics.advanceIonsPosition(&params, &IONS, 0);

        ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0);
    }
    // Repeat 3 times

    hdfObj.saveOutputs(&params, &IONS, &EB, &CS, 0, 0);

    t1 = MPI::Wtime();

    for(int tt=0; tt<params.timeIterations; tt++){ // Time iterations.
        if(tt == 0){
			//*** @tomodiify
            // genFun.checkStability(&params, &mesh, &CS, &IONS);

			ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0.5*params.DT); // Initial condition time level V^(1/2)
        }else{
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, params.DT); // Advance ions' velocity V^(N+1/2).
        }

        ionsDynamics.advanceIonsPosition(&params, &IONS, params.DT); // Advance ions' position in time to level X^(N+1).

        //*** @tomodiify
        fields_solver.advanceBField(&params, &EB, &IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

        //*** @tomodiify
        if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
         	// Using the Bashford-Adams extrapolation.
        	fields_solver.advanceEFieldWithVelocityExtrapolation(&params, &EB, &IONS, 1);
        }else{
			// Using basic velocity extrapolation.
        	fields_solver.advanceEFieldWithVelocityExtrapolation(&params, &EB, &IONS, 0);
	    }

        currentTime += params.DT*CS.time;

        if(fmod((double)(tt + 1), params.outputCadenceIterations) == 0){
			vector<nDimensional::ionSpecies> IONS_OUT = IONS;

            // The ions' velocity is advanced in time in order to obtain V^(N+1)
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS_OUT, 0.5*params.DT);

			hdfObj.saveOutputs(&params, &IONS_OUT, &EB, &CS, outputIterator+1, currentTime);

			outputIterator++;
        }

        //*** @tomodiify
        // if( (params.checkStability == 1) && fmod((double)(tt+1), params.rateOfChecking) == 0 ){
        //	genFun.checkStability(&params, &mesh, &CS, &IONS);
        // }

		//*** @tomodiify
		// genFun.checkEnergy(&params, &mesh, &CS, &IONS, &EB, tt);

        if(tt == numberOfIterationsForEstimator){
            t2 = MPI::Wtime();

			double estimatedSimulationTime = ( (double)params.timeIterations*(t2 - t1)/(double)numberOfIterationsForEstimator )/60.0;

			if(params.mpi.MPI_DOMAIN_NUMBER_CART == 0){
                cout << "ESTIMATED TIME OF COMPLETION: " << estimatedSimulationTime <<" MINUTES" << endl;
            }
        }
    } // Time iterations.

	//*** @tomodiify
    // genFun.saveDiagnosticsVariables(&params);

	mpi_main.finalizeCommunications(&params);

	return(0);
}
