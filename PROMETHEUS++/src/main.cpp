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

// Include intrinsic header files:
// ===============================
#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <ctime>
#include <typeinfo>
#include <utility>

// Include user-defined header files:
// ==================================
#include "structures.h"
#include "collisionOperator.h"
#include "initialize.h"
#include "PIC.h"
#include "fields.h"
#include "units.h"
#include "outputHDF5.h"

// Include headers for parallelization:
// ====================================
#include <omp.h>
#include "mpi_main.h"

using namespace std;
using namespace arma;

// Impletement templates:
// ======================
template <class IT, class FT> void main_run_simulation(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
        
        // Create simulation objects:
        // ==========================
        // MPI object to hold topology information:
	MPI_MAIN mpi_main;
        // Input parameters for simulation:
	simulationParameters params; 	
        // Ion species vector of type "IT":
	vector<IT> IONS; 
	// Characteristic scales:
        characteristicScales CS;
        // Electromagnetic fields of type "FT":
	FT EB; 						 
        // Collision operator object:
        collisionOperator FPCOLL;
        // Initialization object of type "FT" and "FT":
        INITIALIZE<IT, FT> init(&params, argc, argv);
        // UNITS object of type "FT" and "FT":
	UNITS<IT, FT> units;
        
        // Initialize simulation objects:
        // =============================       
        // Create MPI topology:
	mpi_main.createMPITopology(&params);
        // Read "ions_properties.ion" and populate "IONS" vector
	init.loadIonParameters(&params, &IONS);
        // Define characteristic scales and broadcast them to all processes in COMM_WORLD:
	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);
        // Create object and allocate memory according to "params":
	fundamentalScales FS(&params);
        // Define fundamental scales and broadcast them to all processes in COMM_WORLD:
	units.calculateFundamentalScalesAndBcast(&params, &IONS, &FS);

	init.loadMeshGeometry(&params, &FS);
          
        init.loadPlasmaProfiles(&params, &IONS); //Reads & loads plasma profiles from external files

	units.spatialScalesSanityCheck(&params, &FS);

	init.initializeFields(&params, &EB);
          

	init.setupIonsInitialCondition(&params, &CS, &EB, &IONS); // Calculation of IONS[ii].NCP for each species

	HDF<IT, FT> hdfObj(&params, &FS, &IONS); // Outputs in HDF5 format

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

    // Repeat 3 times
    for(int tt=0; tt<3; tt++){
        ionsDynamics.advanceIonsPosition(&params, &EB, &IONS, 0);

        ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0);
    }
    // Repeat 3 times

    hdfObj.saveOutputs(&params, &IONS, &EB, &CS, 0, 0);

    t1 = MPI::Wtime();

    for(int tt=0; tt<params.timeIterations; tt++){ // Time iterations.
        if(tt == 0){
		ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0.5*params.DT); // Initial condition time level V^(1/2)
        }else{
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, params.DT); // Advance ions' velocity V^(N+1/2).
        }
        //if(tt==14){
                //cout<<"There is going to an error error"<<endl;
               // }
        ionsDynamics.advanceIonsPosition(&params,&EB, &IONS, params.DT); // Advance ions' position in time to level X^(N+1).


        //fields_solver.advanceBField(&params, &EB, &IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

        if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
         	// Using the Bashford-Adams extrapolation.
			fields_solver.advanceEField(&params, &EB, &IONS, true, true);
        }else{
			// Using basic velocity extrapolation.
			fields_solver.advanceEField(&params, &EB, &IONS, true, false);
	    }

		currentTime += params.DT*CS.time;

        if(fmod((double)(tt + 1), params.outputCadenceIterations) == 0){
			vector<IT> IONS_OUT = IONS;

            // The ions' velocity is advanced in time in order to obtain V^(N+1)
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS_OUT, 0.5*params.DT);

			hdfObj.saveOutputs(&params, &IONS_OUT, &EB, &CS, outputIterator+1, currentTime);

			outputIterator++;
		}

		// Estimate simulation time
        if(tt == numberOfIterationsForEstimator){
            t2 = MPI::Wtime();

			double estimatedSimulationTime = ( (double)params.timeIterations*(t2 - t1)/(double)numberOfIterationsForEstimator )/60.0;

			if(params.mpi.MPI_DOMAIN_NUMBER == 0){
                cout << "ESTIMATED TIME OF COMPLETION: " << estimatedSimulationTime <<" MINUTES" << endl;
            }
        }
    } // Time iterations.

	mpi_main.finalizeCommunications(&params);
}


// MAIN function:
// ==============
int main(int argc, char* argv[]){

        // Create dummy variables as inputs to variables:
        // ==============================================
        // One-dimensional:
	oneDimensional::ionSpecies dummy_ions_1D;
	oneDimensional::fields dummy_fields_1D;
        // Two-dimensional:
	twoDimensional::ionSpecies dummy_ions_2D;
	twoDimensional::fields dummy_fields_2D;
        
        // Select dimensionality:
        // ======================
	if (strcmp(argv[1],"1-D") == 0)
        {
            main_run_simulation<decltype(dummy_ions_1D), decltype(dummy_fields_1D)>(argc, argv);
	}
        else
        {
            main_run_simulation<decltype(dummy_ions_2D), decltype(dummy_fields_2D)>(argc, argv);
	}

	return(0);
}
