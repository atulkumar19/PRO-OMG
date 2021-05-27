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

// Implement templates:
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
    // Initialize "params" based on input file:
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
    // Define mesh geometry and populate "params" (FS is not used):
    init.loadMeshGeometry(&params, &FS);
    // Read external profile files and load them to "params":
    init.loadPlasmaProfiles(&params, &IONS);
    // Check that mesh size is consistent with hybrid approximation:
    units.spatialScalesSanityCheck(&params, &FS);
    // Initialize electromagnetic field variable:
    init.initializeFields(&params, &EB);
    // Initialize IONS: scalar, bulk and particle arrays
    init.setupIonsInitialCondition(&params, &CS, &EB, &IONS);
    // HDF object constructor:
    HDF<IT, FT> hdfObj(&params, &FS, &IONS);
    // Define time step based on CFL condition: Whistler and ion velocity
    units.defineTimeStep(&params, &IONS);
    // Normalize "params", "IONS", "EB" using "CS"
    units.normalizeVariables(&params, &IONS, &EB, &CS);

    /**************** All the quantities below are dimensionless ****************/

    // Definition of variables for advancing in time particles and fields:
    // ===================================================================
    double t1 = 0.0;
    double t2 = 0.0;
    double currentTime = 0.0;
    int outputIterator = 0;
    int numberOfIterationsForEstimator = 1000;

    // Create objects:
    // ===============
    EMF_SOLVER fields_solver(&params, &CS); // Initializing the EMF_SOLVER class object.
    PIC ionsDynamics; // Initializing the PIC class object.

    // Run 3 dummy cycles to load "n" and "nv" at previous time steps:
    // ===============================================================
    for(int tt=0; tt<3; tt++)
    {
        ionsDynamics.advanceIonsPosition(&params, &EB, &IONS, 0);

        ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0);

        ionsDynamics.extrapolateIonsMoments(&params, &EB, &IONS);
    }

    // Save 1st output:
    // ================
    hdfObj.saveOutputs(&params, &IONS, &EB, &CS, 0, 0);

    // Start timing simulations:
    // ========================
    t1 = MPI::Wtime();

    // Start time iterations:
    // =====================
    for(int tt=0; tt<params.timeIterations; tt++)
    {
        // Advance velocity:
        // ================
        if(tt == 0)
        {
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, 0.5*params.DT); // Initial condition time level V^(1/2)
        }
        else
        {
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS, params.DT); // Advance ions' velocity V^(N+1/2).
        }

        // Advance position:
        // =================
        ionsDynamics.advanceIonsPosition(&params,&EB, &IONS, params.DT); // Advance ions' position in time to level X^(N+1).

        // Calculate ion moments:
        // ======================
        ionsDynamics.extrapolateIonsMoments(&params, &EB, &IONS);

        // Apply collision operator:
        // =========================
        FPCOLL.ApplyCollisionOperator(&params,&CS,&IONS);
        
        // Field solve:
        // ============
        bool EfieldFlag = true;
        bool BfieldFlag = false;

        // Magnetic field:
        if (BfieldFlag)
        {
            fields_solver.advanceBField(&params, &EB, &IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).
        }

        if (EfieldFlag)
        {
            // Electric field:
            if(tt > 2)
            {
                // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
                // Using the Bashford-Adams extrapolation.
                fields_solver.advanceEField(&params, &EB, &IONS, true, true);
            }
            else
            {
                // Using basic velocity extrapolation.
                fields_solver.advanceEField(&params, &EB, &IONS, true, false);
            }
        }
        else
        {
            cout << " E field solve disabled" << endl;
        }

        // Advance time:
        // =============
        currentTime += params.DT*CS.time;

        // Save data:
        // ==========
        if(fmod((double)(tt + 1), params.outputCadenceIterations) == 0)
        {
            vector<IT> IONS_OUT = IONS;

            // The ions' velocity is advanced in time in order to obtain V^(N+1)
            ionsDynamics.advanceIonsVelocity(&params, &CS, &EB, &IONS_OUT, 0.5*params.DT);

            hdfObj.saveOutputs(&params, &IONS_OUT, &EB, &CS, outputIterator+1, currentTime);

            outputIterator++;
        }

        // Estimate simulation time:
        // =========================
        if(tt == numberOfIterationsForEstimator)
        {
            t2 = MPI::Wtime();

            double estimatedSimulationTime = ( (double)params.timeIterations*(t2 - t1)/(double)numberOfIterationsForEstimator )/60.0;

            if(params.mpi.MPI_DOMAIN_NUMBER == 0)
            {
                cout << "ESTIMATED TIME OF COMPLETION: " << estimatedSimulationTime <<" MINUTES" << endl;
            }
        }

    } // Time iterations.

    // Finalizing MPI communications:
    // ==============================
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
