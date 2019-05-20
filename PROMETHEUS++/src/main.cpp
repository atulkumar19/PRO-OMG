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
	vector<ionSpecies> IONS__;  	// Bashford-Adams extrapolation term.
	characteristicScales CS;		// Derived type for keeping info about characteristic scales.
	int outputIterator(0);			//
	meshGeometry mesh; 				// Derived type with info of geometry of the simulation mesh (initially with units).
	emf EB; 						// Derived type with variables of electromagnetic fields.
	double currentTime(0); 			// Current time in simulation.
	double t1;						//
    double t2;						//


	INITIALIZE init(&params, argc, argv);

	mpi_main.createMPITopology(&params);

	init.loadIons(&params, &IONS);

	UNITS units;
	units.defineCharacteristicScalesAndBcast(&params, &IONS, &CS);

	init.loadMeshGeometry(&params, &CS, &mesh);

	init.calculateSuperParticleNumberDensity(&params, &CS, &mesh, &IONS); // Calculation of IONS[ii].NCP for each species

	init.initializeFields(&params, &mesh, &EB, &IONS);

	HDF hdfObj(&params, &mesh, &IONS); // Outputs in HDF5 format

	ALFVENIC alfvenPerturbations(&params, &mesh, &EB, &IONS); // Include Alfvenic perturbations in the initial condition

	units.defineTimeStep(&params, &mesh, &IONS, &EB);

	/*By calling this function we set up some of the simulation parameters and normalize the variables*/
	units.normalizeVariables(&params, &mesh, &IONS, &EB, &CS);

	alfvenPerturbations.normalize(&CS);

	/**************** All the quantities below are dimensionless ****************/

	EMF_SOLVER fields(&params, &CS); // Initializing the emf class object.
	PIC ionsDynamics; // Initializing the PIC class object.
	GENERAL_FUNCTIONS genFun;

	ionsDynamics.advanceIonsPosition(&params, &mesh, &IONS, 0); // Initialization of the particles' density and meshNode.

	ionsDynamics.advanceIonsVelocity(&params, &CS, &mesh, &EB, &IONS, 0);

	alfvenPerturbations.addPerturbations(&params, &IONS, &EB);

	hdfObj.saveOutputs(&params, &IONS, &IONS, &EB, &CS, 0, 0);

	t1 = MPI::Wtime();

	for(int tt=0;tt<params.timeIterations;tt++){ // Time iterations.
		vector<ionSpecies> auxIONS;
		vector<ionSpecies> IONS_;

		ionsDynamics.ionVariables(&IONS, &auxIONS, 0); // The ion density of IONS is copied into auxIONS
		ionsDynamics.ionVariables(&IONS, &IONS_, 1); // The ion flow velocity at the time level nv^(N-1/2) is stored in auxIONS.

		if(tt == 0){
			genFun.checkStability(&params, &mesh, &CS, &IONS);
			ionsDynamics.advanceIonsVelocity(&params, &CS, &mesh, &EB, &IONS, params.DT/2); // Initial condition time level V^(1/2)
		}else{
			ionsDynamics.advanceIonsVelocity(&params, &CS, &mesh, &EB, &IONS, params.DT); // Advance ions' velocity V^(N+1/2).
		}

		ionsDynamics.advanceIonsPosition(&params, &mesh, &IONS, params.DT); // Advance ions' position in time to level X^(N+1).

		ionsDynamics.ionVariables(&IONS, &auxIONS, 2); // Bulk ion variables computed at (N+1/2) and saved to auxIONS

		fields.advanceBField(&params, &mesh, &EB, &auxIONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

		if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
			 // Using the Bashford-Adams extrapolation.
			fields.advanceEFieldWithVelocityExtrapolation(&params, &mesh, &EB, &IONS__, &IONS_, &IONS, 1);
		}else{
			 // Using basic velocity extrapolation.
			fields.advanceEFieldWithVelocityExtrapolation(&params, &mesh, &EB, &IONS__, &IONS_, &IONS, 0);
		}

		currentTime += params.DT*CS.time;

		if(fmod((double)(tt + 1), params.outputCadenceIterations) == 0){
			auxIONS = IONS; // Ions position at level X^(N+1) and ions' velocity at level V^(N+1/2)
			// The ions' velocity is advanced in time in order to obtain V^(N+1)
			ionsDynamics.advanceIonsVelocity(&params, &CS, &mesh, &EB, &auxIONS, params.DT/2);
			hdfObj.saveOutputs(&params, &auxIONS, &IONS, &EB, &CS, outputIterator+1, currentTime);
			outputIterator++;
		}

		if( (params.checkStability == 1) && fmod((double)(tt+1), params.rateOfChecking) == 0 ){
			genFun.checkStability(&params, &mesh, &CS, &IONS);
		}

		IONS__ = IONS_; // Ions variables at time level (N-3/2) for tt>=2

//		genFun.checkEnergy(&params,&mesh,&CS,&IONS,&EB,tt);
		if(tt==100){
			t2 = MPI::Wtime();
			if(params.mpi.rank_cart == 0){
				cout << "ESTIMATED TIME OF COMPLETION: " << (double)params.timeIterations*(t2-t1)/6000.0 <<" minutes\n";
			}
		}
	} // Time iterations.

	/**************** All the quantities above are dimensionless ****************/
	genFun.saveDiagnosticsVariables(&params);

	MPI_Finalize();

	return(0);
}
