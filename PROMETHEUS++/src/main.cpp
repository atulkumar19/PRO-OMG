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

#include <omp.h>
#include "mpi_main.h"

using namespace std;
using namespace arma;

int main(int argc,char* argv[]){

	MPI_Init(&argc,&argv);
	MPI_MAIN mpi_main;

	inputParameters params; // Input parameters for the simulation.
	INITIALIZE init(&params,argc,argv);

	mpi_main.createMPITopology(&params);

	vector<ionSpecies> IONS; // Vector of ionsSpecies structures each of them storing the properties of each ion species.
	init.loadIons(&params,&IONS);

	characteristicScales CS;
	UNITS units;
	units.defineCharacteristicScalesAndBcast(&params,&IONS,&CS);

	meshGeometry mesh; // Geometry of the mesh (still with units).
	init.loadMeshGeometry(&params,&CS,&mesh);

	init.calculateSuperParticleNumberDensity(&params,&CS,&mesh,&IONS); // Calculation of IONS[ii].NCP for each species

	emf EB; //Electric and magnetic fields.
	init.initializeFields(&params,&mesh,&EB,&IONS);

	HDF hdfObj(&params,&mesh,&IONS); // Outputs in HDF5 format

	ALFVENIC alfvenPerturbations(&params,&mesh,&EB,&IONS); // Include Alfvenic perturbations in the initial condition

	/*By calling this function we set up some of the simulation parameters and normalize the variables*/
	units.normalizeVariables(&params,&mesh,&IONS,&EB,&CS);

	alfvenPerturbations.normalize(&CS);

	/**************** All the quantities below are dimensionless ****************/

	EMF_SOLVER fields; // Initializing the emf class object.
	PIC ionsDynamics; // Initializing the PIC class object.
	GENERAL_FUNCTIONS genFun;

	ionsDynamics.advanceIonsPosition(&params,&mesh,&IONS,0); // Initialization of the particles' density and meshNode.

	ionsDynamics.advanceIonsVelocity(&params,&CS,&mesh,&EB,&IONS,0);

	fields.advanceEField(&params,&mesh,&EB,&IONS,&CS);

	fields.advanceBField(&params,&mesh,&EB,&IONS,&CS);

	alfvenPerturbations.addPerturbations(&params,&IONS,&EB);

	hdfObj.saveOutputs(&params,&IONS,&IONS,&EB,&CS,0,0);

	double totalTime(0); // In seconds when it has units.
	int it(0);
	double t1,t2;

	vector<ionSpecies> IONS_BAE;  //  Bashford-Adams extrapolation term.

	t1 = MPI::Wtime();

	for(int tt=0;tt<params.timeIterations;tt++){ // Time iterations.
		vector<ionSpecies> auxIONS;
		vector<ionSpecies> IONS_VE;

		ionsDynamics.ionVariables(&IONS,&auxIONS,0); // The ion density of IONS is copied into auxIONS
		ionsDynamics.ionVariables(&IONS,&IONS_VE,1); // The ion flow velocity at the time level nv^(N-1/2) is stored in auxIONS.

		if(tt == 0){
			genFun.checkStability(&params,&mesh,&CS,&IONS);
			ionsDynamics.advanceIonsVelocity(&params,&CS,&mesh,&EB,&IONS,params.DT/2); // Initial condition time level V^(1/2)
		}else{
			ionsDynamics.advanceIonsVelocity(&params,&CS,&mesh,&EB,&IONS,params.DT); // Advance ions' velocity V^(N+1/2).
		}

		ionsDynamics.advanceIonsPosition(&params,&mesh,&IONS,params.DT); // Advance ions' position in time to level X^(N+1).

		ionsDynamics.ionVariables(&IONS,&auxIONS,2); // Bulk ion variables computed at (N+1/2) and saved to auxIONS

		fields.advanceBField(&params,&mesh,&EB,&auxIONS,&CS); // Use Faraday's law to advance the magnetic field to level B^(N+1/2).

		if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
			 // Using the Bashford-Adams extrapolation.
			fields.advanceEFieldWithVelocityExtrapolation(&params,&mesh,&EB,&IONS_BAE,&IONS_VE,&IONS,&CS,1);
		}else{
			 // Using basic velocity extrapolation.
			fields.advanceEFieldWithVelocityExtrapolation(&params,&mesh,&EB,&IONS_BAE,&IONS_VE,&IONS,&CS,0);
		}

		totalTime += params.DT*CS.time;

		if(fmod((double)(tt + 1),params.saveVariablesEach) == 0){
			auxIONS = IONS; // Ions position at level X^(N+1) and ions' velocity at level V^(N+1/2)
			// The ions' velocity is advanced in time in order to obtain V^(N+1)
			ionsDynamics.advanceIonsVelocity(&params,&CS,&mesh,&EB,&auxIONS,params.DT/2);
			hdfObj.saveOutputs(&params,&auxIONS,&IONS,&EB,&CS,it+1,totalTime);
			it++;
		}

		if( (params.checkStability == 1) && fmod((double)(tt+1),params.rateOfChecking) == 0 ){
			genFun.checkStability(&params,&mesh,&CS,&IONS);
		}

		IONS_BAE = IONS_VE; // Ions variables at time level (N-3/2) for tt>=2

//		genFun.checkEnergy(&params,&mesh,&CS,&IONS,&EB,tt);
		if(tt==100){
			t2 = MPI::Wtime();
			cout << "The time elapsed is: " << t2-t1 <<'\n';
		}
	} // Time iterations.
/*
	MPI_Barrier(params.mpi.mpi_topo);
	cout << "MPI rank: " << params.mpi.rank_cart << " ALL OK!\n";
	MPI_Abort(params.mpi.mpi_topo,-10);
*/
	/**************** All the quantities above are dimensionless ****************/
	genFun.saveDiagnosticsVariables(&params);

	MPI_Finalize();

	return(0);
}
