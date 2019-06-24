#include "timeSteppingMethods.h"

TIME_STEPPING_METHODS::TIME_STEPPING_METHODS(inputParameters * params){
    t1 = 0.0;						//
    t2 = 0.0;
    currentTime = 0.0;
    outputIterator = 0;			//
}

void TIME_STEPPING_METHODS::advanceFullOrbitIonsAndMasslessElectrons(inputParameters * params, meshGeometry * mesh, \
                                characteristicScales * CS, HDF * hdfObj, vector<ionSpecies> * IONS, fields * EB){
    EMF_SOLVER fields_solver(params, CS); // Initializing the emf class object.
	PIC ionsDynamics; // Initializing the PIC class object.
    GENERAL_FUNCTIONS genFun;

    // Repeat 3 times
    for(int tt=0;tt<3;tt++){
        ionsDynamics.advanceIonsPosition(params, mesh, IONS, 0);

        ionsDynamics.advanceIonsVelocity(params, CS, mesh, EB, IONS, 0);
    }

    // Repeat 3 times

    hdfObj->saveOutputs(params, IONS, EB, CS, 0, 0);

    t1 = MPI::Wtime();

    for(int tt=0;tt<params->timeIterations;tt++){ // Time iterations.

        if(tt == 0){
            genFun.checkStability(params, mesh, CS, IONS);
            ionsDynamics.advanceIonsVelocity(params, CS, mesh, EB, IONS, params->DT/2); // Initial condition time level V^(1/2)
        }else{
            ionsDynamics.advanceIonsVelocity(params, CS, mesh, EB, IONS, params->DT); // Advance ions' velocity V^(N+1/2).
        }

        ionsDynamics.advanceIonsPosition(params, mesh, IONS, params->DT); // Advance ions' position in time to level X^(N+1).

        fields_solver.advanceBField(params, mesh, EB, IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

        if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
             // Using the Bashford-Adams extrapolation.
            fields_solver.advanceEFieldWithVelocityExtrapolation(params, mesh, EB, IONS, 1);
        }else{
             // Using basic velocity extrapolation.
            fields_solver.advanceEFieldWithVelocityExtrapolation(params, mesh, EB, IONS, 0);
        }

        currentTime += params->DT*CS->time;

        if(fmod((double)(tt + 1), params->outputCadenceIterations) == 0){
            vector<ionSpecies> IONS_OUT = *IONS;
            // The ions' velocity is advanced in time in order to obtain V^(N+1)
            ionsDynamics.advanceIonsVelocity(params, CS, mesh, EB, &IONS_OUT, params->DT/2);
            hdfObj->saveOutputs(params, &IONS_OUT, EB, CS, outputIterator+1, currentTime);
            outputIterator++;
        }

        if( (params->checkStability == 1) && fmod((double)(tt+1), params->rateOfChecking) == 0 ){
            genFun.checkStability(params, mesh, CS, IONS);
        }

//		genFun.checkEnergy(params,mesh,CS,IONS,EB,tt);
        if(tt==100){
            t2 = MPI::Wtime();
            if(params->mpi.rank_cart == 0){
                cout << "ESTIMATED TIME OF COMPLETION: " << (double)params->timeIterations*(t2 - t1)/6000.0 <<" minutes\n";
            }
        }
    } // Time iterations.

    genFun.saveDiagnosticsVariables(params);
}


void TIME_STEPPING_METHODS::advanceGCIonsAndMasslessElectrons(inputParameters * params, meshGeometry * mesh, characteristicScales * CS, HDF * hdfObj, vector<ionSpecies> * IONS, fields * EB){
    EMF_SOLVER fields_solver(params, CS); // Initializing the emf class object.
    PIC_GC ionsDynamics(params, mesh); // Initializing the PIC class object.

    hdfObj->saveOutputs(params, IONS, EB, CS, 0, 0);

    t1 = MPI::Wtime();

    for(int tt=0;tt<params->timeIterations;tt++){ // Time iterations.
        if(params->mpi.rank_cart == 0)
            cout << "ITERATION: " << tt + 1 << "\n";

        ionsDynamics.advanceGCIons(params, CS, mesh, EB, IONS, params->DT);

//        fields_solver.advanceBField(params, mesh, EB, IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

//        fields_solver.advanceEField(params, mesh, EB, IONS);

        currentTime += params->DT*CS->time;

        if(fmod((double)(tt + 1), params->outputCadenceIterations) == 0){
            hdfObj->saveOutputs(params, IONS, EB, CS, outputIterator + 1, currentTime);
            outputIterator++;
        }

        if(tt==100){
            t2 = MPI::Wtime();
            if(params->mpi.rank_cart == 0){
                cout << "ESTIMATED TIME OF COMPLETION: " << (double)params->timeIterations*(t2 - t1)/6000.0 <<" minutes\n";
            }
        }

    } // Time iterations.

}
