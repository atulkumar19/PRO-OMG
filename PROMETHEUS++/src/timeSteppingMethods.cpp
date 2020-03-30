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

#include "timeSteppingMethods.h"

template <class IT, class FT> TIME_STEPPING_METHODS<IT,FT>::TIME_STEPPING_METHODS(simulationParameters * params){
    t1 = 0.0;						//
    t2 = 0.0;
    currentTime = 0.0;
    outputIterator = 0;			//
}

template <class IT, class FT> void TIME_STEPPING_METHODS<IT,FT>::TIME_STEPPING_METHODS::advanceFullOrbitIonsAndMasslessElectrons(simulationParameters * params, characteristicScales * CS, HDF<IT,FT> * hdfObj, vector<IT> * IONS, FT * EB){
    EMF_SOLVER fields_solver(params, CS); // Initializing the emf class object.
	PIC<IT,FT> ionsDynamics; // Initializing the PIC class object.
    // GENERAL_FUNCTIONS genFun;

    // Repeat 3 times
    for(int tt=0;tt<3;tt++){
        ionsDynamics.advanceIonsPosition(params, IONS, 0);

        ionsDynamics.advanceIonsVelocity(params, CS, EB, IONS, 0);
    }

    // Repeat 3 times

    hdfObj->saveOutputs(params, IONS, EB, CS, 0, 0);

    t1 = MPI::Wtime();

    for(int tt=0;tt<params->timeIterations;tt++){ // Time iterations.
        // cout << "ITERATION: " << tt << endl;
        if(tt == 0){
            // genFun.checkStability(params, mesh, CS, IONS);
            ionsDynamics.advanceIonsVelocity(params, CS, EB, IONS, params->DT/2); // Initial condition time level V^(1/2)
        }else{
            ionsDynamics.advanceIonsVelocity(params, CS, EB, IONS, params->DT); // Advance ions' velocity V^(N+1/2).
        }

        ionsDynamics.advanceIonsPosition(params, IONS, params->DT); // Advance ions' position in time to level X^(N+1).


        fields_solver.advanceBField(params, EB, IONS); // Use Faraday's law to advance the magnetic field to level B^(N+1).

        if(tt > 2){ // We use the generalized Ohm's law to advance in time the Electric field to level E^(N+1).
             // Using the Bashford-Adams extrapolation.
            fields_solver.advanceEFieldWithVelocityExtrapolation(params, EB, IONS, 1);
        }else{
             // Using basic velocity extrapolation.
            fields_solver.advanceEFieldWithVelocityExtrapolation(params, EB, IONS, 0);
        }

        currentTime += params->DT*CS->time;

        if(fmod((double)(tt + 1), params->outputCadenceIterations) == 0){
            vector<ionSpecies> IONS_OUT = *IONS;
            // The ions' velocity is advanced in time in order to obtain V^(N+1)
            ionsDynamics.advanceIonsVelocity(params, CS, EB, &IONS_OUT, params->DT/2);
            hdfObj->saveOutputs(params, &IONS_OUT, EB, CS, outputIterator+1, currentTime);
            outputIterator++;
        }

        //*** @tomodiify
        /*
        if( (params->checkStability == 1) && fmod((double)(tt+1), params->rateOfChecking) == 0 ){
            genFun.checkStability(params, mesh, CS, IONS);
        }
        */

/* This function to monitor energy conservation needs to be implemented in a better way*/
//		genFun.checkEnergy(params,mesh,CS,IONS,EB,tt);
/* This function to monitor energy conservation needs to be implemented in a better way*/

        if(tt==100){
            t2 = MPI::Wtime();
            if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
                cout << "ESTIMATED TIME OF COMPLETION: " << (double)params->timeIterations*(t2 - t1)/6000.0 <<" minutes\n";
            }
        }
    } // Time iterations.

    // genFun.saveDiagnosticsVariables(params);
}


template <class IT, class FT> void TIME_STEPPING_METHODS<IT,FT>::advanceGCIonsAndMasslessElectrons(simulationParameters * params, characteristicScales * CS, HDF<IT,FT> * hdfObj, vector<IT> * IONS, FT * EB){
    MPI_Barrier(params->mpi.MPI_TOPO);
    MPI_Abort(params->mpi.MPI_TOPO, -2000);
}

template class TIME_STEPPING_METHODS<oneDimensional::ionSpecies, oneDimensional::fields>;
