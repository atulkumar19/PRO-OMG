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


#include "units.h"

// For details on the theory behind the restrictions on the time step please see: P. Pritchett, IEEE Transactions on plasma science 28, 1976 (2000).

template <class IT, class FT> void UNITS<IT,FT>::defineTimeStep(simulationParameters * params, vector<IT> * IONS)
{
	// Algorithm for defining time step in simulation:
	// 1) Check whether initial time step satisfies CFL condition for ions.
	// 		1b) If CFL condition for ions is not satisfaced, define new time step.
	// 2) Check whether initial time step satisfies CFL for faster whistler waves in simulation.
	//		2b) If CFL condition for whistler waves is not satisfaced, define new time step.
	// 3) Choose the smaller time step from steps 1) and 2) as the actual time step in simulation.

        // Print to terminal:
        // ==================
	if (params->mpi.IS_PARTICLES_ROOT)
        {
            cout << endl << "* * * * * * * * * * * * COMPUTING SIMULATION TIME STEP * * * * * * * * * * * * * * * * * *" << endl;
        }

        // Define variables:
        // =================
	double ionsMaxVel(0.0);	// Maximum speed of simulated ions
	double DT(0.0); 	// Time step defined by user
	double DT_CFL_I(0.0);	// Minimum time step defined by CFL condition for ions
	double DT_CFL_W(0.0);	// Minimum time step defined by CFL condition for whistler waves
	bool CFL_I(false);
	bool CFL_W(false);

	// Time step given by user:
        // ========================
	DT = params->DTc*params->ionGyroPeriod;

	// Minimum time step defined by CFL condition for whistler waves:
        // =============================================================
	DT_CFL_W = 0.5*pow(params->mesh.DX/params->ionSkinDepth, 2.0)*params->ionGyroPeriod/( M_PI*M_PI*sqrt((double)params->dimensionality) );

	// CFL condition for ions:
        // =======================
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
            {
                    vec V = sqrt( pow(IONS->at(ss).V.col(0), 2.0) + pow(IONS->at(ss).V.col(1), 2.0) + pow(IONS->at(ss).V.col(2), 2.0) );

                    ionsMaxVel = (ionsMaxVel < V.max()) ? V.max() : ionsMaxVel;
            }

            // Minimum time step required by CFL condition for ions:
            DT_CFL_I = params->mesh.DX/( ionsMaxVel*sqrt((double)params->dimensionality) );

            // We gather DT_CFL_I from all MPI processes in particles communicator:
            double * DT_CFL_I_MPI;

            DT_CFL_I_MPI = (double*)malloc( params->mpi.MPIS_PARTICLES*sizeof(double) );

            MPI_Allgather(&DT_CFL_I, 1, MPI_DOUBLE, DT_CFL_I_MPI, 1, MPI_DOUBLE, params->mpi.COMM);

            // Now, in MPI process 0 we find the smallest time step that satisfies CFL condition for ions,
            // then we use it to find the correct time step for avoiding numerical instability and broadcast
            // the correct time step to the rest of MPI processes.

            if (params->mpi.IS_PARTICLES_ROOT)
            {
                for (int ii=0; ii<params->mpi.MPIS_PARTICLES; ii++)
                {
                    DT_CFL_I = (DT_CFL_I > *(DT_CFL_I_MPI + ii)) ? *(DT_CFL_I_MPI + ii) : DT_CFL_I;
                }

                if (DT > DT_CFL_I)
                {
                        DT = DT_CFL_I;

                        CFL_I = true;
                        CFL_W = false;
                }

                if (DT > DT_CFL_W)
                {
                        DT = DT_CFL_W;

                        CFL_I = false;
                        CFL_W = true;
                }

                //params->DT = (CFL_I || CFL_W) ? 0.25*DT : DT; // We use half the CFL time step to ensure numerical stability
                params->DT = 0.5*DT_CFL_I; // We use half the CFL time step to ensure numerical stability

                params->timeIterations = (int)ceil( params->simulationTime*params->ionGyroPeriod/params->DT );

                params->outputCadenceIterations = (int)ceil( params->outputCadence*params->ionGyroPeriod/params->DT );
            }

            free(DT_CFL_I_MPI);
	}

	// Broadcast correct time step, time iterations in simulation, and cadence for generating outputs:
        // ===============================================================================================
	MPI_Bcast(&params->DT, 1, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, MPI_COMM_WORLD);

	MPI_Bcast(&params->timeIterations, 1, MPI_INT, params->mpi.PARTICLES_ROOT_WORLD_RANK, MPI_COMM_WORLD);

	MPI_Bcast(&params->outputCadenceIterations, 1, MPI_INT, params->mpi.PARTICLES_ROOT_WORLD_RANK, MPI_COMM_WORLD);

	// cout << "MPI: " << params->mpi.MPI_DOMAIN_NUMBER << " | DT: " << scientific <<  params->DT << fixed << endl;

	// Print to terminal a brief summary:
        // ==================================
	if (params->mpi.IS_PARTICLES_ROOT)
        {
            if (CFL_I)
                    cout << "+ Simulation time step defined by CFL condition for IONS" << endl;
            else if (CFL_W)
                    cout << "+ Simulation time step defined by CFL condition for WHISTLER WAVES" << endl;
            else
                    cout << "+ Simulation time step defined by USER" << endl;

            cout << "+ Time step defined by CFL condition for ions: " << scientific << DT_CFL_I << fixed << endl;
            cout << "+ Time step defined by CFL condition for whistler waves: " << scientific << DT_CFL_W << fixed << endl;
            cout << "+ Time step used in simulation: " << scientific << params->DT << fixed << endl;
            cout << "+ Time steps in simulation: " << params->timeIterations << endl;
            cout << "+ Simulation time: " << scientific << params->DT*params->timeIterations << fixed << " s" << endl;
            cout << "+ Simulation time: " << params->DT*params->timeIterations/params->ionGyroPeriod << " gyroperiods" << endl;
            cout << "+ Cadence for saving outputs: " << params->outputCadenceIterations << endl;
            cout << "+ Number of outputs: " << floor(params->timeIterations/params->outputCadenceIterations) + 1 << endl;
            cout << "+ Cadence for checking stability: " << params->rateOfChecking << endl;
            cout << "* * * * * * * * * * * * * * * TIME STEP COMPUTED * * * * * * * * * * * * * * * * * * * * *" << endl;
	}

}


template <class IT, class FT> void UNITS<IT,FT>::broadcastCharacteristicScales(simulationParameters * params, characteristicScales * CS)
{
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&CS->time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->velocity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->momentum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->length, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->mass, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->charge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->density, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->eField, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->bField, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->pressure, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->temperature, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->magneticMoment, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->resistivity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->vacuumPermeability, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Bcast(&CS->vacuumPermittivity, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


template <class IT, class FT> void UNITS<IT,FT>::broadcastFundamentalScales(simulationParameters * params, fundamentalScales * FS)
{
    MPI_Barrier(MPI_COMM_WORLD);

    // Electron skin depth
    MPI_Bcast(&FS->electronSkinDepth, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Electron gyro-period
    MPI_Bcast(&FS->electronGyroPeriod, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Electron gyro-radius
    MPI_Bcast(&FS->electronGyroRadius, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Ion skin depth
    MPI_Bcast(FS->ionSkinDepth, params->numberOfParticleSpecies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Ion gyro-period
    MPI_Bcast(FS->ionGyroPeriod, params->numberOfParticleSpecies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Ion gyro-radius
    MPI_Bcast(FS->ionGyroRadius, params->numberOfParticleSpecies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}


template <class IT, class FT> void UNITS<IT,FT>::calculateFundamentalScales(simulationParameters * params, vector<IT> * IONS, fundamentalScales * FS)
{

	cout << endl << "* * * * * * * * * * * * CALCULATING FUNDAMENTAL SCALES IN SIMULATION * * * * * * * * * * * * * * * * * *" << endl;

	/*
	if (params->includeElectronInertia == true){
        cout << " + Including electron inertia: YES" << endl << endl;
    }else{
        cout << " + Including electron inertia: NO" << endl << endl;
    }
	*/

	FS->electronSkinDepth = F_C/sqrt( params->BGP.ne*F_E*F_E/(F_EPSILON*F_ME) );
	FS->electronGyroPeriod = 2.0*M_PI/(F_E*params->BGP.Bo/F_ME);
	FS->electronGyroRadius = sqrt(2.0*F_KB*params->BGP.Te/F_ME)/(F_E*params->BGP.Bo/F_ME);

	cout << " + Electron gyro-period: " << scientific << FS->electronGyroPeriod << fixed << " s" << endl;
	cout << " + Electron skin depth: " << scientific << FS->electronSkinDepth << fixed << " m" << endl;
	cout << " + Electron gyro-radius: " << scientific << FS->electronGyroRadius << fixed << " m" << endl;
	cout << endl;

	for(int ss=0; ss<params->numberOfParticleSpecies; ss++){
		FS->ionGyroPeriod[ss] = 2.0*M_PI/IONS->at(ss).Wc;
		FS->ionSkinDepth[ss] = F_C/IONS->at(ss).Wp;
		FS->ionGyroRadius[ss] = IONS->at(ss).LarmorRadius;

		cout << "ION SPECIES: " << ss << endl;
		cout << " + Ion gyro-period: " << scientific << FS->ionGyroPeriod[ss] << fixed << " s" << endl;
		cout << " + Ion skin depth: " << scientific << FS->ionSkinDepth[ss] << fixed << " m" << endl;
		cout << " + Ion gyro-radius: " << scientific << FS->ionGyroRadius[ss] << fixed << " m" << endl;
		cout << endl;
	}

	cout << "* * * * * * * * * * * * FUNDAMENTAL SCALES IN SIMULATION CALCULATED  * * * * * * * * * * * * * * * * * *" << endl;
}


template <class IT, class FT> void UNITS<IT,FT>::spatialScalesSanityCheck(simulationParameters * params, fundamentalScales * FS)
{
	MPI_Barrier(MPI_COMM_WORLD);

        // Print to terminal:
        // =================
	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << endl << "* * * * * * * * * * * * CHECKING VALIDITY OF HYBRID MODEL FOR THE SIMULATED PLASMA * * * * * * * * * * * * * * * * * *" << endl;
            cout << "Electron skin depth to grid size ratio: " << scientific << FS->electronSkinDepth/params->mesh.DX << fixed << endl;
            cout << "* * * * * * * * * * * * VALIDITY OF HYBRID MODEL FOR THE SIMULATED PLASMA CHECKED  * * * * * * * * * * * * * * * * * *" << endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// Check that DX is larger than the electron skin depth, otherwise, abort simulation:
        // ==================================================================================
	if (params->mesh.DX <= FS->electronSkinDepth)
        {
            cout << "ERROR: Grid size violates assumptions of hybrid model for the plasma -- lenght scales smaller than the electron skind depth can not be resolved." << endl;
            cout << "ABORTING SIMULATION..." << endl;

            MPI_Abort(MPI_COMM_WORLD,-103);
	}
}


template <class IT, class FT> void UNITS<IT,FT>::defineCharacteristicScales(simulationParameters * params, vector<IT> * IONS, characteristicScales * CS)
{
	// The definition of the characteristic quantities is based on:
	// D Winske and N Omidi, Hybrid codes.
	// All the quantities below have units (SI).

	cout << endl << "* * * * * * * * * * * * DEFINING CHARACTERISTIC SCALES IN SIMULATION * * * * * * * * * * * * * * * * * *" << endl;

	for (int ii=0; ii<params->numberOfParticleSpecies; ii++)
        {
            CS->mass += IONS->at(ii).M;
            CS->charge += fabs(IONS->at(ii).Q);
	}

	CS->mass /= params->numberOfParticleSpecies;
	CS->charge /= params->numberOfParticleSpecies;
	CS->density = params->BGP.ne;

	double characteristicPlasmaFrequency(0);//Background ion-plasma frequency.
	characteristicPlasmaFrequency = sqrt( CS->density*CS->charge*CS->charge/(CS->mass*F_EPSILON) );

	CS->time = 1/characteristicPlasmaFrequency;
	CS->velocity = F_C;
	CS->momentum = CS->mass*CS->velocity;
	CS->length = CS->velocity*CS->time;
	CS->eField = ( CS->mass*CS->velocity )/( CS->charge*CS->time );
	CS->bField = CS->eField/CS->velocity; // CS->mass/( CS->charge*CS->time );
	CS->temperature = CS->mass*CS->velocity*CS->velocity/F_KB;
	CS->pressure = CS->bField*CS->velocity*CS->velocity*CS->charge*CS->density*CS->time;
	CS->resistivity = CS->bField/(CS->charge*CS->density);
	CS->magneticMoment = CS->mass*CS->velocity*CS->velocity/CS->bField;
	CS->vacuumPermittivity = (pow(CS->length*CS->charge,2)*CS->density)/(CS->mass*pow(CS->velocity,2));
	CS->vacuumPermeability = CS->mass/( CS->density*pow(CS->charge*CS->velocity*CS->time,2) );

	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		cout << "+ Average mass: " << scientific << CS->mass << fixed << " kg" << endl;
		cout << "+ Average charge: " << scientific << CS->charge << fixed << " C" << endl;
		cout << "+ Density: " << scientific << CS->density << fixed << " m^(-3)" << endl;
		cout << "+ Time: " << scientific << CS->time << fixed << " s" << endl;
		cout << "+ Plasma frequency: " << scientific << characteristicPlasmaFrequency << fixed << " s" << endl;
		cout << "+ Velocity: " << scientific << CS->velocity << fixed << " m/s" << endl;
		cout << "+ Length: " << scientific << CS->length << fixed << " m" << endl;
		cout << "+ Electric field intensity: " << scientific << CS->eField << fixed << " V/m" << endl;
		cout << "+ Magnetic field intensity: " << scientific << CS->bField << fixed << " T" << endl;
		cout << "+ Pressure: " << scientific << CS->pressure << fixed << " Pa" << endl;
		cout << "+ Temperature: " << scientific << CS->temperature << fixed << " K" << endl;
		cout << "+ Magnetic moment: " << scientific << CS->magneticMoment << fixed << " A*m^2" << endl;
		cout << "+ Resistivity: " << scientific << CS->magneticMoment << fixed << " Ohms*m" << endl;
		cout << "+ Vacuum permittivity: " << scientific << CS->vacuumPermittivity << fixed << "" << endl;
		cout << "+ Vacuum permeability: " << scientific << CS->vacuumPermeability << fixed << "" << endl;
		cout << endl << "* * * * * * * * * * * * CHARACTERISTIC SCALES IN SIMULATION DEFINED  * * * * * * * * * * * * * * * * * *" << endl;
	}
}


template <class IT, class FT> void UNITS<IT,FT>::normalizeVariables(simulationParameters * params, vector<IT> * IONS, FT * EB, const characteristicScales * CS)
{
	// Normalizing physical constants:
        // ===============================
	F_E_DS /= CS->charge; 					// Dimensionless electron charge
	F_ME_DS /= CS->mass; 					// Dimensionless electron charge
	// F_MU_DS *= CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass; 	// Dimensionless vacuum permittivity
	F_MU_DS /= CS->vacuumPermeability; 		// Dimensionless vacuum permeability
	F_EPSILON_DS /= CS->vacuumPermittivity;	// Dimensionless vacuum permittivity
	F_C_DS /= CS->velocity; 				// Dimensionless speed of light

	// Normalizing "params":
        // =====================
	params->DT /= CS->time;
	params->BGP.ne /= CS->density;
	params->BGP.Te /= CS->temperature;
	params->BGP.Bo /= CS->bField;
	params->BGP.Bx /= CS->bField;
	params->BGP.By /= CS->bField;
	params->BGP.Bz /= CS->bField;
        params->BGP.Rphi0 /= CS->length;

	params->PP.ne_i /= CS->density;
	params->PP.Tpar_i /= CS->temperature;
        params->PP.Tper_i /= CS->temperature;
	params->PP.Bx_i /= CS->bField;
	params->PP.Br_i /= CS->bField;
	params->PP.dBrdx_i /= CS->bField/CS->length;

	params->ionLarmorRadius /= CS->length;
	params->ionSkinDepth /= CS->length;
	params->ionGyroPeriod /= CS->time;

	//Normalizing the mesh:
	params->mesh.nodes.X = params->mesh.nodes.X/CS->length;
	params->mesh.nodes.Y = params->mesh.nodes.Y/CS->length;
	params->mesh.nodes.Z = params->mesh.nodes.Z/CS->length;
	params->mesh.DX /= CS->length;
	params->mesh.DY /= CS->length;
	params->mesh.DZ /= CS->length;
	params->mesh.LX /= CS->length;
	params->mesh.LY /= CS->length;
	params->mesh.LZ /= CS->length;

	// Normalizing IONS:
        // ================
	for(int ii=0;ii<IONS->size();ii++)
        {
		IONS->at(ii).Q /= CS->charge;
		IONS->at(ii).M /= CS->mass;
		IONS->at(ii).Tpar /= CS->temperature;
		IONS->at(ii).Tper /= CS->temperature;
		IONS->at(ii).LarmorRadius /= CS->length;
		IONS->at(ii).VTpar /= CS->velocity;
		IONS->at(ii).VTper /= CS->velocity;
		IONS->at(ii).Wc *= CS->time;
		IONS->at(ii).Wp *= CS->time;
		IONS->at(ii).avg_mu /= CS->magneticMoment;

		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
                {
                    IONS->at(ii).X = IONS->at(ii).X/CS->length;
                    IONS->at(ii).V = IONS->at(ii).V/CS->velocity;
                    IONS->at(ii).P = IONS->at(ii).P/CS->momentum;
                    IONS->at(ii).Ppar = IONS->at(ii).Ppar/CS->momentum;
                    IONS->at(ii).mu = IONS->at(ii).mu/CS->magneticMoment;
		}
	}

        // Normalizing "EB":
        // =================
	EB->E /= CS->eField;
	EB->B /= CS->bField;
}


template <class IT, class FT> void UNITS<IT,FT>::defineCharacteristicScalesAndBcast(simulationParameters * params, vector<IT> * IONS, characteristicScales * CS)
{
    MPI_Barrier(MPI_COMM_WORLD);

    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        defineCharacteristicScales(params, IONS, CS);
    }

    broadcastCharacteristicScales(params, CS);
}


template <class IT, class FT> void UNITS<IT,FT>::calculateFundamentalScalesAndBcast(simulationParameters * params, vector<IT> * IONS, fundamentalScales * FS)
{
    MPI_Barrier(MPI_COMM_WORLD);

    if(params->mpi.MPI_DOMAIN_NUMBER == 0){
            calculateFundamentalScales(params, IONS, FS);
    }

    broadcastFundamentalScales(params, FS);

    // It is assumed that species 0 is the majority species
    params->ionLarmorRadius = FS->ionGyroRadius[0];
    params->ionSkinDepth = FS->ionSkinDepth[0];
    params->ionGyroPeriod = FS->ionGyroPeriod[0];
}

template class UNITS<oneDimensional::ionSpecies, oneDimensional::fields>;
template class UNITS<twoDimensional::ionSpecies, twoDimensional::fields>;
