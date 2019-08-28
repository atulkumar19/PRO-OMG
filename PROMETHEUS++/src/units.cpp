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

/*All the variables MUST be expresed using the SI units (mks). The charge and
temperature the units are Coulombs (C) and Kelvins (K). The
*/
#include "units.h"

void UNITS::defineTimeStep(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,fields * EB){
	if(params->mpi.rank_cart == 0)
		cout << "\n* * * * * * * * * * * * COMPUTING SIMULATION TIME STEP * * * * * * * * * * * * * * * * * *\n";

	double DT(0);
	double averageB(0);
	double higherIonCyclotronFrequency(0);
	double ion_vmax(0);
	double ionMass(0);
	double VT(0);
	double CFL_ions(0);
	int ionSpecies(0);

	averageB = params->BGP.Bo;
	if(params->mpi.rank_cart == 0)
		cout << "Mean magnetic field in simulation domain: " << scientific << averageB << " T\n";

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iterations over the ion species
		IONS->at(ii).BGP.Wc = IONS->at(ii).Q*averageB/IONS->at(ii).M;
		IONS->at(ii).BGP.Wpi = sqrt( IONS->at(ii).BGP.Dn*params->ne*IONS->at(ii).Q*IONS->at(ii).Q/(F_EPSILON*IONS->at(ii).M) );//Check the definition of the plasma freq for each species!
		IONS->at(ii).BGP.LarmorRadius = IONS->at(ii).BGP.VTper/IONS->at(ii).BGP.Wc;

		if(params->mpi.rank_cart == 0){
			cout << "PROPERTIES OF ION SPECIES No. " << ii + 1 << '\n';
			cout << "+ Charge: " << scientific << IONS->at(ii).Q << fixed << " C\n";//Revisar las cuentas con el factor NCP
			cout << "+ Mass: " << scientific << IONS->at(ii).M << fixed << " kg\n";
			cout << "+ Super-particles: " << IONS->at(ii).NSP << "\n";
			cout << "+ Particles per cell: " << IONS->at(ii).NPC << "\n";
			cout << "+ Charged particles per super-particle: " << scientific << IONS->at(ii).NCP << fixed << "\n";
			cout << "+ Density: " << scientific << IONS->at(ii).BGP.Dn*params->ne << fixed << " m^{-3}\n";
			cout << "+ Parallel temperature: " << scientific << F_KB*IONS->at(ii).BGP.Tpar/F_E << fixed << " eV\n";
			cout << "+ Perpendicular temperature: " << scientific << F_KB*IONS->at(ii).BGP.Tper/F_E << fixed << " eV\n";
			cout << "+ Parallel thermal velocity: " << scientific << IONS->at(ii).BGP.VTpar << fixed << " m/s\n";
			cout << "+ Perpendicular thermal velocity: " << scientific << IONS->at(ii).BGP.VTper << fixed << " m/s\n";
			cout << "+ Cyclotron frequency: " << scientific << IONS->at(ii).BGP.Wc << fixed << " Hz\n";
			cout << "+ Plasma frequency: " << scientific << IONS->at(ii).BGP.Wpi << fixed << " Hz\n";
			cout << "+ Gyroperiod: " << scientific << 2.0*M_PI/IONS->at(ii).BGP.Wc << fixed << " s\n";
			cout << "+ Larmor radius: " << scientific << IONS->at(ii).BGP.LarmorRadius << fixed << " m\n";
			cout << "+ Magnetic moment: " << scientific << IONS->at(ii).BGP.mu << fixed << " A*m^2\n\n";
		}

		//We don't take into account the tracers dynamics of course!
		if(IONS->at(ii).BGP.Wc > higherIonCyclotronFrequency && (IONS->at(ii).SPECIES != 0))
			higherIonCyclotronFrequency = IONS->at(ii).BGP.Wc;

		if(IONS->at(ii).SPECIES != 0){
			ionSpecies++;
			ionMass += IONS->at(ii).M;

			#ifdef ONED
				vec V = abs(IONS->at(ii).V.col(0));
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif

			#ifdef TWOD
				vec V = zeros(IONS->at(ii).NSP);
				V = sqrt( IONS->at(ii).V.col(0) % IONS->at(ii).V.col(0) + IONS->at(ii).V.col(1) % IONS->at(ii).V.col(1) );
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif

			#ifdef THREED
				vec V = zeros(IONS->at(ii).NSP);
				V = sqrt( IONS->at(ii).V.col(0) % IONS->at(ii).V.col(0) + IONS->at(ii).V.col(1) % IONS->at(ii).V.col(1) + IONS->at(ii).V.col(2) % IONS->at(ii).V.col(2));
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif
		}

	}//Iterations over the ion species

	ionMass /=  (double)ionSpecies;
	VT = sqrt(2.0*F_KB*params->BGP.Te/ionMass);//Background thermal velocity

	params->shorterIonGyroperiod = (2.0*M_PI/higherIonCyclotronFrequency); // Shorter ion gyroperiod.

	if(params->BGP.theta == 0.0){
//		CFL_ions = 0.5*(params->DrL*VT/ion_vmax)/(2*M_PI*sqrt((double)params->dimension));
//		DT = CFL_ions*params->shorterIonGyroperiod;

		params->checkStability = 1;
		params->rateOfChecking = 5;

		DT = 0.5*Cmax*mesh->DX/ion_vmax;

		if(params->DTc*params->shorterIonGyroperiod > DT){
			if(params->mpi.rank_cart == 0)
				cout << "Time step defined by Courant–Friedrichs–Lewy condition for ions: " << scientific << DT << fixed << " s\n";
		}else{
			DT = params->DTc*params->shorterIonGyroperiod;
			if(params->mpi.rank_cart == 0)
				cout << "Time step defined by user: " << scientific << DT << fixed << " s\n";
		}
	}else{
		//Here we calculate the timestep following the CFL condition for the reltative high-frequency whistler waves.
		double CFL_w(0), A(0), B(0);//DT = CFL_w*Tc;

		B = params->BGP.Bo;
//		A = params->DrL*sqrt(2*F_KB*params->BGP.Te*params->ne/F_EPSILON)/(M_PI*F_C*B);
		A = (mesh->DX*IONS->at(0).BGP.Wpi)/(M_PI*F_C);//Ion skin depth is calculated using the background ions.

		#ifdef ONED
		CFL_w = 0.5*A*A/2.0;
		#endif

		#ifdef TWOD
		CFL_w = 0.5*A*A/(2.0*sqrt(2.0));
		#endif

		#ifdef THREED
		CFL_w = 0.5*A*A/(2.0*sqrt(3.0));
		#endif

		CFL_ions = (0.5*Cmax*mesh->DX/ion_vmax)/params->shorterIonGyroperiod;

		cout << "CFL (W): " << CFL_w << "CFL (i): " << CFL_ions << "\n";

		if(CFL_w < CFL_ions){
			params->checkStability = 1;
			params->rateOfChecking = 100;
			if(CFL_w < params->DTc){
				DT = CFL_w*params->shorterIonGyroperiod;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by Courant–Friedrichs–Lewy condition for whistler waves: " << scientific << DT << fixed << " s\n";
			}else{
				DT = params->DTc*params->shorterIonGyroperiod;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by user: " << scientific << DT << fixed << " s\n";
			}
		}else{
			params->checkStability = 1;
			params->rateOfChecking = 10;
			DT = CFL_ions*params->shorterIonGyroperiod;

			if(params->DTc*params->shorterIonGyroperiod > DT){
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by Courant–Friedrichs–Lewy condition for ions: " << scientific << DT << fixed << " s\n";
			}else{
				DT = params->DTc*params->shorterIonGyroperiod;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by user: " << scientific << DT << fixed << " s\n";
			}
		}
	}

	// * * * */
	double * DTs;
	double smallest_DT;//smallest timestep accross all MPI processes

	DTs = (double*)malloc(params->mpi.NUMBER_MPI_DOMAINS*sizeof(double));

	MPI_Allgather(&DT, 1, MPI_DOUBLE, DTs, 1, MPI_DOUBLE, params->mpi.mpi_topo);

	smallest_DT = *DTs;
	for(int ii=1; ii<params->mpi.NUMBER_MPI_DOMAINS; ii++){//Notice 'ii' starts at 1 instead of 0
		if( *(DTs + ii) < smallest_DT )
			smallest_DT = *(DTs + ii);
	}
	free(DTs);

	params->DT = smallest_DT;
	// * * * */

	// Using the shorter gyro-period in the simulation we calculate the number of time steps so that:
	// params->simulationTime = params->DT*params->timeIterations/params->shorterIonGyroperiod
	params->timeIterations = (int)ceil( params->simulationTime*params->shorterIonGyroperiod/params->DT);

	params->outputCadenceIterations = (int)ceil( params->outputCadence*params->shorterIonGyroperiod/params->DT);

	if(params->mpi.rank_cart == 0){
		cout << "Time steps in simulation: " << params->timeIterations << "\n";
		cout << "Simulation time: " << scientific << params->DT*params->timeIterations << fixed << " s\n";
		cout << "Simulation time: " << scientific << params->DT*params->timeIterations/params->shorterIonGyroperiod << fixed << " gyroperiods\n";
		cout << "Cadence for saving outputs: " << params->outputCadenceIterations << "\n";
		cout << "Number of outputs: " << floor(params->timeIterations/params->outputCadenceIterations) + 1 << endl;
		cout << "Cadence for checking stability: " << params->rateOfChecking << "\n";
		cout << "* * * * * * * * * * * * * * * TIME STEP COMPUTED * * * * * * * * * * * * * * * * * * * * *\n\n";
	}
}



void UNITS::broadcastCharacteristicScales(inputParameters * params,characteristicScales * CS){

	// Define MPI type for characteristicScales structure.
	int numStructElem(13);
	int structLength[13] = {1,1,1,1,1,1,1,1,1,1,1,1,1};
	MPI_Aint displ[13];
	MPI_Datatype csTypes[13] = {MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE,\
		MPI_DOUBLE};
	MPI_Datatype MPI_CS;

	MPI_Get_address(&CS->time,&displ[0]);
	MPI_Get_address(&CS->velocity,&displ[1]);
	MPI_Get_address(&CS->momentum,&displ[2]);
	MPI_Get_address(&CS->length,&displ[3]);
	MPI_Get_address(&CS->mass,&displ[4]);
	MPI_Get_address(&CS->charge,&displ[5]);
	MPI_Get_address(&CS->density,&displ[6]);
	MPI_Get_address(&CS->eField,&displ[7]);
	MPI_Get_address(&CS->bField,&displ[8]);
	MPI_Get_address(&CS->pressure,&displ[9]);
	MPI_Get_address(&CS->temperature,&displ[10]);
	MPI_Get_address(&CS->magneticMoment,&displ[11]);
	MPI_Get_address(&CS->resistivity,&displ[12]);

	for(int ii=numStructElem-1;ii>=0;ii--)
		displ[ii] -= displ[0];

	MPI_Type_create_struct(numStructElem,structLength,displ,csTypes,&MPI_CS);
	MPI_Type_commit(&MPI_CS);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(CS,1,MPI_CS,0,MPI_COMM_WORLD);

	MPI_Type_free(&MPI_CS);
}


void UNITS::defineCharacteristicScales(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS){
	// The definition of the characteristic quantities is based on:
	// D Winske and N Omidi, Hybrid codes.
	// All the quantities below have units (SI).

	if(params->mpi.rank_cart == 0)
		cout << "\n* * * * * * * * * * * * DEFINING CHARACTERISTIC SCALES IN SIMULATION * * * * * * * * * * * * * * * * * *\n";

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iterations over the ion species.
		CS->mass += IONS->at(ii).M;
		CS->charge += fabs(IONS->at(ii).Q);
	}//Iterations over the ion species.

	CS->mass /= params->numberOfIonSpecies;
	CS->charge /= params->numberOfIonSpecies;
	CS->density = params->ne;

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

	if(params->mpi.rank_cart == 0){
		cout << "Average mass: " << scientific << CS->mass << fixed << " kg\n";
		cout << "Average charge: " << scientific << CS->charge << fixed << " C\n";
		cout << "Density: " << scientific << CS->density << fixed << " m^(-3)\n";
		cout << "Time: " << scientific << CS->time << fixed << " s\n";
		cout << "Plasma frequency: " << scientific << characteristicPlasmaFrequency << fixed << " s\n";
		cout << "Velocity: " << scientific << CS->velocity << fixed << " m/s\n";
		cout << "Length: " << scientific << CS->length << fixed << " m\n";
		cout << "Electric field intensity: " << scientific << CS->eField << fixed << " V/m\n";
		cout << "Magnetic field intensity: " << scientific << CS->bField << fixed << " T\n";
		cout << "Pressure: " << scientific << CS->pressure << fixed << " Pa\n";
		cout << "Temperature: " << scientific << CS->temperature << fixed << " K\n";
		cout << "Magnetic moment: " << scientific << CS->magneticMoment << fixed << " A*m^2\n";
		cout << "Resistivity: " << scientific << CS->magneticMoment << fixed << " Ohms*m\n";
		cout << "* * * * * * * * * * * * CHARACTERISTIC SCALES IN SIMULATION DEFINED  * * * * * * * * * * * * * * * * * *\n\n";
	}
}


void UNITS::dimensionlessForm(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,fields * EB,const characteristicScales * CS){
	// Normalizing physical constants
	F_E_DS /= CS->charge; // Dimensionless electron charge
	F_MU_DS *= CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass; // Dimensionless vacuum permittivity
	F_C_DS /= CS->velocity; // Dimensionless speed of light

	//Normalizing the parameters.
	params->DT /= CS->time;
//	params->BGP.backgroundDensity /= CS->density;
	params->ne /= CS->density;
	params->BGP.Te /= CS->temperature;
	params->BGP.Bo /= CS->bField;
	params->BGP.Bx /= CS->bField;
	params->BGP.By /= CS->bField;
	params->BGP.Bz /= CS->bField;
	//Normalizing the parameters.

	//Normalizing the mesh.
	mesh->nodes.X = mesh->nodes.X/CS->length;
	mesh->nodes.Y = mesh->nodes.Y/CS->length;
	mesh->nodes.Z = mesh->nodes.Z/CS->length;
	mesh->DX /= CS->length;
	mesh->DY /= CS->length;
	mesh->DZ /= CS->length;
	//Normalizing the mesh.

	//Normalizing ions' properties.
	for(int ii=0;ii<IONS->size();ii++){//Iterations over the ion species.
		IONS->at(ii).Q /= CS->charge;
		IONS->at(ii).M /= CS->mass;
		IONS->at(ii).BGP.Tpar /= CS->temperature;
		IONS->at(ii).BGP.Tper /= CS->temperature;
		IONS->at(ii).BGP.LarmorRadius /= CS->length;
		IONS->at(ii).BGP.VTpar /= CS->velocity;
		IONS->at(ii).BGP.VTper /= CS->velocity;
		IONS->at(ii).BGP.Wc *= CS->time;
		IONS->at(ii).BGP.Wpi *= CS->time;//IMPORTANT: Not normalized before!!
		IONS->at(ii).X = IONS->at(ii).X/CS->length;
		IONS->at(ii).V = IONS->at(ii).V/CS->velocity;
		IONS->at(ii).P = IONS->at(ii).P/CS->momentum;
		IONS->at(ii).Ppar = IONS->at(ii).Ppar/CS->momentum;
		IONS->at(ii).mu = IONS->at(ii).mu/CS->magneticMoment;
	}//Iterations over the ion species.
	//Normalizing ions' properties.

	//Normalizing the electromagnetic fields.

	EB->E /= CS->eField;
	EB->B /= CS->bField;
	EB->_B /= CS->bField;

	//Normalizing the electromagnetic fields.
}


void UNITS::normalizeVariables(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,fields * EB,const characteristicScales * CS){

	dimensionlessForm(params,mesh,IONS,EB,CS);
}


void UNITS::defineCharacteristicScalesAndBcast(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS){

	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		defineCharacteristicScales(params,IONS,CS);
	}

	broadcastCharacteristicScales(params,CS);
}
