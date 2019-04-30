/*All the variables MUST be expresed using the SI units (mks). The charge and
temperature the units are Coulombs (C) and Kelvins (K). The
*/
#include "units.h"

double UNITS::defineTimeStep(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB){
	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * COMPUTING SIMULATION TIME STEP * * * * * * * * * * * * * * * * * *\n";

	double DT(0);
	double averageB(0);
	double ionCyclotronFrequency(0);
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
			cout << "+ Larmor radius: " << scientific << IONS->at(ii).BGP.LarmorRadius << fixed << " m\n\n";
		}

		//We don't take into account the tracers dynamics of course!
		if(IONS->at(ii).BGP.Wc > ionCyclotronFrequency && (IONS->at(ii).SPECIES != 0))
			ionCyclotronFrequency = IONS->at(ii).BGP.Wc;

		if(IONS->at(ii).SPECIES != 0){
			ionSpecies++;
			ionMass += IONS->at(ii).M;

			#ifdef ONED
				vec V = abs(IONS->at(ii).velocity.col(0));
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif

			#ifdef TWOD
				vec V = zeros(IONS->at(ii).NSP);
				V = sqrt( IONS->at(ii).velocity.col(0) % IONS->at(ii).velocity.col(0) + IONS->at(ii).velocity.col(1) % IONS->at(ii).velocity.col(1) );
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif

			#ifdef THREED
				vec V = zeros(IONS->at(ii).NSP);
				V = sqrt( IONS->at(ii).velocity.col(0) % IONS->at(ii).velocity.col(0) + IONS->at(ii).velocity.col(1) % IONS->at(ii).velocity.col(1) + IONS->at(ii).velocity.col(2) % IONS->at(ii).velocity.col(2));
				if( max(V) > ion_vmax )
					ion_vmax = max(V);
			#endif
		}

	}//Iterations over the ion species

	ionMass /=  (double)ionSpecies;
	VT = sqrt(2.0*F_KB*params->BGP.Te/ionMass);//Background thermal velocity

	params->backgroundTc = (2.0*M_PI/ionCyclotronFrequency);//Minimum ion gyroperiod.

	if(params->BGP.theta == 0){
//		CFL_ions = 0.5*(params->DrL*VT/ion_vmax)/(2*M_PI*sqrt((double)params->dimension));
//		DT = CFL_ions*params->backgroundTc;

		params->checkStability = 1;
		params->rateOfChecking = 5;

		DT = 0.5*Cmax*mesh->DX/ion_vmax;

		if(params->DTc*params->backgroundTc > DT){
			if(params->mpi.rank_cart == 0)
				cout << "Time step defined by Courant–Friedrichs–Lewy condition for ions: " << scientific << DT << fixed << " s\n";
		}else{
			DT = params->DTc*params->backgroundTc;
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

		CFL_ions = (0.5*Cmax*mesh->DX/ion_vmax)/params->backgroundTc;

		if(CFL_w < CFL_ions){
			params->checkStability = 1;
			params->rateOfChecking = 100;
			if(CFL_w < params->DTc){
				DT = CFL_w*params->backgroundTc;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by Courant–Friedrichs–Lewy condition for whistler waves: " << scientific << DT << fixed << " s\n";
			}else{
				DT = params->DTc*params->backgroundTc;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by user: " << scientific << DT << fixed << " s\n";
			}
		}else{
			params->checkStability = 1;
			params->rateOfChecking = 10;
			DT = CFL_ions*params->backgroundTc;

			if(params->DTc*params->backgroundTc > DT){
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by Courant–Friedrichs–Lewy condition for ions: " << scientific << DT << fixed << " s\n";
			}else{
				DT = params->DTc*params->backgroundTc;
				if(params->mpi.rank_cart == 0)
					cout << "Time step defined by user: " << scientific << DT << fixed << " s\n";
			}
		}
	}


	if(params->mpi.rank_cart == 0){
		cout << "Cadence for saving outputs: " << params->saveVariablesEach << "\n";
		cout << "Cadence for checking stability: " << params->rateOfChecking << "\n";
		cout << "* * * * * * * * * * * * * * * TIME STEP COMPUTED * * * * * * * * * * * * * * * * * * * * *\n\n";
	}
	return(DT);
}


void UNITS::defineCharacteristicScales(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS){
	// The definition of the characteristic quantities is based on:
	// D Winske and N Omidi, Hybrid codes.
	// All the quantities below have units (SI).

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * DEFINING CHARACTERISTIC SCALES IN SIMULATION * * * * * * * * * * * * * * * * * *\n";

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iterations over the ion species.
		CS->mass += IONS->at(ii).M;
		CS->charge += fabs(IONS->at(ii).Q);
	}//Iterations over the ion species.

	CS->mass /= params->numberOfIonSpecies;
	CS->charge /= params->numberOfIonSpecies;
	CS->density = params->ne;

	double plasmaFrequency(0);//Background ion-plasma frequency.
	plasmaFrequency = sqrt( CS->density*CS->charge*CS->charge/(CS->mass*F_EPSILON) );

	CS->time = 1/plasmaFrequency;
	CS->velocity = F_C;
	CS->length = CS->velocity/plasmaFrequency;
	CS->eField = ( plasmaFrequency*CS->mass*CS->velocity )/CS->charge;
	CS->bField = ( plasmaFrequency*CS->mass )/CS->charge;
	CS->temperature = CS->mass*CS->velocity*CS->velocity/F_KB;


	CS->pressure = CS->mass*CS->density*CS->velocity*CS->velocity;

	if(params->mpi.rank_cart == 0){
		cout << "Average mass: " << scientific << CS->mass << " kg\n";
		cout << "Average charge: " << scientific << CS->charge << " C\n";
		cout << "Density: " << scientific << CS->density << " m^(-3)\n";
		cout << "Time: " << scientific << CS->time << " s\n";
		cout << "Plasma frequency: " << scientific << plasmaFrequency << " s\n";
		cout << "Velocity: " << scientific << CS->velocity << " m/s\n";
		cout << "Length: " << scientific << CS->length << " m\n";
		cout << "Electric field intensity: " << scientific << CS->eField << " V/m\n";
		cout << "Magnetic field intensity: " << scientific << CS->bField << " T\n";
		cout << "Pressure: " << scientific << CS->pressure << " Pa\n";
		cout << "Temperature: " << scientific << CS->temperature << " K\n";
		cout << fixed;
		cout << "* * * * * * * * * * * * CHARACTERISTIC SCALES IN SIMULATION DEFINED  * * * * * * * * * * * * * * * * * *\n\n";
	}
}


void UNITS::dimensionlessForm(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB,const characteristicScales * CS){
	// Normalizing physical constants
	F_E_DS /= CS->charge; // Dimensionless electron charge
	F_MU_DS *= CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass; // Dimensionless vacuum permittivity

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
		IONS->at(ii).velocity = IONS->at(ii).velocity/CS->velocity;
	}//Iterations over the ion species.
	//Normalizing ions' properties.

	//Normalizing the electromagnetic fields.

	EB->E /= CS->eField;
	EB->B /= CS->bField;

	//Normalizing the electromagnetic fields.
}


void UNITS::normalizeVariables(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB,const characteristicScales * CS){
	params->DT = defineTimeStep(params,mesh,IONS,EB);//Defining the proper time step in order to resolve the ion gyromotion.

	dimensionlessForm(params,mesh,IONS,EB,CS);
}


void UNITS::defineCharacteristicScalesAndBcast(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS){

	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		defineCharacteristicScales(params,IONS,CS);
	}

	MPI_MAIN mpi_class;

	mpi_class.broadcastCharacteristicScales(params,CS);
}
