/*All the variables MUST be expresed using the SI units (mks). The charge and
temperature the units are Coulombs (C) and Kelvins (K). The
*/

#include "units.h"

double UNITS::defineTimeStep(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB){

	string tmpString = params->PATH + "/ionParameters.txt";
	char *fileName = new char[tmpString.length()+1];
	std::strcpy(fileName,tmpString.c_str());
	std::ofstream ofs(fileName,std::ofstream::out);
	delete[] fileName;

	double DT(0), averageB(0);

	averageB = params->BGP.backgroundBField;
	ofs << "\tThe average magnetic field is: " << scientific << averageB << " T\n";

	double ionCyclotronFrequency(0);
	double ion_vmax(0), ionSpecies(0), ionMass(0), V_T(0), CFL_ions(0);

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iterations over the ion species

		ofs << "\n \tPROPERTIES OF THE ION SPECIES No. " << ii + 1 << '\n';
		ofs << "\tThe charge of the species " << ii+1 << " is: " << IONS->at(ii).Q << " C\n";//Revisar las cuentas con el factor NCP
		ofs << "\tThe mass of the species " << ii+1 << " is: " << IONS->at(ii).M << " kg\n";
		ofs << "\tThe number of super-particles for the species " << ii + 1 << " is: " << IONS->at(ii).NSP << "\n";
		ofs << "\tThe number of particles per cell of the species " << ii + 1 << " is: " << IONS->at(ii).NPC << "\n";
		ofs << "\tThe number of charged particles per super-particle of the species " << ii+1 << " is: " << IONS->at(ii).NCP << "\n";
		ofs << "\tThe simulated density of the species " << ii + 1 << " is: " << IONS->at(ii).BGP.Dn*params->totalDensity << " m^{-3}\n";
		ofs << "\tThe parallel temperature of the species " << ii + 1 << " is: " << IONS->at(ii).BGP.Tpar << " K\n";
		ofs << "\tThe perpendicular temperature of the species " << ii + 1 << " is: " << IONS->at(ii).BGP.Tper << " K\n";
		ofs << "\tThe parallel thermal velocity of the species " << ii + 1 << " is: " << IONS->at(ii).BGP.V_Tpar << " m/s\n";
		ofs << "\tThe perpendicular thermal velocity of the species " << ii + 1 << " is: " << IONS->at(ii).BGP.V_Tper << " m/s\n";
		IONS->at(ii).BGP.Wc = IONS->at(ii).Q*averageB/IONS->at(ii).M;
		ofs << "\tThe cyclotron frequency of the species " << ii + 1 << " is: " << scientific << IONS->at(ii).BGP.Wc << " Hz\n";
		IONS->at(ii).BGP.Wpi = sqrt( IONS->at(ii).BGP.Dn*params->totalDensity*IONS->at(ii).Q*IONS->at(ii).Q/(F_EPSILON*IONS->at(ii).M) );//Check the definition of the plasma freq for each species!
		ofs << "\tThe ion plasma frequency of the species " << ii + 1 << " is: " << scientific << IONS->at(ii).BGP.Wpi << " Hz\n";
		ofs << "\tThe gyroperiod of the species " << ii + 1 << " is: " << scientific << 2.0*M_PI/IONS->at(ii).BGP.Wc << " s\n";
		IONS->at(ii).BGP.LarmorRadius = IONS->at(ii).BGP.V_Tper/IONS->at(ii).BGP.Wc;
		ofs << "\tThe average Larmor radius of the species " << ii + 1 << " is: " << scientific << IONS->at(ii).BGP.LarmorRadius << " m\n";

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

	ionMass /=  ionSpecies;
	V_T = sqrt(2.0*F_KB*params->BGP.backgroundTemperature/ionMass);//Background thermal velocity

	params->backgroundTc = (2.0*M_PI/ionCyclotronFrequency);//Minimum ion gyroperiod.

	if(params->BGP.theta == 0){
//		CFL_ions = 0.5*(params->DrL*V_T/ion_vmax)/(2*M_PI*sqrt((double)params->dimension));
//		DT = CFL_ions*params->backgroundTc;

		params->checkStability = 1;
		params->rateOfChecking = 5;

		DT = 0.5*Cmax*mesh->DX/ion_vmax;

		if(params->DTc*params->backgroundTc > DT){
			ofs << '\n';
			ofs << "\tTimestep following CFL_ions criterium (DT/Tc): " << DT/params->backgroundTc << '\n';
			ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
			ofs << "\t Checking stability each: " << params->rateOfChecking << " iterations\n";
		}else{
			DT = params->DTc*params->backgroundTc;
			ofs << '\n';
			ofs << "\tTimestep given by the user \n";
			ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
			ofs << "\tChecking stability each: " << params->rateOfChecking << " iterations\n";
		}

	}else{
		//Here we calculate the timestep following the CFL condition for the reltative high-frequency whistler waves.
		double CFL_w(0), A(0), B(0);//DT = CFL_w*Tc;

		B = params->BGP.backgroundBField;
//		A = params->DrL*sqrt(2*F_KB*params->BGP.backgroundTemperature*params->totalDensity/F_EPSILON)/(M_PI*F_C*B);
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
				ofs << '\n';
				ofs << "\tTimestep following CFL_wh criterium (DT/Tc): " << CFL_w << '\n';
				ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
			}else{
				DT = params->DTc*params->backgroundTc;
				ofs << '\n';
				ofs << "\tTimestep given by the user  (DT/Tc): " << params->DTc << '\n';
				ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
			}
		}else{
			params->checkStability = 1;
			params->rateOfChecking = 10;
			DT = CFL_ions*params->backgroundTc;
			if(params->DTc*params->backgroundTc > DT){
				ofs << '\n';
				ofs << "\tTimestep following CFL_ions criterium (DT/Tc): " << CFL_ions << '\n';
				ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
				ofs << "\t Checking stability each: " << params->rateOfChecking << " iterations\n";
			}else{
				DT = params->DTc*params->backgroundTc;
				ofs << '\n';
				ofs << "\tTimestep given by the user \n";
				ofs << "\tSaving variables each : " << params->saveVariablesEach << " iterations\n";
				ofs << "\tChecking stability each: " << params->rateOfChecking << " iterations\n";
			}
		}
	}
	ofs.close();
	return(DT);
}


void UNITS::defineCharacteristicScales(inputParameters * params,vector<ionSpecies> * IONS,characteristicScales * CS){
	// The definition of the characteristic quantities is based on:
	// D Winske and N Omidi, Hybrid codes.
	// All the quantities below have units (SI).

	string tmpString = params->PATH + "/characteristicScales.txt";
	char *fileName = new char[tmpString.length()+1];
	std::strcpy(fileName,tmpString.c_str());
	std::ofstream ofs(fileName,std::ofstream::out);
	delete[] fileName;

	ofs << "Status: Defining the characteristic scales of the simulation.\n";

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iterations over the ion species.
		CS->mass += IONS->at(ii).M;
		CS->charge += fabs(IONS->at(ii).Q);
	}//Iterations over the ion species.

	CS->mass /= params->numberOfIonSpecies;
	CS->charge /= params->numberOfIonSpecies;
	CS->density = params->totalDensity;

	double plasmaFrequency(0);//Background ion-plasma frequency.
	plasmaFrequency = sqrt( CS->density*CS->charge*CS->charge/(CS->mass*F_EPSILON) );

	CS->time = 1/plasmaFrequency;
	CS->velocity = F_C;
	CS->length = CS->velocity/plasmaFrequency;
	CS->eField = ( plasmaFrequency*CS->mass*CS->velocity )/CS->charge;
	CS->bField = ( plasmaFrequency*CS->mass )/CS->charge;
	CS->temperature = CS->mass*CS->velocity*CS->velocity/F_KB;


	CS->pressure = CS->mass*CS->density*CS->velocity*CS->velocity;


	ofs << "\tThe average mass of the ions is: " << scientific << CS->mass << " kg\n";
	ofs << "\tThe average charge of the ions is: " << scientific << CS->charge << " C\n";
	ofs << "\tThe characteristic density is: " << scientific << CS->density << " m^(-3)\n";
	ofs << "\tThe characteristic time is: " << scientific << CS->time << " s\n";
	ofs << "\tThe plasma frequency is: " << scientific << plasmaFrequency << " s\n";
	ofs << "\tThe characteristic velocity is: " << scientific << CS->velocity << " m/s\n";
	ofs << "\tThe characteristic length is: " << scientific << CS->length << " m\n";
	ofs << "\tThe characteristic electric field intensity is: " << scientific << CS->eField << " N/C\n";
	ofs << "\tThe characteristic magnetic field intensity is: " << scientific << CS->bField << " T\n";
	ofs << "\tThe characteristic pressure is: " << scientific << CS->pressure << " Pa\n";
	ofs << "\tThe characteristic temperature is: " << scientific << CS->temperature << " K\n";

	ofs << "Status: Characteristic scales defined.\n";

	ofs.close();
}

void UNITS::dimensionlessForm(inputParameters * params,meshGeometry * mesh,vector<ionSpecies> * IONS,emf * EB,const characteristicScales * CS){

	//Normalizing the parameters.
	params->DT /= CS->time;
//	params->BGP.backgroundDensity /= CS->density;
	params->totalDensity /= CS->density;
	params->BGP.backgroundTemperature /= CS->temperature;
	params->BGP.backgroundBField /= CS->bField;
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
		IONS->at(ii).BGP.V_Tpar /= CS->velocity;
		IONS->at(ii).BGP.V_Tper /= CS->velocity;
		IONS->at(ii).BGP.Wc *= CS->time;
		IONS->at(ii).BGP.Wpi *= CS->time;//IMPORTANT: Not normalized before!!
		IONS->at(ii).position = IONS->at(ii).position/CS->length;
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
