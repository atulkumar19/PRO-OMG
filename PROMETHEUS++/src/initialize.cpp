#include "initialize.h"

map<string,float> INITIALIZE::loadParameters(string * inputFile){
	string tmpName;
	float tmpValue;
	fstream fileParameters;

	std::map<string,float> listOfParameters;
	fileParameters.open(inputFile->data(),ifstream::in);

    	if (!fileParameters){
      		cerr << "ERROR: The input file couldn't be opened.\n";
        	exit(1);
    	}

    	while ( fileParameters >> tmpName >> tmpValue ){
          	listOfParameters[ tmpName ] = tmpValue;
    	}

    	fileParameters.close();

    	return listOfParameters;
}

map<string,float> INITIALIZE::loadParameters(const char *  inputFile){
	string tmpName;
	float tmpValue;
	fstream fileParameters;

	cout << inputFile;

	std::map<string,float> listOfParameters;
	fileParameters.open(inputFile ,ifstream::in);

    	if (!fileParameters){
      		cerr << "ERROR: The input file couldn't be opened.\n";
        	exit(1);
    	}

    	while ( fileParameters >> tmpName >> tmpValue ){
          	listOfParameters[ tmpName ] = tmpValue;
    	}

    	fileParameters.close();

    	return listOfParameters;
}

INITIALIZE::INITIALIZE(inputParameters * params,int argc,char* argv[]){
	std::map<string,float> parametersMap;

	MPI_Comm_size(MPI_COMM_WORLD,&params->mpi.NUMBER_MPI_DOMAINS);
	MPI_Comm_rank(MPI_COMM_WORLD,&params->mpi.MPI_DOMAIN_NUMBER);

	if( fmod( (double)params->mpi.NUMBER_MPI_DOMAINS,2.0 ) > 0.0 ){
		if(params->mpi.MPI_DOMAIN_NUMBER == 0)
			cout << "WARNING: The number of MPI processes must be an even number. Terminating simulation!";

		MPI_Abort(MPI_COMM_WORLD,-1);
	}

	//Initialize to zero the variables of the class
	INITIALIZE::ionSkinDepth = 0.0;
	INITIALIZE::LarmorRadius = 0.0;

	params->PATH = argv[1];

	params->argc = argc;
	params->argv = argv;

	if(params->argc > 2){
		string argv(params->argv[2]);
		string name = "inputFiles/input_file_" + argv + ".input";
		parametersMap = loadParameters(&name);

		params->PATH += "/" + argv;
	}else{
		parametersMap = loadParameters("inputFiles/input_file.input");
		params->PATH += "/outputFiles";
	}

	// Create HDF5 folders if they don't exist
	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		string mkdir_outputs_dir = "mkdir " + params->PATH;
		const char * sys = mkdir_outputs_dir.c_str();
		int rsys = system(sys);

		string mkdir_outputs_dir_HDF5 = mkdir_outputs_dir + "/HDF5";
		sys = mkdir_outputs_dir_HDF5.c_str();
		rsys = system(sys);
	}

	params->quietStart = parametersMap["quietStart"];

	params->DTc = parametersMap["DTc"];

	params->restart = (int)parametersMap["restart"];

	params->weightingScheme = (int)parametersMap["weightingScheme"];

	params->BC = (int)parametersMap["BC"];

	params->smoothingParameter = parametersMap["smoothingParameter"];

	params->numberOfRKIterations = (int)parametersMap["numberOfRKIterations"];

	params->filtersPerIteration = (int)parametersMap["filtersPerIteration"];

	params->filtersPerIterationIonsVariables = (int)parametersMap["filtersPerIterationIonsVariables"];

	params->TVF = parametersMap["TVF"];

	params->checkSmoothParameter = (int)parametersMap["checkSmoothParameter"];

	params->timeIterations = (int)parametersMap["timeIterations"];

	params->transient = (unsigned int)parametersMap["transient"];

	params->numberOfIonSpecies = (int)parametersMap["numberOfIonSpecies"];

	params->numberOfTracerSpecies = (int)parametersMap["numberOfTracerSpecies"];

	params->loadModes = (unsigned int)parametersMap["loadModes"];

	params->numberOfAlfvenicModes = (unsigned int)parametersMap["numberOfAlfvenicModes"];

	params->numberOfTestModes = (unsigned int)parametersMap["numberOfTestModes"];

	params->maxAngle = parametersMap["maxAngle"];

	params->shuffleModes = (unsigned int)parametersMap["shuffleModes"];

	params->fracMagEnerInj = parametersMap["fracMagEnerInj"];

	params->simulatedDensityFraction = parametersMap["simulatedDensityFraction"];

	params->totalDensity = parametersMap["totalDensity"];

	params->loadFields = (int)parametersMap["loadFields"];

	params->saveVariablesEach = parametersMap["saveVariablesEach"];

	params->em = new energyMonitor((int)params->numberOfIonSpecies,(int)params->timeIterations);

	params->meshDim.set_size(3);
	params->meshDim(0) = (unsigned int)parametersMap["NX"];
	params->meshDim(1) = (unsigned int)parametersMap["NY"];
	params->meshDim(2) = (unsigned int)parametersMap["NZ"];

	params->DrL = parametersMap["DrL"];

	params->dp = parametersMap["dp"];

	params->BGP.backgroundTemperature = parametersMap["backgroundTemperature"];

	params->BGP.theta = parametersMap["theta"];

	params->BGP.phi = parametersMap["phi"];

	params->BGP.backgroundBField = parametersMap["backgroundBField"];

	params->BGP.Bx = params->BGP.backgroundBField*sin(params->BGP.theta*M_PI/180.0);
	params->BGP.By = 0.0;
	params->BGP.Bz = params->BGP.backgroundBField*cos(params->BGP.theta*M_PI/180.0);
}


void INITIALIZE::loadMeshGeometry(const inputParameters * params,characteristicScales * CS,meshGeometry * mesh){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	string tmpString = params->PATH + "/loadMeshGeometry_D" + domainNumber.str() + ".txt";
	char *fileName = new char[tmpString.length()+1];
	std::strcpy(fileName,tmpString.c_str());
	std::ofstream ofs(fileName,std::ofstream::out);
	delete[] fileName;

	ofs << "Status: Loading the grid.\n";

	ofs << "\tThe mesh will be set up using internal parameters.\n";

	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		mesh->DX = params->DrL*INITIALIZE::LarmorRadius;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		ofs << "\tUsing Larmor radius as parameter.\n";
	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		mesh->DX = params->dp*INITIALIZE::ionSkinDepth;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		ofs << "\tUsing ion skin depth as parameter.\n";
	}

	mesh->dim.set_size(3);

	#ifdef ONED
	mesh->dim(0) = params->meshDim(0);
	mesh->dim(1) = 1;
	mesh->dim(2) = 1;
	#endif

	#ifdef TWOD
	mesh->dim(0) = params->meshDim(0);
	mesh->dim(1) = params->meshDim(1);
	mesh->dim(2) = 1;
	#endif

	#ifdef THREED
	mesh->dim(0) = params->meshDim(0);
	mesh->dim(1) = params->meshDim(1);
	mesh->dim(2) = params->meshDim(2);
	#endif


	mesh->nodes.X.set_size(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);
	mesh->nodes.Y.set_size(mesh->dim(1));
	mesh->nodes.Z.set_size(mesh->dim(2));

	for(int ii=0;ii<(int)(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);ii++){
		mesh->nodes.X(ii) = (double)ii*mesh->DX; //entire simulation domain's mesh grid
	}
	for(int ii=0;ii<mesh->dim(1);ii++){
		mesh->nodes.Y(ii) = (double)ii*mesh->DY; //
	}
	for(int ii=0;ii<mesh->dim(2);ii++){
		mesh->nodes.Z(ii) = (double)ii*mesh->DZ; //
	}

	ofs << "\tThe domain dimension along the x-axis is: " << mesh->nodes.X(mesh->dim(0)-1) + mesh->DX << " m\n";
	ofs << "\tThe domain dimension along the y-axis is: " << mesh->nodes.Y(mesh->dim(1)-1) + mesh->DY << " m\n";
	ofs << "\tThe domain dimension along the z-axis is: " << mesh->nodes.Z(mesh->dim(2)-1) + mesh->DZ << " m\n";

	ofs << "Status: Grid loaded.\n";

	ofs.close();
}


void INITIALIZE::calculateSuperParticleNumberDensity(const inputParameters * params,const characteristicScales * CS,\
	const meshGeometry * mesh,vector<ionSpecies> * IONS){

	double chargeDensityPerCell;

	#ifdef ONED
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->simulatedDensityFraction*params->totalDensity/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->dim(0))*chargeDensityPerCell;
	}
	#endif

	#ifdef TWOD
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->simulatedDensityFraction*params->totalDensity/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->dim(0)*mesh->DY*(double)mesh->dim(1))*chargeDensityPerCell;
	}
	#endif

	#ifdef THREED
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->simulatedDensityFraction*params->totalDensity/IONS->at(ii).NSP;

		IONS->at(ii).NCP = \
		(mesh->DX*(double)mesh->dim(0)*mesh->DY*(double)mesh->dim(1)*mesh->DZ*(double)mesh->dim(0))*chargeDensityPerCell;
	}
	#endif
}

void INITIALIZE::loadIons(inputParameters * params,vector<ionSpecies> * IONS){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	string tmpString = params->PATH + "/loadIons_D" + domainNumber.str() + ".txt";
	char *fileName = new char[tmpString.length()+1];
	std::strcpy(fileName,tmpString.c_str());
	std::ofstream ofs(fileName,std::ofstream::out);
	delete[] fileName;

	ofs <<"Status: Loading ions...\n";
	ofs << "\tThe simulation includes " << params->numberOfIonSpecies << " ion species.\n";
	ofs << "\tThe simulation includes " << params->numberOfTracerSpecies << " tracer species.\n";

	std::map<string,float> parametersMap;
	if(params->argc > 2){
		string argv(params->argv[2]);
		string name = "inputFiles/ions_properties_" + argv + ".ion";
		parametersMap = loadParameters(&name);
	}else{
		parametersMap = loadParameters("inputFiles/ions_properties.ion");
	}

	int totalNumSpecies(params->numberOfIonSpecies + params->numberOfTracerSpecies);

	for(int ii=0;ii<totalNumSpecies;ii++){
		string name;
		ionSpecies ions;
		stringstream ss;

		ss << ii + 1;

		name = "SPECIES" + ss.str();
		ions.SPECIES = (int)parametersMap[name];
		name.clear();

		name = "NPC" + ss.str();
		ions.NPC = parametersMap[name];
		name.clear();

		name = "Tper" + ss.str();
		ions.BGP.Tper = parametersMap[name];
		name.clear();

		name = "Tpar" + ss.str();
		ions.BGP.Tpar = parametersMap[name];
		name.clear();

		name = "Dn" + ss.str();
		ions.BGP.Dn = parametersMap[name];
		name.clear();

		name = "pctSupPartOutput" + ss.str();
		ions.pctSupPartOutput = parametersMap[name];
		name.clear();

		if(ions.SPECIES == 0){//Tracers
			ofs << "\tThe species No " << ii + 1 << " are tracers.\n";
			//The number of charged particles per super-particle (NCP) is defined in
			name = "NCP" + ss.str();
			ions.NCP = parametersMap[name];
			name.clear();
			name = "Z" + ss.str();
			ions.Z = parametersMap[name];//parametersMap[name] = Atomic number.
			ions.Q = F_E*ions.Z;
			name.clear();
			name = "M" + ss.str();
			ions.M = F_U*parametersMap[name]; //parametersMap[name] times the proton mass.
		}else if(ions.SPECIES == 1){ //Electrons
			ofs << "\tThe species No " << ii + 1 << " are electrons.\n";
			ions.Q = -F_E;
//			ions.Z = ? //What to do in this case?! Modify later.
			ions.M = F_ME;
		}else if(ions.SPECIES == 2){ //Protons
			ofs << "\tThe species No " << ii + 1 << " are protons.\n";
			ions.Z = 1.0;
			ions.Q = F_E;
			ions.M = F_MP;
		}else{
			ofs << "\tThe species No " << ii + 1 << " is not in the database.\n";
			name = "Z" + ss.str();
			ions.Z = parametersMap[name]; //parametersMap[name] = Atomic number.
			ions.Q = F_E*ions.Z;
			name.clear();
			name = "M" + ss.str();
			ions.M = F_U*parametersMap[name]; //parametersMap[name] times the atomic mass unit.
		}

		//Definition of the background ion density for each species
		ions.BGP.BG_n = ions.BGP.Dn*(1.0 - params->simulatedDensityFraction)*params->totalDensity;

		//Definition of the initial total number of superparticles for each species
		ions.NSP = ceil(ions.NPC*params->meshDim(0));

		ions.nSupPartOutput = floor( (ions.pctSupPartOutput/100.0)*ions.NSP );

		if(params->restart == 1){
			ofs << "\t Status: Restarting the simulation: loading ions' properties.\n";
			MPI_Abort(MPI_COMM_WORLD,-1);
		}else{
			if(ii == 0){ //Background ions (Protons)
				if(params->quietStart == 0){
	                RANDOMSTART rs;
	    			rs.maxwellianVelocityDistribution(params,&ions,"z");
//					rs.maxwellianVelocityDistribution(params,&ions,"x");
	            }else if(params->quietStart == 1){
					QUIETSTART qs(params,&ions);
					qs.maxwellianVelocityDistribution(params,&ions,"z");
				}

				ions.BGP.BG_UX = 0.0;
				ions.BGP.BG_UY = 0.0;
				ions.BGP.BG_UZ = 0.0;
			}

			if(ii == 1){//Alpha-particles
				if(params->quietStart == 0){
                   	RANDOMSTART rs;
	                rs.ringLikeVelocityDistribution(params,&ions,"z");
//					rs.maxwellianVelocityDistribution(params,&ions,"z");
//					rs.beamVelocityDistribution(params,&ions,"z"); //Perpendicular propagation case
	            }else if(params->quietStart == 1){
	                QUIETSTART qs(params,&ions);
	                qs.ringLikeVelocityDistribution(params,&ions,"z");
	            }

				ions.BGP.BG_UX = 0.0;
				ions.BGP.BG_UY = 0.0;
				ions.BGP.BG_UZ = 0.0;
			}

			if(ii == 2){//Tracer ions
				// Do something
			}

			ofs << "\t The number of superparticles of the species " << ii + 1 << " is: " << ions.NSP << '\n';
		}

		ions.nv.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);

		ions.n.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);

		ions.meshNode.zeros(ions.NSP);

		//Checking the integrity of the initial condition
		if((int)ions.velocity.n_elem != (int)(3*ions.NSP)){
			ofs << "ERROR: loading ions' velocity of species: "<< ii + 1 << '\n';
			ofs << "\tThe velocity array contains a number of elements that it should not have.\n";
			exit(1);
		}
		if((int)ions.position.n_elem != (int)(3*ions.NSP)){
			ofs << "ERROR: loading ions' position of species: "<< ii + 1 << '\n';
			ofs << "\tThe position array contains a number of elements that it should not have.\n";
			exit(1);
		}
		//Checking integrity of the initial condition

		IONS->push_back(ions);
	}//Iteration over ion species

	//The ion skin depth is calculated using the set of values of species No 1, which is assumed to be the backgroud population.
	INITIALIZE::ionSkinDepth =\
	sqrt(IONS->at(0).M/(F_MU*(IONS->at(0).BGP.Dn*params->simulatedDensityFraction*params->totalDensity)))/IONS->at(0).Q;

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iteration over ion species
		double rL;

		rL = sqrt( 2.0*F_KB*IONS->at(ii).BGP.Tper*IONS->at(ii).M )/( IONS->at(ii).Q*params->BGP.backgroundBField );

		if(ii == 0){
			INITIALIZE::LarmorRadius = rL;
		}else if(rL < INITIALIZE::LarmorRadius){
			INITIALIZE::LarmorRadius = rL;
		}

		ofs<<"\tThe Larmor radius of species "<< ii+1 <<" is: "<< rL << " m\n";
	}//Iteration over ion species

	ofs<<"\tThe Larmor radius is: "<< INITIALIZE::LarmorRadius <<" m\n";
	ofs<<"\tThe ion skin depth is: "<< INITIALIZE::ionSkinDepth <<" m\n";

	double HX;
	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		HX = params->DrL*INITIALIZE::LarmorRadius;
	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		HX = params->dp*INITIALIZE::ionSkinDepth;
	}

	if(params->quietStart == 0){
		for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
			IONS->at(ii).position.col(0) = \
			(HX*params->meshDim(0))*(params->mpi.MPI_DOMAIN_NUMBER + IONS->at(ii).position.col(0));
		}
	}else if(params->quietStart == 1){
		for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
			IONS->at(ii).position.col(0) *= HX*params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS;
		}
	}
	ofs<<"Status: The ions were loaded successfully...\n";

	ofs.close();

	MPI_Barrier(MPI_COMM_WORLD);
}

void INITIALIZE::initializeFields(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	string tmpString = params->PATH + "/initializeEMF_D" + domainNumber.str() + ".txt";
	char *fileName = new char[tmpString.length()+1];
	std::strcpy(fileName,tmpString.c_str());
	std::ofstream ofs(fileName,std::ofstream::out);
	delete[] fileName;

	ofs << "Status: Initializing electromagnetic fields...\n";

	if(params->loadFields==1){//The electromagnetic fields are loaded from external files.
		ofs << "\tLoading the initial condition for the electromagnetic fields from external files.\n";
	}else{//The electromagnetic fields are being initialized in the runtime.
		ofs << "\tInitializing the electromagnetic fields in the runtime.\n";
		ofs << "\tThe component of B along the x-axis is: " \
			<< sin(params->BGP.theta*M_PI/180)*params->BGP.backgroundBField <<'\n';
		if(params->BGP.theta == 90.0)
			ofs << "\tThe component of B along the z-axis is: " << 0.0 <<'\n';
		else
			ofs << "\tThe component of B along the z-axis is: " \
				<< cos(params->BGP.theta*M_PI/180.0)*params->BGP.backgroundBField <<'\n';


		int dim(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);
		EB->zeros(dim + 2);//We include the ghost mesh points (+2) in the initialization

		EB->B.X.fill(params->BGP.Bx);//x
		EB->B.Y.fill(params->BGP.By);//y
		EB->B.Z.fill(params->BGP.Bz);//z
	}
	ofs << "Status: The electromagnetic fields were initialized without problems...\n";
	ofs.close();
}
