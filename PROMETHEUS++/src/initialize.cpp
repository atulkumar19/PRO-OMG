#include "initialize.h"

map<string,float> INITIALIZE::loadParameters(string * inputFile){
	string key;
	float value;
	fstream reader;
	std::map<string,float> readMap;


	reader.open(inputFile->data(),ifstream::in);

    if (!reader){
    	cerr << "PRO++ ERROR: The input file couldn't be opened.\n";
    	MPI_Abort(MPI_COMM_WORLD,-123);
    }

    while ( reader >> key >> value ){
      	readMap[ key ] = value;
    }

    reader.close();

    return readMap;
}


map<string,float> INITIALIZE::loadParameters(const char *  inputFile){
	string key;
	float value;
	fstream reader;
	std::map<string,float> readMap;

	reader.open(inputFile ,ifstream::in);

    	if (!reader){
      		cerr << "PRO++ ERROR: The input file couldn't be opened.\n";
        	MPI_Abort(MPI_COMM_WORLD,-123);
    	}

    	while ( reader >> key >> value ){
          	readMap[ key ] = value;
    	}

    	reader.close();

    	return readMap;
}


INITIALIZE::INITIALIZE(inputParameters * params,int argc,char* argv[]){
	std::map<string,float> parametersMap;

	MPI_Comm_size(MPI_COMM_WORLD,&params->mpi.NUMBER_MPI_DOMAINS);
	MPI_Comm_rank(MPI_COMM_WORLD,&params->mpi.MPI_DOMAIN_NUMBER);

	if( fmod( (double)params->mpi.NUMBER_MPI_DOMAINS,2.0 ) > 0.0 ){
		if(params->mpi.MPI_DOMAIN_NUMBER == 0){
			cout << "PRO++ ERROR: The number of MPI processes must be an even number.\n";
		}

		MPI_Abort(MPI_COMM_WORLD,-123);
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
		params->PATH += "/";
	}

	// Create HDF5 folders if they don't exist
	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		string mkdir_outputs_dir = "mkdir " + params->PATH;
		const char * sys = mkdir_outputs_dir.c_str();
		int rsys = system(sys);

		string mkdir_outputs_dir_HDF5 = mkdir_outputs_dir + "/HDF5";
		sys = mkdir_outputs_dir_HDF5.c_str();
		rsys = system(sys);

		string mkdir_outputs_dir_diagnostics = mkdir_outputs_dir + "/diagnostics";
		sys = mkdir_outputs_dir_diagnostics.c_str();
		rsys = system(sys);
	}

	params->particleIntegrator = (int)parametersMap["particleIntegrator"];

	params->quietStart = parametersMap["quietStart"];

	params->DTc = parametersMap["DTc"];

	params->restart = (int)parametersMap["restart"];

	params->weightingScheme = (int)parametersMap["weightingScheme"];

	params->BC = (int)parametersMap["BC"];

	params->smoothingParameter = parametersMap["smoothingParameter"];

	params->numberOfRKIterations = (int)parametersMap["numberOfRKIterations"];

	params->filtersPerIterationFields = (int)parametersMap["filtersPerIterationFields"];

	params->filtersPerIterationIons = (int)parametersMap["filtersPerIterationIons"];

	params->checkSmoothParameter = (int)parametersMap["checkSmoothParameter"];

	params->simulationTime = parametersMap["simulationTime"];

	params->transient = (unsigned int)parametersMap["transient"];

	params->numberOfIonSpecies = (int)parametersMap["numberOfIonSpecies"];

	params->numberOfTracerSpecies = (int)parametersMap["numberOfTracerSpecies"];

	params->loadModes = (unsigned int)parametersMap["loadModes"];

	params->numberOfAlfvenicModes = (unsigned int)parametersMap["numberOfAlfvenicModes"];

	params->numberOfTestModes = (unsigned int)parametersMap["numberOfTestModes"];

	params->maxAngle = parametersMap["maxAngle"];

	params->shuffleModes = (unsigned int)parametersMap["shuffleModes"];

	params->fracMagEnerInj = parametersMap["fracMagEnerInj"];

	params->ne = parametersMap["ne"];

	params->loadFields = (int)parametersMap["loadFields"];

	params->outputCadence = parametersMap["outputCadence"];

	params->em = new energyMonitor((int)params->numberOfIonSpecies,(int)params->timeIterations);

	params->meshDim.set_size(3);
	params->meshDim(0) = (unsigned int)parametersMap["NX"];
	params->meshDim(1) = (unsigned int)parametersMap["NY"];
	params->meshDim(2) = (unsigned int)parametersMap["NZ"];

	params->DrL = parametersMap["DrL"];

	params->dp = parametersMap["dp"];

	params->BGP.Te = parametersMap["Te"]*F_E/F_KB; // Te in eV in input file

	params->BGP.theta = parametersMap["theta"];
	params->BGP.phi = parametersMap["phi"];

	params->BGP.propVectorAngle = parametersMap["propVectorAngle"];

	params->BGP.Bo = parametersMap["Bo"];

	params->BGP.Bx = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0);
	params->BGP.By = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0);
	params->BGP.Bz = params->BGP.Bo*cos(params->BGP.theta*M_PI/180.0);
}


void INITIALIZE::loadMeshGeometry(const inputParameters * params,characteristicScales * CS,meshGeometry * mesh){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * LOADING/COMPUTING SIMULATION GRID * * * * * * * * * * * * * * * * * *\n";

	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		mesh->DX = params->DrL*INITIALIZE::LarmorRadius;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		if(params->mpi.rank_cart == 0)
			cout << "Using LARMOR RADIUS to set up simulation grid.\n";
	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		mesh->DX = params->dp*INITIALIZE::ionSkinDepth;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		if(params->mpi.rank_cart == 0)
			cout << "Using ION SKIN DEPTH to set up simulation grid.\n";
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

	if(params->mpi.rank_cart == 0){
		cout << "Size of simulation domain along the x-axis: " << mesh->nodes.X(mesh->dim(0)-1) + mesh->DX << " m\n";
		cout << "Size of simulation domain along the y-axis: " << mesh->nodes.Y(mesh->dim(1)-1) + mesh->DY << " m\n";
		cout << "Size of simulation domain along the z-axis: " << mesh->nodes.Z(mesh->dim(2)-1) + mesh->DZ << " m\n";
		cout << "* * * * * * * * * * * *  SIMULATION GRID LOADED/COMPUTED  * * * * * * * * * * * * * * * * * *\n\n";
	}
}


void INITIALIZE::calculateSuperParticleNumberDensity(const inputParameters * params,const characteristicScales * CS,\
	const meshGeometry * mesh,vector<ionSpecies> * IONS){

	double chargeDensityPerCell;

	#ifdef ONED
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->dim(0))*chargeDensityPerCell;
	}
	#endif

	#ifdef TWOD
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->dim(0)*mesh->DY*(double)mesh->dim(1))*chargeDensityPerCell;
	}
	#endif

	#ifdef THREED
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).BGP.Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = \
		(mesh->DX*(double)mesh->dim(0)*mesh->DY*(double)mesh->dim(1)*mesh->DZ*(double)mesh->dim(0))*chargeDensityPerCell;
	}
	#endif
}


void INITIALIZE::loadIons(inputParameters * params,vector<ionSpecies> * IONS){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if(params->mpi.rank_cart == 0){
		cout << "* * * * * * * * * * * * INITIALIZING IONS * * * * * * * * * * * * * * * * * *\n";
		cout << "Number of ion species: " << params->numberOfIonSpecies << "\n";
		cout << "Number of tracer species: " << params->numberOfTracerSpecies << "\n";
	}


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
		ions.BGP.Tper = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
		name.clear();

		name = "Tpar" + ss.str();
		ions.BGP.Tpar = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
		name.clear();

		name = "Dn" + ss.str();
		ions.BGP.Dn = parametersMap[name];
		name.clear();

		name = "pctSupPartOutput" + ss.str();
		ions.pctSupPartOutput = parametersMap[name];
		name.clear();

		if(ions.SPECIES == 0){//Tracers
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

			if(params->mpi.rank_cart == 0)
				cout << "Species No "  << ii + 1 << " are tracers\n";
		}else if(ions.SPECIES == 1){ //Electrons
			ions.Q = -F_E;
//			ions.Z = ? //What to do in this case?! Modify later.
			ions.M = F_ME;

			if(params->mpi.rank_cart == 0)
				cout << "Species No "  << ii + 1 << " are electrons\n";
		}else if(ions.SPECIES == 2){ //Protons
			ions.Z = 1.0;
			ions.Q = F_E;
			ions.M = F_MP;

			if(params->mpi.rank_cart == 0)
				cout << "Species No "  << ii + 1 << " are protons\n";
		}else{
			name = "Z" + ss.str();
			ions.Z = parametersMap[name]; //parametersMap[name] = Atomic number.
			ions.Q = F_E*ions.Z;
			name.clear();
			name = "M" + ss.str();
			ions.M = F_U*parametersMap[name]; //parametersMap[name] times the atomic mass unit.

			if(params->mpi.rank_cart == 0){
				cout << "Species No "  << ii + 1 << " are ions with the following parameters:\n";
				cout << "Ion atomic number: " << ions.Z << "\n";
				cout << "Ion mass: " << ions.M << " kg\n";
			}
		}

		//Definition of the initial total number of superparticles for each species
		ions.NSP = ceil(ions.NPC*params->meshDim(0));
		ions.nSupPartOutput = floor( (ions.pctSupPartOutput/100.0)*ions.NSP );

		if(params->restart == 1){
			if(params->mpi.rank_cart == 0)
				cout << "PRO++ MESSAGE: Restarting simulation, loading ions";
			MPI_Abort(MPI_COMM_WORLD,-1);
		}else{
			if(ii == 0){ //Background ions (Protons)
				if(params->quietStart == 0){
	                RANDOMSTART rs(params);
	    			rs.maxwellianVelocityDistribution(params,&ions);
	            }else if(params->quietStart == 1){
					QUIETSTART qs(params,&ions);
					qs.maxwellianVelocityDistribution(params,&ions);
				}
			}

			if(ii == 1){//Alpha-particles
				if(params->quietStart == 0){
                   	RANDOMSTART rs(params);
	                rs.ringLikeVelocityDistribution(params,&ions);
	            }else if(params->quietStart == 1){
	                QUIETSTART qs(params,&ions);
	                qs.ringLikeVelocityDistribution(params,&ions);
	            }
			}

			if(ii == 2){//Tracer ions
				// Do something
			}
		}

		ions.n.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);
		ions.n_.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);
		ions.n__.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);
		ions.n___.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);

		ions.nv.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);
		ions.nv_.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);
		ions.nv__.zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);

		// Setting size and value to zero of arrays for ions' variables
		if(params->mpi.rank_cart == 0)
			cout << "Super-particles used to simulate species No " << ii + 1 << ": " << ions.NSP << '\n';

		ions.meshNode.zeros(ions.NSP);
		ions.wxc.zeros(ions.NSP);
		ions.wxl.zeros(ions.NSP);
		ions.wxr.zeros(ions.NSP);

		//Checking the integrity of the initial condition
		if((int)ions.V.n_elem != (int)(3*ions.NSP)){
			cerr << "PRO++ ERROR: in velocity initial condition of species: " << ii + 1 << '\n';
			MPI_Abort(MPI_COMM_WORLD,-123);
		 	// The velocity array contains a number of elements that it should not have

		}
		if((int)ions.X.n_elem != (int)(3*ions.NSP)){
			cerr << "PRO++ ERROR: in spatial initial condition of species: " << ii + 1 << '\n';
			MPI_Abort(MPI_COMM_WORLD,-123);
			// The position array contains a number of elements that it should not have
		}
		//Checking integrity of the initial condition

		IONS->push_back(ions);
	}//Iteration over ion species

	//The ion skin depth is calculated using the set of values of species No 1, which is assumed to be the backgroud population.
	INITIALIZE::ionSkinDepth =\
	sqrt(IONS->at(0).M/(F_MU*(IONS->at(0).BGP.Dn*params->ne)))/IONS->at(0).Q;

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//Iteration over ion species
		double rL;

		rL = sqrt( 2.0*F_KB*IONS->at(ii).BGP.Tper*IONS->at(ii).M )/( IONS->at(ii).Q*params->BGP.Bo );

		if(ii == 0){
			INITIALIZE::LarmorRadius = rL;
		}else if(rL < INITIALIZE::LarmorRadius){
			INITIALIZE::LarmorRadius = rL;
		}
		if(params->mpi.rank_cart == 0)
			cout <<"Larmor radius of species " << ii+1 << ": "<< rL << " m\n";
	}//Iteration over ion species

	if(params->mpi.rank_cart == 0){
		cout <<"Larmor radius used in simulation " << INITIALIZE::LarmorRadius <<" m\n";
		cout <<"Ion skin depth used in simulation " << INITIALIZE::ionSkinDepth <<" m\n";
	}

	double HX;
	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		HX = params->DrL*INITIALIZE::LarmorRadius;
	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		HX = params->dp*INITIALIZE::ionSkinDepth;
	}

	if(params->quietStart == 0){
		for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
			IONS->at(ii).X.col(0) = \
			(HX*params->meshDim(0))*(params->mpi.MPI_DOMAIN_NUMBER + IONS->at(ii).X.col(0));
		}
	}else if(params->quietStart == 1){
		for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
			IONS->at(ii).X.col(0) *= HX*params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS;
		}
	}

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * IONS INITIALIZED  * * * * * * * * * * * * * * * * * *\n\n";

	MPI_Barrier(MPI_COMM_WORLD);
}


void INITIALIZE::initializeFields(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * INITIALIZING ELECTROMAGNETIC FIELDS * * * * * * * * * * * * * * * * * *\n";

	if(params->loadFields==1){//The electromagnetic fields are loaded from external files.
		if(params->mpi.rank_cart == 0)
			cout << "Loading external electromagnetic fields\n";
		MPI_Abort(MPI_COMM_WORLD,-123);
	}else{//The electromagnetic fields are being initialized in the runtime.
		int dim(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);
		EB->zeros(dim + 2);//We include the ghost mesh points (+2) in the initialization

		EB->B.X.fill(params->BGP.Bx);//x
		EB->B.Y.fill(params->BGP.By);//y
		EB->B.Z.fill(params->BGP.Bz);//z

		if(params->mpi.rank_cart == 0){
			cout << "Initializing electromagnetic fields within simulation\n";
			cout << "Magnetic field component along simulation domain (x-axis): " << scientific << params->BGP.Bx << fixed << " T\n";
			cout << "Magnetic field component perpendicular to simulation domain (y-axis): " << scientific << params->BGP.By << fixed << " T\n";
			cout << "Magnetic field component perpendicular to simulation domain (z-axis): " << scientific << params->BGP.Bz << fixed << " T\n";
		}
	}

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * ELECTROMAGNETIC FIELDS INITIALIZED  * * * * * * * * * * * * * * * * * *\n\n";
}
