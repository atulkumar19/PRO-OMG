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

#include "initialize.h"

vector<string> INITIALIZE::split(const string& str, const string& delim)
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);
        if (pos == string::npos) pos = str.length();
        string token = str.substr(prev, pos-prev);
        if (!token.empty()) tokens.push_back(token);
        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());

    return tokens;
}


map<string,float> INITIALIZE::loadParameters(string * inputFile){
	string key;
	float value;
	fstream reader;
	std::map<string,float> readMap;


	reader.open(inputFile->data(),ifstream::in);

    if (!reader){
        MPI_Barrier(MPI_COMM_WORLD);

    	cerr << "PRO++ ERROR: The input file couldn't be opened." << endl;
    	MPI_Abort(MPI_COMM_WORLD,-101);
    }

    while ( reader >> key >> value ){
      	readMap[ key ] = value;
    }

    reader.close();

    return readMap;
}


map<string,string> INITIALIZE::loadParametersString(string * inputFile){
	string key;
	string value;
	fstream reader;
	std::map<string,string> readMap;


	reader.open(inputFile->data(),ifstream::in);

    if (!reader){
        MPI_Barrier(MPI_COMM_WORLD);

    	cerr << "PRO++ ERROR: The input file couldn't be opened." << endl;
    	MPI_Abort(MPI_COMM_WORLD, -101);
    }

    while ( reader >> key >> value ){
      	readMap[ key ] = value;
    }

    reader.close();

    return readMap;
}


INITIALIZE::INITIALIZE(simulationParameters * params, int argc, char* argv[]){
    // Error codes
    params->errorCodes[-100] = "Odd number of MPI processes";
    params->errorCodes[-101] = "Input file could not be opened";
    params->errorCodes[-102] = "MPI's Cartesian topology could not be created";

	MPI_Comm_size(MPI_COMM_WORLD, &params->mpi.NUMBER_MPI_DOMAINS);
	MPI_Comm_rank(MPI_COMM_WORLD, &params->mpi.MPI_DOMAIN_NUMBER);

    // Copyright and Licence Info
    if (params->mpi.MPI_DOMAIN_NUMBER == 0){
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << endl;
        cout << "* PROMETHEUS++ Copyright (C) 2015-2019  Leopoldo Carbajal               *" << endl;
        cout << "*                                                                       *" << endl;
        cout << "* PROMETHEUS++ is free software: you can redistribute it and/or modify  *" << endl;
        cout << "* it under the terms of the GNU General Public License as published by  *" << endl;
        cout << "* the Free Software Foundation, either version 3 of the License, or     *" << endl;
        cout << "* any later version.                                                    *" << endl;
        cout << "*                                                                       *" << endl;
        cout << "* PROMETHEUS++ is distributed in the hope that it will be useful,       *" << endl;
        cout << "* but WITHOUT ANY WARRANTY; without even the implied warranty of        *" << endl;
        cout << "* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *" << endl;
        cout << "* GNU General Public License for more details.                          *" << endl;
        cout << "*                                                                       *" << endl;
        cout << "* You should have received a copy of the GNU General Public License     *" << endl;
        cout << "* along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/> *" << endl;
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *" << endl;
        cout << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

	if( fmod( (double)params->mpi.NUMBER_MPI_DOMAINS, 2.0 ) > 0.0 ){
        MPI_Barrier(MPI_COMM_WORLD);

		if(params->mpi.MPI_DOMAIN_NUMBER == 0){
			cerr << "PRO++ ERROR: The number of MPI processes must be an even number." << endl;
		}

		MPI_Abort(MPI_COMM_WORLD,-100);
	}

    if(params->mpi.MPI_DOMAIN_NUMBER == 0){
        time_t current_time = std::time(NULL);
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
        cout << "STARTING SIMULATION ON: " << std::ctime(&current_time) << endl;
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
    }

	//Initialize to zero the variables of the class
	INITIALIZE::ionSkinDepth = 0.0;
	INITIALIZE::LarmorRadius = 0.0;

	params->PATH = argv[1];

	params->argc = argc;
	params->argv = argv;

	string name;

	if(params->argc > 2){
		string argv(params->argv[2]);
		name = "inputFiles/input_file_" + argv + ".input";
		params->PATH += "/" + argv;
	}else{
		name = "inputFiles/input_file.input";
		params->PATH += "/";
	}

	std::map<string,string> parametersStringMap;
	parametersStringMap = loadParametersString(&name);

	// Create HDF5 folders if they don't exist
	if(params->mpi.MPI_DOMAIN_NUMBER == 0){
		string mkdir_outputs_dir = "mkdir " + params->PATH;
		const char * sys = mkdir_outputs_dir.c_str();
		int rsys = system(sys);

		string mkdir_outputs_dir_HDF5 = mkdir_outputs_dir + "/HDF5";
		sys = mkdir_outputs_dir_HDF5.c_str();
		rsys = system(sys);

		/*
        string mkdir_outputs_dir_diagnostics = mkdir_outputs_dir + "/diagnostics";
		sys = mkdir_outputs_dir_diagnostics.c_str();
		rsys = system(sys);
        */
	}

    params->dimensionality = std::stoi( parametersStringMap["dimensionality"] );

	params->particleIntegrator = std::stoi( parametersStringMap["particleIntegrator"] );

    if(std::stoi( parametersStringMap["includeElectronInertia"] ) == 1){
        params->includeElectronInertia = true;
    }else{
        params->includeElectronInertia = false;
    }

    if(std::stoi( parametersStringMap["quietStart"] ) == 1){
        params->quietStart = true;
    }else{
        params->quietStart = false;
    }


	params->DTc = std::stod( parametersStringMap["DTc"] );


    if(std::stoi( parametersStringMap["restart"] ) == 1){
        params->restart = true;
    }else{
        params->restart = false;
    }

	params->weightingScheme = std::stoi( parametersStringMap["weightingScheme"] );

	params->BC = std::stoi( parametersStringMap["BC"] );

	params->smoothingParameter = std::stod( parametersStringMap["smoothingParameter"] );

	params->numberOfRKIterations = std::stoi( parametersStringMap["numberOfRKIterations"] );

	params->filtersPerIterationFields = std::stoi( parametersStringMap["filtersPerIterationFields"] );

	params->filtersPerIterationIons = std::stoi( parametersStringMap["filtersPerIterationIons"] );

	params->checkSmoothParameter = std::stoi( parametersStringMap["checkSmoothParameter"] );

	params->simulationTime = std::stod( parametersStringMap["simulationTime"] );

	params->transient = (unsigned int)std::stoi( parametersStringMap["transient"] );

	params->numberOfParticleSpecies = std::stoi( parametersStringMap["numberOfParticleSpecies"] );

	params->numberOfTracerSpecies = std::stoi( parametersStringMap["numberOfTracerSpecies"] );

	params->loadModes = (unsigned int)std::stoi( parametersStringMap["loadModes"] );

	params->numberOfAlfvenicModes = (unsigned int)std::stoi( parametersStringMap["numberOfAlfvenicModes"] );

	params->numberOfTestModes = (unsigned int)std::stoi( parametersStringMap["numberOfTestModes"] );

	params->maxAngle = std::stod( parametersStringMap["maxAngle"] );

	params->shuffleModes = (unsigned int)std::stoi( parametersStringMap["shuffleModes"] );

	params->fracMagEnerInj = std::stod( parametersStringMap["fracMagEnerInj"] );

	params->ne = std::stod( parametersStringMap["ne"] );

	params->loadFields = std::stoi( parametersStringMap["loadFields"] );

	params->outputCadence = std::stod( parametersStringMap["outputCadence"] );

	// params->em = new energyMonitor((int)params->numberOfParticleSpecies,(int)params->timeIterations);

    params->NX_PER_MPI = (unsigned int)std::stoi( parametersStringMap["NX"] );
    params->NY_PER_MPI = (unsigned int)std::stoi( parametersStringMap["NY"] );
    params->NZ_PER_MPI = (unsigned int)std::stoi( parametersStringMap["NZ"] );

	params->DrL = std::stod( parametersStringMap["DrL"] );

	params->dp = std::stod( parametersStringMap["dp"] );

	params->BGP.Te = std::stod( parametersStringMap["Te"] )*F_E/F_KB; // Te in eV in input file

	params->BGP.theta = std::stod( parametersStringMap["theta"] );
	params->BGP.phi = std::stod( parametersStringMap["phi"] );

	params->BGP.propVectorAngle = std::stod( parametersStringMap["propVectorAngle"] );

	params->BGP.Bo = std::stod( parametersStringMap["Bo"] );

	params->BGP.Bx = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0);
	params->BGP.By = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0);
	params->BGP.Bz = params->BGP.Bo*cos(params->BGP.theta*M_PI/180.0);

	// Parsing list of variables in outputs
	std::string nonparsed_variables_list = parametersStringMap["outputs_variables"].substr(1, parametersStringMap["outputs_variables"].length() - 2);
	params->outputs_variables = INITIALIZE::split(nonparsed_variables_list,",");
}


void INITIALIZE::loadMeshGeometry(const simulationParameters * params, fundamentalScales * FS, meshParams * mesh){

    INITIALIZE::LarmorRadius = FS->ionGyroRadius[0];
    INITIALIZE::ionSkinDepth = FS->ionSkinDepth[0];
    if (params->numberOfParticleSpecies > 1){
        for (int ss=1; ss<params->numberOfParticleSpecies; ss++){
            INITIALIZE::LarmorRadius = (INITIALIZE::LarmorRadius < FS->ionGyroRadius[ss]) ? FS->ionGyroRadius[ss] : INITIALIZE::LarmorRadius;
            INITIALIZE::ionSkinDepth = (INITIALIZE::ionSkinDepth < FS->ionSkinDepth[ss]) ? FS->ionSkinDepth[ss] : INITIALIZE::ionSkinDepth;
        }
    }

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if (params->mpi.rank_cart == 0)
		cout << endl << "* * * * * * * * * * * * LOADING/COMPUTING SIMULATION GRID * * * * * * * * * * * * * * * * * *\n";

	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		mesh->DX = params->DrL*INITIALIZE::LarmorRadius;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		if(params->mpi.rank_cart == 0)
			cout << "Using LARMOR RADIUS to set up simulation grid." << endl;

	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		mesh->DX = params->dp*INITIALIZE::ionSkinDepth;
		mesh->DY = mesh->DX;
		mesh->DZ = mesh->DX;
		if(params->mpi.rank_cart == 0)
			cout << "Using ION SKIN DEPTH to set up simulation grid." << endl;
	}

	#ifdef ONED
    mesh->NX_PER_MPI = params->NX_PER_MPI;
    mesh->NY_PER_MPI = 1;
    mesh->NZ_PER_MPI = 1;
	#endif

	#ifdef TWOD
    mesh->NX_PER_MPI = params->NX_PER_MPI;
    mesh->NY_PER_MPI = params->NY_PER_MPI;
    mesh->NZ_PER_MPI = 1;
	#endif

	#ifdef THREED
    mesh->NX_PER_MPI = params->NX_PER_MPI;
    mesh->NY_PER_MPI = params->NY_PER_MPI;
    mesh->NZ_PER_MPI = params->NZ_PER_MPI;
	#endif


	mesh->nodes.X.set_size(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	mesh->nodes.Y.set_size(mesh->NY_PER_MPI);
	mesh->nodes.Z.set_size(mesh->NZ_PER_MPI);

	for(int ii=0;ii<(int)(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);ii++){
		mesh->nodes.X(ii) = (double)ii*mesh->DX; //entire simulation domain's mesh grid
	}
	for(int ii=0;ii<mesh->NY_PER_MPI;ii++){
		mesh->nodes.Y(ii) = (double)ii*mesh->DY; //
	}
	for(int ii=0;ii<mesh->NZ_PER_MPI;ii++){
		mesh->nodes.Z(ii) = (double)ii*mesh->DZ; //
	}

	if(params->mpi.rank_cart == 0){
		cout << "Size of simulation domain along the x-axis: " << mesh->nodes.X(mesh->NX_PER_MPI-1) + mesh->DX << " m\n";
		cout << "Size of simulation domain along the y-axis: " << mesh->nodes.Y(mesh->NY_PER_MPI-1) + mesh->DY << " m\n";
		cout << "Size of simulation domain along the z-axis: " << mesh->nodes.Z(mesh->NZ_PER_MPI-1) + mesh->DZ << " m\n";
		cout << "* * * * * * * * * * * *  SIMULATION GRID LOADED/COMPUTED  * * * * * * * * * * * * * * * * * *\n\n";
	}
}


void INITIALIZE::setupIonsInitialCondition(const simulationParameters * params,const characteristicScales * CS,\
	const meshParams * mesh,vector<ionSpecies> * IONS){

	if(params->mpi.rank_cart == 0){
		cout << "* * * * * * * * * * * * SETTING UP IONS INITIAL CONDITION * * * * * * * * * * * * * * * * * *\n";
	}
	int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

	for(int ii=0;ii<totalNumSpecies;ii++){
		if(params->restart){
			if(params->mpi.rank_cart == 0)
				cout << "PRO++ MESSAGE: Restarting simulation, loading ions";
			MPI_Abort(MPI_COMM_WORLD,-1);
		}else{
			switch (IONS->at(ii).IC) {
				case(1):{
						if(params->quietStart){
                            QUIETSTART qs(params,&IONS->at(ii));
							qs.maxwellianVelocityDistribution(params,&IONS->at(ii));

						}else{
                            RANDOMSTART rs(params);
							rs.maxwellianVelocityDistribution(params,&IONS->at(ii));
						}

						break;
						}
				case(2):{
						if(params->quietStart){
                            QUIETSTART qs(params,&IONS->at(ii));
							qs.ringLikeVelocityDistribution(params,&IONS->at(ii));
						}else{
                            RANDOMSTART rs(params);
							rs.ringLikeVelocityDistribution(params,&IONS->at(ii));
						}

						break;
						}
				default:{
                        if(params->quietStart){
                            QUIETSTART qs(params,&IONS->at(ii));
                            qs.maxwellianVelocityDistribution(params,&IONS->at(ii));

                        }else{
                            RANDOMSTART rs(params);
                            rs.maxwellianVelocityDistribution(params,&IONS->at(ii));
                        }
						}
			} // switch
		} // if(params->restart)

		IONS->at(ii).n.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		IONS->at(ii).n_.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		IONS->at(ii).n__.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		IONS->at(ii).n___.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);

		IONS->at(ii).nv.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		IONS->at(ii).nv_.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		IONS->at(ii).nv__.zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);

		// Setting size and value to zero of arrays for ions' variables
		if(params->mpi.rank_cart == 0)
			cout << "Super-particles used to simulate species No " << ii + 1 << ": " << IONS->at(ii).NSP << '\n';

		IONS->at(ii).meshNode.zeros(IONS->at(ii).NSP);
		IONS->at(ii).wxc.zeros(IONS->at(ii).NSP);
		IONS->at(ii).wxl.zeros(IONS->at(ii).NSP);
		IONS->at(ii).wxr.zeros(IONS->at(ii).NSP);

		//Checking the integrity of the initial condition
		if((int)IONS->at(ii).V.n_elem != (int)(3*IONS->at(ii).NSP)){
            MPI_Barrier(MPI_COMM_WORLD);

			cerr << "PRO++ ERROR: in velocity initial condition of species: " << ii + 1 << endl;
			MPI_Abort(MPI_COMM_WORLD,-104);
		 	// The velocity array contains a number of elements that it should not have

		}
		if((int)IONS->at(ii).X.n_elem != (int)(3*IONS->at(ii).NSP)){
            MPI_Barrier(MPI_COMM_WORLD);

			cerr << "PRO++ ERROR: in spatial initial condition of species: " << ii + 1 << endl;
			MPI_Abort(MPI_COMM_WORLD,-105);
			// The position array contains a number of elements that it should not have
		}
		//Checking integrity of the initial condition
	}//Iteration over ion species


	double HX;
	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		HX = params->DrL*INITIALIZE::LarmorRadius;
	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		HX = params->dp*INITIALIZE::ionSkinDepth;
	}

	if(params->quietStart){
        for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
			IONS->at(ii).X.col(0) *= HX*params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;
		}
	}else{
        for(int ii=0;ii<IONS->size();ii++){//Iteration over ion species
            IONS->at(ii).X.col(0) = \
            (HX*params->NX_PER_MPI)*(params->mpi.MPI_DOMAIN_NUMBER + IONS->at(ii).X.col(0));
        }
	}

	double chargeDensityPerCell;

	#ifdef ONED
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->NX_PER_MPI)*chargeDensityPerCell;
	}
	#endif

	#ifdef TWOD
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = (mesh->DX*(double)mesh->NX_PER_MPI*mesh->DY*(double)mesh->NY_PER_MPI)*chargeDensityPerCell;
	}
	#endif

	#ifdef THREED
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		chargeDensityPerCell = \
		IONS->at(ii).Dn*params->ne/IONS->at(ii).NSP;

		IONS->at(ii).NCP = \
		(mesh->DX*(double)mesh->NX_PER_MPI*mesh->DY*(double)mesh->NY_PER_MPI*mesh->DZ*(double)mesh->NZ_PER_MPI)*chargeDensityPerCell;
	}
	#endif

	PIC pic;
	for(int ii=0;ii<totalNumSpecies;ii++){
		pic.assignCell(params, mesh, &IONS->at(ii), 1);
	}

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * * IONS INITIAL CONDITION SET UP * * * * * * * * * * * * * * * * * * *\n";

}


void INITIALIZE::loadIonParameters(simulationParameters * params, vector<ionSpecies> * IONS,  vector<GCSpecies> * GCP){
	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if(params->mpi.rank_cart == 0){
		cout << "* * * * * * * * * * * * LOADING ION PARAMETERS * * * * * * * * * * * * * * * * * *\n";
		cout << "Number of ion species: " << params->numberOfParticleSpecies << endl;
		cout << "Number of tracer species: " << params->numberOfTracerSpecies << endl;
	}


	string name;
	if(params->argc > 2){
		string argv(params->argv[2]);
		name = "inputFiles/ions_properties_" + argv + ".ion";
	}else{
		name = "inputFiles/ions_properties.ion";
	}

	std::map<string,float> parametersMap;
	parametersMap = loadParameters(&name);

	int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

	for(int ii=0;ii<totalNumSpecies;ii++){
		string name;
		ionSpecies ions;
        GCSpecies gcp;
        int SPECIES;
		stringstream ss;

		ss << ii + 1;

        name = "SPECIES" + ss.str();
        SPECIES = (int)parametersMap[name];
        name.clear();

        if (SPECIES == 0 || SPECIES == 1){
            ions.SPECIES = SPECIES;

            name = "NPC" + ss.str();
    		ions.NPC = parametersMap[name];
    		name.clear();

    		name = "IC" + ss.str();
    		ions.IC = (int)parametersMap[name];
    		name.clear();

    		name = "Tper" + ss.str();
    		ions.Tper = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
    		name.clear();

    		name = "Tpar" + ss.str();
    		ions.Tpar = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
    		name.clear();

    		name = "Dn" + ss.str();
    		ions.Dn = parametersMap[name];
    		name.clear();

    		name = "pctSupPartOutput" + ss.str();
    		ions.pctSupPartOutput = parametersMap[name];
    		name.clear();

            name = "Z" + ss.str();
			ions.Z = parametersMap[name]; //parametersMap[name] = Atomic number.
            name.clear();

            ions.Q = F_E*ions.Z;

			name = "M" + ss.str();
			ions.M = F_U*parametersMap[name]; //parametersMap[name] times the atomic mass unit.
            name.clear();

            ions.Wc = ions.Q*params->BGP.Bo/ions.M;

            ions.Wp = sqrt( ions.Dn*params->ne*ions.Q*ions.Q/(F_EPSILON*ions.M) );//Check the definition of the plasma freq for each species!

            ions.VTper = sqrt(2.0*F_KB*ions.Tper/ions.M);
        	ions.VTpar = sqrt(2.0*F_KB*ions.Tpar/ions.M);

            ions.LarmorRadius = ions.VTper/ions.Wc;

            //Definition of the initial total number of superparticles for each species
    		ions.NSP = ceil(ions.NPC*params->NX_PER_MPI);
    		ions.nSupPartOutput = floor( (ions.pctSupPartOutput/100.0)*ions.NSP );

    		IONS->push_back(ions);

            if(params->mpi.rank_cart == 0){
                if (ions.SPECIES == 0){
                    cout << endl << "Species No "  << ii + 1 << " are tracers with the following parameters:" << endl;
                }else{
                    cout << endl << "Species No "  << ii + 1 << " are full-orbit particles with the following parameters:" << endl;
                }
                cout << "Atomic number: " << ions.Z << endl;
                cout << "Mass: " << ions.M << " kg" << endl;
                cout << "Parallel temperature: " << ions.Tpar*F_KB/F_E << " eV" << endl;
                cout << "Perpendicular temperature: " << ions.Tper*F_KB/F_E << " eV" << endl;
                cout << "Cyclotron frequency: " << ions.Wc << " Hz" << endl;
                cout << "Plasma frequency: " << ions.Wp << " Hz" << endl;
                cout << "Parallel thermal velocity: " << ions.VTpar << " m/s" << endl;
                cout << "Perpendicular thermal velocity: " << ions.VTper << " m/s" << endl;
                cout << "Larmor radius: " << ions.LarmorRadius << " m" << endl;
            }
        }else if (SPECIES == -1){
            gcp.SPECIES = SPECIES;

            name = "NPC" + ss.str();
    		gcp.NPC = parametersMap[name];
    		name.clear();

    		name = "IC" + ss.str();
    		gcp.IC = (int)parametersMap[name];
    		name.clear();

    		name = "Tper" + ss.str();
    		gcp.Tper = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
    		name.clear();

    		name = "Tpar" + ss.str();
    		gcp.Tpar = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
    		name.clear();

    		name = "Dn" + ss.str();
    		gcp.Dn = parametersMap[name];
    		name.clear();

    		name = "pctSupPartOutput" + ss.str();
    		gcp.pctSupPartOutput = parametersMap[name];
    		name.clear();

            name = "Z" + ss.str();
			gcp.Z = parametersMap[name]; //parametersMap[name] = Atomic number.
            name.clear();

            gcp.Q = F_E*gcp.Z;

			name = "M" + ss.str();
			gcp.M = F_U*parametersMap[name]; //parametersMap[name] times the atomic mass unit.
            name.clear();

            ions.Wc = ions.Q*params->BGP.Bo/ions.M;

            ions.VTper = sqrt(2.0*F_KB*ions.Tper/ions.M);
        	ions.VTpar = sqrt(2.0*F_KB*ions.Tpar/ions.M);

            ions.LarmorRadius = ions.VTper/ions.Wc;

            //Definition of the initial total number of superparticles for each species
    		gcp.NSP = ceil(gcp.NPC*params->NX_PER_MPI);
    		gcp.nSupPartOutput = floor( (gcp.pctSupPartOutput/100.0)*gcp.NSP );

    		GCP->push_back(gcp);

            if(params->mpi.rank_cart == 0){
                cout << endl << "Species No "  << ii + 1 << " are guiding-center particles with the following parameters:" << endl;
                cout << "Atomic number: " << gcp.Z << endl;
                cout << "Mass: " << gcp.M << " kg" << endl;
                cout << "Parallel temperature: " << gcp.Tpar*F_KB/F_E << " eV" << endl;
                cout << "Perpendicular temperature: " << gcp.Tper*F_KB/F_E << " eV" << endl;
                cout << "Cyclotron frequency: " << gcp.Wc << " Hz" << endl;
                cout << "Parallel thermal velocity: " << gcp.VTpar << " m/s" << endl;
                cout << "Perpendicular thermal velocity: " << gcp.VTper << " m/s" << endl;
                cout << "Larmor radius: " << gcp.LarmorRadius << " m" << endl;
            }
        }else{
            MPI_Barrier(MPI_COMM_WORLD);

            if(params->mpi.MPI_DOMAIN_NUMBER == 0){
    			cerr << "PRO++ ERROR: Enter a valid type of species -- options are 0 = tracers, 1 = full orbit, -1 = guiding center" << endl;
    		}
    		MPI_Abort(MPI_COMM_WORLD,-106);
        }

	}//Iteration over ion species

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * ION PARAMETERS LOADED * * * * * * * * * * * * * * * * * *\n";

	MPI_Barrier(MPI_COMM_WORLD);
}


void INITIALIZE::initializeFields(const simulationParameters * params, const meshParams * mesh, fields * EB){

	stringstream domainNumber;
	domainNumber << params->mpi.MPI_DOMAIN_NUMBER;

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * INITIALIZING ELECTROMAGNETIC FIELDS * * * * * * * * * * * * * * * * * *\n";

	if(params->loadFields==1){//The electromagnetic fields are loaded from external files.
		if(params->mpi.rank_cart == 0)
			cout << "Loading external electromagnetic fields\n";
		MPI_Abort(MPI_COMM_WORLD,-107);
	}else{//The electromagnetic fields are being initialized in the runtime.
		int NX(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);
		EB->zeros(NX);//We include the ghost mesh points (+2) in the initialization

		EB->B.X.fill(params->BGP.Bx);//x
		EB->B.Y.fill(params->BGP.By);//y
		EB->B.Z.fill(params->BGP.Bz);//z

		EB->_B.X.subvec(1,NX-2) = sqrt( EB->B.X.subvec(1,NX-2) % EB->B.X.subvec(1,NX-2) \
						+ 0.25*( ( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) ) % ( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) ) ) \
						+ 0.25*( ( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) ) % ( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) ) ) );

		EB->_B.Y.subvec(1,NX-2) = sqrt( 0.25*( ( EB->B.X.subvec(1,NX-2) + EB->B.X.subvec(0,NX-3) ) % ( EB->B.X.subvec(1,NX-2) + EB->B.X.subvec(0,NX-3) ) ) \
						+ EB->B.Y.subvec(1,NX-2) % EB->B.Y.subvec(1,NX-2) + EB->B.Z.subvec(1,NX-2) % EB->B.Z.subvec(1,NX-2) );

		EB->_B.Z.subvec(1,NX-2) = sqrt( 0.25*( ( EB->B.X.subvec(1,NX-2) + EB->B.X.subvec(0,NX-3) ) % ( EB->B.X.subvec(1,NX-2) + EB->B.X.subvec(0,NX-3) ) ) \
						+ EB->B.Y.subvec(1,NX-2) % EB->B.Y.subvec(1,NX-2) + EB->B.Z.subvec(1,NX-2) % EB->B.Z.subvec(1,NX-2) );

		EB->b = EB->B/EB->_B;

		EB->b_ = EB->b;

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
