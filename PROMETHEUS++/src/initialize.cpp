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

template <class IT, class FT> vector<string> INITIALIZE<IT,FT>::split(const string& str, const string& delim)
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


template <class IT, class FT> map<string,float> INITIALIZE<IT,FT>::loadParameters(string * inputFile){
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


template <class IT, class FT> map<string,string>INITIALIZE<IT,FT>::loadParametersString(string * inputFile){
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

template <class IT, class FT> INITIALIZE<IT,FT>::INITIALIZE(simulationParameters * params, int argc, char* argv[]){
    MPI_Comm_size(MPI_COMM_WORLD, &params->mpi.NUMBER_MPI_DOMAINS);
	MPI_Comm_rank(MPI_COMM_WORLD, &params->mpi.MPI_DOMAIN_NUMBER);

    // Error codes
    params->errorCodes[-100] = "Odd number of MPI processes";
    params->errorCodes[-101] = "Input file could not be opened";
    params->errorCodes[-102] = "MPI's Cartesian topology could not be created";
    params->errorCodes[-103] = "Grid size violates assumptions of hybrid model for the plasma -- DX smaller than the electron skind depth can not be resolved";
    params->errorCodes[-104] = "Loading external electromagnetic fields not implemented yet";
    params->errorCodes[-105] = "Restart not implemented yet";
    params->errorCodes[-106] = "Inconsistency in iniital ion's velocity distribution function";
    params->errorCodes[-107] = "Inconsistency in iniital ion's spatial distribution function";
    params->errorCodes[-108] = "Non-finite value in meshNode";
    params->errorCodes[-109] = "Number of nodes in either direction of simulation domain need to be a multiple of 2";
    params->errorCodes[-110] = "Non finite values in Ex";
    params->errorCodes[-111] = "Non finite values in Ey";
    params->errorCodes[-112] = "Non finite values in Ez";
    params->errorCodes[-113] = "Non finite values in Bx";
    params->errorCodes[-114] = "Non finite values in By";
    params->errorCodes[-115] = "Non finite values in Bz";


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

    // Arguments and paths to main function
    params->PATH = argv[2];

	params->argc = argc;
	params->argv = argv;


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
        cout << "STARTING " << params->argv[1] << " SIMULATION ON: " << std::ctime(&current_time) << endl;
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
    }

	string name;
	if(params->argc > 3){
		string argv(params->argv[3]);
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
	}

    params->dimensionality = std::stoi( parametersStringMap["dimensionality"] );

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

	params->BC = std::stoi( parametersStringMap["BC"] );

	params->smoothingParameter = std::stod( parametersStringMap["smoothingParameter"] );

	params->numberOfRKIterations = std::stoi( parametersStringMap["numberOfRKIterations"] );

	params->filtersPerIterationFields = std::stoi( parametersStringMap["filtersPerIterationFields"] );

	params->filtersPerIterationIons = std::stoi( parametersStringMap["filtersPerIterationIons"] );

	params->simulationTime = std::stod( parametersStringMap["simulationTime"] );

	params->numberOfParticleSpecies = std::stoi( parametersStringMap["numberOfParticleSpecies"] );

	params->numberOfTracerSpecies = std::stoi( parametersStringMap["numberOfTracerSpecies"] );

	params->BGP.ne = std::stod( parametersStringMap["ne"] );

	params->loadFields = std::stoi( parametersStringMap["loadFields"] );

	params->outputCadence = std::stod( parametersStringMap["outputCadence"] );

    // Number of nodes in entire simulation domain
    unsigned int NX = (unsigned int)std::stoi( parametersStringMap["NX"] );
    unsigned int NY = (unsigned int)std::stoi( parametersStringMap["NY"] );
    unsigned int NZ = (unsigned int)std::stoi( parametersStringMap["NZ"] );

    // Sanity check: if NX and/or NY is not a multiple of 2, the simulation aborts
    if (params->dimensionality == 1){
        if (fmod(NX, 2.0) > 0.0){
            MPI_Barrier(params->mpi.MPI_TOPO);
            MPI_Abort(params->mpi.MPI_TOPO,-109);
        }

        params->mesh.NX_IN_SIM = NX;
        params->mesh.NX_PER_MPI = (int)( (double)NX/(double)params->mpi.NUMBER_MPI_DOMAINS );
        params->mesh.SPLIT_DIRECTION = 0;

        params->mesh.NY_IN_SIM = 1;
        params->mesh.NY_PER_MPI = 1;

        params->mesh.NZ_IN_SIM = 1;
        params->mesh.NZ_PER_MPI = 1;

        params->mesh.NUM_NODES_PER_MPI = params->mesh.NX_PER_MPI;
        params->mesh.NUM_NODES_IN_SIM = params->mesh.NX_IN_SIM;
    }else{
        if ((fmod(NX, 2.0) > 0.0) || (fmod(NY, 2.0) > 0.0)){
            MPI_Barrier(params->mpi.MPI_TOPO);
            MPI_Abort(params->mpi.MPI_TOPO,-109);
        }

        if (NX >= NY){
            params->mesh.NX_IN_SIM = NX;
            params->mesh.NX_PER_MPI = (int)( (double)NX/(double)params->mpi.NUMBER_MPI_DOMAINS );

            params->mesh.NY_IN_SIM = NY;
            params->mesh.NY_PER_MPI = NY;

            params->mesh.SPLIT_DIRECTION = 0;
        }else{
            params->mesh.NX_IN_SIM = NX;
            params->mesh.NX_PER_MPI = NX;

            params->mesh.NY_IN_SIM = NY;
            params->mesh.NY_PER_MPI = (int)( (double)NY/(double)params->mpi.NUMBER_MPI_DOMAINS );

            params->mesh.SPLIT_DIRECTION = 1;
        }

        params->mesh.NZ_IN_SIM = 1;
        params->mesh.NZ_PER_MPI = 1;

        params->mesh.NUM_NODES_PER_MPI = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI;
        params->mesh.NUM_NODES_IN_SIM = params->mesh.NX_IN_SIM*params->mesh.NY_IN_SIM;
    }


	params->DrL = std::stod( parametersStringMap["DrL"] );

	params->dp = std::stod( parametersStringMap["dp"] );

	params->BGP.Te = std::stod( parametersStringMap["Te"] )*F_E/F_KB; // Te in eV in input file

	params->BGP.theta = std::stod( parametersStringMap["theta"] );
	params->BGP.phi = std::stod( parametersStringMap["phi"] );

	params->BGP.Bo = std::stod( parametersStringMap["Bo"] );

	params->BGP.Bx = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0);
	params->BGP.By = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0);
	params->BGP.Bz = params->BGP.Bo*cos(params->BGP.theta*M_PI/180.0);

	// Parsing list of variables in outputs
	std::string nonparsed_variables_list = parametersStringMap["outputs_variables"].substr(1, parametersStringMap["outputs_variables"].length() - 2);
	params->outputs_variables = INITIALIZE::split(nonparsed_variables_list,",");
}


template <class IT, class FT> void INITIALIZE<IT,FT>::loadMeshGeometry(simulationParameters * params, fundamentalScales * FS){

    MPI_Barrier(params->mpi.MPI_TOPO);

	if (params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << endl << "* * * * * * * * * * * * LOADING/COMPUTING SIMULATION GRID * * * * * * * * * * * * * * * * * *\n";

	if( (params->DrL > 0.0) && (params->dp < 0.0) ){
		params->mesh.DX = params->DrL*params->ionLarmorRadius;
		params->mesh.DY = params->mesh.DX;
		params->mesh.DZ = params->mesh.DX;
		if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
			cout << "Using LARMOR RADIUS to set up simulation grid." << endl;

	}else if( (params->DrL < 0.0) && (params->dp > 0.0) ){
		params->mesh.DX = params->dp*params->ionSkinDepth;
		params->mesh.DY = params->mesh.DX;
		params->mesh.DZ = params->mesh.DX;
		if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
			cout << "Using ION SKIN DEPTH to set up simulation grid." << endl;
	}

	params->mesh.nodes.X.set_size(params->mesh.NX_IN_SIM);
	params->mesh.nodes.Y.set_size(params->mesh.NY_IN_SIM);
	params->mesh.nodes.Z.set_size(params->mesh.NZ_IN_SIM);

    //cout << params->mpi.MPI_DOMAIN_NUMBER_CART << " | NX:" << params->mpi.MPI_DOMAINS_ALONG_X_AXIS << " | NY:" << params->mpi.MPI_DOMAINS_ALONG_Y_AXIS << " | NZ:" << params->mpi.MPI_DOMAINS_ALONG_Z_AXIS << endl;

	for(int ii=0; ii<params->mesh.NX_IN_SIM; ii++){
		params->mesh.nodes.X(ii) = (double)ii*params->mesh.DX; //entire simulation domain's mesh grid
	}

    for(int ii=0; ii<params->mesh.NY_IN_SIM; ii++){
		params->mesh.nodes.Y(ii) = (double)ii*params->mesh.DY; //
	}

	for(int ii=0; ii<params->mesh.NZ_IN_SIM; ii++){
		params->mesh.nodes.Z(ii) = (double)ii*params->mesh.DZ; //
	}

    params->mesh.LX = params->mesh.DX*params->mesh.NX_IN_SIM;
    params->mesh.LY = params->mesh.DY*params->mesh.NY_IN_SIM;
    params->mesh.LZ = params->mesh.DZ*params->mesh.NZ_IN_SIM;

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
        cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
        cout << "+ Number of mesh nodes along y-axis: " << params->mesh.NY_IN_SIM << endl;
        cout << "+ Number of mesh nodes along z-axis: " << params->mesh.NZ_IN_SIM << endl;

		cout << "+ Size of simulation domain along the x-axis: " << params->mesh.LX << " m" << endl;
		cout << "+ Size of simulation domain along the y-axis: " << params->mesh.LY << " m" << endl;
		cout << "+ Size of simulation domain along the z-axis: " << params->mesh.LZ << " m" << endl;
		cout << "* * * * * * * * * * * *  SIMULATION GRID LOADED/COMPUTED  * * * * * * * * * * * * * * * * * *" << endl;
	}
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeIonsArrays(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
    IONS->n.zeros(params->mesh.NX_IN_SIM + 2);       // Ghost cells are included (+2)
    IONS->n_.zeros(params->mesh.NX_IN_SIM + 2);      // Ghost cells are included (+2)
    IONS->n__.zeros(params->mesh.NX_IN_SIM + 2);     // Ghost cells are included (+2)
    IONS->n___.zeros(params->mesh.NX_IN_SIM + 2);    // Ghost cells are included (+2)

    IONS->nv.zeros(params->mesh.NX_IN_SIM + 2);      // Ghost cells are included (+2)
    IONS->nv_.zeros(params->mesh.NX_IN_SIM + 2);     // Ghost cells are included (+2)
    IONS->nv__.zeros(params->mesh.NX_IN_SIM + 2);    // Ghost cells are included (+2)

    // Setting size and value to zero of arrays for ions' variables
    IONS->mn.zeros(IONS->NSP);

    IONS->E.zeros(IONS->NSP,3);
    IONS->B.zeros(IONS->NSP,3);

    IONS->wxc.zeros(IONS->NSP);
    IONS->wxl.zeros(IONS->NSP);
    IONS->wxr.zeros(IONS->NSP);

    IONS->wxc_.zeros(IONS->NSP);
    IONS->wxl_.zeros(IONS->NSP);
    IONS->wxr_.zeros(IONS->NSP);

    //Checking the integrity of the initial condition
    if((int)IONS->V.n_elem != (int)(3*IONS->NSP)){
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-106);
        // The velocity array contains a number of elements that it should not have
    }

    if((int)IONS->X.n_elem != (int)(3*IONS->NSP)){
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-107);
        // The position array contains a number of elements that it should not have
    }
    //Checking integrity of the initial condition

    if(params->quietStart){
        IONS->X.col(0) *= params->mesh.DX*params->mesh.NX_IN_SIM;
    }else{
        IONS->X.col(0) = (params->mesh.DX*params->mesh.NX_PER_MPI)*(params->mpi.MPI_DOMAIN_NUMBER + IONS->X.col(0));
    }

    IONS->NCP = (params->BGP.ne*params->mesh.LX)/(IONS->NSP*params->mpi.NUMBER_MPI_DOMAINS);

    PIC ionsDynamics;
    ionsDynamics.assignCell(params, IONS);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeIonsArrays(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
    IONS->n.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);       // Ghost cells are included (+2)
    IONS->n_.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);      // Ghost cells are included (+2)
    IONS->n__.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);     // Ghost cells are included (+2)
    IONS->n___.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);    // Ghost cells are included (+2)

    IONS->nv.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);      // Ghost cells are included (+2)
    IONS->nv_.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);     // Ghost cells are included (+2)
    IONS->nv__.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);    // Ghost cells are included (+2)

    // Setting size and value to zero of arrays for ions' variables
    IONS->mn.zeros(IONS->NSP, 2);

    IONS->E.zeros(IONS->NSP,3);
    IONS->B.zeros(IONS->NSP,3);

    IONS->wxc.zeros(IONS->NSP);
    IONS->wxl.zeros(IONS->NSP);
    IONS->wxr.zeros(IONS->NSP);

    IONS->wyc.zeros(IONS->NSP);
    IONS->wyl.zeros(IONS->NSP);
    IONS->wyr.zeros(IONS->NSP);

    //Checking the integrity of the initial condition
    if((int)IONS->V.n_elem != (int)(3*IONS->NSP)){
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-106);
        // The velocity array contains a number of elements that it should not have
    }

    if((int)IONS->X.n_elem != (int)(3*IONS->NSP)){
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-107);
        // The position array contains a number of elements that it should not have
    }
    //Checking integrity of the initial condition


    if(params->quietStart){
        IONS->X.col(0) *= params->mesh.DX*params->mesh.NX_IN_SIM;
        IONS->X.col(1) *= params->mesh.DY*params->mesh.NY_IN_SIM;
    }else{
        IONS->X.col(0) = (params->mesh.DX*params->mesh.NX_PER_MPI)*(params->mpi.MPI_CART_COORDS_2D[0] + IONS->X.col(0)); //*** @tomodify
        IONS->X.col(1) = (params->mesh.DY*params->mesh.NY_PER_MPI)*(params->mpi.MPI_CART_COORDS_2D[1] + IONS->X.col(1)); //*** @tomodify
    }

    IONS->NCP = (params->BGP.ne*params->mesh.LX*params->mesh.LY)/(IONS->NSP*params->mpi.NUMBER_MPI_DOMAINS);

    PIC ionsDynamics;
    ionsDynamics.assignCell(params, IONS);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::setupIonsInitialCondition(const simulationParameters * params, const characteristicScales * CS, vector<IT> * IONS){

    int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

    if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << endl << "* * * * * * * * * * * * SETTING UP IONS INITIAL CONDITION * * * * * * * * * * * * * * * * * *" << endl;

	for(int ii=0; ii<totalNumSpecies; ii++){
		if(params->restart){
			if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
				cout << "Restart not implemented yet" << endl;

			MPI_Abort(MPI_COMM_WORLD,-105);
		}else{
			switch (IONS->at(ii).IC) {
				case(1):{
						if(params->quietStart){
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
							qs.maxwellianVelocityDistribution(params, &IONS->at(ii));
						}else{
                            RANDOMSTART<IT> rs(params);
							rs.maxwellianVelocityDistribution(params,&IONS->at(ii));
						}

						break;
						}
				case(2):{
						if(params->quietStart){
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
							qs.ringLikeVelocityDistribution(params, &IONS->at(ii));
						}else{
                            RANDOMSTART<IT> rs(params);
							rs.ringLikeVelocityDistribution(params, &IONS->at(ii));
						}

						break;
						}
				default:{
                        if(params->quietStart){
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
                            qs.maxwellianVelocityDistribution(params, &IONS->at(ii));

                        }else{
                            RANDOMSTART<IT> rs(params);
                            rs.maxwellianVelocityDistribution(params, &IONS->at(ii));
                        }
						}
			} // switch
		} // if(params->restart)

        if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
			cout << "Super-particles used to simulate species No " << ii + 1 << ": " << IONS->at(ii).NSP*params->mpi.NUMBER_MPI_DOMAINS << '\n';

        initializeIonsArrays(params, &IONS->at(ii));
    }//Iteration over ion species

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "* * * * * * * * * * * * * IONS INITIAL CONDITION SET UP * * * * * * * * * * * * * * * * * * *\n";

}


template <class IT, class FT> void INITIALIZE<IT,FT>::loadIonParameters(simulationParameters * params, vector<IT> * IONS){

    MPI_Barrier(params->mpi.MPI_TOPO);

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
		cout << "* * * * * * * * * * * * LOADING ION PARAMETERS * * * * * * * * * * * * * * * * * *\n";
		cout << "+ Number of ion species: " << params->numberOfParticleSpecies << endl;
		cout << "+ Number of tracer species: " << params->numberOfTracerSpecies << endl;
	}


	string name;
	if(params->argc > 3){
		string argv(params->argv[3]);
		name = "inputFiles/ions_properties_" + argv + ".ion";
	}else{
		name = "inputFiles/ions_properties.ion";
	}

	std::map<string,float> parametersMap;
	parametersMap = loadParameters(&name);

	int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

	for(int ii=0;ii<totalNumSpecies;ii++){
		string name;
		IT ions;
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

    		name = "densityFraction" + ss.str();
    		ions.densityFraction = parametersMap[name];
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

            ions.Wp = sqrt( ions.densityFraction*params->BGP.ne*ions.Q*ions.Q/(F_EPSILON*ions.M) );//Check the definition of the plasma freq for each species!

            ions.VTper = sqrt(2.0*F_KB*ions.Tper/ions.M);
        	ions.VTpar = sqrt(2.0*F_KB*ions.Tpar/ions.M);

            ions.LarmorRadius = ions.VTper/ions.Wc;

            //Definition of the initial total number of superparticles for each species
            if (params->dimensionality == 1)
    		    ions.NSP = ceil(ions.NPC*params->mesh.NX_PER_MPI);
            else
                ions.NSP = ceil(ions.NPC*params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI);

    		ions.nSupPartOutput = floor( (ions.pctSupPartOutput/100.0)*ions.NSP );

    		IONS->push_back(ions);

            if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
                if (ions.SPECIES == 0){
                    cout << endl << "Species No "  << ii + 1 << " are tracers with the following parameters:" << endl;
                }else{
                    cout << endl << "Species No "  << ii + 1 << " are full-orbit particles with the following parameters:" << endl;
                }
                cout << "+ Atomic number: " << ions.Z << endl;
                cout << "+ Mass: " << ions.M << " kg" << endl;
                cout << "+ Parallel temperature: " << ions.Tpar*F_KB/F_E << " eV" << endl;
                cout << "+ Perpendicular temperature: " << ions.Tper*F_KB/F_E << " eV" << endl;
                cout << "+ Cyclotron frequency: " << ions.Wc << " Hz" << endl;
                cout << "+ Plasma frequency: " << ions.Wp << " Hz" << endl;
                cout << "+ Parallel thermal velocity: " << ions.VTpar << " m/s" << endl;
                cout << "+ Perpendicular thermal velocity: " << ions.VTper << " m/s" << endl;
                cout << "+ Larmor radius: " << ions.LarmorRadius << " m" << endl;
            }
        }else{
            MPI_Barrier(MPI_COMM_WORLD);

            if(params->mpi.MPI_DOMAIN_NUMBER == 0){
    			cerr << "PRO++ ERROR: Enter a valid type of species -- options are 0 = tracers, 1 = full orbit, -1 = guiding center" << endl;
    		}
    		MPI_Abort(MPI_COMM_WORLD,-106);
        }

	}//Iteration over ion species

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "* * * * * * * * * * * * ION PARAMETERS LOADED * * * * * * * * * * * * * * * * * *\n";

	MPI_Barrier(MPI_COMM_WORLD);
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(const simulationParameters * params, oneDimensional::fields * EB){
    int NX(params->mesh.NX_IN_SIM + 2); // Ghost mesh points (+2) included

    EB->zeros(NX);

    EB->B.X.fill(params->BGP.Bx); // x
    EB->B.Y.fill(params->BGP.By); // y
    EB->B.Z.fill(params->BGP.Bz); // z
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(const simulationParameters * params, twoDimensional::fields * EB){
    int NX(params->mesh.NX_IN_SIM + 2); // Ghost mesh points (+2) included
    int NY(params->mesh.NY_IN_SIM + 2); // Ghost mesh points (+2) included

    EB->zeros(NX,NY);

    EB->B.X.fill(params->BGP.Bx); // x
    EB->B.Y.fill(params->BGP.By); // y
    EB->B.Z.fill(params->BGP.Bz); // z

    //*** @todelete
    /*
    arma::vec xAxis = params->mesh.nodes.X + 0.5*params->mesh.DX;
    for(int ii=1; ii<(NY-1); ii++){
        EB->B.Z.col(ii).subvec(1,NX-2) += 0.5*params->BGP.Bz*cos(2.0*M_PI*xAxis/params->mesh.LX);
    }
    */

    //*** @todelete
    /*
    arma::vec xAxis = params->mesh.nodes.X;
    for(int ii=1; ii<(NY-1); ii++){
        EB->B.X.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
        EB->E.Y.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
        EB->E.Z.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
    }

    xAxis = params->mesh.nodes.X + 0.5*params->mesh.DX;
    for(int ii=1; ii<(NY-1); ii++){
        EB->E.X.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
        EB->B.Y.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
        EB->B.Z.col(ii).subvec(1,NX-2) = cos(2.0*M_PI*xAxis/params->mesh.LX);
    }
    */

    //*** @todelete
    /*
    arma::rowvec yAxis = conv_to<rowvec>::from( params->mesh.nodes.Y + 0.5*params->mesh.DY );
    for(int ii=1; ii<(NX-1); ii++){
        EB->B.X.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
        EB->B.Z.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
        EB->E.Y.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
    }

    yAxis = conv_to<rowvec>::from( params->mesh.nodes.Y );
    for(int ii=1; ii<(NX-1); ii++){
        EB->E.X.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
        EB->E.Z.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
        EB->B.Y.row(ii).subvec(1,NY-2) = cos(2.0*M_PI*yAxis/params->mesh.LY);
    }
    */
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFields(const simulationParameters * params, FT * EB){

    MPI_Barrier(params->mpi.MPI_TOPO);

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << endl << "* * * * * * * * * * * * INITIALIZING ELECTROMAGNETIC FIELDS * * * * * * * * * * * * * * * * * *" << endl;

	if(params->loadFields==1){//The electromagnetic fields are loaded from external files.
        if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
			cout << "Loading external electromagnetic fields..." << endl;

		MPI_Abort(params->mpi.MPI_TOPO,-104);
	}else{//The electromagnetic fields are being initialized in the runtime.
        initializeFieldsSizeAndValue(params, EB);

		if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
			cout << "Initializing electromagnetic fields within simulation" << endl;
			cout << "+ Magnetic field along x-axis: " << scientific << params->BGP.Bx << fixed << " T" << endl;
			cout << "+ Magnetic field along y-axis: " << scientific << params->BGP.By << fixed << " T" << endl;
			cout << "+ Magnetic field along z-axis: " << scientific << params->BGP.Bz << fixed << " T" << endl;
		}
	}

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "* * * * * * * * * * * * ELECTROMAGNETIC FIELDS INITIALIZED  * * * * * * * * * * * * * * * * * *" << endl;
}


template class INITIALIZE<oneDimensional::ionSpecies, oneDimensional::fields>;
template class INITIALIZE<twoDimensional::ionSpecies, twoDimensional::fields>;
