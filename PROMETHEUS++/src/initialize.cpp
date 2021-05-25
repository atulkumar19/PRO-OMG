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

template <class IT, class FT> INITIALIZE<IT,FT>::INITIALIZE(simulationParameters * params, int argc, char* argv[])
{
    // Get RANK and SIZE of nodes within COMM_WORLD:
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


	if( fmod( (double)params->mpi.NUMBER_MPI_DOMAINS, 2.0 ) > 0.0 )
    {
        MPI_Barrier(MPI_COMM_WORLD);

		if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
			cerr << "PRO++ ERROR: The number of MPI processes must be an even number." << endl;
		}

		MPI_Abort(MPI_COMM_WORLD,-100);
	}

    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        time_t current_time = std::time(NULL);
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
        cout << "STARTING " << params->argv[1] << " SIMULATION ON: " << std::ctime(&current_time) << endl;
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
    }

	string name;
	if(params->argc > 3)
    {
		string argv(params->argv[3]);
		name = "inputFiles/input_file_" + argv + ".input";
		params->PATH += "/" + argv;
	}
    else
    {
		name = "inputFiles/input_file.input";
		params->PATH += "/";
	}

	std::map<string,string> parametersStringMap;
	parametersStringMap = loadParametersString(&name);

	// Create HDF5 folders if they don't exist
	if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
		string mkdir_outputs_dir = "mkdir " + params->PATH;
		const char * sys = mkdir_outputs_dir.c_str();
		int rsys = system(sys);

		string mkdir_outputs_dir_HDF5 = mkdir_outputs_dir + "/HDF5";
		sys = mkdir_outputs_dir_HDF5.c_str();
		rsys = system(sys);
	}

    params->dimensionality = std::stoi( parametersStringMap["dimensionality"] );

    params->mpi.MPIS_FIELDS = std::stoi( parametersStringMap["mpisForFields"] );
    params->mpi.MPIS_PARTICLES = params->mpi.NUMBER_MPI_DOMAINS - params->mpi.MPIS_FIELDS;

    if(std::stoi( parametersStringMap["includeElectronInertia"] ) == 1)
    {
        params->includeElectronInertia = true;
    }
    else
    {
        params->includeElectronInertia = false;
    }

    if(std::stoi( parametersStringMap["quietStart"] ) == 1)
    {
        params->quietStart = true;
    }
    else
    {
        params->quietStart = false;
    }

	params->DTc = std::stod( parametersStringMap["DTc"] );

    if(std::stoi( parametersStringMap["restart"] ) == 1)
    {
        params->restart = true;
    }
    else
    {
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

	params->BGP.ne    = std::stod( parametersStringMap["ne"] );

    params->PP.nTable = std::stod( parametersStringMap["nTable"] );

    params->BGP.Rphi0 = std::stod( parametersStringMap["Rphi0"] );

	params->loadFields = std::stoi( parametersStringMap["loadFields"] );

	params->outputCadence = std::stod( parametersStringMap["outputCadence"] );

    // Number of nodes in entire simulation domain
    unsigned int NX = (unsigned int)std::stoi( parametersStringMap["NX"] );
    unsigned int NY = (unsigned int)std::stoi( parametersStringMap["NY"] );
    unsigned int NZ = (unsigned int)std::stoi( parametersStringMap["NZ"] );

    // Sanity check: if NX and/or NY is not a multiple of 2, the simulation aborts
    if (params->dimensionality == 1)
    {
        if (fmod(NX, 2.0) > 0.0)
        {
            MPI_Barrier(params->mpi.MPI_TOPO);
            MPI_Abort(params->mpi.MPI_TOPO,-109);
        }

        params->mesh.NX_IN_SIM = NX;
        params->mesh.NX_PER_MPI = (int)( (double)NX/(double)params->mpi.MPIS_FIELDS );
        params->mesh.SPLIT_DIRECTION = 0;

        params->mesh.NY_IN_SIM = 1;
        params->mesh.NY_PER_MPI = 1;

        params->mesh.NZ_IN_SIM = 1;
        params->mesh.NZ_PER_MPI = 1;

        params->mesh.NUM_CELLS_PER_MPI = params->mesh.NX_PER_MPI;
        params->mesh.NUM_CELLS_IN_SIM = params->mesh.NX_IN_SIM;
    }
    else
    {
        if ((fmod(NX, 2.0) > 0.0) || (fmod(NY, 2.0) > 0.0))
        {
            MPI_Barrier(params->mpi.MPI_TOPO);
            MPI_Abort(params->mpi.MPI_TOPO,-109);
        }

        if (NX >= NY)
        {
            params->mesh.NX_IN_SIM = NX;
            params->mesh.NX_PER_MPI = (int)( (double)NX/(double)params->mpi.MPIS_FIELDS );

            params->mesh.NY_IN_SIM = NY;
            params->mesh.NY_PER_MPI = NY;

            params->mesh.SPLIT_DIRECTION = 0;
        }
        else
        {
            params->mesh.NX_IN_SIM = NX;
            params->mesh.NX_PER_MPI = NX;

            params->mesh.NY_IN_SIM = NY;
            params->mesh.NY_PER_MPI = (int)( (double)NY/(double)params->mpi.MPIS_FIELDS );

            params->mesh.SPLIT_DIRECTION = 1;
        }

        params->mesh.NZ_IN_SIM = 1;
        params->mesh.NZ_PER_MPI = 1;

        params->mesh.NUM_CELLS_PER_MPI = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI;
        params->mesh.NUM_CELLS_IN_SIM = params->mesh.NX_IN_SIM*params->mesh.NY_IN_SIM;
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

    params->BGP.Bx = (abs(params->BGP.Bx) < PRO_ZERO) ? 0.0 : params->BGP.Bx;
    params->BGP.By = (abs(params->BGP.By) < PRO_ZERO) ? 0.0 : params->BGP.By;
    params->BGP.Bz = (abs(params->BGP.Bz) < PRO_ZERO) ? 0.0 : params->BGP.Bz;

	// Parsing list of variables in outputs
	std::string nonparsed_variables_list = parametersStringMap["outputs_variables"].substr(1, parametersStringMap["outputs_variables"].length() - 2);
	params->outputs_variables = INITIALIZE::split(nonparsed_variables_list,",");

    // loading and rescaling plasma profiles obtained from external files
}


template <class IT, class FT> void INITIALIZE<IT,FT>::loadMeshGeometry(simulationParameters * params, fundamentalScales * FS){

    MPI_Barrier(MPI_COMM_WORLD);

    // Print to terminal:
    // ==================
    if (params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << endl << "* * * * * * * * * * * * LOADING/COMPUTING SIMULATION GRID * * * * * * * * * * * * * * * * * *\n";
    }

    // Select grid size: based on Larmour radius or ion skin depth:
    // ============================================================
    if( (params->DrL > 0.0) && (params->dp < 0.0) )
    {
	params->mesh.DX = params->DrL*params->ionLarmorRadius;
	params->mesh.DY = params->mesh.DX;
	params->mesh.DZ = params->mesh.DX;

	if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << "Using LARMOR RADIUS to set up simulation grid." << endl;
        }
    }
    else if( (params->DrL < 0.0) && (params->dp > 0.0) )
    {
        params->mesh.DX = params->dp*params->ionSkinDepth;
        params->mesh.DY = params->mesh.DX;
        params->mesh.DZ = params->mesh.DX;

        if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << "Using ION SKIN DEPTH to set up simulation grid." << endl;
        }
    }

    // Set size of mesh and allocate memory:
    // =====================================
    params->mesh.nodes.X.set_size(params->mesh.NX_IN_SIM);
    params->mesh.nodes.Y.set_size(params->mesh.NY_IN_SIM);
    params->mesh.nodes.Z.set_size(params->mesh.NZ_IN_SIM);

    // Create mesh nodes: X domain
    // ============================
    for(int ii=0; ii<params->mesh.NX_IN_SIM; ii++)
    {
        params->mesh.nodes.X(ii) = (double)ii*params->mesh.DX;
    }

    // Create mesh nodes: Y domain
    // ===========================
    for(int ii=0; ii<params->mesh.NY_IN_SIM; ii++)
    {
        params->mesh.nodes.Y(ii) = (double)ii*params->mesh.DY;
    }

    // Create mesh nodes: Z domain
    // ===========================
    for(int ii=0; ii<params->mesh.NZ_IN_SIM; ii++)
    {
        params->mesh.nodes.Z(ii) = (double)ii*params->mesh.DZ;
    }

    // Define total length of each mesh:
    // =================================
    params->mesh.LX = params->mesh.DX*params->mesh.NX_IN_SIM;
    params->mesh.LY = params->mesh.DY*params->mesh.NY_IN_SIM;
    params->mesh.LZ = params->mesh.DZ*params->mesh.NZ_IN_SIM;

    // Print to terminal:
    // ==================
    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
        cout << "+ Number of mesh nodes along y-axis: " << params->mesh.NY_IN_SIM << endl;
        cout << "+ Number of mesh nodes along z-axis: " << params->mesh.NZ_IN_SIM << endl;

    	cout << "+ Size of simulation domain along the x-axis: " << params->mesh.LX << " m" << endl;
    	cout << "+ Size of simulation domain along the y-axis: " << params->mesh.LY << " m" << endl;
    	cout << "+ Size of simulation domain along the z-axis: " << params->mesh.LZ << " m" << endl;
    	cout << "* * * * * * * * * * * *  SIMULATION GRID LOADED/COMPUTED  * * * * * * * * * * * * * * * * * *" << endl;
    }
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeParticlesArrays(const simulationParameters * params, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
    // Set size and value to zero of arrays for ions' variables:
    // ========================================================
    IONS->mn.zeros(IONS->NSP);

    IONS->E.zeros(IONS->NSP,3);
    IONS->B.zeros(IONS->NSP,3);

    IONS->wxc.zeros(IONS->NSP);
    IONS->wxl.zeros(IONS->NSP);
    IONS->wxr.zeros(IONS->NSP);

    IONS->wxc_.zeros(IONS->NSP);
    IONS->wxl_.zeros(IONS->NSP);
    IONS->wxr_.zeros(IONS->NSP);

    // Initialize particle-defined quantities:
    // ==================================
    IONS->n_p.zeros(IONS->NSP);
    IONS->nv_p.zeros(IONS->NSP);
    IONS->Tpar_p.zeros(IONS->NSP);
    IONS->Tper_p.zeros(IONS->NSP);

    // Check integrity of the initial condition:
    // ========================================
    if((int)IONS->V.n_elem != (int)(3*IONS->NSP))
    {
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-106);
        // The velocity array contains a number of elements that it should not have
    }

    if((int)IONS->X.n_elem != (int)(3*IONS->NSP))
    {
        MPI_Barrier(params->mpi.MPI_TOPO);
        MPI_Abort(params->mpi.MPI_TOPO,-107);
        // The position array contains a number of elements that it should not have
    }

    // Assign cell:
    // ===========
    // Populates wxc,wxl and wxr only
    PIC ionsDynamics;
    ionsDynamics.assignCell(params, EB, IONS);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeParticlesArrays(const simulationParameters * params, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{
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

    PIC ionsDynamics;
    ionsDynamics.assignCell(params, EB, IONS);
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeBulkVariablesArrays(const simulationParameters * params, oneDimensional::ionSpecies * IONS)
{
    // Initialize mesh-defined quantities:
    // ==================================
    // Ion density:
    IONS->n.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->n_.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->n__.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->n___.zeros(params->mesh.NX_IN_SIM + 2);

    // Ion flux density:
    IONS->nv.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->nv_.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->nv__.zeros(params->mesh.NX_IN_SIM + 2);

    // Pressure tensors:
    IONS->P11.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->P22.zeros(params->mesh.NX_IN_SIM + 2);

    // Derived quantities:
    IONS->Tpar_m.zeros(params->mesh.NX_IN_SIM + 2);
    IONS->Tper_m.zeros(params->mesh.NX_IN_SIM + 2);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeBulkVariablesArrays(const simulationParameters * params, twoDimensional::ionSpecies * IONS)
{
    // Initialize plasma density:
    // Include Ghost cells (+2)
    IONS->n.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
    IONS->n_.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
    IONS->n__.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
    IONS->n___.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);

    // Initialize plasma flux:
    // Include Ghost cells (+2)
    IONS->nv.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
    IONS->nv_.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
    IONS->nv__.zeros(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::setupIonsInitialCondition(const simulationParameters * params, const characteristicScales * CS, FT * EB, vector<IT> * IONS)
{
    // Define total number of ions species:
    // ====================================
    int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

    // Print to terminal:
    // ==================
    if (params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << endl << "* * * * * * * * * * * * SETTING UP IONS INITIAL CONDITION * * * * * * * * * * * * * * * * * *" << endl;
    }

    // Loop over all species:
    // ======================
    for (int ii=0; ii<totalNumSpecies; ii++)
    {
        if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            if(params->restart)
            {
                if(params->mpi.MPI_DOMAIN_NUMBER == 0)
                {
                    cout << "Restart not implemented yet" << endl;
                }

                MPI_Abort(MPI_COMM_WORLD,-105);
            }
            else
            {
                switch (IONS->at(ii).IC)
                {
                    case(1):
                    {
                        if (params->quietStart)
                        {
                            // Create quiet start object:
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
                            // Use quiet start object:
                            qs.maxwellianVelocityDistribution(params, &IONS->at(ii));
                        }
                        else
                        {
                            // Create random start object:
                            RANDOMSTART<IT> rs(params);
                            // Apply random start object: USES MH algorithm
                            rs.maxwellianVelocityDistribution_nonhomogeneous(params, &IONS->at(ii));
                        }
                        break;
                    }
                    case(2):
                    {
                        if (params->quietStart)
                        {
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
                            qs.ringLikeVelocityDistribution(params, &IONS->at(ii));
                        }
                        else
                        {
                            RANDOMSTART<IT> rs(params);
                            rs.ringLikeVelocityDistribution(params, &IONS->at(ii));
                        }
                        break;
                    }
                    default:
                    {
                        if (params->quietStart)
                        {
                            QUIETSTART<IT> qs(params, &IONS->at(ii));
                            qs.maxwellianVelocityDistribution(params, &IONS->at(ii));
                        }
                        else
                        {
                            RANDOMSTART<IT> rs(params);
                            rs.maxwellianVelocityDistribution(params, &IONS->at(ii));
                        }
                    }
                } // switch
            } // if(params->restart)

            initializeParticlesArrays(params, EB, &IONS->at(ii));

            initializeBulkVariablesArrays(params, &IONS->at(ii));

        }
        else if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
        {
            initializeBulkVariablesArrays(params, &IONS->at(ii));
        }

        // Broadcast NSP (Number of super particles per process) and nSupPartPutput from ROOTS to COMM_WORLD:
        MPI_Bcast(&IONS->at(ii).NSP, 1, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, MPI_COMM_WORLD);
        MPI_Bcast(&IONS->at(ii).nSupPartOutput, 1, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, MPI_COMM_WORLD);

        // Calculate NCP: conversion factor from number of TOTAL super-particles NSP*NPROC to total number of real-particles
        // NCP = NR/(NSP*NPROC):
        if (params->dimensionality == 1)
        {
            double Ds=(params->mesh.LX)/(params->PP.ne.n_elem);
            double SNeDx=sum((params->BGP.ne)*(params->PP.ne)/(params->PP.Bx/params->BGP.Bo))*Ds;
            IONS->at(ii).NCP=((IONS->at(ii).densityFraction)*(SNeDx)/(IONS->at(ii).NSP*params->mpi.MPIS_PARTICLES));
        }
        else
        {
            IONS->at(ii).NCP = (IONS->at(ii).densityFraction*params->BGP.ne*params->mesh.LX*params->mesh.LY)/(IONS->at(ii).NSP*params->mpi.MPIS_PARTICLES);
        }

        // Print to the terminal:
        if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << "iON SPECIES: " << (ii + 1) << endl;

            if (params->quietStart)
            {
                cout << "+ Using quiet start: YES" << endl;
            }
            else
            {
                cout << "+ Using quiet start: NO" << endl;
            }

            cout << "+ Super-particles used in simulation: " << IONS->at(ii).NSP*params->mpi.MPIS_PARTICLES << endl;
        }

    }//Iteration over ion species

    // Print to terminal:
    // ==================
    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << "* * * * * * * * * * * * * IONS INITIAL CONDITION SET UP * * * * * * * * * * * * * * * * * * *" << endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
}


template <class IT, class FT> void INITIALIZE<IT,FT>::loadIonParameters(simulationParameters * params, vector<IT> * IONS)
{
    MPI_Barrier(MPI_COMM_WORLD);

    // Print to terminal:
    // ==================
    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << "* * * * * * * * * * * * LOADING ION PARAMETERS * * * * * * * * * * * * * * * * * *\n";
    	cout << "+ Number of ion species: " << params->numberOfParticleSpecies << endl;
    	cout << "+ Number of tracer species: " << params->numberOfTracerSpecies << endl;
    }

    // Assemble path to "ion_properties.ion":
    // ======================================
    string name;
    if(params->argc > 3)
    {
    	string argv(params->argv[3]);
    	name = "inputFiles/ions_properties_" + argv + ".ion";
    }
    else
    {
    	name = "inputFiles/ions_properties.ion";
    }

    // Read data from "ion_properties.ion" into "parametersMap":
    // =========================================================
    std::map<string,float> parametersMap;
    parametersMap = loadParameters(&name);

    // Determine the total number of ION species:
    // =========================================
    int totalNumSpecies(params->numberOfParticleSpecies + params->numberOfTracerSpecies);

    // Loop over all ION species and extract data from "parametersMap":
    // ================================================================
    for(int ii=0;ii<totalNumSpecies;ii++)
    {
        string name;
        IT ions;
        int SPECIES;
        stringstream ss;

        ss << ii + 1;

        name = "SPECIES" + ss.str();
        SPECIES = (int)parametersMap[name];
        name.clear();

        if (SPECIES == 0 || SPECIES == 1)
        {
            // Species type, -1: Guiding center, 0: Tracer, 1: Full orbit
            ions.SPECIES = SPECIES;

            // Number
            name = "NPC" + ss.str();
            ions.NPC = parametersMap[name];
            name.clear();

            //
            name = "IC" + ss.str();
            ions.IC = (int)parametersMap[name];
            name.clear();

            // Perpendicular temperature:
            name = "Tper" + ss.str();
            ions.Tper = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
            name.clear();

            // Parallel temperature:
            name = "Tpar" + ss.str();
            ions.Tpar = parametersMap[name]*F_E/F_KB; // Tpar in eV in input file
            name.clear();

            // Density fraction relative to 1:
            name = "densityFraction" + ss.str();
            ions.densityFraction = parametersMap[name];
            name.clear();

            //
            name = "pctSupPartOutput" + ss.str();
            ions.pctSupPartOutput = parametersMap[name];
            name.clear();

            // Charge state:
            name = "Z" + ss.str();
            ions.Z = parametersMap[name];
            name.clear();

            // AMU mass number:
            name = "M" + ss.str();
            ions.M = F_U*parametersMap[name];
            name.clear();

            // Derived quantities:
            ions.Q = F_E*ions.Z;
            ions.Wc = ions.Q*params->BGP.Bo/ions.M;
            ions.Wp = sqrt( ions.densityFraction*params->BGP.ne*ions.Q*ions.Q/(F_EPSILON*ions.M) );//Check the definition of the plasma freq for each species!
            ions.VTper = sqrt(2.0*F_KB*ions.Tper/ions.M);
            ions.VTpar = sqrt(2.0*F_KB*ions.Tpar/ions.M);
            ions.LarmorRadius = ions.VTper/ions.Wc;

            //Initializing the events counter:
            ions.pCount.zeros(1);
            ions.eCount.zeros(1);

            //Definition of the initial total number of superparticles for each species
            ions.NSP = ceil( ions.NPC*(double)params->mesh.NUM_CELLS_IN_SIM/(double)params->mpi.MPIS_PARTICLES );

            //
            ions.nSupPartOutput = floor( (ions.pctSupPartOutput/100.0)*ions.NSP );

            // Create new element on IONS vector:
            IONS->push_back(ions);

            // Print ion parameters to terminal:
            // =================================
            if(params->mpi.MPI_DOMAIN_NUMBER == 0)
            {
                if (ions.SPECIES == 0)
                {
                    cout << endl << "Species No "  << ii + 1 << " are tracers with the following parameters:" << endl;
                }
                else
                {
                    cout << endl << "Species No "  << ii + 1 << " are full-orbit particles with the following parameters:" << endl;
                }

                // Stream to terminal:
                // ===================
                cout << "+ User-defined number of particles per MPI: " << ions.NSP << endl;
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
        }
        else
        {
            MPI_Barrier(MPI_COMM_WORLD);

            if(params->mpi.MPI_DOMAIN_NUMBER == 0)
            {
                cerr << "PRO++ ERROR: Enter a valid type of species -- options are 0 = tracers, 1 = full orbit, -1 = guiding center" << endl;
            }
            MPI_Abort(MPI_COMM_WORLD,-106);
        }

    }//Iteration over ion species

    // Print to terminal:
    // ==================
    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << "* * * * * * * * * * * * ION PARAMETERS LOADED * * * * * * * * * * * * * * * * * *\n";
        }

    MPI_Barrier(MPI_COMM_WORLD);
}

template <class IT, class FT> void INITIALIZE<IT,FT>::loadPlasmaProfiles(simulationParameters * params, vector<IT> * IONS)
{
    // Define number of mesh points with Ghost cells included:
    // ======================================================
    int NX(params->mesh.NX_IN_SIM + 2);

    // Define number of elements in external profiles:
    // ===============================================
    int nTable = params->PP.nTable;

    // Initialize variables:
    // =====================
    params->PP.ne.zeros(nTable);
    params->PP.Tpar.zeros(nTable);
    params->PP.Tper.zeros(nTable);
    params->PP.Bx.zeros(nTable);
    params->PP.Bx_i.zeros(NX);
    params->PP.Br_i.zeros(NX);
    params->PP.dBrdx_i.zeros(NX);

if (params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        // Get file name containing initial condition plasma profiles:
        // ===========================================================
        int nn = params->PATH.length();
        std::string inputFilePath = params->PATH.substr(0,nn-13);
        std::string fileName1= inputFilePath + "/inputFiles/ne_norm_profile.txt"; //Path to load the external file
        std::string fileName2= inputFilePath + "/inputFiles/Tpar_norm_profile.txt"; //Path to load the external file
        std::string fileName3= inputFilePath + "/inputFiles/Tper_norm_profile.txt"; //Path to load the external file
        std::string fileName4= inputFilePath + "/inputFiles/Bx_norm_profile.txt"; //Path to load the external file

        // REad normalized plasma profiles from external files:
        // =====================================================
        params->PP.ne.load(fileName1);
        params->PP.Tpar.load(fileName2);
        params->PP.Tper.load(fileName3);
        params->PP.Bx.load(fileName4);

        //Rescale profiles:
        // ================
        //params->PP.ne   *= params->BGP.ne;
        params->PP.Tpar *= IONS->at(0).Tpar;
        params->PP.Tper *= IONS->at(0).Tper;
        params->PP.Bx   *= params->BGP.Bo;

        //Interpolate at mesh points:
        // ==========================
        // Query points:
        arma::vec xq = linspace(0,params->mesh.LX,NX);
        arma::vec yq(xq.size());
        // Sample points:
        arma::vec xt = linspace(0,params->mesh.LX,nTable); // x-vector from the table
        arma:: vec yt(xt.size());

        // Bx profile:
        // ===========
        yt = params->PP.Bx;
        interp1(xt,yt,xq,yq);
        params->PP.Bx_i = yq;

        // Br profile:
        // ===========
        arma::vec Br(nTable,1);
        Br.subvec(0,nTable-2) = -0.5*(params->BGP.Rphi0)*diff(params->PP.Bx)/(params->mesh.LX/nTable);
        Br(nTable-1) = Br(nTable-2);

        yt = Br;
        interp1(xt,yt,xq,yq);
        params->PP.Br_i = yq;

        // dBr profile:
        // ===========
        arma::vec dBr(nTable,1);
        dBr.subvec(0,nTable-2) = diff(Br)/(params->mesh.LX/nTable);
        dBr(nTable-1) = dBr(nTable-2);

        yt = dBr;
        interp1(xt,yt,xq,yq);
        params->PP.dBrdx_i = yq;

        //interp1(xt,dBr,xq,params->PP.dBrdx_i);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Plasma density:
    MPI_Bcast(params->PP.ne.memptr()  ,params->PP.ne.size() , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Parallel temperature:
    MPI_Bcast(params->PP.Tpar.memptr(),params->PP.Tpar.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perpendicular temperature:
    MPI_Bcast(params->PP.Tper.memptr(),params->PP.Tper.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Magnetic field profile, X component:
    MPI_Bcast(params->PP.Bx.memptr()  ,params->PP.Bx.size()  , MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Quantities interpolated at the mesh points:
    MPI_Bcast(params->PP.Bx_i.memptr()   ,params->PP.Bx_i.size()   , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(params->PP.Br_i.memptr()   ,params->PP.Br_i.size()   , MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(params->PP.dBrdx_i.memptr(),params->PP.dBrdx_i.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(const simulationParameters * params, oneDimensional::fields * EB)
{

    // Number of mesh points with ghost cells included:
    // ================================================
    int NX(params->mesh.NX_IN_SIM + 2);

    // Select how to initialize electromagnetic fields:
    // ================================================
    EB->zeros(NX);
    if (params->quietStart)
    {
        // From "params" and assumes uniform fields:
        EB->B.Z.fill(params->BGP.Bz);
        EB->B.Y.fill(params->BGP.By);
        EB->B.X.fill(params->BGP.Bx);
    }
    else
    {
        // From externally supplied files:
        EB->B.X = params->PP.Bx_i;
        EB->B.Y = params->PP.Br_i;
        EB->B.Z.fill(params->BGP.Bz);
    }
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(const simulationParameters * params, twoDimensional::fields * EB){
    int NX(params->mesh.NX_IN_SIM + 2); // Ghost mesh points (+2) included
    int NY(params->mesh.NY_IN_SIM + 2); // Ghost mesh points (+2) included

    EB->zeros(NX,NY);

    EB->E.X.fill(0.0); // x
    EB->E.Y.fill(0.0); // x
    EB->E.Z.fill(0.0); // x

    EB->B.X.fill(params->BGP.Bx); // x
    EB->B.Y.fill(params->BGP.By); // y
    EB->B.Z.fill(params->BGP.Bz); // z
}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFields(const simulationParameters * params, FT * EB){

    MPI_Barrier(MPI_COMM_WORLD);

        // Print to terminal:
        // ==================
	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << endl << "* * * * * * * * * * * * INITIALIZING ELECTROMAGNETIC FIELDS * * * * * * * * * * * * * * * * * *" << endl;
        }

        // Select how to initialize fields:
        // ================================
	if (params->loadFields == 1)
        {
            // From external files.
            if(params->mpi.MPI_DOMAIN_NUMBER == 0)
                    cout << "Loading external electromagnetic fields..." << endl;
                    MPI_Abort(params->mpi.MPI_TOPO,-104);
	}
        else
        {
            //The electromagnetic fields are being initialized in the runtime.
            initializeFieldsSizeAndValue(params, EB);

            // Print to terminal:
            if (params->mpi.MPI_DOMAIN_NUMBER == 0)
            {
                cout << "Initializing electromagnetic fields within simulation" << endl;
                cout << "+ Magnetic field along x-axis: " << scientific << params->BGP.Bx << fixed << " T" << endl;
                cout << "+ Magnetic field along y-axis: " << scientific << params->BGP.By << fixed << " T" << endl;
                cout << "+ Magnetic field along z-axis: " << scientific << params->BGP.Bz << fixed << " T" << endl;
            }
	}

        // Print to terminal:
        // ==================
	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            cout << "* * * * * * * * * * * * ELECTROMAGNETIC FIELDS INITIALIZED  * * * * * * * * * * * * * * * * * *" << endl;
        }
}


template class INITIALIZE<oneDimensional::ionSpecies, oneDimensional::fields>;
template class INITIALIZE<twoDimensional::ionSpecies, twoDimensional::fields>;
