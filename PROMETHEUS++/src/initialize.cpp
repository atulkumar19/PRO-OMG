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


// Private supporting functions:
// =============================================================================
template <class IT, class FT> vector<string> INITIALIZE<IT,FT>::split(const string& str, const string& delim)
{
    vector<string> tokens;
    size_t prev = 0, pos = 0;
    do
    {
        pos = str.find(delim, prev);

        if (pos == string::npos)
        {
            pos = str.length();
        }

        string token = str.substr(prev, pos-prev);

        if (!token.empty())
        {
            tokens.push_back(token);
        }

        prev = pos + delim.length();
    }
    while (pos < str.length() && prev < str.length());

    return tokens;
}

template <class IT, class FT> map<string,string>INITIALIZE<IT,FT>::ReadAndloadInputFile(string * inputFile)
{
    // Create stream object:
    // =====================
    fstream reader;

    // Create map object:
    // ==================
    std::map<string,string> readMap;

    // Open input file using reader object:
    // ====================================
    reader.open(inputFile->data(),ifstream::in);

    // Handle error:
    // =============
    if (!reader){
        MPI_Barrier(MPI_COMM_WORLD);

    	cerr << "PRO++ ERROR: The input file couldn't be opened." << endl;
    	MPI_Abort(MPI_COMM_WORLD, -101);
    }

    // Parse through file:
    // ===================
    string lineContent;
    vector<string> keyValuePair;
    while ( reader.good() )
    {
        // Read entire line:
        getline(reader,lineContent);

        // Search for comment symbol:
        size_t commentCharPos = lineContent.find("//",0);

        // Check for comment symbol:
        if (commentCharPos == 0 || lineContent.empty())
        {
            // Skip line
        }
        else
        {
            // Get value pair:
            keyValuePair = INITIALIZE::split(lineContent," ");

            // Update map:
            readMap[ keyValuePair[0] ] = keyValuePair[1];
        }
    }

    // Close stream object:
    // ===================
    reader.close();

    // Return map:
    // ==========
    return readMap;
}

// Constructor:
// =============================================================================
template <class IT, class FT> INITIALIZE<IT,FT>::INITIALIZE(simulationParameters * params, int argc, char* argv[])
{
    // Get RANK and SIZE of nodes within COMM_WORLD:
    // =============================================
    MPI_Comm_size(MPI_COMM_WORLD, &params->mpi.NUMBER_MPI_DOMAINS);
    MPI_Comm_rank(MPI_COMM_WORLD, &params->mpi.MPI_DOMAIN_NUMBER);

    // Error codes:
    // ============
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


    // Copyright and Licence Info:
    // ===========================
    if (params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
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

    // Arguments and paths to main function:
    // =====================================
    params->PATH = argv[2];
	params->argc = argc;
	params->argv = argv;

    // Check number of MPI domains:
    // ============================
	if( fmod( (double)params->mpi.NUMBER_MPI_DOMAINS, 2.0 ) > 0.0 )
    {
        MPI_Barrier(MPI_COMM_WORLD);

		if(params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
			cerr << "PRO++ ERROR: The number of MPI processes must be an even number." << endl;
		}

		MPI_Abort(MPI_COMM_WORLD,-100);
	}

    // Stream date when simulation is started:
    // =======================================
    if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        time_t current_time = std::time(NULL);
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
        cout << "STARTING " << params->argv[1] << " SIMULATION ON: " << std::ctime(&current_time) << endl;
        cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * " << endl;
    }

    // Get name of path to input file:
    // ===============================
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

    // Read input file and assemble map:
    // ================================
	std::map<string,string> parametersStringMap;
	parametersStringMap = ReadAndloadInputFile(&name);

	// Create HDF5 folders if they don't exist:
    // ========================================
	if(params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
		string mkdir_outputs_dir = "mkdir " + params->PATH;
		const char * sys = mkdir_outputs_dir.c_str();
		int rsys = system(sys);

		string mkdir_outputs_dir_HDF5 = mkdir_outputs_dir + "/HDF5";
		sys = mkdir_outputs_dir_HDF5.c_str();
		rsys = system(sys);
	}

    // Populate "params" with data from input file:
    // ============================================

    // Assign input data from map to "params":
    // -------------------------------------------------------------------------
    params->dimensionality   = stoi( parametersStringMap["dimensionality"] );
    params->mpi.MPIS_FIELDS  = stoi( parametersStringMap["mpisForFields"] );

    if(stoi( parametersStringMap["quietStart"] ) == 1)
    {
        params->quietStart = true;
    }
    else
    {
        params->quietStart = false;
    }

    if(std::stoi( parametersStringMap["restart"] ) == 1)
    {
        params->restart = true;
    }
    else
    {
        params->restart = false;
    }

    params->loadFields              = stoi( parametersStringMap["loadFields"] );
    params->numberOfRKIterations    = stoi( parametersStringMap["numberOfRKIterations"] );
    params->numberOfParticleSpecies = stoi( parametersStringMap["numberOfParticleSpecies"] );
  	params->numberOfTracerSpecies   = stoi( parametersStringMap["numberOfTracerSpecies"] );

    // Characteristic values:
    // -------------------------------------------------------------------------
    params->CV.ne   = stod( parametersStringMap["CV_ne"] );
    params->CV.Te   = stod( parametersStringMap["CV_Te"] )*F_E/F_KB;
    params->CV.B    = stod( parametersStringMap["CV_B"] );
    params->CV.Tpar = stod( parametersStringMap["CV_Tpar"] )*F_E/F_KB;
    params->CV.Tper = stod( parametersStringMap["CV_Tper"] )*F_E/F_KB;

    // Simulation time:
    // -------------------------------------------------------------------------
    params->DTc            = stod( parametersStringMap["DTc"] );
    params->simulationTime = std::stod( parametersStringMap["simulationTime"] );

    // Switches:
    // -------------------------------------------------------------------------
    params->SW.EfieldSolve   = stoi( parametersStringMap["SW_EfieldSolve"] );
    params->SW.HallTermSolve = stoi( parametersStringMap["SW_HallTermSolve"] );
    params->SW.BfieldSolve   = stoi( parametersStringMap["SW_BfieldSolve"] );
    params->SW.Collisions    = stoi( parametersStringMap["SW_Collisions"] );
    params->SW.RFheating     = stoi( parametersStringMap["SW_RFheating"] );
    params->SW.advancePos    = stoi( parametersStringMap["SW_advancePos"] );
    params->SW.linearSolve   = stoi( parametersStringMap["SW_linearSolve"] );

    // Magnetic field initial conditions:
    // -------------------------------------------------------------------------
    params->em_IC.BX          = stod( parametersStringMap["IC_BX"] );
    params->em_IC.BY          = stod( parametersStringMap["IC_BY"] );
    params->em_IC.BZ          = stod( parametersStringMap["IC_BZ"] );
    params->em_IC.BX_NX       = stoi( parametersStringMap["IC_BX_NX"] );
    params->em_IC.BX_fileName = parametersStringMap["IC_BX_fileName"];

    // Geometry:
    // -------------------------------------------------------------------------
    unsigned int NX      = (unsigned int)stoi( parametersStringMap["NX"] );
    unsigned int NY      = (unsigned int)stoi( parametersStringMap["NY"] );
    unsigned int NZ      = (unsigned int)stoi( parametersStringMap["NZ"] );
    params->DrL          = stod( parametersStringMap["DrL"] );
	  params->dp           = stod( parametersStringMap["dp"] );
  	params->geometry.r1  = stod( parametersStringMap["r1"] );
    params->geometry.r2  = stod( parametersStringMap["r2"] );

    // Electron initial conditions:
    // -------------------------------------------------------------------------
    params->f_IC.ne      = stod( parametersStringMap["IC_ne"] );
    params->f_IC.Te      = stod( parametersStringMap["IC_Te"] )*F_E/F_KB; // Te in eV in input file

    // RF parameters
    // -------------------------------------------------------------------------
    params->RF.Prf       = stod( parametersStringMap["RF_Prf"] );
    params->RF.freq      = stod( parametersStringMap["RF_freq"]);
    params->RF.x1        = stod( parametersStringMap["RF_x1"]  );
    params->RF.x2        = stod( parametersStringMap["RF_x2"]  );
    params->RF.kpar      = stod( parametersStringMap["RF_kpar"]);
    params->RF.kper      = stod( parametersStringMap["RF_kper"]);
    params->RF.handedness= stoi( parametersStringMap["RF_handedness"]);
    params->RF.Prf_NS    = stoi( parametersStringMap["RF_Prf_NS"] );
    params->RF.Prf_fileName = parametersStringMap["RF_Prf_fileName"];
    params->RF.numit     = stoi( parametersStringMap["RF_numit"]);

    // Output variables:
    // -------------------------------------------------------------------------
    params->outputCadence           = stod( parametersStringMap["outputCadence"] );
    string nonparsed_variables_list = parametersStringMap["outputs_variables"].substr(1, parametersStringMap["outputs_variables"].length() - 2);
	  params->outputs_variables       = split(nonparsed_variables_list,",");

    // Data smoothing:
    // -------------------------------------------------------------------------
    params->smoothingParameter        = stod( parametersStringMap["smoothingParameter"] );
    params->filtersPerIterationFields = stoi( parametersStringMap["filtersPerIterationFields"] );
    params->filtersPerIterationIons   = stoi( parametersStringMap["filtersPerIterationIons"] );

    // Derived parameters:
    // ===================
    params->mpi.MPIS_PARTICLES = params->mpi.NUMBER_MPI_DOMAINS - params->mpi.MPIS_FIELDS;
    params->geometry.A_0 = M_PI*(pow(params->geometry.r2,2) -pow(params->geometry.r1,2));

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

    // Magnetic field data:
    //params->em_IC.B0 = sqrt( pow(params->em_IC.BX,2.0) + pow(params->em_IC.BY,2.0) + pow(params->em_IC.BZ,2.0) );
	//params->BGP.Bx = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0);
	//params->BGP.By = params->BGP.Bo*sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0);
	//params->BGP.Bz = params->BGP.Bo*cos(params->BGP.theta*M_PI/180.0);

    //params->BGP.Bx = (abs(params->BGP.Bx) < PRO_ZERO) ? 0.0 : params->BGP.Bx;
    //params->BGP.By = (abs(params->BGP.By) < PRO_ZERO) ? 0.0 : params->BGP.By;
    //params->BGP.Bz = (abs(params->BGP.Bz) < PRO_ZERO) ? 0.0 : params->BGP.Bz;

}


template <class IT, class FT> void INITIALIZE<IT,FT>::loadMeshGeometry(simulationParameters * params, fundamentalScales * FS){

    MPI_Barrier(MPI_COMM_WORLD);

    // Print to terminal:
    // ==================
    if (params->mpi.MPI_DOMAIN_NUMBER == 0)
    {
        cout << endl << "* * * * * * * * * * * * LOADING/COMPUTING SIMULATION GRID * * * * * * * * * * * * * * * * * *\n";
    }

    // Select grid size: based on Larmour radius or  skin depth:
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
        params->mesh.nodes.X(ii) = (double)ii*params->mesh.DX + params->mesh.DX/2;
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

    // Initialize particle defined flags:
    // ==================================
    IONS->f1.zeros(IONS->NSP);
    IONS->f2.zeros(IONS->NSP);
    IONS->f3.zeros(IONS->NSP);

    // Initialize particle kinetic energy at boundaries:
    // ================================================
    IONS->dE1.zeros(IONS->NSP);
    IONS->dE2.zeros(IONS->NSP);
    IONS->dE3.zeros(IONS->NSP);

    // Initialize particle weight:
    // ===========================
    IONS->a.ones(IONS->NSP);

    // Initialize RF terms:
    // ====================
    IONS->p_RF.rho.zeros(IONS->NSP);
    IONS->p_RF.cosPhi.zeros(IONS->NSP);
    IONS->p_RF.sinPhi.zeros(IONS->NSP);
    IONS->p_RF.phase.zeros(IONS->NSP);
    IONS->p_RF.udE3.zeros(IONS->NSP);

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
{}

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
{}


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

        if (params->dimensionality == 1)
        {
            double Ds = (params->mesh.LX)/(params->em_IC.BX_NX);
            double NR = (IONS->at(ii).p_IC.densityFraction)*sum(((params->em_IC.BX*params->geometry.A_0)/(params->em_IC.Bx_profile))*(params->f_IC.ne))*Ds;
            IONS->at(ii).NCP = (NR/(IONS->at(ii).NSP*params->mpi.MPIS_PARTICLES));
        }
        else
        {
            IONS->at(ii).NCP = (IONS->at(ii).p_IC.densityFraction*params->f_IC.ne*params->mesh.LX*params->mesh.LY)/(IONS->at(ii).NSP*params->mpi.MPIS_PARTICLES);
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
    std::map<string,string> parametersMap;
    parametersMap = ReadAndloadInputFile(&name);

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
        SPECIES = stoi(parametersMap[name]);
        name.clear();

        if (SPECIES == 0 || SPECIES == 1)
        {
            // General:
            // =================================================================
            // Species type, -1: Guiding center, 0: Tracer, 1: Full orbit
            ions.SPECIES = SPECIES;

            // Number of particles per cell:
            name = "NPC" + ss.str();
            ions.NPC = stoi(parametersMap[name]);
            name.clear();

            name = "pctSupPartOutput" + ss.str();
            ions.pctSupPartOutput = stod(parametersMap[name]);
            name.clear();


            // Charge state:
            name = "Z" + ss.str();
            ions.Z = stod(parametersMap[name]);
            name.clear();

            // AMU mass number:
            name = "M" + ss.str();
            ions.M = F_U*stod(parametersMap[name]);
            name.clear();

            // Initial condition:
            // =================================================================
            name = "IC_type_" + ss.str();
            ions.IC = stoi(parametersMap[name]);
            ions.p_IC.IC_type = stoi(parametersMap[name]);
            name.clear();

            // Perpendicular temperature:
            name = "IC_Tper_" + ss.str();
            ions.p_IC.Tper = stod(parametersMap[name])*F_E/F_KB;
            name.clear();

            name = "IC_Tper_fileName_" + ss.str();
            ions.p_IC.Tper_fileName = parametersMap[name];
            name.clear();



            name = "IC_Tper_NX_" + ss.str();
            ions.p_IC.Tper_NX = stoi(parametersMap[name]);
            name.clear();

            // Parallel temperature:
            name = "IC_Tpar_" + ss.str();
            ions.p_IC.Tpar = stod(parametersMap[name])*F_E/F_KB; // Tpar in eV in input file
            name.clear();


            name = "IC_Tpar_fileName_" + ss.str();
            ions.p_IC.Tpar_fileName = parametersMap[name];
            name.clear();



            name = "IC_Tpar_NX_" + ss.str();
            ions.p_IC.Tpar_NX = stoi(parametersMap[name]);
            name.clear();

            // Density fraction relative to 1:
            name = "IC_densityFraction_" + ss.str();
            //ions.densityFraction = stod(parametersMap[name]);
            ions.p_IC.densityFraction = stod(parametersMap[name]);
            name.clear();


            name = "IC_densityFraction_fileName_" + ss.str();
            ions.p_IC.densityFraction_fileName = parametersMap[name];
            name.clear();


            name = "IC_densityFraction_NX_" + ss.str();
            ions.p_IC.densityFraction_NX = stoi(parametersMap[name]);
            name.clear();

            // Boundary conditions:
            // =================================================================
            name = "BC_type_" + ss.str();
            ions.p_BC.BC_type = stoi(parametersMap[name]);
            name.clear();

            name = "BC_T_" + ss.str();
            ions.p_BC.T = stod(parametersMap[name])*F_E/F_KB;
            name.clear();


            name = "BC_E_" + ss.str();
            ions.p_BC.E = stod(parametersMap[name])*F_E/F_KB;
            name.clear();

            name = "BC_eta_" + ss.str();
            ions.p_BC.eta = stod(parametersMap[name]);
            name.clear();

            name = "BC_G_" + ss.str();
            ions.p_BC.G = stod(parametersMap[name]);
            name.clear();

            name = "BC_mean_x_" + ss.str();
            ions.p_BC.mean_x = stod(parametersMap[name]);
            name.clear();

            name = "BC_sigma_x_" + ss.str();
            ions.p_BC.sigma_x = stod(parametersMap[name]);
            name.clear();

            name = "BC_G_fileName_" + ss.str();
            ions.p_BC.G_fileName = parametersMap[name];
            name.clear();

            name = "BC_G_NS_" + ss.str();
            ions.p_BC.G_NS = stoi(parametersMap[name]);
            name.clear();

            // Derived quantities:
            // =================================================================
            ions.Q     = F_E*ions.Z;
            ions.Wc    = ions.Q*params->CV.B/ions.M;
            ions.Wp    = sqrt( ions.p_IC.densityFraction*params->CV.ne*ions.Q*ions.Q/(F_EPSILON*ions.M) );//Check the definition of the plasma freq for each species!
            ions.VTper = sqrt(2.0*F_KB*params->CV.Tper/ions.M);
            ions.VTpar = sqrt(2.0*F_KB*params->CV.Tpar/ions.M);
            ions.LarmorRadius = ions.VTper/ions.Wc;

            // Initializing the events counter:
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
                cout << "+ Parallel temperature: " << params->CV.Tpar*F_KB/F_E << " eV" << endl;
                cout << "+ Perpendicular temperature: " << params->CV.Tper*F_KB/F_E << " eV" << endl;
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
    int nn = params->PATH.length();
    std::string inputFilePath = params->PATH.substr(0,nn-13);

    for(int ii=0;ii<IONS->size();ii++)
    {

        // Assemble  external filenames
        std::string fileName  = params->PATH.substr(0,nn-13);
        std::string fileName2 = fileName + "/inputFiles/" + IONS->at(ii).p_IC.Tper_fileName;
        std::string fileName3 = fileName + "/inputFiles/" + IONS->at(ii).p_IC.Tpar_fileName;
        std::string fileName4 = fileName + "/inputFiles/" + IONS->at(ii).p_IC.densityFraction_fileName;


        // Load data from external file:
        // ============================
        IONS->at(ii).p_IC.Tper_profile.load(fileName2);
        IONS->at(ii).p_IC.Tpar_profile.load(fileName3);
        IONS->at(ii).p_IC.densityFraction_profile.load(fileName4);

        // Rescale the plasma Profiles
        // ================================
        IONS->at(ii).p_IC.Tper_profile *= IONS->at(ii).p_IC.Tper;
        IONS->at(ii).p_IC.Tpar_profile *= IONS->at(ii).p_IC.Tpar;
        IONS->at(ii).p_IC.densityFraction_profile *= params->f_IC.ne;
    }

}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(simulationParameters * params, oneDimensional::fields * EB)
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
        EB->B.Z.fill(params->em_IC.BZ);
        EB->B.Y.fill(params->em_IC.BY);
        EB->B.X.fill(params->em_IC.BX);
    }
    else
    {
        // Read filename from input file:
        // =============================
        int nn = params->PATH.length();
        std::string fileName = params->PATH.substr(0,nn-13);
        fileName = fileName + "/inputFiles/" + params->em_IC.BX_fileName;

        // Load data from external file:
        // ============================
        params->em_IC.Bx_profile.load(fileName);
        params->em_IC.Bx_profile *= params->em_IC.BX;

        //Interpolate at mesh points:
        // ==========================
        // Query points:
        arma::vec xq = zeros(NX);
        arma::vec yq = zeros(NX);

          for(int ii=0; ii<NX; ii++)
          {
              xq(ii) = (double)ii*params->mesh.DX - (0.5*params->mesh.DX) ;
          }
        // arma::vec xq = linspace(dx,params->mesh.LX+dx,NX);
        // yq(xq.size());
        // Sample points:
        int BX_NX  = params->em_IC.BX_NX;

        double dX = params->mesh.LX/((double)BX_NX);

        double dBX_NX = params->mesh.LX/BX_NX;

        arma::vec xt = linspace(0,params->mesh.LX,BX_NX); // x-vector from the table
        arma:: vec yt(xt.size());

        // Bx profile:
        // ===========
        yt = params->em_IC.Bx_profile;
        interp1(xt,yt,xq,yq);
        EB->B.X = yq;

        // By profile:
        // ===========
        arma::vec Br(BX_NX,1);
        arma::vec BX = params->em_IC.Bx_profile;
        Br.subvec(1,BX_NX-2) = -0.5*(params->geometry.r2)*((BX.subvec(2,BX_NX-1) - BX.subvec(0,BX_NX-3))/(2*dX));
        Br(BX_NX-1) = Br(BX_NX-2);
        Br(0) = Br(1);

        yt = Br;
        interp1(xt,yt,xq,yq);
        EB->B.Y = yq;

        // Bz profile:
        // ===========
        EB->B.Z.fill(params->em_IC.BZ);
    }
}

template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFieldsSizeAndValue(simulationParameters * params, twoDimensional::fields * EB)
{}


template <class IT, class FT> void INITIALIZE<IT,FT>::initializeFields(simulationParameters * params, FT * EB){

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
            cout << "+ Magnetic field along x-axis: " << scientific << params->em_IC.BX << fixed << " T" << endl;
            cout << "+ Magnetic field along y-axis: " << scientific << params->em_IC.BY << fixed << " T" << endl;
            cout << "+ Magnetic field along z-axis: " << scientific << params->em_IC.BZ << fixed << " T" << endl;
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
