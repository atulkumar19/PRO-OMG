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

#include "mpi_main.h"


void MPI_MAIN::createMPITopology(simulationParameters * params)
{
    // Generation of communicators for fields and particles:
    // =====================================================
    // params->mpi.MPI_DOMAIN_NUMBER is defined in constructor is INITIALIZE

	// Define the color of each MPI sub-communicator:
	if (params->mpi.MPI_DOMAIN_NUMBER < params->mpi.MPIS_FIELDS)
        {
		params->mpi.COMM_COLOR = FIELDS_MPI_COLOR; // Color of fields' communicator
	}
        else
        {
		params->mpi.COMM_COLOR = PARTICLES_MPI_COLOR; // Color of particles' communicator
	}

	// Generate communicators for fields and particle MPI processes:
	// key = params->mpi.MPI_DOMAIN_NUMBER
    MPI_Comm_split(MPI_COMM_WORLD, params->mpi.COMM_COLOR, params->mpi.MPI_DOMAIN_NUMBER, &params->mpi.COMM);

	// Get RANK and SIZE in the context of the new communicators:
    MPI_Comm_rank(params->mpi.COMM, &params->mpi.COMM_RANK);
	MPI_Comm_size(params->mpi.COMM, &params->mpi.COMM_SIZE);

	// Define PARTICLES master nodes and distribute info among all MPIs in MPI_COMM_WORLD:
	// ==================================================================================
	// Indentify PARTICLES master node:
    if ((params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR) && (params->mpi.COMM_RANK == 0))
    {
		params->mpi.IS_PARTICLES_ROOT = true;
		params->mpi.PARTICLES_ROOT_WORLD_RANK = params->mpi.MPI_DOMAIN_NUMBER;
	}
    else
    {
		params->mpi.IS_PARTICLES_ROOT = false;
		params->mpi.PARTICLES_ROOT_WORLD_RANK = -1;
	}

	// Broadcast PARTICLES_ROOT_WORLD_RANK fom PARTICLE root to all processes in PARTICLE COMM:
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
	MPI_Bcast(&params->mpi.PARTICLES_ROOT_WORLD_RANK, 1, MPI_INT, 0, params->mpi.COMM);

	// Broadcast from a PARTICLE node to ALL nodes in COMM_WORLD:
	MPI_Bcast(&params->mpi.PARTICLES_ROOT_WORLD_RANK, 1, MPI_INT, (params->mpi.NUMBER_MPI_DOMAINS - 1), MPI_COMM_WORLD);

	// Define FIELDS master nodes and distribute info among all MPIs in MPI_COMM_WORLD:
	// ================================================================================
	// Indentify FIELDS master node:
	if ((params->mpi.COMM_COLOR == FIELDS_MPI_COLOR) && (params->mpi.COMM_RANK == 0))
    {
		params->mpi.IS_FIELDS_ROOT = true;
		params->mpi.FIELDS_ROOT_WORLD_RANK = params->mpi.MPI_DOMAIN_NUMBER;
	}
    else
    {
		params->mpi.IS_FIELDS_ROOT = false;
		params->mpi.FIELDS_ROOT_WORLD_RANK = -1;
	}

	// Broadcast FIELDS_ROOT_WORLD_RANK fom FIELDS root to all processes in FIELDS COMM:
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
    {
	MPI_Bcast(&params->mpi.FIELDS_ROOT_WORLD_RANK, 1, MPI_INT, 0, params->mpi.COMM);
    }

	// Broadcast from a FIELDS node to ALL nodes in COMM_WORLD:
	MPI_Bcast(&params->mpi.FIELDS_ROOT_WORLD_RANK, 1, MPI_INT, 0, MPI_COMM_WORLD);

	// cout << "WORLD RANK: " << params->mpi.MPI_DOMAIN_NUMBER << " | IS PM: " << params->mpi.IS_PARTICLES_ROOT << " | IS FM: " << params->mpi.IS_FIELDS_ROOT << " | PMWR: " <<  params->mpi.PARTICLES_ROOT_WORLD_RANK << " | FMWR:" << params->mpi.FIELDS_ROOT_WORLD_RANK << endl;

    // Cartesian topology generation for MPIs evolving the fields in the simulation:
	// ============================================================================
    if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
    {
        int ndims;
        int dims_1D[1] = {params->mpi.MPIS_FIELDS};
        int dims_2D[2];
        int reorder(0);
        int periods_1D[1] = {1};
        int periods_2D[2] = {1, 1};
        int src;
        int coord;
        int coords_1D[1];
        int coords_2D[2];
        int topo_status;

        if (params->dimensionality == 1)
        {
            ndims = 1;

            params->mpi.MPI_DOMAINS_ALONG_X_AXIS = params->mpi.MPIS_FIELDS;
            params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = 1;
            params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

            MPI_Cart_create(params->mpi.COMM, ndims, dims_1D, periods_1D, reorder, &params->mpi.MPI_TOPO);
        }
        else
        {
			ndims = 2;

			if (params->mesh.SPLIT_DIRECTION == 0)
			{
				dims_2D[0] = params->mpi.MPIS_FIELDS; 	// x-axis
				dims_2D[1] = 1; 						// y-axis
			}
			else
			{
				dims_2D[0] = 1; 						// x-axis
				dims_2D[1] = params->mpi.MPIS_FIELDS;	// y-axis
			}

			params->mpi.MPI_DOMAINS_ALONG_X_AXIS = dims_2D[0];
			params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = dims_2D[1];
			params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

			MPI_Cart_create(params->mpi.COMM, ndims, dims_2D, periods_2D, reorder, &params->mpi.MPI_TOPO);
        }


        MPI_Topo_test(params->mpi.MPI_TOPO, &topo_status);

        if (topo_status == MPI_CART)
        {
            MPI_Comm_rank(params->mpi.MPI_TOPO, &params->mpi.MPI_DOMAIN_NUMBER_CART);

			if (params->dimensionality == 1)
			{ // 1-D
				MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.MPI_DOMAIN_NUMBER_CART, ndims, params->mpi.MPI_CART_COORDS_1D);
			}
            else
            { // 2-D
                MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.MPI_DOMAIN_NUMBER_CART, ndims, params->mpi.MPI_CART_COORDS_2D);
			}

			for (int mpis=0; mpis<params->mpi.MPIS_FIELDS; mpis++)
			{
				int * COORDS = new int[2*params->mpi.MPIS_FIELDS];

				MPI_Allgather(&params->mpi.MPI_CART_COORDS_2D, 2, MPI_INT, COORDS, 2, MPI_INT, params->mpi.MPI_TOPO);

				params->mpi.MPI_CART_COORDS.push_back(new int[2]);
				*(params->mpi.MPI_CART_COORDS.at(mpis)) = *(COORDS + 2*mpis);
				*(params->mpi.MPI_CART_COORDS.at(mpis) + 1) = *(COORDS + 2*mpis + 1);
			}

			// Calculate neighboring RANKS:
			// ===========================
			// Right side:
			src = params->mpi.MPI_DOMAIN_NUMBER_CART;
			MPI_Cart_shift(params->mpi.MPI_TOPO, 0, 1, &src, &params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART);

			// Left side:
			src = params->mpi.MPI_DOMAIN_NUMBER_CART;
			MPI_Cart_shift(params->mpi.MPI_TOPO, 0, -1, &src, &params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART);

            // Up and down for 2D:
			if (params->dimensionality == 2)
			{
				src = params->mpi.MPI_DOMAIN_NUMBER_CART;
				MPI_Cart_shift(params->mpi.MPI_TOPO, 1, 1, &src, &params->mpi.UP_MPI_DOMAIN_NUMBER_CART);

				src = params->mpi.MPI_DOMAIN_NUMBER_CART;
				MPI_Cart_shift(params->mpi.MPI_TOPO, 1, -1, &src, &params->mpi.DOWN_MPI_DOMAIN_NUMBER_CART);
			}

			// Calculate mesh index range (iIndex,fIndex) associated with each RANK:
			if (params->dimensionality == 1)
			{
				params->mpi.iIndex = params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1;
				params->mpi.fIndex = params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1);
			}
			else
			{
				params->mpi.irow = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART))*params->mesh.NX_PER_MPI + 1;
				params->mpi.frow = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)) + 1)*params->mesh.NX_PER_MPI;
				params->mpi.icol = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1)*params->mesh.NY_PER_MPI + 1;
				params->mpi.fcol = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1) + 1)*params->mesh.NY_PER_MPI;
			}

			if (params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
            {
                cout << endl << "* * * * * * * * * * * * GENERATING MPI TOPOLOGY FOR FIELDS * * * * * * * * * * * * * * * * * *" << endl;
                if (params->dimensionality == 1)
				{
					cout << "+ Number of MPI processes along the x-axis: " << dims_1D[0] << endl;
					cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
				}
                else
                {
					cout << "+ Number of MPI processes along the x-axis: " << dims_2D[0] << endl;
					cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
					cout << "+ Number of MPI processes along the y-axis: " << dims_2D[1] << endl;
					cout << "+ Number of mesh nodes along y-axis: " << params->mesh.NY_IN_SIM << endl;
                }

				cout << "+ Number of MPI processes for FIELDS: " << params->mpi.MPIS_FIELDS << endl;
				cout << "+ Number of MPI processes for PARTICLES: " << params->mpi.MPIS_PARTICLES << endl;
				cout << "* * * * * * * * * * * * MPI TOPOLOGY FOR FIELDS GENERATED  * * * * * * * * * * * * * * * * * *" << endl << endl;
			}
		}
        else
        {
			cerr << "ERROR: MPI topology could not be created!" << endl;

			MPI_Barrier(MPI_COMM_WORLD);
			MPI_Abort(MPI_COMM_WORLD,-102);
		}
	}

}

// Implement methods: finalizeCommunications
// ==================================================================================================================================
void MPI_MAIN::finalizeCommunications(simulationParameters * params){
	bool finalized = false;

	if(params->mpi.MPI_DOMAIN_NUMBER == 0)
		cout << endl << "* * * * * * * * * * * * FINALIZING MPI COMMUNICATIONS * * * * * * * * * * * * * * * * * *" << endl;

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Comm_free(&params->mpi.COMM);

	MPI_Finalize();
	int temp;
	MPI_Finalized(&temp);
	finalized = (bool)temp;

	if(finalized)
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER << " FINALIZED" << endl;
	else
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER << " NOT FINALIZED - ERROR" << endl;

	if(params->mpi.MPI_DOMAIN_NUMBER == 0)
		cout << "* * * * * * * * * * * * MPI COMMUNICATIONS FINALIZED * * * * * * * * * * * * * * * * * *" << endl;
}
