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

void MPI_MAIN::mpi_function(simulationParameters * params){

	int nthreads,thread;

	#pragma omp parallel private(nthreads,thread)
	{
	#pragma omp critical
	{
	nthreads = omp_get_num_threads();
	thread = omp_get_thread_num();
	cout << "Hello from thread " << thread << " in MPI process " << params->mpi.MPI_DOMAIN_NUMBER << " of " << params->mpi.NUMBER_MPI_DOMAINS << " MPI processes" << endl;
	}
	}

	MPI_Barrier(params->mpi.MPI_TOPO);
	MPI_Abort(params->mpi.MPI_TOPO,-7);
}


void MPI_MAIN::createMPITopology(simulationParameters * params){
	int ndims;
	int dims_1D[1] = {params->mpi.NUMBER_MPI_DOMAINS};
	int dims_2D[2];
	int reorder(0);
	int periods_1D[1] = {1};
	int periods_2D[2] = {1, 1};
	int src;
	int coord;
	int coords_1D[1];
	int coords_2D[2];
	int topo_status;

	if(params->dimensionality == 1){
		ndims = 1;

		params->mpi.MPI_DOMAINS_ALONG_X_AXIS = params->mpi.NUMBER_MPI_DOMAINS;
		params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = 1;
		params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

		MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_1D, periods_1D, reorder, &params->mpi.MPI_TOPO);
	}else{
		ndims = 2;

		if(params->mesh.SPLIT_DIRECTION == 0){
			dims_2D[0] = params->mpi.NUMBER_MPI_DOMAINS; 	// x-axis
			dims_2D[1] = 1; 								// y-axis
		}else{
			dims_2D[0] = 1; 								// x-axis
			dims_2D[1] = params->mpi.NUMBER_MPI_DOMAINS;	// y-axis
		}

		params->mpi.MPI_DOMAINS_ALONG_X_AXIS = dims_2D[0];
		params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = dims_2D[1];
		params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

		MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_2D, periods_2D, reorder, &params->mpi.MPI_TOPO);
	}


	MPI_Topo_test(params->mpi.MPI_TOPO, &topo_status);

	if (topo_status == MPI_CART){
		MPI_Comm_rank(params->mpi.MPI_TOPO, &params->mpi.MPI_DOMAIN_NUMBER_CART);

		if (params->dimensionality == 1){ // 1-D
			MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.MPI_DOMAIN_NUMBER_CART, ndims, params->mpi.MPI_CART_COORDS_1D);
		}else{ // 2-D
			MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.MPI_DOMAIN_NUMBER_CART, ndims, params->mpi.MPI_CART_COORDS_2D);
		}

		for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
			int * COORDS = new int[2*params->mpi.NUMBER_MPI_DOMAINS];

			MPI_Allgather(&params->mpi.MPI_CART_COORDS_2D, 2, MPI_INT, COORDS, 2, MPI_INT, params->mpi.MPI_TOPO);

			params->mpi.MPI_CART_COORDS.push_back(new int[2]);
			*(params->mpi.MPI_CART_COORDS.at(mpis)) = *(COORDS + 2*mpis);
			*(params->mpi.MPI_CART_COORDS.at(mpis) + 1) = *(COORDS + 2*mpis + 1);
		}

		src = params->mpi.MPI_DOMAIN_NUMBER_CART;
		MPI_Cart_shift(params->mpi.MPI_TOPO, 0, 1, &src, &params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART);

		src = params->mpi.MPI_DOMAIN_NUMBER_CART;
		MPI_Cart_shift(params->mpi.MPI_TOPO, 0, -1, &src, &params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART);

		if(params->dimensionality == 2){
			src = params->mpi.MPI_DOMAIN_NUMBER_CART;
			MPI_Cart_shift(params->mpi.MPI_TOPO, 1, 1, &src, &params->mpi.UP_MPI_DOMAIN_NUMBER_CART);

			src = params->mpi.MPI_DOMAIN_NUMBER_CART;
			MPI_Cart_shift(params->mpi.MPI_TOPO, 1, -1, &src, &params->mpi.DOWN_MPI_DOMAIN_NUMBER_CART);
		}

		if(params->dimensionality == 1){
			params->mpi.iIndex = params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1;
			params->mpi.fIndex = params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1);
		}else{
			params->mpi.irow = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART))*params->mesh.NX_PER_MPI + 1;
			params->mpi.frow = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)) + 1)*params->mesh.NX_PER_MPI;
			params->mpi.icol = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1)*params->mesh.NY_PER_MPI + 1;
			params->mpi.fcol = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1) + 1)*params->mesh.NY_PER_MPI;
		}

		if (params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
			cout << endl << "* * * * * * * * * * * * GENERATING MPI TOPOLOGY * * * * * * * * * * * * * * * * * *" << endl;
			if(params->dimensionality == 1){
				cout << "+ Number of MPI processes along the x-axis: " << dims_1D[0] << endl;
				cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
			}else{
				cout << "+ Number of MPI processes along the x-axis: " << dims_2D[0] << endl;
				cout << "+ Number of mesh nodes along x-axis: " << params->mesh.NX_IN_SIM << endl;
				cout << "+ Number of MPI processes along the y-axis: " << dims_2D[1] << endl;
				cout << "+ Number of mesh nodes along y-axis: " << params->mesh.NY_IN_SIM << endl;
			}
			cout << "* * * * * * * * * * * * MPI TOPOLOGY GENERATED  * * * * * * * * * * * * * * * * * *" << endl << endl;
		}
	}else{
		cerr << "ERROR: MPI topology could not be created!" << endl;

		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Abort(MPI_COMM_WORLD,-102);
	}
}


void MPI_MAIN::finalizeCommunications(simulationParameters * params){
	bool finalized = false;

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "\n* * * * * * * * * * * * FINALIZING MPI COMMUNICATIONS * * * * * * * * * * * * * * * * * *" << endl;

	MPI_Barrier(params->mpi.MPI_TOPO);

	MPI_Finalize();

	finalized = MPI::Is_finalized();

	if(finalized)
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " FINALIZED" << endl;
	else
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " NOT FINALIZED - ERROR" << endl;

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "* * * * * * * * * * * * MPI COMMUNICATIONS FINALIZED * * * * * * * * * * * * * * * * * *" << endl;
}
