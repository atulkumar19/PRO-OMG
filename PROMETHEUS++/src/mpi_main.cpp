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
	cout << "Hello from thread " << thread << " in MPI process " << params->mpi.MPI_DOMAIN_NUMBER << " of " << params->mpi.NUMBER_MPI_DOMAINS << " MPI processes\n";
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
	int topo_status;

	if(params->dimensionality == 1){
		ndims = 1;

		params->mpi.MPI_DOMAINS_ALONG_X_AXIS = params->mpi.NUMBER_MPI_DOMAINS;
		params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = 1;
		params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

		params->NX_IN_SIM = params->NX_PER_MPI*params->mpi.MPI_DOMAINS_ALONG_X_AXIS;
		params->NY_IN_SIM = 1;
		params->NZ_IN_SIM = 1;

		MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_1D, periods_1D, reorder, &params->mpi.MPI_TOPO);
	}else{
		ndims = 2;

		double n(0.0); // Exponent of 2^n
		double x;

		do{
			x = ((double)params->mpi.NUMBER_MPI_DOMAINS)/2.0;
			n += 1.0;
		} while(x > 1.0);

		// We check whether n is a even number or an odd number
		if( fmod(n, 2.0) > 0.0 ){	// n is an odd number
			// The Cartesian topology of MPIs will be of size 2 x 2^(n-1),
			// where 2 MPI processes are used along the direction with less nodes.
			if(params->NX_PER_MPI > params->NY_PER_MPI){
				if(params->mpi.NUMBER_MPI_DOMAINS > 2){
					dims_2D[0] = (int)pow(2.0, n-1.0); 	// x-axis
					dims_2D[1] = 2;						// y-axis
				}else{
					dims_2D[0] = 2; 	// x-axis
					dims_2D[1] = 1;						// y-axis
				}
			}else{
				if(params->mpi.NUMBER_MPI_DOMAINS > 2){
					dims_2D[0] = 2;						// x-axis
					dims_2D[1] = (int)pow(2.0, n-1.0);	// y-axis
				}else{
					dims_2D[0] = 1;						// x-axis
					dims_2D[1] = 2;	// y-axis
				}

			}
		}else{ // n is an even number
			// The Cartesian topology of MPIs will be of size 2^(n/2) x 2^(n/2).
			dims_2D[0] = (int)pow(2.0, n/2.0); 	// x-axis
			dims_2D[1] = (int)pow(2.0, n/2.0); 	// y-axis
		}

		params->mpi.MPI_DOMAINS_ALONG_X_AXIS = dims_2D[0];
		params->mpi.MPI_DOMAINS_ALONG_Y_AXIS = dims_2D[1];
		params->mpi.MPI_DOMAINS_ALONG_Z_AXIS = 1;

		params->NX_IN_SIM = params->NX_PER_MPI*params->mpi.MPI_DOMAINS_ALONG_X_AXIS;
		params->NY_IN_SIM = params->NY_PER_MPI*params->mpi.MPI_DOMAINS_ALONG_Y_AXIS;
		params->NZ_IN_SIM = 1;

		MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_2D, periods_2D, reorder, &params->mpi.MPI_TOPO);
	}


	MPI_Topo_test(params->mpi.MPI_TOPO, &topo_status);

	if(topo_status == MPI_CART){
		MPI_Comm_rank(params->mpi.MPI_TOPO, &params->mpi.MPI_DOMAIN_NUMBER_CART);

		MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.MPI_DOMAIN_NUMBER_CART, ndims, &coord);

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

		if (params->mpi.MPI_DOMAIN_NUMBER_CART == 0){
			cout << endl << "* * * * * * * * * * * * GENERATING MPI TOPOLOGY * * * * * * * * * * * * * * * * * *" << endl;
			if(params->dimensionality == 1){
				cout << "+ Number of MPI processes along the x-axis: " << dims_1D[0] << endl;
				cout << "+ Number of mesh nodes along x-axis: " << params->NX_IN_SIM << endl;
			}else{
				cout << "+ Number of MPI processes along the x-axis: " << dims_2D[0] << endl;
				cout << "+ Number of mesh nodes along x-axis: " << params->NX_IN_SIM << endl;
				cout << "+ Number of MPI processes along the y-axis: " << dims_2D[1] << endl;
				cout << "+ Number of mesh nodes along y-axis: " << params->NY_IN_SIM << endl;
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
		cout << "\n* * * * * * * * * * * * FINALIZING MPI COMMUNICATIONS * * * * * * * * * * * * * * * * * *\n";

	MPI_Barrier(params->mpi.MPI_TOPO);

	MPI_Finalize();

	finalized = MPI::Is_finalized();

	if(finalized)
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " FINALIZED\n";
	else
		cout << "MPI process: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " NOT FINALIZED - ERROR\n";

	if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		cout << "* * * * * * * * * * * * MPI COMMUNICATIONS FINALIZED * * * * * * * * * * * * * * * * * *\n";
}
