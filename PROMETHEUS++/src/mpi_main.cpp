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
	int dims_1D[1];
	int dims_2D[2];
	int * dims_ptr;
	int reorder(0);
	int periods_1D[1] = {1};
	int periods_2D[2] = {1, 1};
	int * periods_ptr;
	// int coord_1D[1]; //*** @tocheck
	// int coord_2D[2]; //*** @tocheck
	// int * coord_prt; //*** @tocheck
	// int coords; //*** @tocheck
	int topo_status;

	MPI_Comm_rank(MPI_COMM_WORLD, &params->mpi.rank); // Get MPI rank in new topology

	// A Cartesian, periodic topology is generated in 1-D or 2-D
	// In this Cartesian topology direction 0 is along the x-axis, and direction 1 is along
	// the y-axis. Left and right are along the x-axis, and up and down along the y-direction.
	if(params->dimensionality == 1){
		ndims = 1;
		dims_ptr = dims_1D;
		periods_ptr = periods_1D;
		// coord_prt = coord_1D; //*** @tocheck

		dims_1D[0] = params->mpi.NUMBER_MPI_DOMAINS;
	}else{
		double n(0.0); // Exponent of 2^n
		double x;

		ndims = 2;
		dims_ptr = dims_2D;
		periods_ptr = periods_2D;
		// coord_prt = coord_2D; //*** @tocheck

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
	}

	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims_ptr, periods_ptr, reorder, &params->mpi.MPI_TOPO);

	MPI_Topo_test(params->mpi.MPI_TOPO, &topo_status);

	if(topo_status == MPI_CART){
		MPI_Comm_rank(params->mpi.MPI_TOPO, &params->mpi.rank_cart); // Get MPI rank in new topology

		// MPI_Cart_coords(params->mpi.MPI_TOPO, params->mpi.rank_cart, ndims, &coords); // Get MPI rank in new topology

		MPI_Cart_shift(params->mpi.MPI_TOPO, 0, 1, &params->mpi.rank_cart, &params->mpi.rRank);

		MPI_Cart_shift(params->mpi.MPI_TOPO, 0, -1, &params->mpi.rank_cart, &params->mpi.lRank);

		if(ndims == 2){
			MPI_Cart_shift(params->mpi.MPI_TOPO, 1, 1, &params->mpi.rank_cart, &params->mpi.uRank);

			MPI_Cart_shift(params->mpi.MPI_TOPO, 1, -1, &params->mpi.rank_cart, &params->mpi.dRank);
		}

		//cout << "+ MPI: " << params->mpi.rank_cart << " | L: " << params->mpi.lRank << " | R: " << params->mpi.rRank \
		//		<< " | U: " << params->mpi.uRank << " | D: " << params->mpi.dRank << endl;

		if (params->mpi.rank_cart == 0){
			cout << endl << "* * * * * * * * * * * * GENERATING MPI TOPOLOGY * * * * * * * * * * * * * * * * * *" << endl;
			if(params->dimensionality == 1){
				cout << "+ Number of MPI processes along the x-axis: " << dims_1D[0] << endl;
			}else{
				cout << "+ Number of MPI processes along the x-axis: " << dims_2D[0] << endl;
				cout << "+ Number of MPI processes along the y-axis: " << dims_2D[1] << endl;
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

	if(params->mpi.rank_cart == 0)
		cout << "\n* * * * * * * * * * * * FINALIZING MPI COMMUNICATIONS * * * * * * * * * * * * * * * * * *\n";

	MPI_Barrier(params->mpi.MPI_TOPO);

	MPI_Finalize();

	finalized = MPI::Is_finalized();

	if(finalized)
		cout << "MPI process: " << params->mpi.rank_cart << " FINALIZED\n";
	else
		cout << "MPI process: " << params->mpi.rank_cart << " NOT FINALIZED - ERROR\n";

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * MPI COMMUNICATIONS FINALIZED * * * * * * * * * * * * * * * * * *\n";
}
