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

	MPI_Barrier(params->mpi.mpi_topo);
	MPI_Abort(params->mpi.mpi_topo,-7);
}


void MPI_MAIN::createMPITopology(simulationParameters * params){

	int ndims(1), dims[1] = {params->mpi.NUMBER_MPI_DOMAINS};
	int reorder(0), periods[1] = {1};
	int src, coord, topo_status;

	MPI_Cart_create(MPI_COMM_WORLD,ndims,dims,periods,reorder,&params->mpi.mpi_topo);

	MPI_Topo_test(params->mpi.mpi_topo,&topo_status);

	if(topo_status == MPI_CART){
		MPI_Comm_rank(params->mpi.mpi_topo,&params->mpi.rank_cart);
		MPI_Cart_coords(params->mpi.mpi_topo,params->mpi.rank_cart,ndims,&coord);
		src = params->mpi.rank_cart;
		MPI_Cart_shift(params->mpi.mpi_topo,0,1,&src,&params->mpi.rRank);
		src = params->mpi.rank_cart;
		MPI_Cart_shift(params->mpi.mpi_topo,0,-1,&src,&params->mpi.lRank);
/*		cout << "Coordinate and rank " << params->mpi.MPI_DOMAIN_NUMBER << '\t' \
		<< params->mpi.rank_cart << " coordinate " << coord << " left & right " \
		<< params->mpi.lRank << '\t' << params->mpi.rRank << '\n';
*/
	}
}


void MPI_MAIN::finalizeCommunications(simulationParameters * params){
	bool finalized = false;

	if(params->mpi.rank_cart == 0)
		cout << "\n* * * * * * * * * * * * FINALIZING MPI COMMUNICATIONS * * * * * * * * * * * * * * * * * *\n";

	MPI_Barrier(params->mpi.mpi_topo);

	MPI_Finalize();

	finalized = MPI::Is_finalized();

	if(finalized)
		cout << "MPI process: " << params->mpi.rank_cart << " FINALIZED\n";
	else
		cout << "MPI process: " << params->mpi.rank_cart << " NOT FINALIZED - ERROR\n";

	if(params->mpi.rank_cart == 0)
		cout << "* * * * * * * * * * * * MPI COMMUNICATIONS FINALIZED * * * * * * * * * * * * * * * * * *\n";
}
