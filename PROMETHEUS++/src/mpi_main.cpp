#include "mpi_main.h"

void MPI_MAIN::broadcastCharacteristicScales(inputParameters * params,characteristicScales * CS){

	// Define MPI type for characteristicScales structure.
	int numStructElem(10);
	int structLength[10] = {1,1,1,1,1,1,1,1,1,1};
	MPI_Aint displ[10];
	MPI_Datatype csTypes[10] = {MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE,MPI_DOUBLE};
	MPI_Datatype MPI_CS;

	MPI_Get_address(&CS->time,&displ[0]);
	MPI_Get_address(&CS->velocity,&displ[1]);
	MPI_Get_address(&CS->length,&displ[2]);
	MPI_Get_address(&CS->mass,&displ[3]);
	MPI_Get_address(&CS->charge,&displ[4]);
	MPI_Get_address(&CS->density,&displ[5]);
	MPI_Get_address(&CS->eField,&displ[6]);
	MPI_Get_address(&CS->bField,&displ[7]);
	MPI_Get_address(&CS->pressure,&displ[8]);
	MPI_Get_address(&CS->temperature,&displ[9]);

	for(int ii=numStructElem-1;ii>=0;ii--)
		displ[ii] -= displ[0];

	MPI_Type_create_struct(numStructElem,structLength,displ,csTypes,&MPI_CS);
	MPI_Type_commit(&MPI_CS);

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Bcast(CS,1,MPI_CS,0,MPI_COMM_WORLD);

	MPI_Type_free(&MPI_CS);
}


void MPI_MAIN::mpi_function(inputParameters * params){

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


void MPI_MAIN::createMPITopology(inputParameters * params){

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
//		cout << "Coordinate and rank " << params->mpi.MPI_DOMAIN_NUMBER << '\t' \
//		<< params->mpi.rank_cart << " coordinate " << coord << " left & right " \
//		<< params->mpi.lRank << '\t' << params->mpi.rRank << '\n';
	}

}
