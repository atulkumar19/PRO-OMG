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

#include "PIC.h"

PIC::PIC(){

}

void PIC::MPI_BcastDensity(const simulationParameters * params, ionSpecies * IONS){

	arma::vec nSend = zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);
	arma::vec nRecv = zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);
	nSend = IONS->n;

	MPI_ARMA_VEC mpi_n(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);

	MPI_Barrier(params->mpi.mpi_topo);

	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(nSend.memptr(), 1, mpi_n.type, params->mpi.rRank, 0, nRecv.memptr(), 1, mpi_n.type, params->mpi.lRank, 0, params->mpi.mpi_topo, MPI_STATUS_IGNORE);
		IONS->n += nRecv;
		nSend = nRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::MPI_BcastBulkVelocity(const simulationParameters * params, ionSpecies * IONS){

	arma::vec bufSend = zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);
	arma::vec bufRecv = zeros(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);

	MPI_ARMA_VEC mpi_n(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS+2);

	MPI_Barrier(params->mpi.mpi_topo);

	//x-component
	bufSend = IONS->nv.X;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(), 1, mpi_n.type, params->mpi.rRank, 0, bufRecv.memptr(), 1, mpi_n.type, params->mpi.lRank, 0, params->mpi.mpi_topo, MPI_STATUS_IGNORE);
		IONS->nv.X += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);

	//x-component
	bufSend = IONS->nv.Y;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(), 1, mpi_n.type, params->mpi.rRank, 0, bufRecv.memptr(), 1, mpi_n.type, params->mpi.lRank, 0, params->mpi.mpi_topo, MPI_STATUS_IGNORE);
		IONS->nv.Y += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);

	//x-component
	bufSend = IONS->nv.Z;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(), 1, mpi_n.type, params->mpi.rRank, 0, bufRecv.memptr(), 1, mpi_n.type, params->mpi.lRank, 0, params->mpi.mpi_topo, MPI_STATUS_IGNORE);
		IONS->nv.Z += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::MPI_AllgatherField(const simulationParameters * params, vfield_vec * field){

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.rank_cart+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.rank_cart+1));
	arma::vec recvBuf(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->NX_PER_MPI);

	MPI_Barrier(params->mpi.mpi_topo);

	//Allgather for x-component
	sendBuf = field->X.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.mpi_topo);
	field->X.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.mpi_topo);

	//Allgather for y-component
	sendBuf = field->Y.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.mpi_topo);
	field->Y.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.mpi_topo);

	//Allgather for z-component
	sendBuf = field->Z.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.mpi_topo);
	field->Z.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::MPI_AllgatherField(const simulationParameters * params, arma::vec * field){

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.rank_cart+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.rank_cart+1));
	arma::vec recvBuf(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->NX_PER_MPI);

	MPI_Barrier(params->mpi.mpi_topo);

	//Allgather for x-component
	sendBuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.mpi_topo);
	field->subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::smooth_TOS(arma::vec * v, double as){

	//Step 1: Averaging process
	int NX(v->n_elem);
	int N(v->n_elem + 2);
	arma::vec b = zeros(NX+2);
	double w0(23.0/48.0);
	double w1(0.25);
	double w2(1.0/96.0);//weights

	b.subvec(2, N-3) = v->subvec(1, NX-2);

	forwardPBC_1D_TOS(&b);

	b.subvec(2, N-3) = w0*b.subvec(2, N-3) + w1*b.subvec(3, N-2) + w1*b.subvec(1, N-4) + w2*b.subvec(4, N-1) + w2*b.subvec(0, N-5);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1-as)*v->subvec(1, NX-2) + as*b.subvec(2, N-3);
}


void PIC::smooth_TOS(vfield_vec * vf, double as){

	int NX(vf->X.n_elem),  N(vf->X.n_elem + 2);
	arma::vec b = zeros(NX+2);
	double w0(23.0/48.0),  w1(0.25),  w2(1.0/96.0);//weights

	//Step 1: Averaging process
	b.subvec(2, N-3) = vf->X.subvec(1, NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2, N-3) = w0*b.subvec(2, N-3) + w1*b.subvec(3, N-2) + w1*b.subvec(1, N-4) + w2*b.subvec(4, N-1) + w2*b.subvec(0, N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->X.subvec(1, NX-2) = (1-as)*vf->X.subvec(1, NX-2) + as*b.subvec(2, N-3);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(2, N-3) = vf->Y.subvec(1, NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2, N-3) = w0*b.subvec(2, N-3) + w1*b.subvec(3, N-2) + w1*b.subvec(1, N-4) + w2*b.subvec(4, N-1) + w2*b.subvec(0, N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->Y.subvec(1, NX-2) = (1-as)*vf->Y.subvec(1, NX-2) + as*b.subvec(2, N-3);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(2, N-3) = vf->Z.subvec(1, NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2, N-3) = w0*b.subvec(2, N-3) + w1*b.subvec(3, N-2) + w1*b.subvec(1, N-4) + w2*b.subvec(4, N-1) + w2*b.subvec(0, N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->Z.subvec(1, NX-2) = (1-as)*vf->Z.subvec(1, NX-2) + as*b.subvec(2, N-3);
}


void PIC::smooth_TOS(vfield_mat * vf, double as){

}


void PIC::smooth_TSC(arma::vec * v, double as){

	//Step 1: Averaging process

	int NX(v->n_elem);
	arma::vec b = zeros(NX);
	double w0(0.75),  w1(0.125);//weights

	b.subvec(1, NX-2) = v->subvec(1, NX-2);
	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1-as)*v->subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void PIC::smooth_TSC(vfield_vec * vf, double as){

	int NX(vf->X.n_elem);
	arma::vec b = zeros(NX);
	double w0(0.75);
	double w1(0.125);//weights


	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->X.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(1, NX-2) = (1-as)*vf->X.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Y.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(1, NX-2) = (1-as)*vf->Y.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Z.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(1, NX-2) = (1-as)*vf->Z.subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void PIC::smooth_TSC(vfield_mat * vf, double as){

}


void PIC::smooth(arma::vec * v, double as){

	//Step 1: Averaging process

	int NX(v->n_elem);
	arma::vec b = zeros(NX);
	double w0(0.5);
	double w1(0.25);//weights

	b.subvec(1, NX-2) = v->subvec(1, NX-2);
	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1-as)*v->subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void PIC::smooth(vfield_vec * vf, double as){

	int NX(vf->X.n_elem);
	arma::vec b = zeros(NX);
	double w0(0.5);
	double w1(0.25);//weights


	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->X.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(1, NX-2) = (1-as)*vf->X.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Y.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(1, NX-2) = (1-as)*vf->Y.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Z.subvec(1, NX-2);
	forwardPBC_1D(&b);
	b.subvec(1, NX-2) = w0*b.subvec(1, NX-2) + w1*b.subvec(2, NX-1) + w1*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(1, NX-2) = (1-as)*vf->Z.subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void PIC::smooth(vfield_mat * vf, double as){

}


void PIC::assignCell_TOS(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS, int dim){
	//This function assigns the particles to the closest mesh node depending in their position and
	//calculate the weights for the charge extrapolation and force interpolation

	//wxc = 23/48 - (x/H)^2/4
	//wxr = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxrr = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12
	//wxl = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxll = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12

	#define CC1 23.0/48.0
	#define CC2 0.25
	#define CC3 1.0/6.0
	#define CC4 2.5
	#define CC5 0.5
	#define CC6 1.0/12.0
	#define CC7 3.5
	#define CC8 3.0/4.0

	switch (dim){
		case(1):{
			int ii;
			int NSP(IONS->NSP);//number of superparticles
			int NC(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);//number of grid cells
			arma::vec X;
			uvec LOGIC;

			#pragma omp parallel shared(mesh, IONS, X, LOGIC) private(ii) firstprivate(NSP, NC)
			{
				#pragma omp for
				for(ii=0;ii<NSP;ii++)
					IONS->meshNode(ii) = floor((IONS->X(ii, 0) + 0.5*mesh->DX)/mesh->DX);

				#pragma omp single
				{
					IONS->wxc = zeros(NSP);
					IONS->wxl = zeros(NSP);
					IONS->wxr = zeros(NSP);
					IONS->wxll = zeros(NSP);
					IONS->wxrr = zeros(NSP);
					X = zeros(NSP);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(IONS->meshNode(ii) != NC){
						X(ii) = IONS->X(ii, 0) - mesh->nodes.X(IONS->meshNode(ii));
					}else{
						X(ii) =  IONS->X(ii, 0) - (mesh->nodes.X(NC-1) + mesh->DX);
					}
				}

				#pragma omp single
				{
					LOGIC = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
					X = abs(X);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					IONS->wxc(ii) = CC1 - CC2*(X(ii)/mesh->DX)*(X(ii)/mesh->DX);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(LOGIC(ii) == 1){
						IONS->wxl(ii) = CC3*((mesh->DX + X(ii))/mesh->DX - 1)*((mesh->DX + X(ii))/mesh->DX - CC4)*((mesh->DX + X(ii))/mesh->DX + CC5) + CC2;
						IONS->wxr(ii) = CC3*((mesh->DX - X(ii))/mesh->DX - 1)*((mesh->DX - X(ii))/mesh->DX - CC4)*((mesh->DX - X(ii))/mesh->DX + CC5) + CC2;
						IONS->wxll(ii) = CC6*(CC7 - (2*mesh->DX + X(ii))/mesh->DX)*((2 - (2*mesh->DX + X(ii))/mesh->DX)*(2 - (2*mesh->DX + X(ii))/mesh->DX) + CC8) - CC6;
						IONS->wxrr(ii) = CC6*(CC7 - (2*mesh->DX - X(ii))/mesh->DX)*((2 - (2*mesh->DX - X(ii))/mesh->DX)*(2 - (2*mesh->DX - X(ii))/mesh->DX) + CC8) - CC6;
					}else{
						IONS->wxl(ii) = CC3*((mesh->DX - X(ii))/mesh->DX - 1)*((mesh->DX - X(ii))/mesh->DX - CC4)*((mesh->DX - X(ii))/mesh->DX + CC5) + CC2;
						IONS->wxr(ii) = CC3*((mesh->DX + X(ii))/mesh->DX - 1)*((mesh->DX + X(ii))/mesh->DX - CC4)*((mesh->DX + X(ii))/mesh->DX + CC5) + CC2;
						IONS->wxll(ii) = CC6*(CC7 - (2*mesh->DX - X(ii))/mesh->DX)*((2 - (2*mesh->DX - X(ii))/mesh->DX)*(2 - (2*mesh->DX - X(ii))/mesh->DX) + CC8) - CC6;
						IONS->wxrr(ii) = CC6*(CC7 - (2*mesh->DX + X(ii))/mesh->DX)*((2 - (2*mesh->DX + X(ii))/mesh->DX)*(2 - (2*mesh->DX + X(ii))/mesh->DX) + CC8) - CC6;
					}
				}

			}//End of the parallel region
			break;
		}
		case(2):{
			break;
		}
		case(3):{
			break;
		}
		default:{
			std::ofstream ofs("errors/assignCell_TSC.txt", std::ofstream::out);
			ofs << "assignCell_TSC: Introduce a valid option!\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!IONS->meshNode.is_finite()){
		std::ofstream ofs("errors/assignCell_TSC.txt", std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::assignCell_TSC(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS, int dim){
	//This function assigns the particles to the closest mesh node depending in their position and
	//calculate the weights for the charge extrapolation and force interpolation

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x
	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	switch (dim){
		case(1):{
			int ii;
			int NSP(IONS->NSP);//number of superparticles
			int NC;//number of grid cells
			arma::vec X;
			uvec LOGIC;
			#pragma omp parallel shared(mesh, IONS, X, NSP, LOGIC) private(ii, NC)
			{
				#pragma omp for
				for(ii=0;ii<NSP;ii++)
					IONS->meshNode(ii) = floor((IONS->X(ii, 0) + 0.5*mesh->DX)/mesh->DX);

				NC = mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;

				#pragma omp single
				{
					IONS->wxc = zeros(NSP);
					IONS->wxl = zeros(NSP);
					IONS->wxr = zeros(NSP);
					X = zeros(NSP);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(IONS->meshNode(ii) != NC){
						X(ii) = IONS->X(ii, 0) - mesh->nodes.X(IONS->meshNode(ii));
					}else{
						X(ii) = IONS->X(ii, 0) - (mesh->nodes.X(NC-1) + mesh->DX);
					}
				}

				#pragma omp single
				{
					LOGIC = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
					X = abs(X);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					IONS->wxc(ii) = 0.75 - (X(ii)/mesh->DX)*(X(ii)/mesh->DX);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(LOGIC(ii) == 1){
						IONS->wxl(ii) = 0.5*(1.5 - (mesh->DX + X(ii))/mesh->DX)*(1.5 - (mesh->DX + X(ii))/mesh->DX);
						IONS->wxr(ii) = 0.5*(1.5 - (mesh->DX - X(ii))/mesh->DX)*(1.5 - (mesh->DX - X(ii))/mesh->DX);
					}else{
						IONS->wxl(ii) = 0.5*(1.5 - (mesh->DX - X(ii))/mesh->DX)*(1.5 - (mesh->DX - X(ii))/mesh->DX);
						IONS->wxr(ii) = 0.5*(1.5 - (mesh->DX + X(ii))/mesh->DX)*(1.5 - (mesh->DX + X(ii))/mesh->DX);
					}
				}

			}//End of the parallel region
			break;
		}
		case(2):{
			// To do something
			break;
		}
		case(3):{
			// To do something
			break;
		}
		default:{
			std::ofstream ofs("errors/assignCell_TSC.txt",std::ofstream::out);
			ofs << "assignCell_TSC: Introduce a valid option!\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!IONS->meshNode.is_finite()){
		std::ofstream ofs("errors/assignCell_TSC.txt",std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::assignCell_NNS(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS, int dim){

	switch (dim){
		case(1):{
			/*Periodic boundary condition*/
			int aux(0);
			for(int ii=0;ii<IONS->NSP;ii++){
				aux = floor(IONS->X(ii, 0)/mesh->DX);
				if(aux == mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS){
					IONS->meshNode(ii) = 0;
					IONS->X(ii) = 0;
				}else{
					IONS->meshNode(ii) = aux;
				}
				aux = 0;
			}
			/*Periodic boundary condition*/
			break;
		}
		case(2):{
			break;
		}
		case(3):{
			break;
		}
		default:{
			std::ofstream ofs("errors/assignCell_NNS.txt", std::ofstream::out);
			ofs << "ERROR: Invalid option.\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!IONS->meshNode.is_finite()){
		std::ofstream ofs("errors/assignCell_NNS.txt", std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::assignCell(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS, int dim){
	switch (params->weightingScheme){
		case(0):{
				PIC::assignCell_TOS(params, mesh, IONS, dim);
				break;
				}
		case(1):{
				PIC::assignCell_TSC(params, mesh, IONS, dim);
				break;
				}
		case(2):{
				PIC::assignCell_NNS(params, mesh, IONS, dim);
				break;
				}
		case(3):{
				PIC::assignCell_TOS(params, mesh, IONS, dim);
				break;
				}
		case(4):{
				PIC::assignCell_TSC(params, mesh, IONS, dim);
				break;
				}
		default:{
				PIC::assignCell_TSC(params, mesh, IONS, dim);
				}
	}
}


void PIC::crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB){
	if(A->n_elem != B->n_elem){
		cerr<<"\nERROR: The number of elements of A and B, unable to calculate AxB.\n";
		exit(1);
	}

	AxB->set_size(A->n_rows, 3);//Here we set up the size of the matrix AxB.

	AxB->col(0) = A->col(1)%B->col(2) - A->col(2)%B->col(1);//(AxB)_x
	AxB->col(1) = A->col(2)%B->col(0) - A->col(0)%B->col(2);//(AxB)_y
	AxB->col(2) = A->col(0)%B->col(1) - A->col(1)%B->col(0);//(AxB)_z
}


#ifdef ONED
void PIC::eivTOS_1D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	int NC(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);
	vfield_vec nv;

	IONS->nv.zeros(NC);//Setting to zero the ions' bulk velocity
	nv.zeros(NC);

	#pragma omp parallel shared(mesh, IONS) firstprivate(NC, nv) private(ii)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = IONS->meshNode(ii) + 1;
			if(ix == (NC-2)){//For the particles on the right side boundary.
				nv.X(NC-4) += IONS->wxll(ii)*IONS->V(ii,0);
				nv.X(NC-3) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(NC-2) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(NC-1) += IONS->wxr(ii)*IONS->V(ii,0);
				nv.X(0) += IONS->wxrr(ii)*IONS->V(ii,0);

				nv.Y(NC-4) += IONS->wxll(ii)*IONS->V(ii,1);
				nv.Y(NC-3) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(NC-2) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(NC-1) += IONS->wxr(ii)*IONS->V(ii,1);
				nv.Y(0) += IONS->wxrr(ii)*IONS->V(ii,1);

				nv.Z(NC-4) += IONS->wxll(ii)*IONS->V(ii,2);
				nv.Z(NC-3) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(NC-2) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(NC-1) += IONS->wxr(ii)*IONS->V(ii,2);
				nv.Z(0) += IONS->wxrr(ii)*IONS->V(ii,2);
			}else if(ix == (NC-1)){//For the particles on the right side boundary.
				nv.X(NC-3) += IONS->wxll(ii)*IONS->V(ii,0);
				nv.X(NC-2) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(NC-1) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(2) += IONS->wxr(ii)*IONS->V(ii,0);
				nv.X(3) += IONS->wxrr(ii)*IONS->V(ii,0);

				nv.Y(NC-3) += IONS->wxll(ii)*IONS->V(ii,1);
				nv.Y(NC-2) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(NC-1) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(2) += IONS->wxr(ii)*IONS->V(ii,1);
				nv.Y(3) += IONS->wxrr(ii)*IONS->V(ii,1);

				nv.Z(NC-3) += IONS->wxll(ii)*IONS->V(ii,2);
				nv.Z(NC-2) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(NC-1) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(2) += IONS->wxr(ii)*IONS->V(ii,2);
				nv.Z(3) += IONS->wxrr(ii)*IONS->V(ii,2);
			}else if(ix == 1){
				nv.X(NC-1) += IONS->wxll(ii)*IONS->V(ii,0);
				nv.X(0) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(ix) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(ix+1) += IONS->wxr(ii)*IONS->V(ii,0);
				nv.X(ix+2) += IONS->wxrr(ii)*IONS->V(ii,0);

				nv.Y(NC-1) += IONS->wxll(ii)*IONS->V(ii,1);
				nv.Y(0) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(ix) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(ix+1) += IONS->wxr(ii)*IONS->V(ii,1);
				nv.Y(ix+2) += IONS->wxrr(ii)*IONS->V(ii,1);

				nv.Z(NC-1) += IONS->wxll(ii)*IONS->V(ii,2);
				nv.Z(0) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(ix) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(ix+1) += IONS->wxr(ii)*IONS->V(ii,2);
				nv.Z(ix+2) += IONS->wxrr(ii)*IONS->V(ii,2);
			}else{
				nv.X(ix-2) += IONS->wxll(ii)*IONS->V(ii,0);
				nv.X(ix-1) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(ix) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(ix+1) += IONS->wxr(ii)*IONS->V(ii,0);
				nv.X(ix+2) += IONS->wxrr(ii)*IONS->V(ii,0);

				nv.Y(ix-2) += IONS->wxll(ii)*IONS->V(ii,1);
				nv.Y(ix-1) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(ix) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(ix+1) += IONS->wxr(ii)*IONS->V(ii,1);
				nv.Y(ix+2) += IONS->wxrr(ii)*IONS->V(ii,1);

				nv.Z(ix-2) += IONS->wxll(ii)*IONS->V(ii,2);
				nv.Z(ix-1) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(ix) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(ix+1) += IONS->wxr(ii)*IONS->V(ii,2);
				nv.Z(ix+2) += IONS->wxrr(ii)*IONS->V(ii,2);
			}
		}

		#pragma omp critical (update_bulk_velocity)
		{
		IONS->nv += nv;
		}

	}//End of the parallel region

	backwardPBC_1D(&IONS->nv.X);
	backwardPBC_1D(&IONS->nv.Y);
	backwardPBC_1D(&IONS->nv.Z);

	IONS->nv *= IONS->NCP/mesh->DX;

}
#endif


#ifdef ONED
void PIC::eivTSC_1D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NC(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);
	vfield_vec nv;
	IONS->nv.zeros(NC);//Setting to zero the ions' bulk velocity

	#pragma omp parallel shared(mesh, IONS) private(ii, NC, nv)
	{
		NC = mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2;
		nv.zeros(NC);
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = IONS->meshNode(ii) + 1;
			if(ix == (NC-1)){//For the particles on the right side boundary.
				nv.X(NC-2) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(NC-1) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(2) += IONS->wxr(ii)*IONS->V(ii,0);

				nv.Y(NC-2) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(NC-1) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(2) += IONS->wxr(ii)*IONS->V(ii,1);

				nv.Z(NC-2) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(NC-1) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(2) += IONS->wxr(ii)*IONS->V(ii,2);
			}else if(ix != (NC-1)){
				nv.X(ix-1) += IONS->wxl(ii)*IONS->V(ii,0);
				nv.X(ix) += IONS->wxc(ii)*IONS->V(ii,0);
				nv.X(ix+1) += IONS->wxr(ii)*IONS->V(ii,0);

				nv.Y(ix-1) += IONS->wxl(ii)*IONS->V(ii,1);
				nv.Y(ix) += IONS->wxc(ii)*IONS->V(ii,1);
				nv.Y(ix+1) += IONS->wxr(ii)*IONS->V(ii,1);

				nv.Z(ix-1) += IONS->wxl(ii)*IONS->V(ii,2);
				nv.Z(ix) += IONS->wxc(ii)*IONS->V(ii,2);
				nv.Z(ix+1) += IONS->wxr(ii)*IONS->V(ii,2);
			}
		}

		#pragma omp critical (update_bulk_velocity)
		{
		IONS->nv += nv;
		}

	}//End of the parallel region

	backwardPBC_1D(&IONS->nv.X);
	backwardPBC_1D(&IONS->nv.Y);
	backwardPBC_1D(&IONS->nv.Z);

	IONS->nv *= IONS->NCP/mesh->DX;

}
#endif


#ifdef TWOD
void PIC::eivTSC_2D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

}
#endif


#ifdef THREED
void PIC::eivTSC_3D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

}
#endif


void PIC::extrapolateIonVelocity(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	switch (params->weightingScheme){
		case(0):{
				#ifdef ONED
				eivTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(1):{
				#ifdef ONED
				eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eivTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				#ifdef ONED
				eivTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(4):{
				#ifdef ONED
				eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eivTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		default:{
				#ifdef ONED
				eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eivTSC_3D(params, mesh, IONS);
				#endif
				}
	}
}


#ifdef ONED
void PIC::eidTOS_1D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){
	int NC(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);//Grid cells along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);
	arma::vec n;

	IONS->n = zeros(NC);//Setting to zero the ion density.
	n.zeros(NC);

	#pragma omp parallel shared(mesh, IONS) private(ii) firstprivate(NSP, NC, n)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = IONS->meshNode(ii) + 1;
			if(ix == (NC-2)){//For the particles on the right side boundary.
				n(NC-4) += IONS->wxll(ii);
				n(NC-3) += IONS->wxl(ii);
				n(NC-2) += IONS->wxc(ii);
				n(NC-1) += IONS->wxr(ii);
				n(0) += IONS->wxrr(ii);
			}else if(ix == (NC-1)){//For the particles on the right side boundary.
				n(NC-3) += IONS->wxll(ii);
				n(NC-2) += IONS->wxl(ii);
				n(NC-1) += IONS->wxc(ii);
				n(2) += IONS->wxr(ii);
				n(3) += IONS->wxrr(ii);
			}else if(ix == 1){
				n(NC-1) += IONS->wxll(ii);
				n(0) += IONS->wxl(ii);
				n(ix) += IONS->wxc(ii);
				n(ix+1) += IONS->wxr(ii);
				n(ix+2) += IONS->wxrr(ii);
			}else{
				n(ix-2) += IONS->wxll(ii);
				n(ix-1) += IONS->wxl(ii);
				n(ix) += IONS->wxc(ii);
				n(ix+1) += IONS->wxr(ii);
				n(ix+2) += IONS->wxrr(ii);
			}
		}

		#pragma omp critical (update_density)
		{
		IONS->n += n;
		}

	}//End of the parallel region

	backwardPBC_1D(&IONS->n);
	IONS->n *= IONS->NCP/mesh->DX;
}
#endif


#ifdef ONED
void PIC::eidTSC_1D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NC(mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2);//Grid cells along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);
	arma::vec n;

	IONS->n = zeros(NC);//Setting to zero the ion density.
	n.zeros(NC);

	#pragma omp parallel shared(mesh, IONS) private(ii) firstprivate(NSP, NC, n)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = IONS->meshNode(ii) + 1;

			if(ix == (NC-1)){//For the particles on the right side boundary.
				n(NC-2) += IONS->wxl(ii);
				n(NC-1) += IONS->wxc(ii);
				n(2) += IONS->wxr(ii);
			}else if(ix != (NC-1)){
				n(ix-1) += IONS->wxl(ii);
				n(ix) += IONS->wxc(ii);
				n(ix+1) += IONS->wxr(ii);
			}
		}

		#pragma omp critical (update_density)
		{
		IONS->n += n;
		}

	}//End of the parallel region

	backwardPBC_1D(&IONS->n);
	IONS->n *= IONS->NCP/mesh->DX;

}
#endif


#ifdef TWOD
void PIC::eidTSC_2D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

}
#endif


#ifdef THREED
void PIC::eidTSC_3D(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

}
#endif


void PIC::extrapolateIonDensity(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	switch (params->weightingScheme){
		case(0):{
				#ifdef ONED
				eidTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(1):{
				#ifdef ONED
				eidTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eidTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eidTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				#ifdef ONED
				eidTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(4):{
				#ifdef ONED
				eidTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eidTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eidTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		default:{
				#ifdef ONED
				eidTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				eidTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				eidTSC_3D(params, mesh, IONS);
				#endif
				}
	}
}


#ifdef ONED
void PIC::EMF_TOS_1D(const simulationParameters * params, const ionSpecies * IONS, vfield_vec * EMF, arma::mat * F){

	//wxc = 23/48 - (x/H)^2/4
	//wxr = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxrr = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12
	//wxl = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxll = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12

	int N =  params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2;//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);

	//Contrary to what may be thought,  F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.

	#pragma omp parallel for private(ii) shared(IONS, EMF, F) firstprivate(N, NSP)
	for(ii=0;ii<NSP;ii++){
		int ix = IONS->meshNode(ii) + 1;
		if(ix == (N-2)){//For the particles on the right side boundary.
			(*F)(ii,0) += IONS->wxll(ii)*EMF->X(N-4);
			(*F)(ii,1) += IONS->wxll(ii)*EMF->Y(N-4);
			(*F)(ii,2) += IONS->wxll(ii)*EMF->Z(N-4);

			(*F)(ii,0) += IONS->wxl(ii)*EMF->X(N-3);
			(*F)(ii,1) += IONS->wxl(ii)*EMF->Y(N-3);
			(*F)(ii,2) += IONS->wxl(ii)*EMF->Z(N-3);

			(*F)(ii,0) += IONS->wxc(ii)*EMF->X(N-2);
			(*F)(ii,1) += IONS->wxc(ii)*EMF->Y(N-2);
			(*F)(ii,2) += IONS->wxc(ii)*EMF->Z(N-2);

			(*F)(ii,0) += IONS->wxr(ii)*EMF->X(N-1);
			(*F)(ii,1) += IONS->wxr(ii)*EMF->Y(N-1);
			(*F)(ii,2) += IONS->wxr(ii)*EMF->Z(N-1);

			(*F)(ii,0) += IONS->wxrr(ii)*EMF->X(0);
			(*F)(ii,1) += IONS->wxrr(ii)*EMF->Y(0);
			(*F)(ii,2) += IONS->wxrr(ii)*EMF->Z(0);
		}else if(ix == (N-1)){//For the particles on the right side boundary.
			(*F)(ii,0) += IONS->wxll(ii)*EMF->X(N-3);
			(*F)(ii,1) += IONS->wxll(ii)*EMF->Y(N-3);
			(*F)(ii,2) += IONS->wxll(ii)*EMF->Z(N-3);

			(*F)(ii,0) += IONS->wxl(ii)*EMF->X(N-2);
			(*F)(ii,1) += IONS->wxl(ii)*EMF->Y(N-2);
			(*F)(ii,2) += IONS->wxl(ii)*EMF->Z(N-2);

			(*F)(ii,0) += IONS->wxc(ii)*EMF->X(N-1);
			(*F)(ii,1) += IONS->wxc(ii)*EMF->Y(N-1);
			(*F)(ii,2) += IONS->wxc(ii)*EMF->Z(N-1);

			(*F)(ii,0) += IONS->wxr(ii)*EMF->X(2);
			(*F)(ii,1) += IONS->wxr(ii)*EMF->Y(2);
			(*F)(ii,2) += IONS->wxr(ii)*EMF->Z(2);

			(*F)(ii,0) += IONS->wxrr(ii)*EMF->X(3);
			(*F)(ii,1) += IONS->wxrr(ii)*EMF->Y(3);
			(*F)(ii,2) += IONS->wxrr(ii)*EMF->Z(3);
		}else if(ix == 1){
			(*F)(ii,0) += IONS->wxll(ii)*EMF->X(N-1);
			(*F)(ii,1) += IONS->wxll(ii)*EMF->Y(N-1);
			(*F)(ii,2) += IONS->wxll(ii)*EMF->Z(N-1);

			(*F)(ii,0) += IONS->wxl(ii)*EMF->X(0);
			(*F)(ii,1) += IONS->wxl(ii)*EMF->Y(0);
			(*F)(ii,2) += IONS->wxl(ii)*EMF->Z(0);

			(*F)(ii,0) += IONS->wxc(ii)*EMF->X(ix);
			(*F)(ii,1) += IONS->wxc(ii)*EMF->Y(ix);
			(*F)(ii,2) += IONS->wxc(ii)*EMF->Z(ix);

			(*F)(ii,0) += IONS->wxr(ii)*EMF->X(ix+1);
			(*F)(ii,1) += IONS->wxr(ii)*EMF->Y(ix+1);
			(*F)(ii,2) += IONS->wxr(ii)*EMF->Z(ix+1);

			(*F)(ii,0) += IONS->wxrr(ii)*EMF->X(ix+2);
			(*F)(ii,1) += IONS->wxrr(ii)*EMF->Y(ix+2);
			(*F)(ii,2) += IONS->wxrr(ii)*EMF->Z(ix+2);
		}else{
			(*F)(ii,0) += IONS->wxll(ii)*EMF->X(ix-2);
			(*F)(ii,1) += IONS->wxll(ii)*EMF->Y(ix-2);
			(*F)(ii,2) += IONS->wxll(ii)*EMF->Z(ix-2);

			(*F)(ii,0) += IONS->wxl(ii)*EMF->X(ix-1);
			(*F)(ii,1) += IONS->wxl(ii)*EMF->Y(ix-1);
			(*F)(ii,2) += IONS->wxl(ii)*EMF->Z(ix-1);

			(*F)(ii,0) += IONS->wxc(ii)*EMF->X(ix);
			(*F)(ii,1) += IONS->wxc(ii)*EMF->Y(ix);
			(*F)(ii,2) += IONS->wxc(ii)*EMF->Z(ix);

			(*F)(ii,0) += IONS->wxr(ii)*EMF->X(ix+1);
			(*F)(ii,1) += IONS->wxr(ii)*EMF->Y(ix+1);
			(*F)(ii,2) += IONS->wxr(ii)*EMF->Z(ix+1);

			(*F)(ii,0) += IONS->wxrr(ii)*EMF->X(ix+2);
			(*F)(ii,1) += IONS->wxrr(ii)*EMF->Y(ix+2);
			(*F)(ii,2) += IONS->wxrr(ii)*EMF->Z(ix+2);
		}
	}//End of the parallel region
}
#endif


#ifdef ONED
void PIC::EMF_TSC_1D(const simulationParameters * params, const ionSpecies * IONS, vfield_vec * emf, arma::mat * F){
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int N =  params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2;//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	int ii(0);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for private(ii) shared(N, NSP, params, IONS, emf, F)
	for(ii=0;ii<NSP;ii++){
		int ix = IONS->meshNode(ii) + 1;
		if(ix == (N-1)){//For the particles on the right side boundary.
			(*F)(ii,0) += IONS->wxl(ii)*emf->X(ix-1);
			(*F)(ii,1) += IONS->wxl(ii)*emf->Y(ix-1);
			(*F)(ii,2) += IONS->wxl(ii)*emf->Z(ix-1);

			(*F)(ii,0) += IONS->wxc(ii)*emf->X(ix);
			(*F)(ii,1) += IONS->wxc(ii)*emf->Y(ix);
			(*F)(ii,2) += IONS->wxc(ii)*emf->Z(ix);

			(*F)(ii,0) += IONS->wxr(ii)*emf->X(2);
			(*F)(ii,1) += IONS->wxr(ii)*emf->Y(2);
			(*F)(ii,2) += IONS->wxr(ii)*emf->Z(2);
		}else{
			(*F)(ii,0) += IONS->wxl(ii)*emf->X(ix-1);
			(*F)(ii,1) += IONS->wxl(ii)*emf->Y(ix-1);
			(*F)(ii,2) += IONS->wxl(ii)*emf->Z(ix-1);

			(*F)(ii,0) += IONS->wxc(ii)*emf->X(ix);
			(*F)(ii,1) += IONS->wxc(ii)*emf->Y(ix);
			(*F)(ii,2) += IONS->wxc(ii)*emf->Z(ix);

			(*F)(ii,0) += IONS->wxr(ii)*emf->X(ix+1);
			(*F)(ii,1) += IONS->wxr(ii)*emf->Y(ix+1);
			(*F)(ii,2) += IONS->wxr(ii)*emf->Z(ix+1);
		}
	}//End of the parallel region

}
#endif


void PIC::interpolateElectromagneticFields_1D(const simulationParameters * params, const ionSpecies * IONS, fields * EB, arma::mat * E, arma::mat * B){
	switch (params->weightingScheme){
		case(0):{
				EMF_TOS_1D(params, IONS, &EB->E, E);
				EMF_TOS_1D(params, IONS, &EB->B, B);
				break;
				}
		case(1):{
				EMF_TSC_1D(params, IONS, &EB->E, E);
				EMF_TSC_1D(params, IONS, &EB->B, B);
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				EMF_TOS_1D(params, IONS, &EB->E, E);
				EMF_TOS_1D(params, IONS, &EB->B, B);
				break;
				}
		case(4):{
				EMF_TSC_1D(params, IONS, &EB->E, E);
				EMF_TSC_1D(params, IONS, &EB->B, B);
				break;
				}
		default:{
				EMF_TSC_1D(params, IONS, &EB->E, E);
				EMF_TSC_1D(params, IONS, &EB->B, B);
				}
	}
}


#ifdef TWOD
void PIC::EMF_TSC_2D(const meshGeometry * mesh, const ionSpecies * IONS, vfield_cube * emf, arma::mat * F){

}
#endif


#ifdef THREED
void PIC::EMF_TSC_3D(const meshGeometry * mesh, const ionSpecies * IONS, vfield_cube * emf, arma::mat * F){

}
#endif




#ifdef ONED
void PIC::aiv_Vay_1D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){


	MPI_AllgatherField(params, &EB->E);
	MPI_AllgatherField(params, &EB->B);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	fields EB_;
	EB_.zeros(NX);

	EB_.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	EB_.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	EB_.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	EB_.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	EB_.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	EB_.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	forwardPBC_1D(&EB_.E.X);
	forwardPBC_1D(&EB_.E.Y);
	forwardPBC_1D(&EB_.E.Z);

	forwardPBC_1D(&EB_.B.X);
	forwardPBC_1D(&EB_.B.Y);
	forwardPBC_1D(&EB_.B.Z);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
		arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

		interpolateElectromagneticFields_1D(params, &IONS->at(ii), &EB_, &Ep, &Bp);

		//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		int NSP(IONS->at(ii).NSP);
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)
		arma::vec gp(IONS->at(ii).NSP);
		arma::vec sigma(IONS->at(ii).NSP);
		arma::vec us(IONS->at(ii).NSP);
		arma::vec s(IONS->at(ii).NSP);
		arma::mat U(IONS->at(ii).NSP, 3);
		arma::mat VxB(IONS->at(ii).NSP, 3);
		arma::mat tau(IONS->at(ii).NSP, 3);
		arma::mat up(IONS->at(ii).NSP, 3);
		arma::mat t(IONS->at(ii).NSP, 3);
		arma::mat upxt(IONS->at(ii).NSP, 3);

		crossProduct(&IONS->at(ii).V, &Bp, &VxB);//VxB

		#pragma omp parallel shared(IONS, Ep, Bp, U, gp, sigma, us, s, VxB, tau, up, t, upxt) firstprivate(A, NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
				U.row(ip) = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

				U.row(ip) += 0.5*A*(Ep.row(ip) + VxB.row(ip)); // U_hs = U_L + 0.5*a*(E + cross(V, B)); % Half step for velocity
				tau.row(ip) = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
				up.row(ip) = U.row(ip) + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
				gp(ip) = sqrt( 1.0 + dot(up.row(ip), up.row(ip))/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
				sigma(ip) = gp(ip)*gp(ip) - dot(tau.row(ip), tau.row(ip)); // sigma = gammap^2 - tau*tau';
				us(ip) = dot(up.row(ip), tau.row(ip))/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
				IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma(ip) + sqrt( sigma(ip)*sigma(ip) + 4.0*( dot(tau.row(ip), tau.row(ip)) + us(ip)*us(ip) ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
				t.row(ip) = tau.row(ip)/IONS->at(ii).g(ip); 			// t = tau/gamma;
				s(ip) = 1.0/( 1.0 + dot(t.row(ip), t.row(ip)) ); // s = 1/(1 + t*t'); % variable 's' in paper
			}

			#pragma omp critical
			crossProduct(&up, &t, &upxt);

			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				U.row(ip) = s(ip)*( up.row(ip) + dot(up.row(ip), t.row(ip))*t.row(ip)+ upxt.row(ip) ); 	// U_L = s*(up + (up*t')*t + cross(up, t));
				IONS->at(ii).V.row(ip) = U.row(ip)/IONS->at(ii).g(ip);	// V = U_L/gamma;
			}
		} // End of parallel region

		extrapolateIonVelocity(params, mesh, &IONS->at(ii));

		PIC::MPI_BcastBulkVelocity(params, &IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(2):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(3):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(4):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv, params->smoothingParameter);
					}
		}

	}//structure to iterate over all the ion species.

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	restoreVector(&EB->E.X);
	restoreVector(&EB->E.Y);
	restoreVector(&EB->E.Z);

	restoreVector(&EB->B.X);
	restoreVector(&EB->B.Y);
	restoreVector(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

}
#endif


#ifdef TWOD
void PIC::aiv_Vay_2D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}
#endif


#ifdef THREED
void PIC::aiv_Vay_3D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}
#endif


#ifdef ONED
void PIC::aiv_Boris_1D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

	MPI_AllgatherField(params, &EB->E);
	MPI_AllgatherField(params, &EB->B);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	fields EB_;
	EB_.zeros(NX);

	EB_.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	EB_.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	EB_.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	EB_.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	EB_.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	EB_.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	forwardPBC_1D(&EB_.E.X);
	forwardPBC_1D(&EB_.E.Y);
	forwardPBC_1D(&EB_.E.Z);

	forwardPBC_1D(&EB_.B.X);
	forwardPBC_1D(&EB_.B.Y);
	forwardPBC_1D(&EB_.B.Z);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
		arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

		switch (params->weightingScheme){
			case(0):{
					EMF_TOS_1D(params, &IONS->at(ii), &EB_.E, &Ep);
					EMF_TOS_1D(params, &IONS->at(ii), &EB_.B, &Bp);
					break;
					}
			case(1):{
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.E, &Ep);
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.B, &Bp);
					break;
					}
			case(2):{
					exit(0);
					break;
					}
			case(3):{
					EMF_TOS_1D(params, &IONS->at(ii), &EB_.E, &Ep);
					EMF_TOS_1D(params, &IONS->at(ii), &EB_.B, &Bp);
					break;
					}
			case(4):{
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.E, &Ep);
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.B, &Bp);
					break;
					}
			default:{
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.E, &Ep);
					EMF_TSC_1D(params, &IONS->at(ii), &EB_.B, &Bp);
					}
		}

		//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)
		arma::vec C1;
		arma::vec C2;
		arma::vec C3;
		arma::vec C4;
		arma::vec BB;
		arma::vec VB;
		arma::vec EB;
		arma::mat ExB;
		arma::mat VxB;

		#pragma omp parallel sections shared(IONS, BB, VB, EB, ExB, VxB, Ep, Bp)
		{
			#pragma omp section
			BB = sum(Bp % Bp, 1);//B\dotB evaluated at each particle position.
			#pragma omp section
			VB = sum(IONS->at(ii).V % Bp, 1);//V\dotB
			#pragma omp section
			EB = sum(Ep % Bp, 1);//E\dotB
			#pragma omp section
			crossProduct(&Ep, &Bp, &ExB);//E\times B
			#pragma omp section
			crossProduct(&IONS->at(ii).V, &Bp, &VxB);//V\times B
		}//end of the parallel region

		C1 = ( 1.0 - (A*A)*BB/4.0 )/( 1.0 + (A*A)*BB/4.0 );
		C2 = A/( 1.0 + (A*A)*BB/4.0 );
		C3 = ((A*A)/2.0)/( 1.0 + (A*A)*BB/4.0 );
		C4 = ((A*A)/4.0)/( 1.0 + (A*A)*BB/4.0 );

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(IONS, BB, VB, EB, ExB, VxB, Ep, Bp, C1, C2, C3, C4) firstprivate(NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).V(ip,0) = C1(ip)*IONS->at(ii).V(ip,0);
				IONS->at(ii).V(ip,0) += C2(ip)*( Ep(ip,0) + VxB(ip,0) );
				IONS->at(ii).V(ip,0) += C3(ip)*( ExB(ip,0) + VB(ip)*Bp(ip,0) );
				IONS->at(ii).V(ip,0) += C4(ip)*( EB(ip)*Bp(ip,0) );
			}

			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).V(ip,1) = C1(ip)*IONS->at(ii).V(ip,1);
				IONS->at(ii).V(ip,1) += C2(ip)*( Ep(ip,1) + VxB(ip,1) );
				IONS->at(ii).V(ip,1) += C3(ip)*( ExB(ip,1) + VB(ip)*Bp(ip,1) );
				IONS->at(ii).V(ip,1) += C4(ip)*( EB(ip)*Bp(ip,1) );
			}

			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).V(ip,2) = C1(ip)*IONS->at(ii).V(ip,2);
				IONS->at(ii).V(ip,2) += C2(ip)*( Ep(ip,2) + VxB(ip,2) );
				IONS->at(ii).V(ip,2) += C3(ip)*( ExB(ip,2) + VB(ip)*Bp(ip,2) );
				IONS->at(ii).V(ip,2) += C4(ip)*( EB(ip)*Bp(ip,2) );
			}
		} //End of the parallel region

		extrapolateIonVelocity(params, mesh, &IONS->at(ii));

		PIC::MPI_BcastBulkVelocity(params, &IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(2):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(3):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(4):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv, params->smoothingParameter);
					}
		}

	}//structure to iterate over all the ion species.

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	restoreVector(&EB->E.X);
	restoreVector(&EB->E.Y);
	restoreVector(&EB->E.Z);

	restoreVector(&EB->B.X);
	restoreVector(&EB->B.Y);
	restoreVector(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

}
#endif


#ifdef TWOD
void PIC::aiv_Boris_2D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}
#endif


#ifdef THREED
void PIC::aiv_Boris_3D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}
#endif


void PIC::aip_1D(const simulationParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT){

	double lx = mesh->DX*mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;//

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(IONS) firstprivate(DT, lx, NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);

                IONS->at(ii).X(ip,0) = fmod(IONS->at(ii).X(ip,0),lx);//x

                if(IONS->at(ii).X(ip,0) < 0)
        			IONS->at(ii).X(ip,0) += lx;
			}
		}//End of the parallel region

		PIC::assignCell(params, mesh, &IONS->at(ii), 1);

		extrapolateIonDensity(params, mesh, &IONS->at(ii));//Once the ions have been pushed,  we extrapolate the density at the node grids.

		PIC::MPI_BcastDensity(params, &IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).n, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).n, params->smoothingParameter);
					break;
					}
			case(2):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n, params->smoothingParameter);
					break;
					}
			case(3):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n, params->smoothingParameter);
					break;
					}
			case(4):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n, params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n, params->smoothingParameter);
					}
		}

	}//structure to iterate over all the ion species.
}


void PIC::aip_2D(const simulationParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT){

	double lx = mesh->nodes.X(mesh->NX_PER_MPI-1) + mesh->DX;//
	double ly = mesh->nodes.Y(mesh->NY_PER_MPI-1) + mesh->DY;//

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)


		IONS->at(ii).X.col(0) += DT*IONS->at(ii).V.col(0);//x-component
		IONS->at(ii).X.col(1) += DT*IONS->at(ii).V.col(1);//y-component

		for(int jj=0;jj<IONS->at(ii).NSP;jj++){//Periodic boundary condition for the ions
			IONS->at(ii).X(jj,0) = fmod(IONS->at(ii).X(jj,0),lx);//x
			if(IONS->at(ii).X(jj,0) < 0)
				IONS->at(ii).X(jj,0) += lx;

			IONS->at(ii).X(jj,1) = fmod(IONS->at(ii).X(jj, 1), ly);//y
			if(IONS->at(ii).X(jj,1) < 0)
				IONS->at(ii).X(jj,1) += ly;
		}//Periodic boundary condition for the ions

		assignCell_TSC(params, mesh, &IONS->at(ii), 2);

		extrapolateIonDensity(params, mesh, &IONS->at(ii));//Once the ions have been pushed, we extrapolate the density at the node grids.

	}//structure to iterate over all the ion species.
}


void PIC::aip_3D(const simulationParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT){

}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

	//cout << "Status: Advancing the ions' velocity...\n";

	#ifdef ONED
	switch (params->particleIntegrator){
		case(1):{
				aiv_Boris_1D(params, CS, mesh, EB, IONS, DT);
				break;
				}
		case(2):{
				aiv_Vay_1D(params, CS, mesh, EB, IONS, DT);
				break;
				}
		case(3):{
				exit(0);
				break;
				}
		default:{
				aiv_Vay_1D(params, CS, mesh, EB, IONS, DT);
				}
	}
	#endif

	#ifdef TWOD
	aiv_Boris_2D(params, CS, mesh, EB, IONS, DT);
	#endif

	#ifdef THREED
	aiv_Boris_3D(params, CS, mesh, EB, IONS, DT);
	#endif
}


void PIC::advanceIonsPosition(const simulationParameters * params, const meshGeometry * mesh, vector<ionSpecies> * IONS, const double DT){

	//cout << "Status: Advancing the ions' position...\n";

	#ifdef ONED
	aip_1D(params, mesh, IONS, DT);
	#endif

	#ifdef TWOD
	aip_2D(params, mesh, IONS, DT);
	#endif

	#ifdef THREED
	aip_3D(params, mesh, IONS, DT);
	#endif
}


/* * * * * * * * CLASS PIC_GC OBJECTS * * * * * * * */

void PIC_GC::set_to_zero_RK45_variables(){
	K1.fill(0.0);
	K2.fill(0.0);
	K3.fill(0.0);
	K4.fill(0.0);
	K5.fill(0.0);
	K6.fill(0.0);
	K7.fill(0.0);

	S4.fill(0.0);
	S5.fill(0.0);
}


void PIC_GC::set_GC_vars(ionSpecies * IONS, PIC_GC::GC_VARS * gcv, int pp){
	gcv->Q = IONS->Q;
	gcv->M = IONS->M;

	gcv->wx = { IONS->wxl(pp), IONS->wxc(pp), IONS->wxr(pp) };
	gcv->B = zeros(3);
	gcv->Bs = zeros(3);
	gcv->E = zeros(3);
	gcv->Es = zeros(3);
	gcv->b = zeros(3);

	gcv->mn = IONS->meshNode(pp);
	gcv->mu = IONS->mu(pp);

	gcv->Xo = IONS->X(pp,0);
	gcv->Pparo = IONS->Ppar(pp);
	gcv->go = IONS->g(pp);

	gcv->X = gcv->Xo;
	gcv->Ppar = gcv->Pparo;
	gcv->g = gcv->go;
}


void PIC_GC::set_to_zero_GC_vars(PIC_GC::GC_VARS * gcv){
	gcv->wx = zeros(3);
	gcv->B = zeros(3);
	gcv->Bs = zeros(3);
	gcv->b = zeros(3);
	gcv->E = zeros(3);
	gcv->Es = zeros(3);

	gcv->mn = 0;
	gcv->mu = 0.0;

	gcv->Xo = 0.0;
	gcv->Pparo = 0.0;
	gcv->go = 0.0;

	gcv->X = 0.0;
	gcv->Ppar = 0.0;
	gcv->g = 0.0;
}


void PIC_GC::reset_GC_vars(PIC_GC::GC_VARS * gcv){
	gcv->wx.fill(0.0);
	gcv->B.fill(0.0);
	gcv->Bs.fill(0.0);
	gcv->E.fill(0.0);
	gcv->Es.fill(0.0);
	gcv->b.fill(0.0);

	gcv->mn = 0;

	gcv->Xo = gcv->Xo_;
	gcv->Pparo = gcv->Pparo_;
	gcv->go = gcv->go_;

	gcv->X = gcv->Xo;
	gcv->Ppar = gcv->Pparo;
	gcv->g = gcv->go;
}


void PIC_GC::depositIonDensityAndBulkVelocity(const simulationParameters * params, const meshGeometry * mesh, ionSpecies * IONS){

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;
	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	switch (params->weightingScheme){
		case(0):{
				#ifdef ONED
				PIC::eidTOS_1D(params, mesh, IONS);
				PIC::eivTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(1):{
				#ifdef ONED
				PIC::eidTSC_1D(params, mesh, IONS);
				PIC::eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				PIC::eidTSC_2D(params, mesh, IONS);
				PIC::eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				PIC::eidTSC_3D(params, mesh, IONS);
				PIC::eivTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				#ifdef ONED
				PIC::eidTOS_1D(params, mesh, IONS);
				PIC::eivTOS_1D(params, mesh, IONS);
				#endif
				break;
				}
		case(4):{
				#ifdef ONED
				PIC::eidTSC_1D(params, mesh, IONS);
				PIC::eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				PIC::eidTSC_2D(params, mesh, IONS);
				PIC::eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				PIC::eidTSC_3D(params, mesh, IONS);
				PIC::eivTSC_3D(params, mesh, IONS);
				#endif
				break;
				}
		default:{
				#ifdef ONED
				PIC::eidTSC_1D(params, mesh, IONS);
				PIC::eivTSC_1D(params, mesh, IONS);
				#endif

				#ifdef TWOD
				PIC::eidTSC_2D(params, mesh, IONS);
				PIC::eivTSC_2D(params, mesh, IONS);
				#endif

				#ifdef THREED
				PIC::eidTSC_3D(params, mesh, IONS);
				PIC::eivTSC_3D(params, mesh, IONS);
				#endif
				}
	}// Switch
}


void PIC_GC::EFF_EMF_TSC_1D(const double DT, const double DX, GC_VARS * gcv, const fields * EB){
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x
	// wxl = wx(0)
	// wxc = wx(1)
	// wxr = wx(2)
	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int ix = gcv->mn + 1;
	gcv->b.fill(0.0);
	gcv->B.fill(0.0);
	gcv->E.fill(0.0);
	gcv->Bs.fill(0.0);
	gcv->Es.fill(0.0);

	// Computation of B, E and b fields.
	if(ix == (NX-2)){
		// Unitary, parallel vector b
		// wxl
		gcv->b(0) += gcv->wx(0)*EB->b.X(ix-1);
		gcv->b(1) += gcv->wx(0)*EB->b.Y(ix-1);
		gcv->b(2) += gcv->wx(0)*EB->b.Z(ix-1);

		// wxc
		gcv->b(0) += gcv->wx(1)*EB->b.X(ix);
		gcv->b(1) += gcv->wx(1)*EB->b.Y(ix);
		gcv->b(2) += gcv->wx(1)*EB->b.Z(ix);

		// wxr
		gcv->b(0) += gcv->wx(2)*EB->b.X(ix+1);
		gcv->b(1) += gcv->wx(2)*EB->b.Y(ix+1);
		gcv->b(2) += gcv->wx(2)*EB->b.Z(ix+1);

		// Effective magnetic field
		// wxl
		gcv->B(0) += gcv->wx(0)*EB->B.X(ix-1);
		gcv->B(1) += gcv->wx(0)*EB->B.Y(ix-1);
		gcv->B(2) += gcv->wx(0)*EB->B.Z(ix-1);

		// wxc
		gcv->B(0) += gcv->wx(1)*EB->B.X(ix);
		gcv->B(1) += gcv->wx(1)*EB->B.Y(ix);
		gcv->B(2) += gcv->wx(1)*EB->B.Z(ix);

		// wxr
		gcv->B(0) += gcv->wx(2)*EB->B.X(ix+1);
		gcv->B(1) += gcv->wx(2)*EB->B.Y(ix+1);
		gcv->B(2) += gcv->wx(2)*EB->B.Z(ix+1);

		// Effective electric field
		// wxl
		gcv->E(0) += gcv->wx(0)*EB->E.X(ix-1);
		gcv->E(1) += gcv->wx(0)*EB->E.Y(ix-1);
		gcv->E(2) += gcv->wx(0)*EB->E.Z(ix-1);

		// wxc
		gcv->E(0) += gcv->wx(1)*EB->E.X(ix);
		gcv->E(1) += gcv->wx(1)*EB->E.Y(ix);
		gcv->E(2) += gcv->wx(1)*EB->E.Z(ix);

		// wxr
		gcv->E(0) += gcv->wx(2)*EB->E.X(ix+1);
		gcv->E(1) += gcv->wx(2)*EB->E.Y(ix+1);
		gcv->E(2) += gcv->wx(2)*EB->E.Z(ix+1);
	}else if(ix == (NX-1)){
		// Unitary, parallel vector b
		// wxl
		gcv->b(0) += gcv->wx(0)*EB->b.X(ix-1);
		gcv->b(1) += gcv->wx(0)*EB->b.Y(ix-1);
		gcv->b(2) += gcv->wx(0)*EB->b.Z(ix-1);

		// wxc
		gcv->b(0) += gcv->wx(1)*EB->b.X(ix);
		gcv->b(1) += gcv->wx(1)*EB->b.Y(ix);
		gcv->b(2) += gcv->wx(1)*EB->b.Z(ix);

		// wxr
		gcv->b(0) += gcv->wx(2)*EB->b.X(2);
		gcv->b(1) += gcv->wx(2)*EB->b.Y(2);
		gcv->b(2) += gcv->wx(2)*EB->b.Z(2);

		// Effective magnetic field
		// wxl
		gcv->B(0) += gcv->wx(0)*EB->B.X(ix-1);
		gcv->B(1) += gcv->wx(0)*EB->B.Y(ix-1);
		gcv->B(2) += gcv->wx(0)*EB->B.Z(ix-1);

		// wxc
		gcv->B(0) += gcv->wx(1)*EB->B.X(ix);
		gcv->B(1) += gcv->wx(1)*EB->B.Y(ix);
		gcv->B(2) += gcv->wx(1)*EB->B.Z(ix);

		// wxr
		gcv->B(0) += gcv->wx(2)*EB->B.X(2);
		gcv->B(1) += gcv->wx(2)*EB->B.Y(2);
		gcv->B(2) += gcv->wx(2)*EB->B.Z(2);

		// Effective electric field
		// wxl
		gcv->E(0) += gcv->wx(0)*EB->E.X(ix-1);
		gcv->E(1) += gcv->wx(0)*EB->E.Y(ix-1);
		gcv->E(2) += gcv->wx(0)*EB->E.Z(ix-1);

		// wxc
		gcv->E(0) += gcv->wx(1)*EB->E.X(ix);
		gcv->E(1) += gcv->wx(1)*EB->E.Y(ix);
		gcv->E(2) += gcv->wx(1)*EB->E.Z(ix);

		// wxr
		gcv->E(0) += gcv->wx(2)*EB->E.X(2);
		gcv->E(1) += gcv->wx(2)*EB->E.Y(2);
		gcv->E(2) += gcv->wx(2)*EB->E.Z(2);
	}else{
		// Unitary, parallel vector b
		// wxl
		gcv->b(0) += gcv->wx(0)*EB->b.X(ix-1);
		gcv->b(1) += gcv->wx(0)*EB->b.Y(ix-1);
		gcv->b(2) += gcv->wx(0)*EB->b.Z(ix-1);

		// wxc
		gcv->b(0) += gcv->wx(1)*EB->b.X(ix);
		gcv->b(1) += gcv->wx(1)*EB->b.Y(ix);
		gcv->b(2) += gcv->wx(1)*EB->b.Z(ix);

		// wxr
		gcv->b(0) += gcv->wx(2)*EB->b.X(ix+1);
		gcv->b(1) += gcv->wx(2)*EB->b.Y(ix+1);
		gcv->b(2) += gcv->wx(2)*EB->b.Z(ix+1);

		// Effective magnetic field
		// wxl
		gcv->B(0) += gcv->wx(0)*EB->B.X(ix-1);
		gcv->B(1) += gcv->wx(0)*EB->B.Y(ix-1);
		gcv->B(2) += gcv->wx(0)*EB->B.Z(ix-1);

		// wxc
		gcv->B(0) += gcv->wx(1)*EB->B.X(ix);
		gcv->B(1) += gcv->wx(1)*EB->B.Y(ix);
		gcv->B(2) += gcv->wx(1)*EB->B.Z(ix);

		// wxr
		gcv->B(0) += gcv->wx(2)*EB->B.X(ix+1);
		gcv->B(1) += gcv->wx(2)*EB->B.Y(ix+1);
		gcv->B(2) += gcv->wx(2)*EB->B.Z(ix+1);

		// Effective electric field
		// wxl
		gcv->E(0) += gcv->wx(0)*EB->E.X(ix-1);
		gcv->E(1) += gcv->wx(0)*EB->E.Y(ix-1);
		gcv->E(2) += gcv->wx(0)*EB->E.Z(ix-1);

		// wxc
		gcv->E(0) += gcv->wx(1)*EB->E.X(ix);
		gcv->E(1) += gcv->wx(1)*EB->E.Y(ix);
		gcv->E(2) += gcv->wx(1)*EB->E.Z(ix);

		// wxr
		gcv->E(0) += gcv->wx(2)*EB->E.X(ix+1);
		gcv->E(1) += gcv->wx(2)*EB->E.Y(ix+1);
		gcv->E(2) += gcv->wx(2)*EB->E.Z(ix+1);
	}

	// Computation of relativistic factor gamma (gcv->g) using B field computed above
	gcv->g = sqrt( 1.0 + 2.0*gcv->mu*sqrt( dot(gcv->B, gcv->B) )/( gcv->M*F_C_DS*F_C_DS ) + gcv->Ppar*gcv->Ppar/( gcv->M*gcv->M*F_C_DS*F_C_DS ) );

	// Computation of effective fields Bs, Es.
	if(ix == (NX-2)){
		// Effective magnetic field
		// wxl
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(0)*( -( EB->b.Z(ix) - EB->b.Z(ix-1))/DX );
		gcv->Bs(2) += gcv->wx(0)*( ( EB->b.Y(ix) - EB->b.Y(ix-1))/DX );

		// wxc
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(1)*( -( EB->b.Z(ix+1) - EB->b.Z(ix))/DX );
		gcv->Bs(2) += gcv->wx(1)*( ( EB->b.Y(ix+1) - EB->b.Y(ix))/DX );

		// wxr
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(2)*( -( EB->b.Z(2) - EB->b.Z(ix+1))/DX );
		gcv->Bs(2) += gcv->wx(2)*( ( EB->b.Y(2) - EB->b.Y(ix+1))/DX );

		// Effective electric field
		// wxl
		//gcv->E(0) += gcv->wx(0)*( EB->E.X(ix-1) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX - gcv->Ppar*(EB->b.X(ix-1) - EB->b_.X(ix-1))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(0)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Y(ix-1) - EB->b_.Y(ix-1))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Z(ix-1) - EB->b_.Z(ix-1))/DT )/gcv->Q );

		// wxc
		//gcv->E(0) += gcv->wx(1)*( EB->E.X(ix) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(ix+1) - EB->_B.X(ix))/DX - gcv->Ppar*(EB->b.X(ix) - EB->b_.X(ix))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(1)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix+1) - EB->_B.X(ix))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Y(ix) - EB->b_.Y(ix))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Z(ix) - EB->b_.Z(ix))/DT )/gcv->Q );

		// wxr
		//gcv->E(0) += gcv->wx(2)*( EB->E.X(ix+1) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(2) - EB->_B.X(ix+1))/DX - gcv->Ppar*(EB->b.X(ix+1) - EB->b_.X(ix+1))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(2)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(2) - EB->_B.X(ix+1))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Y(ix+1) - EB->b_.Y(ix+1))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Z(ix+1) - EB->b_.Z(ix+1))/DT )/gcv->Q );
	}else if(ix == (NX-1)){
		// Effective magnetic field
		// wxl
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(0)*( -( EB->b.Z(ix) - EB->b.Z(ix-1))/DX );
		gcv->Bs(2) += gcv->wx(0)*( ( EB->b.Y(ix) - EB->b.Y(ix-1))/DX );

		// wxc
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(1)*( -( EB->b.Z(2) - EB->b.Z(ix))/DX );
		gcv->Bs(2) += gcv->wx(1)*( ( EB->b.Y(2) - EB->b.Y(ix))/DX );

		// wxr
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(2)*( -( EB->b.Z(3) - EB->b.Z(2))/DX );
		gcv->Bs(2) += gcv->wx(2)*( ( EB->b.Y(3) - EB->b.Y(2))/DX );

		// Effective electric field
		// wxl
		//gcv->E(0) += gcv->wx(0)*( EB->E.X(ix-1) - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX - gcv->Ppar*(EB->b.X(ix-1) - EB->b_.X(ix-1))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(0)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Y(ix-1) - EB->b_.Y(ix-1))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Z(ix-1) - EB->b_.Z(ix-1))/DT )/gcv->Q );

		// wxc
		//gcv->E(0) += gcv->wx(1)*( EB->E.X(ix) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(2) - EB->_B.X(ix))/DX - gcv->Ppar*(EB->b.X(ix) - EB->b_.X(ix))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(1)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(2) - EB->_B.X(ix))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Y(ix) - EB->b_.Y(ix))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Z(ix) - EB->b_.Z(ix))/DT )/gcv->Q );

		// wxr
		//gcv->E(0) += gcv->wx(2)*( EB->E.X(2) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(3) - EB->_B.X(2))/DX - gcv->Ppar*(EB->b.X(2) - EB->b_.X(2))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(2)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(3) - EB->_B.X(2))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Y(2) - EB->b_.Y(2))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Z(2) - EB->b_.Z(2))/DT )/gcv->Q );
	}else{
		// Effective magnetic field
		// wxl
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(0)*( -( EB->b.Z(ix) - EB->b.Z(ix-1))/DX );
		gcv->Bs(2) += gcv->wx(0)*( ( EB->b.Y(ix) - EB->b.Y(ix-1))/DX );

		// wxc
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(1)*( -( EB->b.Z(ix+1) - EB->b.Z(ix))/DX );
		gcv->Bs(2) += gcv->wx(1)*( ( EB->b.Y(ix+1) - EB->b.Y(ix))/DX );

		// wxr
		gcv->Bs(0) += 0.0;
		gcv->Bs(1) += gcv->wx(2)*( -( EB->b.Z(ix+2) - EB->b.Z(ix+1))/DX );
		gcv->Bs(2) += gcv->wx(2)*( ( EB->b.Y(ix+2) - EB->b.Y(ix+1))/DX );

		// Effective electric field
		// wxl
		//gcv->E(0) += gcv->wx(0)*( EB->E.X(ix-1) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX - gcv->Ppar*(EB->b.X(ix-1) - EB->b_.X(ix-1))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(0)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix) - EB->_B.X(ix-1))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Y(ix-1) - EB->b_.Y(ix-1))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(0)*( ( gcv->Ppar*(EB->b.Z(ix-1) - EB->b_.Z(ix-1))/DT )/gcv->Q );

		// wxc
		//gcv->E(0) += gcv->wx(1)*( EB->E.X(ix) - ( (2*gcv->mu/gcv->g)*(EB->_B.X(ix+1) - EB->_B.X(ix))/DX - gcv->Ppar*(EB->b.X(ix) - EB->b_.X(ix))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(1)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix+1) - EB->_B.X(ix))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Y(ix) - EB->b_.Y(ix))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(1)*( ( gcv->Ppar*(EB->b.Z(ix) - EB->b_.Z(ix))/DT )/gcv->Q );

		// wxr
		//gcv->E(0) += gcv->wx(2)*( EB->E.X(ix+1) - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix+2) - EB->_B.X(ix+1))/DX - gcv->Ppar*(EB->b.X(ix+1) - EB->b_.X(ix+1))/DT )/gcv->Q );
		gcv->Es(0) += gcv->wx(2)*( - ( (2.0*gcv->mu/gcv->g)*(EB->_B.X(ix+2) - EB->_B.X(ix+1))/DX )/gcv->Q );
		gcv->Es(1) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Y(ix+1) - EB->b_.Y(ix+1))/DT )/gcv->Q );
		gcv->Es(2) += gcv->wx(2)*( ( gcv->Ppar*(EB->b.Z(ix+1) - EB->b_.Z(ix+1))/DT )/gcv->Q );
	}

	gcv->Bs *= gcv->Ppar/gcv->Q;

	gcv->Bs += gcv->B;
	gcv->Es += gcv->E;
}


void PIC_GC::assignCell_TSC(const simulationParameters * params, const meshGeometry * mesh, GC_VARS * gcv, int dim){
	//This function assigns the particles to the closest mesh node depending in their position and
	//calculate the weights for the charge extrapolation and force interpolation

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x
	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	switch (dim){
		case(1):{
			int NC = mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;

			gcv->mn = floor( (gcv->X + 0.5*mesh->DX)/mesh->DX );

			double X = 0.0;
			if(gcv->mn != NC){
				X = gcv->X - mesh->nodes.X(gcv->mn);
			}else{
				X = gcv->X - ( mesh->nodes.X(NC-1) + mesh->DX );
			}

			if(X > 0.0){
				X = abs(X);

				gcv->wx(0) = 0.5*(1.5 - (mesh->DX + X)/mesh->DX)*(1.5 - (mesh->DX + X)/mesh->DX);
				gcv->wx(1) = 0.75 - ( X/mesh->DX )*( X/mesh->DX );
				gcv->wx(2) = 0.5*(1.5 - (mesh->DX - X)/mesh->DX)*(1.5 - (mesh->DX - X)/mesh->DX);
			}else{
				X = abs(X);

				gcv->wx(0) = 0.5*(1.5 - (mesh->DX - X)/mesh->DX)*(1.5 - (mesh->DX - X)/mesh->DX);
				gcv->wx(1) = 0.75 - ( X/mesh->DX )*( X/mesh->DX );
				gcv->wx(2) = 0.5*(1.5 - (mesh->DX + X)/mesh->DX)*(1.5 - (mesh->DX + X)/mesh->DX);
			}

			break;
		}
		case(2):{
			// To do something
			break;
		}
		case(3):{
			// To do something
			break;
		}
		default:{
			std::ofstream ofs("errors/assignCell_TSC_GC.txt",std::ofstream::out);
			ofs << "assignCell_TSC_GC: Introduce a valid option!\n";
			ofs.close();
			exit(1);
		}
	}
}


void PIC_GC::assignCell(const simulationParameters * params, const meshGeometry * mesh, GC_VARS * gcv, int dim){
	switch (params->weightingScheme){
		case(0):{
				//assignCell_TOS(params, mesh, gcv, 1);
				break;
				}
		case(1):{
				PIC_GC::assignCell_TSC(params, mesh, gcv, dim);
				break;
				}
		case(2):{
				//assignCell_NNS(params, mesh, gcv, 1);
				break;
				}
		case(3):{
				//assignCell_TOS(params, mesh, gcv, 1);
				break;
				}
		case(4):{
				PIC_GC::assignCell_TSC(params, mesh, gcv, dim);
				break;
				}
		default:{
				PIC_GC::assignCell_TSC(params, mesh, gcv, dim);
				}
	}
}


void PIC_GC::computeFullOrbitVelocity(const simulationParameters * params, const meshGeometry * mesh, const fields * EB, GC_VARS * gcv, arma::rowvec * V, int dim){

	PIC_GC::assignCell(params, mesh, gcv, dim);

	// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
	PIC_GC::EFF_EMF_TSC_1D(params->DT, mesh->DX, gcv, EB);

	double Bspar = dot(gcv->b, gcv->Bs);

	(*V)(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
	(*V)(1) = gcv->Ppar*gcv->Bs(1)/(gcv->g*gcv->M*Bspar) - (gcv->Es(0)*gcv->b(2) - gcv->Es(2)*gcv->b(0))/Bspar;
	(*V)(2) = gcv->Ppar*gcv->Bs(2)/(gcv->g*gcv->M*Bspar) + (gcv->Es(0)*gcv->b(1) - gcv->Es(1)*gcv->b(0))/Bspar;
}


void PIC_GC::advanceRungeKutta45Stages_1D(const simulationParameters * params, const meshGeometry * mesh, double * DT_RK, GC_VARS * gcv, const fields * EB, int STG){
	// We interpolate the effective fields to GC particles position
	switch (STG) {
		case(1):{
			// We keep a copy of X and Ppar from the previous successful time step
			gcv->Xo_ = gcv->Xo;
			gcv->Pparo_ = gcv->Pparo;
			gcv->go_ = gcv->go;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->Bs);

			//K1
			K1(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K1(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(2):{
			gcv->X = gcv->Xo + *DT_RK*A(1,0)*K1(0);
			gcv->Ppar = gcv->Pparo + *DT_RK*A(1,0)*K1(1);

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K2
			K2(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K2(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(3):{
			gcv->X = gcv->Xo + *DT_RK*( A(2,0)*K1(0) + A(2,1)*K2(0) );
			gcv->Ppar = gcv->Pparo + *DT_RK*( A(2,0)*K1(1) + A(2,1)*K2(1) );

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K3
			K3(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K3(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(4):{
			gcv->X = gcv->Xo + *DT_RK*( A(3,0)*K1(0) + A(3,1)*K2(0) + A(3,2)*K3(0) );
			gcv->Ppar = gcv->Pparo + *DT_RK*( A(3,0)*K1(1) + A(3,1)*K2(1) + A(3,2)*K3(1)) ;

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K4
			K4(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K4(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(5):{
			gcv->X = gcv->Xo + *DT_RK*( A(4,0)*K1(0) + A(4,1)*K2(0) + A(4,2)*K3(0) + A(4,3)*K4(0) );
			gcv->Ppar = gcv->Pparo + *DT_RK*( A(4,0)*K1(1) + A(4,1)*K2(1) + A(4,2)*K3(1) + A(4,3)*K4(1) );

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K5
			K5(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K5(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(6):{
			gcv->X = gcv->Xo + *DT_RK*( A(5,0)*K1(0) + A(5,1)*K2(0) + A(5,2)*K3(0) + A(5,3)*K4(0) + A(5,4)*K5(0) );
			gcv->Ppar = gcv->Pparo + *DT_RK*( A(5,0)*K1(1) + A(5,1)*K2(1) + A(5,2)*K3(1) + A(5,3)*K4(1) + A(5,4)*K5(1) );

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::assignCell(params, mesh, gcv, 1);

			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K6
			K6(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K6(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
		case(7):{
			gcv->X = gcv->Xo + *DT_RK*( A(6,0)*K1(0) + A(6,1)*K2(0) + A(6,2)*K3(0) + A(6,3)*K4(0) + A(6,4)*K5(0) + A(6,5)*K6(0) );
			gcv->Ppar = gcv->Pparo + *DT_RK*( A(6,0)*K1(1) + A(6,1)*K2(1) + A(6,2)*K3(1) + A(6,3)*K4(1) + A(6,4)*K5(1) + A(6,5)*K6(1) );

			// Periodic boundary condition
			gcv->X = fmod(gcv->X, LX);
			gcv->X = (gcv->X < 0) ? gcv->X + LX : gcv->X;

			PIC_GC::assignCell(params, mesh, gcv, 1);

			// NOTE: The relativistic factor (gamma) is updated within EFF_EMF_TSC_1D, this using the updated magnetic field and Ppar
			PIC_GC::EFF_EMF_TSC_1D(*DT_RK, mesh->DX, gcv, EB);

			double Bspar = dot(gcv->b, gcv->B);

			//K7
			K7(0) = gcv->Ppar*gcv->Bs(0)/(gcv->g*gcv->M*Bspar) + (gcv->Es(1)*gcv->b(2) - gcv->Es(2)*gcv->b(1))/Bspar;
			K7(1) = gcv->Q*dot(gcv->Es, gcv->Bs)/Bspar;
			break;
		}
	}

}


void PIC_GC::ai_GC_1D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){
//	int NX(EB->E.X.n_elem);

	MPI_AllgatherField(params, &EB->E);
	MPI_AllgatherField(params, &EB->B);
	MPI_AllgatherField(params, &EB->b);
	MPI_AllgatherField(params, &EB->b_);
	MPI_AllgatherField(params, &EB->_B);


	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);

	forwardPBC_1D(&EB->b.X);
	forwardPBC_1D(&EB->b.Y);
	forwardPBC_1D(&EB->b.Z);

	forwardPBC_1D(&EB->b_.X);
	forwardPBC_1D(&EB->b_.Y);
	forwardPBC_1D(&EB->b_.Z);

	forwardPBC_1D(&EB->_B.X);
	forwardPBC_1D(&EB->_B.Y);
	forwardPBC_1D(&EB->_B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	fields EB_;
	EB_.zeros(NX);

	EB_.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	EB_.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	EB_.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	EB_.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	EB_.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	EB_.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	EB_.b.X.subvec(1,NX-2) = EB->b.X.subvec(1,NX-2);
	EB_.b.Y.subvec(1,NX-2) = 0.5*( EB->b.Y.subvec(1,NX-2) + EB->b.Y.subvec(0,NX-3) );
	EB_.b.Z.subvec(1,NX-2) = 0.5*( EB->b.Z.subvec(1,NX-2) + EB->b.Z.subvec(0,NX-3) );

	EB_.b_.X.subvec(1,NX-2) = EB->b_.X.subvec(1,NX-2);
	EB_.b_.Y.subvec(1,NX-2) = 0.5*( EB->b_.Y.subvec(1,NX-2) + EB->b_.Y.subvec(0,NX-3) );
	EB_.b_.Z.subvec(1,NX-2) = 0.5*( EB->b_.Z.subvec(1,NX-2) + EB->b_.Z.subvec(0,NX-3) );

	EB_._B = EB->_B;

	forwardPBC_1D(&EB_.E.X);
	forwardPBC_1D(&EB_.E.Y);
	forwardPBC_1D(&EB_.E.Z);

	forwardPBC_1D(&EB_.B.X);
	forwardPBC_1D(&EB_.B.Y);
	forwardPBC_1D(&EB_.B.Z);

	forwardPBC_1D(&EB_.b.X);
	forwardPBC_1D(&EB_.b.Y);
	forwardPBC_1D(&EB_.b.Z);

	forwardPBC_1D(&EB_.b_.X);
	forwardPBC_1D(&EB_.b_.Y);
	forwardPBC_1D(&EB_.b_.Z);

	forwardPBC_1D(&EB_._B.X);
	forwardPBC_1D(&EB_._B.Y);
	forwardPBC_1D(&EB_._B.Z);


	for(int ss=0; ss<IONS->size(); ss++){// Loop over species

		int NSP = IONS->at(ss).NSP;

		#pragma omp parallel for shared(params, IONS, mesh, EB_, A, B4, B5) firstprivate(DT, ss, NX, LX, NSP, K1, K2, K3, K4, K5, K6, K7, S4, S5)
		// #pragma omp parallel for shared(params, IONS, mesh, EB_) private(K1, K2, K3, K4, K5, K6, K7, S4, S5) firstprivate(DT, ss, NX, LX, NSP)
		for(int pp=0; pp<NSP; pp++){// Loop over particles

			PIC_GC::GC_VARS gcv;
			double DT_RK = DT;
			double TRK = 0.0;
			double TRK_ = 0.0;
			double DS45 = 0.0;
			double s = 0.0;

			set_GC_vars(&IONS->at(ss), &gcv, pp);

			set_to_zero_RK45_variables();

			while(TRK < DT){ // Sub-cycling time loop
				for(int stg=1; stg<8; stg++){
					advanceRungeKutta45Stages_1D(params, mesh, &DT_RK, &gcv, &EB_, stg);
				}

				S4(0) = gcv.Xo + DT_RK*( B4(0)*K1(0) + B4(1)*K2(0) + B4(2)*K3(0) + B4(3)*K4(0) + B4(4)*K5(0) + B4(5)*K6(0) + B4(6)*K7(0) );
				// Periodic boundary condition
				S4(0) = fmod(S4(0), LX);
				S4(0) = (S4(0) < 0) ? S4(0) + LX : S4(0);
				// Periodic boundary condition
				S4(1) = gcv.Pparo + DT_RK*( B4(0)*K1(1) + B4(1)*K2(1) + B4(2)*K3(1) + B4(3)*K4(1) + B4(4)*K5(1) + B4(5)*K6(1) + B4(6)*K7(1) );

				S5(0) = gcv.Xo + DT_RK*( B5(0)*K1(0) + B5(1)*K2(0) + B5(2)*K3(0) + B5(3)*K4(0) + B5(4)*K5(0) + B5(5)*K6(0) + B5(6)*K7(0) );
				// Periodic boundary condition
				S5(0) = fmod(S5(0), LX);
				S5(0) = (S5(0) < 0) ? S5(0) + LX : S5(0);
				// Periodic boundary condition
				S5(1) = gcv.Pparo + DT_RK*( B5(0)*K1(1) + B5(1)*K2(1) + B5(2)*K3(1) + B5(3)*K4(1) + B5(4)*K5(1) + B5(5)*K6(1) + B5(6)*K7(1) );

				DS45 = sqrt( dot(S5 - S4, S5 - S4) );

				s = (DS45 < double_zero) ? 3.0 : pow(0.5*DT_RK*Tol/(DT*DS45), 0.25);

				if(s >= 2.0){
					TRK_ = TRK;
					TRK += DT_RK;

					DT_RK = (2.0*DT_RK < DT) ? 2.0*DT_RK : DT_RK;

					gcv.Xo = S4(0);
					gcv.Pparo = S4(1);
				}else if(s >= 1.0){
					TRK_ = TRK;
					TRK += DT_RK;

					gcv.Xo = S4(0);
					gcv.Pparo = S4(1);
				}else if(s < 1.0){
					DT_RK *= 0.5;
				}
			} // Sub-cycling time loop


			if(TRK > DT){
				DT_RK = DT - TRK_;

				reset_GC_vars(&gcv);

				set_to_zero_RK45_variables();

				for(int stg=1; stg<8; stg++){
					advanceRungeKutta45Stages_1D(params, mesh, &DT_RK, &gcv, &EB_, stg);
				}

				S4(0) = gcv.Xo + DT_RK*( B4(0)*K1(0) + B4(1)*K2(0) + B4(2)*K3(0) + B4(3)*K4(0) + B4(4)*K5(0) + B4(5)*K6(0) + B4(6)*K7(0) );
				// Periodic boundary condition
				S4(0) = fmod(S4(0), LX);
				S4(0) = (S4(0) < 0) ? S4(0) + LX : S4(0);
				// Periodic boundary condition
				S4(1) = gcv.Pparo + DT_RK*( B4(0)*K1(1) + B4(1)*K2(1) + B4(2)*K3(1) + B4(3)*K4(1) + B4(4)*K5(1) + B4(5)*K6(1) + B4(6)*K7(1) );

				gcv.X = S4(0);
				gcv.Ppar = S4(1);
			}

			// Compute particles' (full-orbit) velocities
			arma::rowvec V(3);
			// Within the function 'computeFullOrbitVelocity' is calculated the relativistic factor gamma.
			PIC_GC::computeFullOrbitVelocity(params, mesh, &EB_, &gcv, &V, 1);

			IONS->at(ss).V.row(pp) = V;

			IONS->at(ss).X(pp,0) = gcv.X;
			IONS->at(ss).Ppar(pp) = gcv.Ppar;
			IONS->at(ss).g(pp) = gcv.g;

			IONS->at(ss).meshNode(pp) = gcv.mn;
			IONS->at(ss).wxl(pp) = gcv.wx(0);
			IONS->at(ss).wxc(pp) = gcv.wx(1);
			IONS->at(ss).wxr(pp) = gcv.wx(2);
		} // Loop over particles

		depositIonDensityAndBulkVelocity(params, mesh, &IONS->at(ss));

		PIC::MPI_BcastDensity(params, &IONS->at(ss));

		PIC::MPI_BcastBulkVelocity(params, &IONS->at(ss));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ss).n, params->smoothingParameter);
						smooth_TOS(&IONS->at(ss).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ss).n, params->smoothingParameter);
						smooth_TSC(&IONS->at(ss).nv, params->smoothingParameter);
					break;
					}
			case(2):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ss).n, params->smoothingParameter);
						smooth(&IONS->at(ss).nv, params->smoothingParameter);
					break;
					}
			case(3):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ss).n, params->smoothingParameter);
						smooth(&IONS->at(ss).nv, params->smoothingParameter);
					break;
					}
			case(4):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ss).n, params->smoothingParameter);
						smooth(&IONS->at(ss).nv, params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ss).n, params->smoothingParameter);
						smooth(&IONS->at(ss).nv, params->smoothingParameter);
					}
		}
	} // Loop over species

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	restoreVector(&EB->E.X);
	restoreVector(&EB->E.Y);
	restoreVector(&EB->E.Z);

	restoreVector(&EB->B.X);
	restoreVector(&EB->B.Y);
	restoreVector(&EB->B.Z);

	restoreVector(&EB->b.X);
	restoreVector(&EB->b.Y);
	restoreVector(&EB->b.Z);

	restoreVector(&EB->b_.X);
	restoreVector(&EB->b_.Y);
	restoreVector(&EB->b_.Z);

	restoreVector(&EB->_B.X);
	restoreVector(&EB->_B.Y);
	restoreVector(&EB->_B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

}


void PIC_GC::ai_GC_2D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}


void PIC_GC::ai_GC_3D(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

}


PIC_GC::PIC_GC(const simulationParameters * params, const meshGeometry * mesh){

	NX =  params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2;//Mesh size along the X axis (considering the gosht cell)
	LX = mesh->DX*mesh->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;

/*
	A = { 	{0.0,           0.0,            0.0,        0.0,        0.0,            0.0,    0.0},
	 		{1/5,           0.0,            0.0,        0.0,        0.0,            0.0,    0.0},
	 		{3/40,          9/40,           0.0,        0.0,        0.0,            0.0,    0.0},
	 		{44/45,         -56/15,         32/9,       0.0,        0.0,            0.0,    0.0},
	 		{19372/6561,    -25360/2187,    64448/6561, -212/729,   0.0,            0.0,    0.0},
	 		{9017/3168,     -355/33,        46732/5247, 49/176,     -5103/18656,    0.0,    0.0},
	 		{35/384,        0.0,            500/1113,   125/192,    -2187/6784,     11/84,  0.0}	};
*/

	A(0,0) = 0.0;
	A(0,1) = 0.0;
	A(0,2) = 0.0;
	A(0,3) = 0.0;
	A(0,4) = 0.0;
	A(0,5) = 0.0;
	A(0,6) = 0.0;

	A(1,0) = 1.0/5.0;
	A(1,1) = 0.0;
	A(1,2) = 0.0;
	A(1,3) = 0.0;
	A(1,4) = 0.0;
	A(1,5) = 0.0;
	A(1,6) = 0.0;

	A(2,0) = 3.0/40.0;
	A(2,1) = 9.0/40.0;
	A(2,2) = 0.0;
	A(2,3) = 0.0;
	A(2,4) = 0.0;
	A(2,5) = 0.0;
	A(2,6) = 0.0;

	A(3,0) = 44.0/45.0;
	A(3,1) = -56.0/15.0;
	A(3,2) = 32.0/9.0;
	A(3,3) = 0.0;
	A(3,4) = 0.0;
	A(3,5) = 0.0;
	A(3,6) = 0.0;

	A(4,0) = 19372.0/6561.0;
	A(4,1) = -25360.0/2187.0;
	A(4,2) = 64448.0/6561.0;
	A(4,3) = -212.0/729.0;
	A(4,4) = 0.0;
	A(4,5) = 0.0;
	A(4,6) = 0.0;

	A(5,0) = 9017.0/3168.0;
	A(5,1) = -355.0/33.0;
	A(5,2) = 46732.0/5247.0;
	A(5,3) = 49.0/176.0;
	A(5,4) = -5103.0/18656.0;
	A(5,5) = 0.0;
	A(5,6) = 0.0;

	A(6,0) = 35.0/384.0;
	A(6,1) = 0.0;
	A(6,2) = 500.0/1113.0;
	A(6,3) = 125.0/192.0;
	A(6,4) = -2187.0/6784.0;
	A(6,5) = 11.0/84.0;
	A(6,6) = 0.0;

	B4 = {	5179.0/57600.0, 0.0, 7571.0/16695.0, 393.0/640.0, -92097.0/339200.0, 187.0/2100.0, 1.0/40.0	};

	B5 = {	35.0/384.0, 0.0, 500.0/1113.0, 125.0/192.0, -2187.0/6784.0, 11.0/84.0, 0.0	};

	#ifdef ONED
	K1 = zeros(2);
	K2 = zeros(2);
	K3 = zeros(2);
	K4 = zeros(2);
	K5 = zeros(2);
	K6 = zeros(2);
	K7 = zeros(2);

	S4 = zeros(2);
	S5 = zeros(2);
	#endif

	#ifdef TWOD
	K1 = zeros(3);
	K2 = zeros(3);
	K3 = zeros(3);
	K4 = zeros(3);
	K5 = zeros(3);
	K6 = zeros(3);
	K7 = zeros(3);

	S4 = zeros(3);
	S5 = zeros(3);
	#endif

	#ifdef THREED
	K1 = zeros(4);
	K2 = zeros(4);
	K3 = zeros(4);
	K4 = zeros(4);
	K5 = zeros(4);
	K6 = zeros(4);
	K7 = zeros(4);

	S4 = zeros(4);
	S5 = zeros(4);
	#endif

/*
	gcv.wx = zeros(3);
	gcv.B = zeros(3);
	gcv.Bs = zeros(3);
	gcv.b = zeros(3);
	gcv.E = zeros(3);
	gcv.Es = zeros(3);

	gcv.mn = 0;
	gcv.mu = 0.0;

	gcv.Xo = 0.0;
	gcv.Pparo = 0.0;
	gcv.go = 0.0;

	gcv.X = 0.0;
	gcv.Ppar = 0.0;
	gcv.g = 0.0;
*/
}


void PIC_GC::advanceGCIons(const simulationParameters * params, const characteristicScales * CS, const meshGeometry * mesh, fields * EB, vector<ionSpecies> * IONS, const double DT){

	#ifdef ONED
	ai_GC_1D(params, CS, mesh, EB, IONS, DT);
	#endif

	#ifdef TWOD
	ai_GC_2D(params, CS, mesh, EB, IONS, DT);
	#endif

	#ifdef THREED
	ai_GC_3D(params, CS, mesh, EB, IONS, DT);
	#endif
}
