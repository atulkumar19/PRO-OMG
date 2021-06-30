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

#include "fields.h"

EMF_SOLVER::EMF_SOLVER(const simulationParameters * params, characteristicScales * CS)
{
	NX_S = params->mesh.NX_PER_MPI + 2;
	NX_T = params->mesh.NX_IN_SIM + 2;
	NX_R = params->mesh.NX_IN_SIM;

	NY_S = params->mesh.NY_PER_MPI + 2;
	NY_T = params->mesh.NY_IN_SIM + 2;
	NY_R = params->mesh.NY_IN_SIM;

	if (params->dimensionality == 1)
        {
            n_cs = CS->length*CS->density;

            V1D.ne.zeros(NX_S);
            V1D._n.zeros(NX_S);
            V1D.n.zeros(NX_S);
            V1D.n_.zeros(NX_S);
            V1D.n__.zeros(NX_S);

            V1D.V.zeros(NX_S);
            V1D.U.zeros(NX_S);
            V1D.U_.zeros(NX_S);
            V1D.U__.zeros(NX_S);
	}
        else if (params->dimensionality == 2)
        {
            n_cs = CS->length*CS->length*CS->density;

            V2D.ne.zeros(NX_S,NY_S);
            V2D._n.zeros(NX_S,NY_S);
            V2D.n.zeros(NX_S,NY_S);
            V2D.n_.zeros(NX_S,NY_S);
            V2D.n__.zeros(NX_S,NY_S);

            V2D.V.zeros(NX_S,NY_S);
            V2D.U.zeros(NX_S,NY_S);
            V2D.U_.zeros(NX_S,NY_S);
            V2D.U__.zeros(NX_S,NY_S);
	}
}


void EMF_SOLVER::MPI_Allgathervec(const simulationParameters * params, arma::vec * field)
{
    unsigned int iIndex = params->mpi.iIndex;
    unsigned int fIndex = params->mpi.fIndex;

    arma::vec recvbuf(params->mesh.NX_IN_SIM);
    arma::vec sendbuf(params->mesh.NX_PER_MPI);

    //Allgather for x-component
    sendbuf = field->subvec(iIndex, fIndex);
    MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
    field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
}


void EMF_SOLVER::MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield){
	MPI_Allgathervec(params, &vfield->X);
	MPI_Allgathervec(params, &vfield->Y);
	MPI_Allgathervec(params, &vfield->Z);
}


void EMF_SOLVER::MPI_Allgathermat(const simulationParameters * params, arma::mat * field){
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;

	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	arma::vec recvbuf = zeros(params->mesh.NX_IN_SIM*params->mesh.NY_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI);

	//Allgather for x-component
	sendbuf = vectorise(field->submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.MPIS_FIELDS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;

		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NY_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NY_PER_MPI;

		field->submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}

}


void EMF_SOLVER::MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield){
	MPI_Allgathermat(params, &vfield->X);
	MPI_Allgathermat(params, &vfield->Y);
	MPI_Allgathermat(params, &vfield->Z);
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, arma::vec * F){
	unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

	double sendbuf;
	double recvbuf;

	sendbuf = (*F)(fIndex);
	MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART, 0, &recvbuf, 1, MPI_DOUBLE, params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART, 0, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
	(*F)(iIndex-1) = recvbuf;

	sendbuf = (*F)(iIndex);
	MPI_Sendrecv(&sendbuf, 1, MPI_DOUBLE, params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART, 1, &recvbuf, 1, MPI_DOUBLE, params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART, 1, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
	(*F)(fIndex+1) = recvbuf;
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, vfield_vec * F){
	MPI_passGhosts(params, &F->X);
	MPI_passGhosts(params, &F->Y);
	MPI_passGhosts(params, &F->Z);
}


void EMF_SOLVER::MPI_SendVec(const simulationParameters * params, arma::vec * v){
	// We send the vector from root process of fields to root process of particles
	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Send(v->memptr(), v->n_elem, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, FIELDS_TAG, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Recv(v->memptr(), v->n_elem, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, FIELDS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Then, the vevtor is broadcasted to all processes in the particles communicator
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
		MPI_Bcast(v->memptr(), v->n_elem, MPI_DOUBLE, 0, params->mpi.COMM);
}


void EMF_SOLVER::MPI_SendMat(const simulationParameters * params, arma::mat * m){
	// We send the vector from root process of fields to root process of particles
	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Send(m->memptr(), m->n_elem, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, FIELDS_TAG, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Recv(m->memptr(), m->n_elem, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, FIELDS_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Then, the vevtor is broadcasted to all processes in the particles communicator
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
		MPI_Bcast(m->memptr(), m->n_elem, MPI_DOUBLE, 0, params->mpi.COMM);
}


void EMF_SOLVER::MPI_Gathervec(const simulationParameters * params, arma::vec * field){
	unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	// First, we gather the vector at the root (rank 0) process of the COMM MPI communicator for fields
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
		sendbuf = field->subvec(iIndex, fIndex);

		MPI_Gather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, 0, params->mpi.COMM);

		if (params->mpi.IS_FIELDS_ROOT)
			field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


void EMF_SOLVER::MPI_Gathervfield_vec(const simulationParameters * params, vfield_vec * vfield){
	MPI_Gathervec(params, &vfield->X);
	MPI_Gathervec(params, &vfield->Y);
	MPI_Gathervec(params, &vfield->Z);
}


void EMF_SOLVER::MPI_Gathermat(const simulationParameters * params, arma::mat * field){
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;

	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	arma::vec recvbuf = zeros(params->mesh.NUM_CELLS_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NUM_CELLS_PER_MPI);

	// First, we gather the vector at the root (rank 0) process of the COMM MPI communicator for fields
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
		sendbuf = vectorise(field->submat(irow,icol,frow,fcol));

		MPI_Gather(sendbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, 0, params->mpi.COMM);

		if (params->mpi.IS_FIELDS_ROOT){
			for (int mpis=0; mpis<params->mpi.MPIS_FIELDS; mpis++){
				unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
				unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*(mpis+1) - 1;

				unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
				unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;

				unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NY_PER_MPI + 1;
				unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NY_PER_MPI;

				field->submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}


void EMF_SOLVER::MPI_Gathervfield_mat(const simulationParameters * params, vfield_mat * vfield){
	MPI_Gathermat(params, &vfield->X);
	MPI_Gathermat(params, &vfield->Y);
	MPI_Gathermat(params, &vfield->Z);
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, arma::mat * F){
	if (params->mesh.SPLIT_DIRECTION == 0){ // Split along x-axis
		unsigned int irow = params->mpi.irow;
		unsigned int frow = params->mpi.frow;

		int NY = params->mesh.NY_IN_SIM + 2;

		arma::rowvec recvbuf(NY);
		arma::rowvec sendbuf(NY);

		// Up to down ghost cells and viceversa
		F->submat(irow,0,frow,0) = F->submat(irow,NY-2,frow,NY-2);
		F->submat(irow,NY-1,frow,NY-1) = F->submat(irow,1,frow,1);

		// We first pass the ghosts cells at the right end to the right MPI process in the Cartesian topology
		sendbuf = F->row(frow);
		sendbuf(0) = sendbuf(NY-2);
		sendbuf(NY-1) = sendbuf(1);
		MPI_Sendrecv(sendbuf.memptr(), NY, MPI_DOUBLE, params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART, 0, recvbuf.memptr(), NY, MPI_DOUBLE, params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART, 0, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
		F->row(irow-1) = recvbuf;

		sendbuf = F->row(irow);
		sendbuf(0) = sendbuf(NY-2);
		sendbuf(NY-1) = sendbuf(1);
		MPI_Sendrecv(sendbuf.memptr(), NY, MPI_DOUBLE, params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART, 1, recvbuf.memptr(), NY, MPI_DOUBLE, params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART, 1, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
		F->row(frow+1) = recvbuf;
	}else{ // Split along y-axis
		unsigned int icol = params->mpi.icol;
		unsigned int fcol = params->mpi.fcol;

		int NX = params->mesh.NX_IN_SIM + 2;

		arma::colvec recvbuf(NX);
		arma::colvec sendbuf(NX);

		// Left to right ghost cells and viceversa
		F->submat(0,icol,0,fcol) = F->submat(NX-2,icol,NX-2,fcol);
		F->submat(NX-1,icol,NX-1,fcol) = F->submat(1,icol,1,fcol);

		// We first pass the ghosts cells at the right end to the right MPI process in the Cartesian topology
		sendbuf = F->col(fcol);
		sendbuf(0) = sendbuf(NX-2);
		sendbuf(NX-1) = sendbuf(1);
		MPI_Sendrecv(sendbuf.memptr(), NX, MPI_DOUBLE, params->mpi.UP_MPI_DOMAIN_NUMBER_CART, 0, recvbuf.memptr(), NX, MPI_DOUBLE, params->mpi.DOWN_MPI_DOMAIN_NUMBER_CART, 0, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
		F->col(icol-1) = recvbuf;

		sendbuf = F->col(icol);
		sendbuf(0) = sendbuf(NX-2);
		sendbuf(NX-1) = sendbuf(1);
		MPI_Sendrecv(sendbuf.memptr(), NX, MPI_DOUBLE, params->mpi.DOWN_MPI_DOMAIN_NUMBER_CART, 1, recvbuf.memptr(), NX, MPI_DOUBLE, params->mpi.UP_MPI_DOMAIN_NUMBER_CART, 1, params->mpi.MPI_TOPO, MPI_STATUS_IGNORE);
		F->col(fcol+1) = recvbuf;
	}

}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, vfield_mat * F){
	MPI_passGhosts(params, &F->X);
	MPI_passGhosts(params, &F->Y);
	MPI_passGhosts(params, &F->Z);
}


// * * * Smoothing * * *
void EMF_SOLVER::smooth(arma::vec * v, double as){
	int NX = v->n_elem;

	arma::vec b = zeros(NX);
	/*
	double wc(0.75); 	// center weight
	double ws(0.125);	// sides weight
	*/

	double wc(0.5); 	// center weight
	double ws(0.25);	// sides weight

	//Step 1: Averaging process
	b.subvec(1, NX-2) = v->subvec(1, NX-2);

	fillGhosts(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1.0 - as)*v->subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void EMF_SOLVER::smooth(arma::mat * m, double as){
	int NX = m->n_rows;
	int NY = m->n_cols;

	arma::mat b = zeros(NX,NY);
	/*
	double wc(9.0/16.0);
	double ws(3.0/32.0);
	double wcr(1.0/64.0);
	*/
	double wc(0.25);
	double ws(0.1250);
	double wcr(0.0625);

	// Step 1: Averaging
	b.submat(1,1,NX-2,NY-2) = m->submat(1,1,NX-2,NY-2);

	fillGhosts(&b);

	b.submat(1,1,NX-2,NY-2) = wc*b.submat(1,1,NX-2,NY-2) + \
								ws*b.submat(2,1,NX-1,NY-2) + ws*b.submat(0,1,NX-3,NY-2) + \
								ws*b.submat(1,2,NX-2,NY-1) + ws*b.submat(1,0,NX-2,NY-3) + \
								wcr*b.submat(2,2,NX-1,NY-1) + wcr*b.submat(0,2,NX-3,NY-1) + \
								wcr*b.submat(0,0,NX-3,NY-3) + wcr*b.submat(2,0,NX-1,NY-3);

	// Step 2: Averaged weighted variable estimation
	m->submat(1,1,NX-2,NY-2) = (1.0 - as)*m->submat(1,1,NX-2,NY-2) + as*b.submat(1,1,NX-2,NY-2);

}


void EMF_SOLVER::smooth(vfield_vec * vf, double as){
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}


void EMF_SOLVER::smooth(vfield_mat * vf, double as){
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}
// * * * Smoothing * * *


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params, oneDimensional::fields * EB){//This function calculates -culr(EB->E)
	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	// Indices of subdomain
	unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

	//There is not x-component of curl(B)
	EB->B.X.fill(0);

	//y-component
	EB->B.Y.subvec(iIndex,fIndex) =   0.5*( EB->E.Z.subvec(iIndex+1,fIndex+1) - EB->E.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

	//z-component
	EB->B.Z.subvec(iIndex,fIndex) = - 0.5*( EB->E.Y.subvec(iIndex+1,fIndex+1) - EB->E.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params, twoDimensional::fields * EB){
	MPI_passGhosts(params, &EB->E);
	MPI_passGhosts(params, &EB->B);

	// Indices of subdomain
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;
	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	//There is not x-component of curl(B)
	EB->B.X.submat(irow,icol,frow,fcol)  = - 0.5*( EB->E.Z.submat(irow,icol+1,frow,fcol+1) - EB->E.Z.submat(irow,icol-1,frow,fcol-1) )/params->mesh.DY;

	//y-component
	EB->B.Y.submat(irow,icol,frow,fcol)  =   0.5*( EB->E.Z.submat(irow+1,icol,frow+1,fcol) - EB->E.Z.submat(irow-1,icol,frow-1,fcol) )/params->mesh.DX;

	//z-component
	EB->B.Z.submat(irow,icol,frow,fcol)  = - 0.5*( EB->E.Y.submat(irow+1,icol,frow+1,fcol) - EB->E.Y.submat(irow-1,icol,frow-1,fcol) )/params->mesh.DX;
	EB->B.Z.submat(irow,icol,frow,fcol) +=   0.5*( EB->E.X.submat(irow,icol+1,frow,fcol+1) - EB->E.X.submat(irow,icol-1,frow,fcol-1) )/params->mesh.DY;
}


void EMF_SOLVER::advanceBField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS){
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
		//Using the RK4 scheme to advance B.
		//B^(N+1) = B^(N) + dt( K1^(N) + 2*K2^(N) + 2*K3^(N) + K4^(N) )/6
		dt = params->DT/((double)params->numberOfRKIterations);

		for(int RKit=0; RKit<params->numberOfRKIterations; RKit++){ // Runge-Kutta iterations

			V1D.K1 = *EB; 									// The value of the fields at the time level (N-1/2)
			advanceEField(params, &V1D.K1, IONS, false, false);			// E1 (using B^(N-1/2))
			FaradaysLaw(params, &V1D.K1);					// K1

			V1D.K2.B.X = EB->B.X;							// B^(N-1/2) + 0.5*dt*K1
			V1D.K2.B.Y = EB->B.Y + (0.5*dt)*V1D.K1.B.Y;
			V1D.K2.B.Z = EB->B.Z + (0.5*dt)*V1D.K1.B.Z;
			V1D.K2.E = EB->E;
			advanceEField(params, &V1D.K2, IONS, false, false);			// E2 (using B^(N-1/2) + 0.5*dt*K1)
			FaradaysLaw(params, &V1D.K2);					// K2

			V1D.K3.B.X = EB->B.X;							// B^(N-1/2) + 0.5*dt*K2
			V1D.K3.B.Y = EB->B.Y + (0.5*dt)*V1D.K2.B.Y;
			V1D.K3.B.Z = EB->B.Z + (0.5*dt)*V1D.K2.B.Z;
			V1D.K3.E = EB->E;
			advanceEField(params, &V1D.K3, IONS, false, false);			// E3 (using B^(N-1/2) + 0.5*dt*K2)
			FaradaysLaw(params, &V1D.K3);					// K3

			V1D.K4.B.X = EB->B.X;							// B^(N-1/2) + dt*K2
			V1D.K4.B.Y = EB->B.Y + dt*V1D.K3.B.Y;
			V1D.K4.B.Z = EB->B.Z + dt*V1D.K3.B.Z;
			V1D.K4.E = EB->E;
			advanceEField(params, &V1D.K4, IONS, false, false);			// E4 (using B^(N-1/2) + dt*K3)
			FaradaysLaw(params, &V1D.K4);					// K4

			EB->B += (dt/6.0)*( V1D.K1.B + 2.0*V1D.K2.B + 2.0*V1D.K3.B + V1D.K4.B );
		} // Runge-Kutta iterations

		/*
		if (params->includeElectronInertia){

		}
		*/

		#ifdef CHECKS_ON
		if(!EB->B.X.is_finite()){
			cout << "Non finite values in Bx" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -113);
		}else if(!EB->B.Y.is_finite()){
			cout << "Non finite values in By" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -114);
		}else if(!EB->B.Z.is_finite()){
			cout << "Non finite values in Bz" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -115);
		}
		#endif

		MPI_Allgathervfield_vec(params, &EB->B);

		smooth(&EB->B, params->smoothingParameter);
	}

	MPI_SendVec(params, &EB->B.X);
	MPI_SendVec(params, &EB->B.Y);
	MPI_SendVec(params, &EB->B.Z);
}


void EMF_SOLVER::advanceBField(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS){
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
		//Using the RK4 scheme to advance B.
		//B^(N+1) = B^(N) + dt( K1^(N) + 2*K2^(N) + 2*K3^(N) + K4^(N) )/6
		dt = params->DT/((double)params->numberOfRKIterations);

		for(int RKit=0; RKit<params->numberOfRKIterations; RKit++){ // Runge-Kutta iterations

			V2D.K1 = *EB; 									// The value of the fields at the time level (N-1/2)
			advanceEField(params, &V2D.K1, IONS, false, false);			// E1 (using B^(N-1/2))
			FaradaysLaw(params, &V2D.K1);					// K1

			V2D.K2.B = EB->B + (0.5*dt)*V2D.K1.B;			// B^(N-1/2) + 0.5*dt*K1
			V2D.K2.E = EB->E;
			advanceEField(params, &V2D.K2, IONS, false, false);			// E2 (using B^(N-1/2) + 0.5*dt*K1)
			FaradaysLaw(params, &V2D.K2);					// K2

			V2D.K3.B = EB->B + (0.5*dt)*V2D.K2.B;			// B^(N-1/2) + 0.5*dt*K2
			V2D.K3.E = EB->E;
			advanceEField(params, &V2D.K3, IONS, false, false);			// E3 (using B^(N-1/2) + 0.5*dt*K2)
			FaradaysLaw(params, &V2D.K3);					// K3

			V2D.K4.B = EB->B + dt*V2D.K3.B;					// B^(N-1/2) + dt*K2
			V2D.K4.E = EB->E;
			advanceEField(params, &V2D.K4, IONS, false, false);			// E4 (using B^(N-1/2) + dt*K3)
			FaradaysLaw(params, &V2D.K4);					// K4

			EB->B += (dt/6.0)*( V2D.K1.B + 2.0*V2D.K2.B + 2.0*V2D.K3.B + V2D.K4.B );
		} // Runge-Kutta iterations

		/*
		if (params->includeElectronInertia){
			//*** @toimplement
		}
		*/

		#ifdef CHECKS_ON
		if(!EB->B.X.is_finite()){
			cout << "Non finite values in Bx" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -113);
		}else if(!EB->B.Y.is_finite()){
			cout << "Non finite values in By" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -114);
		}else if(!EB->B.Z.is_finite()){
			cout << "Non finite values in Bz" << endl;
			MPI_Abort(params->mpi.MPI_TOPO, -115);
		}
		#endif

		MPI_Allgathervfield_mat(params, &EB->B);

		smooth(&EB->B, params->smoothingParameter);
	}

	MPI_SendMat(params, &EB->B.X);
	MPI_SendMat(params, &EB->B.Y);
	MPI_SendMat(params, &EB->B.Z);
}


/* In this function the different ionSpecies vectors represent the following:
	+ IONS__: Ions' variables at time level "l - 3/2"
	+ IONS_: Ions' variables at time level "l - 1/2"
	+ IONS: Ions' variables at time level "l + 1/2"
*/
void EMF_SOLVER::advanceEField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, bool extrap, bool BAE)
{
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
	{
		MPI_passGhosts(params,&EB->E);
		MPI_passGhosts(params,&EB->B);

		// Indices of subdomain`
		unsigned int iIndex = params->mpi.iIndex;
		unsigned int fIndex = params->mpi.fIndex;

		V1D.ne.zeros();
		V1D.V.zeros();

		// Calculate the number density and bulk velocities at time level "l + 1/2":
		// =========================================================================
		V1D.n.zeros();
		V1D.U.zeros();
		for(int ii=0; ii<params->numberOfParticleSpecies; ii++)
		{
			fillGhosts(&IONS->at(ii).n);
			fillGhosts(&IONS->at(ii).nv.X);
			fillGhosts(&IONS->at(ii).nv.Y);
			fillGhosts(&IONS->at(ii).nv.Z);

			// Ions density at time level "l + 1/2"
			// n(l+1/2) = ( n(l+1) + n(l) )/2
			V1D.n += 0.5*IONS->at(ii).Z*( IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1) + IONS->at(ii).n_.subvec(iIndex - 1, fIndex + 1) );

			// Ions bulk velocity at time level "l + 1/2"
			//sum_k[ Z_k*n_k*u_k ]
			V1D.U.X += IONS->at(ii).Z*IONS->at(ii).nv.X.subvec(iIndex - 1, fIndex + 1);
			V1D.U.Y += IONS->at(ii).Z*IONS->at(ii).nv.Y.subvec(iIndex - 1, fIndex + 1);
			V1D.U.Z += IONS->at(ii).Z*IONS->at(ii).nv.Z.subvec(iIndex - 1, fIndex + 1);
		}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

		V1D.U.X /= V1D.n;
		V1D.U.Y /= V1D.n;
		V1D.U.Z /= V1D.n;

		V1D.n /= n_cs;//Dimensionless density

		// We perform the following computations of the number density and bulk velocity at different past times:
		// ======================================================================================================
		if (extrap)
		{
			V1D._n.zeros();
			V1D.n_.zeros();
			V1D.U_.zeros();
			for(int ii=0; ii<params->numberOfParticleSpecies; ii++)
			{
					fillGhosts(&IONS->at(ii).n);
					fillGhosts(&IONS->at(ii).n_);
					fillGhosts(&IONS->at(ii).n__);
					fillGhosts(&IONS->at(ii).nv_.X);
					fillGhosts(&IONS->at(ii).nv_.Y);
					fillGhosts(&IONS->at(ii).nv_.Z);

					// Electron density at time level "l + 1"
					V1D._n += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1);

					// Ions density at time level "l - 1/2"
					// n(l-1/2) = ( n(l) + n(l-1) )/2
					V1D.n_ += 0.5*IONS->at(ii).Z*( IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) + IONS->at(ii).n_.subvec(iIndex - 1,fIndex + 1) );

					// Ions bulk velocity at time level "l - 1/2"
					//sum_k[ Z_k*n_k*u_k ]
					V1D.U_.X += IONS->at(ii).Z*IONS->at(ii).nv_.X.subvec(iIndex-1,fIndex+1);
					V1D.U_.Y += IONS->at(ii).Z*IONS->at(ii).nv_.Y.subvec(iIndex-1,fIndex+1);
					V1D.U_.Z += IONS->at(ii).Z*IONS->at(ii).nv_.Z.subvec(iIndex-1,fIndex+1);
			}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

			V1D.U_.X /= V1D.n_;
			V1D.U_.Y /= V1D.n_;
			V1D.U_.Z /= V1D.n_;

			V1D._n /= n_cs;

			if(BAE)
			{
				V1D.n__.zeros();
				V1D.U__.zeros();
				for(int ii=0; ii<params->numberOfParticleSpecies; ii++)
				{
					fillGhosts(&IONS->at(ii).n___);
					fillGhosts(&IONS->at(ii).nv__.X);
					fillGhosts(&IONS->at(ii).nv__.Y);
					fillGhosts(&IONS->at(ii).nv__.Z);

					// Ions density at time level "l - 3/2"
					// n(l-3/2) = ( n(l-1) + n(l-2) )/2
					V1D.n__ += 0.5*IONS->at(ii).Z*( IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) + IONS->at(ii).n___.subvec(iIndex - 1,fIndex + 1) );

					//sum_k[ Z_k*n_k*u_k ]
					V1D.U__.X += IONS->at(ii).Z*IONS->at(ii).nv__.X.subvec(iIndex-1,fIndex+1);
					V1D.U__.Y += IONS->at(ii).Z*IONS->at(ii).nv__.Y.subvec(iIndex-1,fIndex+1);
					V1D.U__.Z += IONS->at(ii).Z*IONS->at(ii).nv__.Z.subvec(iIndex-1,fIndex+1);
				}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

				V1D.U__.X /= V1D.n__;
				V1D.U__.Y /= V1D.n__;
				V1D.U__.Z /= V1D.n__;

				//Here we use the velocity extrapolation U^(N+1) = 2*U^(N+1/2) - 1.5*U^(N-1/2) + 0.5*U^(N-3/2)
				V1D.V = 2.0*V1D.U - 1.5*V1D.U_ + 0.5*V1D.U__;
			}
			else
			{
				//Here we use the velocity extrapolation U^(N+1) = 1.5*U^(N+1/2) - 0.5*U^(N-1/2)
				V1D.V = 1.5*V1D.U - 0.5*V1D.U_;
			}

			V1D.ne = V1D._n;
		}
		else
		{
			V1D.V = V1D.U;
			V1D.ne = V1D.n;
		}

		// Electric field solve:
		// =======================================================================================================
    	//The E-field has three terms: (1) CurlB terms, (2) Hall terms and, (3) Pressure gradient terms

		bool curlBSolve = false;
		bool hallTermSolve = false;
		bool pressureTermSolve = true;

		// Initialize electric field:
		// =========================
		EB->E.zeros();

		// Curl B terms:
		// =====================
		if (curlBSolve)
		{
			// Calculate Curl B:
			vfield_vec curlB(NX_T);
			curlB.Y.subvec(iIndex,fIndex) = - 0.5*( EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
			curlB.Z.subvec(iIndex,fIndex) =   0.5*( EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

			// "x" component:
			EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y.subvec(iIndex,fIndex) % EB->B.Z.subvec(iIndex,fIndex) - curlB.Z.subvec(iIndex,fIndex) % EB->B.Y.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D.ne.subvec(1,NX_S-2) );

			// "y" component:
			EB->E.Y.subvec(iIndex,fIndex) =( ( curlB.Z.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D.ne.subvec(1,NX_S-2) ));

			// "z" component:
			EB->E.Z.subvec(iIndex,fIndex) = ( - ( curlB.Y.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*V1D.ne.subvec(1,NX_S-2)));
		}

		// Hall terms:
		// ===========
		if (hallTermSolve)
		{
			// "x" component:
			// ########################################################################
			// Bz and By need to be written in term of Bx in the paraxial approximation
			// #######################################################################

			EB->E.X.subvec(iIndex,fIndex) += (- V1D.V.Y.subvec(1,NX_S-2) % EB->B.Z.subvec(iIndex,fIndex));
			EB->E.X.subvec(iIndex,fIndex) += sqrt(params->em_IC.BX/(EB->B.X.subvec(iIndex,fIndex))) % ( V1D.V.Z.subvec(1,NX_S-2) % EB->B.Y.subvec(iIndex,fIndex));

			// "y" component:
			// ########################################################################
			// Bz and By need to be written in term of Bx in the paraxial approximation
			// ########################################################################
			EB->E.Y.subvec(iIndex,fIndex) += ( V1D.V.X.subvec(1,NX_S-2) % EB->B.Z.subvec(iIndex,fIndex));
			EB->E.Y.subvec(iIndex,fIndex) += (- V1D.V.Z.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex));

			// "z" component:
			// ########################################################################
			// Bz and By need to be written in term of Bx in the paraxial approximation
			// ########################################################################
			EB->E.Z.subvec(iIndex,fIndex) += sqrt(params->em_IC.BX/(EB->B.X.subvec(iIndex,fIndex))) % ( - V1D.V.X.subvec(1,NX_S-2) % EB->B.Y.subvec(iIndex,fIndex));
			EB->E.Z.subvec(iIndex,fIndex) += (V1D.V.Y.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex));
		}

		// Pressure term:
		// ==============
		if (pressureTermSolve)
		{
			// The partial derivative dn/dx is computed using centered finite differences with grid size DX/2
			V1D.dndx = 0.5*( V1D.ne.subvec(2,NX_S-1) - V1D.ne.subvec(0,NX_S-3) )/params->mesh.DX;

			// "x" component:
			EB->E.X.subvec(iIndex,fIndex) += - (params->f_IC.Te/F_E_DS)*V1D.dndx/V1D.ne.subvec(1,NX_S-2);

			// With scalar Te in 1D3V, there are not "y" and "z" components.
			// If Te becomes tensor, then mirror forces arise in "x" and then may also cause "y" and "z" components to be finite.

			// "y" component:

			// "z" component:
		}

		// Error checks:
		// ===============
		#ifdef CHECKS_ON
			if(!EB->E.X.is_finite())
			{
				cout << "Non finite values in Ex" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -110);
			}else if(!EB->E.Y.is_finite()){
				cout << "Non finite values in Ey" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -111);
			}else if(!EB->E.Z.is_finite()){
				cout << "Non finite values in Ez" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -112);
			}
		#endif

		MPI_Allgathervfield_vec(params, &EB->E);

		smooth(&EB->E, params->smoothingParameter);
	}

	// Extrapolation:
	// ==============
	if (extrap)
	{
		MPI_SendVec(params, &EB->E.X);
		MPI_SendVec(params, &EB->E.Y);
		MPI_SendVec(params, &EB->E.Z);
	}
}


void EMF_SOLVER::advanceEField(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, bool extrap, bool BAE){
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
		MPI_passGhosts(params,&EB->E);
		MPI_passGhosts(params,&EB->B);

		// Indices of subdomain
		unsigned int irow = params->mpi.irow;
		unsigned int frow = params->mpi.frow;
		unsigned int icol = params->mpi.icol;
		unsigned int fcol = params->mpi.fcol;

		V2D.ne.zeros();
		V2D.V.zeros();

		// We calculate the number density and bulk velocities at time level "l + 1/2"
		V2D.n.zeros();
		V2D.U.zeros();
		for(int ii=0; ii<params->numberOfParticleSpecies; ii++){
				fillGhosts(&IONS->at(ii).n);
				fillGhosts(&IONS->at(ii).nv.X);
				fillGhosts(&IONS->at(ii).nv.Y);
				fillGhosts(&IONS->at(ii).nv.Z);

				// Ions density at time level "l + 1/2"
				// n(l+1/2) = ( n(l+1) + n(l) )/2
				V2D.n += 0.5*IONS->at(ii).Z*( IONS->at(ii).n.submat(irow-1,icol-1,frow+1,fcol+1) + IONS->at(ii).n_.submat(irow-1,icol-1,frow+1,fcol+1) );

				// Ions bulk velocity at time level "l + 1/2"
				//sum_k[ Z_k*n_k*u_k ]
				V2D.U.X += IONS->at(ii).Z*IONS->at(ii).nv.X.submat(irow-1,icol-1,frow+1,fcol+1);
				V2D.U.Y += IONS->at(ii).Z*IONS->at(ii).nv.Y.submat(irow-1,icol-1,frow+1,fcol+1);
				V2D.U.Z += IONS->at(ii).Z*IONS->at(ii).nv.Z.submat(irow-1,icol-1,frow+1,fcol+1);
		}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

		V2D.U.X /= V2D.n;
		V2D.U.Y /= V2D.n;
		V2D.U.Z /= V2D.n;

		V2D.n /= n_cs;

		// We perform the following computations of the number density and bulk velocity at different past times
		if (extrap){
			V2D._n.zeros();
			V2D.n_.zeros();
			V2D.U_.zeros();
			for(int ii=0; ii<params->numberOfParticleSpecies; ii++){
				fillGhosts(&IONS->at(ii).n);
				fillGhosts(&IONS->at(ii).n_);
				fillGhosts(&IONS->at(ii).n__);
				fillGhosts(&IONS->at(ii).nv_.X);
				fillGhosts(&IONS->at(ii).nv_.Y);
				fillGhosts(&IONS->at(ii).nv_.Z);

				// Electron density at time level "l + 1"
				V2D._n += IONS->at(ii).Z*IONS->at(ii).n.submat(irow-1,icol-1,frow+1,fcol+1);

				// Ions density at time level "l - 1/2"
				// n(l-1/2) = ( n(l) + n(l-1) )/2
				V2D.n_ += 0.5*IONS->at(ii).Z*( IONS->at(ii).n__.submat(irow-1,icol-1,frow+1,fcol+1) + IONS->at(ii).n_.submat(irow-1,icol-1,frow+1,fcol+1) );

				// Ions bulk velocity at time level "l - 1/2"
				//sum_k[ Z_k*n_k*u_k ]
				V2D.U_.X += IONS->at(ii).Z*IONS->at(ii).nv_.X.submat(irow-1,icol-1,frow+1,fcol+1);
				V2D.U_.Y += IONS->at(ii).Z*IONS->at(ii).nv_.Y.submat(irow-1,icol-1,frow+1,fcol+1);
				V2D.U_.Z += IONS->at(ii).Z*IONS->at(ii).nv_.Z.submat(irow-1,icol-1,frow+1,fcol+1);
			}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

			V2D.U_.X /= V2D.n_;
			V2D.U_.Y /= V2D.n_;
			V2D.U_.Z /= V2D.n_;

			V2D._n /= n_cs;

			if(BAE){
				V2D.n__.zeros();
				V2D.U__.zeros();
				for(int ii=0; ii<params->numberOfParticleSpecies; ii++){
					fillGhosts(&IONS->at(ii).n___);
					fillGhosts(&IONS->at(ii).nv__.X);
					fillGhosts(&IONS->at(ii).nv__.Y);
					fillGhosts(&IONS->at(ii).nv__.Z);

					// Ions density at time level "l - 3/2"
					// n(l-3/2) = ( n(l-1) + n(l-2) )/2
					V2D.n__ += 0.5*IONS->at(ii).Z*( IONS->at(ii).n__.submat(irow-1,icol-1,frow+1,fcol+1) + IONS->at(ii).n___.submat(irow-1,icol-1,frow+1,fcol+1) );

					//sum_k[ Z_k*n_k*u_k ]
					V2D.U__.X += IONS->at(ii).Z*IONS->at(ii).nv__.X.submat(irow-1,icol-1,frow+1,fcol+1);
					V2D.U__.Y += IONS->at(ii).Z*IONS->at(ii).nv__.Y.submat(irow-1,icol-1,frow+1,fcol+1);
					V2D.U__.Z += IONS->at(ii).Z*IONS->at(ii).nv__.Z.submat(irow-1,icol-1,frow+1,fcol+1);
				}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

				V2D.U__.X /= V2D.n__;
				V2D.U__.Y /= V2D.n__;
				V2D.U__.Z /= V2D.n__;

				//Here we use the velocity extrapolation U^(N+1) = 2*U^(N+1/2) - 1.5*U^(N-1/2) + 0.5*U^(N-3/2)
				V2D.V = 2.0*V2D.U - 1.5*V2D.U_ + 0.5*V2D.U__;
			}else{
				//Here we use the velocity extrapolation U^(N+1) = 1.5*U^(N+1/2) - 0.5*U^(N-1/2)
				V2D.V = 1.5*V2D.U - 0.5*V2D.U_;
			}

			V2D.ne = V2D._n;
		}else{
			V2D.V = V2D.U;
			V2D.ne = V2D.n;
		}

		// We compute the curl(B).
		vfield_mat curlB(NX_T,NY_T);

		// x-component
		curlB.X.submat(irow,icol,frow,fcol) =    0.5*(EB->B.Z.submat(irow,icol+1,frow,fcol+1) - EB->B.Z.submat(irow,icol-1,frow,fcol-1))/params->mesh.DY;
		// y-component
		curlB.Y.submat(irow,icol,frow,fcol) =  - 0.5*(EB->B.Z.submat(irow+1,icol,frow+1,fcol) - EB->B.Z.submat(irow-1,icol,frow-1,fcol))/params->mesh.DX;
		// z-component
		curlB.Z.submat(irow,icol,frow,fcol) =    0.5*(EB->B.Y.submat(irow+1,icol,frow+1,fcol) - EB->B.Y.submat(irow-1,icol,frow-1,fcol))/params->mesh.DX;
		curlB.Z.submat(irow,icol,frow,fcol) += - 0.5*(EB->B.X.submat(irow,icol+1,frow,fcol+1) - EB->B.X.submat(irow,icol-1,frow,fcol-1))/params->mesh.DY;

		EB->E.zeros();

		// x-component

		EB->E.X.submat(irow,icol,frow,fcol) =   ( curlB.Y.submat(irow,icol,frow,fcol) % EB->B.Z.submat(irow,icol,frow,fcol) - curlB.Z.submat(irow,icol,frow,fcol) % EB->B.Y.submat(irow,icol,frow,fcol) )/( F_MU_DS*F_E_DS*V2D.ne.submat(1,1,NX_S-2,NY_S-2) );

		EB->E.X.submat(irow,icol,frow,fcol) += - V2D.V.Y.submat(1,1,NX_S-2,NY_S-2) % EB->B.Z.submat(irow,icol,frow,fcol);

		EB->E.X.submat(irow,icol,frow,fcol) +=   V2D.V.Z.submat(1,1,NX_S-2,NY_S-2) % EB->B.Y.submat(irow,icol,frow,fcol);

		V2D.dndx = 0.5*( V2D.ne.submat(2,1,NX_S-1,NY_S-2) - V2D.ne.submat(0,1,NX_S-3,NY_S-2) )/params->mesh.DX;

		EB->E.X.submat(irow,icol,frow,fcol) += - ( params->f_IC.Te/F_E_DS )*V2D.dndx/V2D.ne.submat(1,1,NX_S-2,NY_S-2);

		// y-component

		EB->E.Y.submat(irow,icol,frow,fcol) = - ( curlB.X.submat(irow,icol,frow,fcol) % EB->B.Z.submat(irow,icol,frow,fcol) - curlB.Z.submat(irow,icol,frow,fcol) % EB->B.X.submat(irow,icol,frow,fcol) )/( F_MU_DS*F_E_DS*V2D.ne.submat(1,1,NX_S-2,NY_S-2) );

		EB->E.Y.submat(irow,icol,frow,fcol) +=   V2D.V.X.submat(1,1,NX_S-2,NY_S-2) % EB->B.Z.submat(irow,icol,frow,fcol);

		EB->E.Y.submat(irow,icol,frow,fcol) += - V2D.V.Z.submat(1,1,NX_S-2,NY_S-2) % EB->B.X.submat(irow,icol,frow,fcol);

		V2D.dndy = 0.5*( V2D.ne.submat(1,2,NX_S-2,NY_S-1) - V2D.ne.submat(1,0,NX_S-2,NY_S-3) )/params->mesh.DY;

		EB->E.Y.submat(irow,icol,frow,fcol) += - ( params->f_IC.Te/F_E_DS )*V2D.dndy/V2D.ne.submat(1,1,NX_S-2,NY_S-2);

		// z-component

		EB->E.Z.submat(irow,icol,frow,fcol) =   ( curlB.X.submat(irow,icol,frow,fcol) % EB->B.Y.submat(irow,icol,frow,fcol) - curlB.Y.submat(irow,icol,frow,fcol) % EB->B.X.submat(irow,icol,frow,fcol) )/( F_MU_DS*F_E_DS*V2D.ne.submat(1,1,NX_S-2,NY_S-2) );

		EB->E.Z.submat(irow,icol,frow,fcol) = - V2D.V.X.submat(1,1,NX_S-2,NY_S-2) % EB->B.Y.submat(irow,icol,frow,fcol);

		EB->E.Z.submat(irow,icol,frow,fcol) =   V2D.V.Y.submat(1,1,NX_S-2,NY_S-2) % EB->B.X.submat(irow,icol,frow,fcol);

		#ifdef CHECKS_ON
			if(!EB->E.X.is_finite()){
				cout << "Non finite values in Ex" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -110);
			}else if(!EB->E.Y.is_finite()){
				cout << "Non finite values in Ey" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -111);
			}else if(!EB->E.Z.is_finite()){
				cout << "Non finite values in Ez" << endl;
				MPI_Abort(params->mpi.MPI_TOPO, -112);
			}
		#endif

		MPI_Allgathervfield_mat(params, &EB->E);

		smooth(&EB->E, params->smoothingParameter);
	}

	if (extrap){
		MPI_SendMat(params, &EB->E.X);
		MPI_SendMat(params, &EB->E.Y);
		MPI_SendMat(params, &EB->E.Z);
	}
}
