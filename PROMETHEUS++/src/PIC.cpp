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


void PIC::MPI_AllreduceVec(const simulationParameters * params, arma::vec * v){
	arma::vec recvbuf = zeros(v->n_elem);

	MPI_Allreduce(v->memptr(), recvbuf.memptr(), v->n_elem, MPI_DOUBLE, MPI_SUM, params->mpi.MPI_TOPO);

	*v = recvbuf;
}


void PIC::MPI_AllreduceMat(const simulationParameters * params, arma::mat * m){
	arma::mat recvbuf = zeros(m->n_rows, m->n_cols);

	MPI_Allreduce(m->memptr(), recvbuf.memptr(), m->n_elem, MPI_DOUBLE, MPI_SUM, params->mpi.MPI_TOPO);

	*m = recvbuf;
}


void PIC::MPI_SendVec(const simulationParameters * params, arma::vec * v){
	// We send the vector from root process of particles to root process of fields
	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Send(v->memptr(), v->n_elem, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, PARTICLES_TAG, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Recv(v->memptr(), v->n_elem, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, PARTICLES_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Then, the vevtor is broadcasted to all processes in the fields communicator
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
		MPI_Bcast(v->memptr(), v->n_elem, MPI_DOUBLE, 0, params->mpi.COMM);
}


void PIC::MPI_ReduceVec(const simulationParameters * params, arma::vec * v){
	arma::vec recvbuf = zeros(v->n_elem);

	// We reduce the vector at the root process of particles
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
		MPI_Reduce(v->memptr(), recvbuf.memptr(), v->n_elem, MPI_DOUBLE, MPI_SUM, 0, params->mpi.COMM);

		if (params->mpi.IS_PARTICLES_ROOT)
			*v = recvbuf;
	}
}


void PIC::MPI_SendMat(const simulationParameters * params, arma::mat * m){
	// We send the vector from root process of particles to root process of fields
	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Send(m->memptr(), m->n_elem, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, PARTICLES_TAG, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Recv(m->memptr(), m->n_elem, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, PARTICLES_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}

	// Then, the vevtor is broadcasted to all processes in the fields communicator
	if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR)
		MPI_Bcast(m->memptr(), m->n_elem, MPI_DOUBLE, 0, params->mpi.COMM);
}


void PIC::MPI_ReduceMat(const simulationParameters * params, arma::mat * m){
	arma::mat recvbuf = zeros(m->n_rows, m->n_cols);

	// We reduce the vector at the root process of particles
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
		MPI_Reduce(m->memptr(), recvbuf.memptr(), m->n_elem, MPI_DOUBLE, MPI_SUM, 0, params->mpi.COMM);

		if (params->mpi.IS_PARTICLES_ROOT)
			*m = recvbuf;
	}
}


void PIC::MPI_Allgathervec(const simulationParameters * params, arma::vec * field){
	unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	//Allgather for x-component
	sendbuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
}


void PIC::MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield){
	MPI_Allgathervec(params, &vfield->X);
	MPI_Allgathervec(params, &vfield->Y);
	MPI_Allgathervec(params, &vfield->Z);
}


void PIC::MPI_Allgathermat(const simulationParameters * params, arma::mat * field){
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;

	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	arma::vec recvbuf = zeros(params->mesh.NUM_CELLS_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NUM_CELLS_PER_MPI);

	sendbuf = vectorise(field->submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;

		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NY_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NY_PER_MPI;

		field->submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}

}


void PIC::MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield){
	MPI_Allgathermat(params, &vfield->X);
	MPI_Allgathermat(params, &vfield->Y);
	MPI_Allgathermat(params, &vfield->Z);
}


void PIC::MPI_Recvvec(const simulationParameters * params, arma::vec * field){
	// We send the vector from root process of fields to root process of particles
	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_IN_SIM);

	sendbuf = field->subvec(1, params->mesh.NX_IN_SIM);

	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Send(sendbuf.memptr(), params->mesh.NX_IN_SIM, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, 0, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Recv(recvbuf.memptr(), params->mesh.NX_IN_SIM, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
	}

	// Then, the fields is broadcasted to all processes in the particles communicator COMM
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
		sendbuf = field->subvec(1, params->mesh.NX_IN_SIM);

		MPI_Bcast(sendbuf.memptr(), params->mesh.NX_IN_SIM, MPI_DOUBLE, 0, params->mpi.COMM);

		field->subvec(1, params->mesh.NX_IN_SIM) = sendbuf;
	}
}



void PIC::MPI_Recvvfield_vec(const simulationParameters * params, vfield_vec * vfield){
	MPI_Recvvec(params, &vfield->X);
	MPI_Recvvec(params, &vfield->Y);
	MPI_Recvvec(params, &vfield->Z);
}


void PIC::MPI_Recvmat(const simulationParameters * params, arma::mat * field){
	// Then, we send the vector from root process of fields to root process of particles
	arma::vec recvbuf = zeros(params->mesh.NUM_CELLS_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NUM_CELLS_IN_SIM);

	sendbuf = vectorise(field->submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM));

	if (params->mpi.IS_FIELDS_ROOT){
		MPI_Send(sendbuf.memptr(), params->mesh.NUM_CELLS_IN_SIM, MPI_DOUBLE, params->mpi.PARTICLES_ROOT_WORLD_RANK, 0, MPI_COMM_WORLD);
	}

	if (params->mpi.IS_PARTICLES_ROOT){
		MPI_Recv(recvbuf.memptr(), params->mesh.NUM_CELLS_IN_SIM, MPI_DOUBLE, params->mpi.FIELDS_ROOT_WORLD_RANK, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		field->submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) = reshape(recvbuf, params->mesh.NX_IN_SIM, params->mesh.NY_IN_SIM);
	}

	// Finally, the fields is broadcasted to all processes in the particles communicator COMM
	if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
		sendbuf = vectorise(field->submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM));

		MPI_Bcast(sendbuf.memptr(), params->mesh.NUM_CELLS_IN_SIM, MPI_DOUBLE, 0, params->mpi.COMM);

		field->submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) = reshape(sendbuf, params->mesh.NX_IN_SIM, params->mesh.NY_IN_SIM);
	}
}


void PIC::MPI_Recvvfield_mat(const simulationParameters * params, vfield_mat * vfield){
	MPI_Recvmat(params, &vfield->X);
	MPI_Recvmat(params, &vfield->Y);
	MPI_Recvmat(params, &vfield->Z);
}


// * * * Ghost contributions * * *
void PIC::fill4Ghosts(arma::vec * v){
	int N = v->n_elem;

	v->subvec(N-2,N-1) = v->subvec(2,3);
	v->subvec(0,1) = v->subvec(N-4,N-3);
}

void PIC::include4GhostsContributions(arma::vec * v){
	int N = v->n_elem;

	v->subvec(2,3) += v->subvec(N-2,N-1);
	v->subvec(N-4,N-3) += v->subvec(0,1);
}


void PIC::fill4Ghosts(arma::mat * m){
	int NX = m->n_rows;
	int NY = m->n_cols;

	// Sides

	m->submat(NX-2,2,NX-1,NY-3) = m->submat(2,2,3,NY-3); // left size along x-axis
	m->submat(0,2,1,NY-3) = m->submat(NX-4,2,NX-3,NY-3); // right size along x-axis

	m->submat(2,NY-2,NX-3,NY-1) = m->submat(2,2,NX-3,3); // left size along y-axis
	m->submat(2,0,NX-3,1) = m->submat(2,NY-4,NX-3,NY-3); // right size along y-axis

	// Corners

	m->submat(NX-2,NY-2,NX-1,NY-1) = m->submat(2,2,3,3); // left x-axis, left y-axis
	m->submat(0,NY-2,1,NY-1) = m->submat(NX-4,2,NX-3,3); // right x-axis, left y-axis
	m->submat(NX-2,0,NX-1,1) = m->submat(2,NY-4,3,NY-3); // left x-axis, right y-axis
	m->submat(0,0,1,1) = m->submat(NX-4,NY-4,NX-3,NY-3); // right x-axis, right y-axis
}


void PIC::include4GhostsContributions(arma::mat * m){
	int NX = m->n_rows;
	int NY = m->n_cols;

	// Sides

	m->submat(2,2,3,NY-3) += m->submat(NX-2,2,NX-1,NY-3); // left size along x-axis
	m->submat(NX-4,2,NX-3,NY-3) += m->submat(0,2,1,NY-3); // right size along x-axis

	m->submat(2,2,NX-3,3) += m->submat(2,NY-2,NX-3,NY-1); // left size along y-axis
	m->submat(2,NY-4,NX-3,NY-3) += m->submat(2,0,NX-3,1); // right size along y-axis

	// Corners

	m->submat(2,2,3,3) += m->submat(NX-2,NY-2,NX-1,NY-1); // left x-axis, left y-axis
	m->submat(NX-4,2,NX-3,3) += m->submat(0,NY-2,1,NY-1); // right x-axis, left y-axis
	m->submat(2,NY-4,3,NY-3) += m->submat(NX-2,0,NX-1,1); // left x-axis, right y-axis
	m->submat(NX-4,NY-4,NX-3,NY-3) += m->submat(0,0,1,1); // right x-axis, right y-axis
}

// * * * Ghost contributions * * *


// * * * Smoothing * * *
void PIC::smooth(arma::vec * v, double as){
	int NX(v->n_elem);

	arma::vec b = zeros(NX);

	double wc(0.75); 	// center weight
	double ws(0.125);	// sides weight

	//Step 1: Averaging process
	b.subvec(1, NX-2) = v->subvec(1, NX-2);

	fillGhosts(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1.0 - as)*v->subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


void PIC::smooth(arma::mat * m, double as){
	int NX(m->n_rows);
	int NY(m->n_cols);

	arma::mat b = zeros(NX,NY);

	double wc(9.0/16.0);
	double ws(3.0/32.0);
	double wcr(1.0/64.0);
	/*
	double wc(0.25);
	double ws(0.1250);
	double wcr(0.0625);
	*/

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


void PIC::smooth(vfield_vec * vf, double as){
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}


void PIC::smooth(vfield_mat * vf, double as){
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}
// * * * Smoothing * * *


void PIC::assignCell(const simulationParameters * params,  oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS){
	//This function assigns the particles to the closest mesh node depending in their position and
	//calculate the weights for the charge extrapolation and force interpolation
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x
	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);//number of superparticles

	double X = 0.0;
	bool LOGIC;

	IONS->wxc.zeros();
	IONS->wxl.zeros();
	IONS->wxr.zeros();

	#pragma omp parallel for default(none) shared(IONS, params) private(X, LOGIC) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++){
		IONS->mn(ii) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);

		if(IONS->mn(ii) != params->mesh.NX_IN_SIM){
			X = IONS->X(ii,0) - params->mesh.nodes.X(IONS->mn(ii));
		}else{
			X = IONS->X(ii,0) - params->mesh.LX;
		}

		// If , X > 0, then the particle is on the right of the meshnode
		LOGIC = X > 0.0;
		X = abs(X);

		IONS->wxc(ii) = 0.75 - (X/params->mesh.DX)*(X/params->mesh.DX);

		if(LOGIC){
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX + X)/params->mesh.DX)*(1.5 - (params->mesh.DX + X)/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX - X)/params->mesh.DX)*(1.5 - (params->mesh.DX - X)/params->mesh.DX);
		}else{
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX - X)/params->mesh.DX)*(1.5 - (params->mesh.DX - X)/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX + X)/params->mesh.DX)*(1.5 - (params->mesh.DX + X)/params->mesh.DX);
		}
	}

    #ifdef CHECKS_ON
	if(!IONS->mn.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
}


void PIC::assignCell(const simulationParameters * params,  twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS){
	//This function assigns the particles to the closest mesh node depending in their position and
	//calculate the weights for the charge extrapolation and force interpolation
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x
	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);//number of superparticles

	double X;
	double Y;
	bool LOGIC_X;
	bool LOGIC_Y;

	IONS->wxc.zeros();
	IONS->wxl.zeros();
	IONS->wxr.zeros();

	IONS->wyc.zeros();
	IONS->wyl.zeros();
	IONS->wyr.zeros();

	#pragma omp parallel for default(none) shared(IONS, params) private(X, Y, LOGIC_X, LOGIC_Y) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++){
		IONS->mn(ii,0) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);
		IONS->mn(ii,1) = floor((IONS->X(ii,1) + 0.5*params->mesh.DY)/params->mesh.DY);

		if(IONS->mn(ii,0) != params->mesh.NX_IN_SIM){
			X = IONS->X(ii,0) - params->mesh.nodes.X(IONS->mn(ii,0));
		}else{
			X = IONS->X(ii,0) - params->mesh.LX;
		}

		if(IONS->mn(ii,1) != params->mesh.NY_IN_SIM){
			Y = IONS->X(ii,1) - params->mesh.nodes.Y(IONS->mn(ii,1));
		}else{
			Y = IONS->X(ii,1) - params->mesh.LY;
		}

		LOGIC_X = X > 0;
		// If X > 0, then the particle is on the right of the meshnode
		X = abs(X);

		// If Y > 0, then the particle is on the right of the meshnode
		LOGIC_Y = Y > 0;
		Y = abs(Y);

		IONS->wxc(ii) = 0.75 - (X/params->mesh.DX)*(X/params->mesh.DX);
		IONS->wyc(ii) = 0.75 - (Y/params->mesh.DY)*(Y/params->mesh.DY);

		if(LOGIC_X){
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX + X)/params->mesh.DX)*(1.5 - (params->mesh.DX + X)/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX - X)/params->mesh.DX)*(1.5 - (params->mesh.DX - X)/params->mesh.DX);
		}else{
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX - X)/params->mesh.DX)*(1.5 - (params->mesh.DX - X)/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX + X)/params->mesh.DX)*(1.5 - (params->mesh.DX + X)/params->mesh.DX);
		}

		if(LOGIC_Y){
			IONS->wyl(ii) = 0.5*(1.5 - (params->mesh.DY + Y)/params->mesh.DY)*(1.5 - (params->mesh.DY + Y)/params->mesh.DY);
			IONS->wyr(ii) = 0.5*(1.5 - (params->mesh.DY - Y)/params->mesh.DY)*(1.5 - (params->mesh.DY - Y)/params->mesh.DY);
		}else{
			IONS->wyl(ii) = 0.5*(1.5 - (params->mesh.DY - Y)/params->mesh.DY)*(1.5 - (params->mesh.DY - Y)/params->mesh.DY);
			IONS->wyr(ii) = 0.5*(1.5 - (params->mesh.DY + Y)/params->mesh.DY)*(1.5 - (params->mesh.DY + Y)/params->mesh.DY);
		}
	}

    #ifdef CHECKS_ON
	if(!IONS->mn.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
}


void PIC::crossProduct(const arma::vec A, const arma::vec B, arma::vec * AxB){
	(*AxB)(0) = A(1)*B(2) - A(2)*B(1);
	(*AxB)(1) = A(2)*B(0) - A(0)*B(2);
	(*AxB)(2) = A(0)*B(1) - A(1)*B(0);
}


void PIC::crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB){
	AxB->set_size(A->n_rows, 3);//Here we set up the size of the matrix AxB.

	AxB->col(0) = A->col(1)%B->col(2) - A->col(2)%B->col(1);//(AxB)_x
	AxB->col(1) = A->col(2)%B->col(0) - A->col(0)%B->col(2);//(AxB)_y
	AxB->col(2) = A->col(0)%B->col(1) - A->col(1)%B->col(0);//(AxB)_z
}


void PIC::eiv(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);
	IONS->nv.zeros(); // Setting to zero the ions' bulk velocity

	#pragma omp parallel default(none) shared(params, IONS) firstprivate(NSP)
	{
		vfield_vec nv(params->mesh.NX_IN_SIM + 4);

		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->mn(ii) + 2;

			nv.X(ix-1) 	+= IONS->wxl(ii)*IONS->V(ii,0);
			nv.X(ix) 	+= IONS->wxc(ii)*IONS->V(ii,0);
			nv.X(ix+1) 	+= IONS->wxr(ii)*IONS->V(ii,0);

			nv.Y(ix-1) 	+= IONS->wxl(ii)*IONS->V(ii,1);
			nv.Y(ix) 	+= IONS->wxc(ii)*IONS->V(ii,1);
			nv.Y(ix+1) 	+= IONS->wxr(ii)*IONS->V(ii,1);

			nv.Z(ix-1) 	+= IONS->wxl(ii)*IONS->V(ii,2);
			nv.Z(ix) 	+= IONS->wxc(ii)*IONS->V(ii,2);
			nv.Z(ix+1) 	+= IONS->wxr(ii)*IONS->V(ii,2);
		}

		include4GhostsContributions(&nv.X);
		include4GhostsContributions(&nv.Y);
		include4GhostsContributions(&nv.Z);

		#pragma omp critical (update_bulk_velocity)
		{
		IONS->nv.X.subvec(1,params->mesh.NX_IN_SIM) += nv.X.subvec(2,params->mesh.NX_IN_SIM + 1);
		IONS->nv.Y.subvec(1,params->mesh.NX_IN_SIM) += nv.Y.subvec(2,params->mesh.NX_IN_SIM + 1);
		IONS->nv.Z.subvec(1,params->mesh.NX_IN_SIM) += nv.Z.subvec(2,params->mesh.NX_IN_SIM + 1);
		}

	}//End of the parallel region

	IONS->nv *= IONS->NCP/params->mesh.DX;
}


void PIC::extrapolateIonVelocity(const simulationParameters * params, oneDimensional::ionSpecies * IONS){

	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	eiv(params, IONS);
}


void PIC::eiv(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);
	IONS->nv.zeros(); // Setting to zero the ions' bulk velocity

	#pragma omp parallel default(none) shared(params, IONS) firstprivate(NSP)
	{
		vfield_mat nv(params->mesh.NX_IN_SIM + 4, params->mesh.NY_IN_SIM + 4);

		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->mn(ii,0) + 2;
			int iy = IONS->mn(ii,1) + 2;

			// x component
			nv.X(ix-1,iy) 	+= IONS->wxl(ii)*IONS->wyc(ii)*IONS->V(ii,0);
			nv.X(ix,iy)   	+= IONS->wxc(ii)*IONS->wyc(ii)*IONS->V(ii,0);
			nv.X(ix+1,iy) 	+= IONS->wxr(ii)*IONS->wyc(ii)*IONS->V(ii,0);

			nv.X(ix,iy-1) 	+= IONS->wxc(ii)*IONS->wyl(ii)*IONS->V(ii,0);
			nv.X(ix,iy+1) 	+= IONS->wxc(ii)*IONS->wyr(ii)*IONS->V(ii,0);

			nv.X(ix+1,iy-1) += IONS->wxr(ii)*IONS->wyl(ii)*IONS->V(ii,0);
			nv.X(ix+1,iy+1) += IONS->wxr(ii)*IONS->wyr(ii)*IONS->V(ii,0);

			nv.X(ix-1,iy-1) += IONS->wxl(ii)*IONS->wyl(ii)*IONS->V(ii,0);
			nv.X(ix-1,iy+1) += IONS->wxl(ii)*IONS->wyr(ii)*IONS->V(ii,0);

			// y component
			nv.Y(ix-1,iy) 	+= IONS->wxl(ii)*IONS->wyc(ii)*IONS->V(ii,1);
			nv.Y(ix,iy)   	+= IONS->wxc(ii)*IONS->wyc(ii)*IONS->V(ii,1);
			nv.Y(ix+1,iy) 	+= IONS->wxr(ii)*IONS->wyc(ii)*IONS->V(ii,1);

			nv.Y(ix,iy-1) 	+= IONS->wxc(ii)*IONS->wyl(ii)*IONS->V(ii,1);
			nv.Y(ix,iy+1) 	+= IONS->wxc(ii)*IONS->wyr(ii)*IONS->V(ii,1);

			nv.Y(ix+1,iy-1) += IONS->wxr(ii)*IONS->wyl(ii)*IONS->V(ii,1);
			nv.Y(ix+1,iy+1) += IONS->wxr(ii)*IONS->wyr(ii)*IONS->V(ii,1);

			nv.Y(ix-1,iy-1) += IONS->wxl(ii)*IONS->wyl(ii)*IONS->V(ii,1);
			nv.Y(ix-1,iy+1) += IONS->wxl(ii)*IONS->wyr(ii)*IONS->V(ii,1);

			// z component
			nv.Z(ix-1,iy) 	+= IONS->wxl(ii)*IONS->wyc(ii)*IONS->V(ii,2);
			nv.Z(ix,iy) 	+= IONS->wxc(ii)*IONS->wyc(ii)*IONS->V(ii,2);
			nv.Z(ix+1,iy) 	+= IONS->wxr(ii)*IONS->wyc(ii)*IONS->V(ii,2);

			nv.Z(ix,iy-1) 	+= IONS->wxc(ii)*IONS->wyl(ii)*IONS->V(ii,2);
			nv.Z(ix,iy+1) 	+= IONS->wxc(ii)*IONS->wyr(ii)*IONS->V(ii,2);

			nv.Z(ix+1,iy-1) += IONS->wxr(ii)*IONS->wyl(ii)*IONS->V(ii,2);
			nv.Z(ix+1,iy+1) += IONS->wxr(ii)*IONS->wyr(ii)*IONS->V(ii,2);

			nv.Z(ix-1,iy-1) += IONS->wxl(ii)*IONS->wyl(ii)*IONS->V(ii,2);
			nv.Z(ix-1,iy+1) += IONS->wxl(ii)*IONS->wyr(ii)*IONS->V(ii,2);
		}

		include4GhostsContributions(&nv.X);
		include4GhostsContributions(&nv.Y);
		include4GhostsContributions(&nv.Z);

		#pragma omp critical (update_bulk_velocity)
		{
		IONS->nv.X.submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) += nv.X.submat(2,2,params->mesh.NX_IN_SIM+1,params->mesh.NY_IN_SIM+1);
		IONS->nv.Y.submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) += nv.Y.submat(2,2,params->mesh.NX_IN_SIM+1,params->mesh.NY_IN_SIM+1);
		IONS->nv.Z.submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) += nv.Z.submat(2,2,params->mesh.NX_IN_SIM+1,params->mesh.NY_IN_SIM+1);
		}

	}//End of the parallel region

	IONS->nv *= IONS->NCP/(params->mesh.DX*params->mesh.DY);
}


void PIC::extrapolateIonVelocity(const simulationParameters * params, twoDimensional::ionSpecies * IONS){

	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	eiv(params, IONS);
}


void PIC::eid(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);

	IONS->n.zeros(); // Setting to zero the ion density.

	#pragma omp parallel default(none) shared(params, IONS) firstprivate(NSP)
	{
		arma::vec n = zeros(params->mesh.NX_IN_SIM + 4); // Four ghosht cells considereds

		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->mn(ii) + 2;

			n(ix-1) += IONS->wxl(ii);
			n(ix)   += IONS->wxc(ii);
			n(ix+1) += IONS->wxr(ii);
		}

		include4GhostsContributions(&n);

		#pragma omp critical (update_density)
		{
		IONS->n.subvec(1,params->mesh.NX_IN_SIM) += n.subvec(2,params->mesh.NX_IN_SIM + 1);
		}

	}//End of the parallel region
          
        
           //Adds compression effect
         IONS->n.subvec(1,params->mesh.NX_IN_SIM) = IONS->n.subvec(1,params->mesh.NX_IN_SIM) % (params->PP.Bx_i.subvec(1,params->mesh.NX_IN_SIM)/params->BGP.Bo);
          //Scaling the ion density with proper dimension
	IONS->n *= IONS->NCP/params->mesh.DX;
          
          
}


void PIC::eid(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);

	IONS->n.zeros(); // Setting to zero the ion density.

	#pragma omp parallel default(none) shared(params, IONS) firstprivate(NSP)
	{
		arma::mat n = zeros(params->mesh.NX_IN_SIM + 4, params->mesh.NY_IN_SIM + 4); // Four ghosht cells considereds

		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->mn(ii,0) + 2;
			int iy = IONS->mn(ii,1) + 2;

			n(ix-1,iy) 		+= IONS->wxl(ii)*IONS->wyc(ii);
			n(ix,iy) 		+= IONS->wxc(ii)*IONS->wyc(ii);
			n(ix+1,iy) 		+= IONS->wxr(ii)*IONS->wyc(ii);

			n(ix,iy-1) 		+= IONS->wxc(ii)*IONS->wyl(ii);
			n(ix,iy+1) 		+= IONS->wxc(ii)*IONS->wyr(ii);

			n(ix+1,iy-1) 	+= IONS->wxr(ii)*IONS->wyl(ii);
			n(ix+1,iy+1) 	+= IONS->wxr(ii)*IONS->wyr(ii);

			n(ix-1,iy-1) 	+= IONS->wxl(ii)*IONS->wyl(ii);
			n(ix-1,iy+1) 	+= IONS->wxl(ii)*IONS->wyr(ii);
		}

		include4GhostsContributions(&n);

		#pragma omp critical (update_density)
		{
		IONS->n.submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) += n.submat(2,2,params->mesh.NX_IN_SIM+1,params->mesh.NY_IN_SIM+1);
		}

	}//End of the parallel region

	IONS->n *= IONS->NCP/(params->mesh.DX*params->mesh.DY);
}


void PIC::extrapolateIonDensity(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
	// First, the particle density of time steps (it - 1) is kept in n_, (it - 2) is kept in n__,
	// and at the time iteration (it - 3) in n___.

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	// Then, the ions' properties are distributed in the grid following the charge deposition of the Triangular
	// Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	eid(params, IONS);
}


void PIC::extrapolateIonDensity(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
	// First, the particle density of time steps (it - 1) is kept in n_, (it - 2) is kept in n__,
	// and at the time iteration (it - 3) in n___.

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	// Then, the ions' properties are distributed in the grid following the charge deposition of the Triangular
	// Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	eid(params, IONS);
}


void PIC::interpolateVectorField(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, vfield_vec * field, arma::mat * F){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NX =  params->mesh.NX_IN_SIM + 4;//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);

	arma::vec field_X = zeros(NX);
	arma::vec field_Y = zeros(NX);
	arma::vec field_Z = zeros(NX);

	field_X.subvec(1,NX-2) = field->X;
	field_Y.subvec(1,NX-2) = field->Y;
	field_Z.subvec(1,NX-2) = field->Z;

	fill4Ghosts(&field_X);
	fill4Ghosts(&field_Y);
	fill4Ghosts(&field_Z);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for default(none) shared(params, IONS, F, field_X, field_Y, field_Z) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++){
		int ix = IONS->mn(ii) + 2;

		(*F)(ii,0) += IONS->wxl(ii)*field_X(ix-1);
		(*F)(ii,1) += IONS->wxl(ii)*field_Y(ix-1);
		(*F)(ii,2) += IONS->wxl(ii)*field_Z(ix-1);

		(*F)(ii,0) += IONS->wxc(ii)*field_X(ix);
		(*F)(ii,1) += IONS->wxc(ii)*field_Y(ix);
		(*F)(ii,2) += IONS->wxc(ii)*field_Z(ix);

		(*F)(ii,0) += IONS->wxr(ii)*field_X(ix+1);
		(*F)(ii,1) += IONS->wxr(ii)*field_Y(ix+1);
		(*F)(ii,2) += IONS->wxr(ii)*field_Z(ix+1);
	}//End of the parallel region
}


void PIC::interpolateElectromagneticFields(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, oneDimensional::fields * EB, arma::mat * E, arma::mat * B){
	interpolateVectorField(params, IONS, &EB->E, E);
	interpolateVectorField(params, IONS, &EB->B, B);
}


void PIC::interpolateVectorField(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, vfield_mat * field, arma::mat * F){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NX =  params->mesh.NX_IN_SIM + 4; // Mesh size along the X axis (considering 4 gosht cell)
	int NY =  params->mesh.NY_IN_SIM + 4; // Mesh size along the Y axis (considering 4 gosht cell)
	int NSP(IONS->NSP);

	arma::mat field_X = zeros(NX,NY);
	arma::mat field_Y = zeros(NX,NY);
	arma::mat field_Z = zeros(NX,NY);

	field_X.submat(1,1,NX-2,NY-2) = field->X;
	field_Y.submat(1,1,NX-2,NY-2) = field->Y;
	field_Z.submat(1,1,NX-2,NY-2) = field->Z;

	fill4Ghosts(&field_X);
	fill4Ghosts(&field_Y);
	fill4Ghosts(&field_Z);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for default(none) shared(params, IONS, F, field_X, field_Y, field_Z) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++){
		int ix = IONS->mn(ii,0) + 2;
		int iy = IONS->mn(ii,1) + 2;

		// x component
		(*F)(ii,0) += IONS->wxl(ii)*IONS->wyc(ii)*field_X(ix-1,iy);
		(*F)(ii,0) += IONS->wxc(ii)*IONS->wyc(ii)*field_X(ix,iy);
		(*F)(ii,0) += IONS->wxr(ii)*IONS->wyc(ii)*field_X(ix+1,iy);

		(*F)(ii,0) += IONS->wxc(ii)*IONS->wyl(ii)*field_X(ix,iy-1);
		(*F)(ii,0) += IONS->wxc(ii)*IONS->wyr(ii)*field_X(ix,iy+1);

		(*F)(ii,0) += IONS->wxr(ii)*IONS->wyl(ii)*field_X(ix+1,iy-1);
		(*F)(ii,0) += IONS->wxr(ii)*IONS->wyr(ii)*field_X(ix+1,iy+1);

		(*F)(ii,0) += IONS->wxl(ii)*IONS->wyl(ii)*field_X(ix-1,iy-1);
		(*F)(ii,0) += IONS->wxl(ii)*IONS->wyr(ii)*field_X(ix-1,iy+1);

		// y component
		(*F)(ii,1) += IONS->wxl(ii)*IONS->wyc(ii)*field_Y(ix-1,iy);
		(*F)(ii,1) += IONS->wxc(ii)*IONS->wyc(ii)*field_Y(ix,iy);
		(*F)(ii,1) += IONS->wxr(ii)*IONS->wyc(ii)*field_Y(ix+1,iy);

		(*F)(ii,1) += IONS->wxc(ii)*IONS->wyl(ii)*field_Y(ix,iy-1);
		(*F)(ii,1) += IONS->wxc(ii)*IONS->wyr(ii)*field_Y(ix,iy+1);

		(*F)(ii,1) += IONS->wxr(ii)*IONS->wyl(ii)*field_Y(ix+1,iy-1);
		(*F)(ii,1) += IONS->wxr(ii)*IONS->wyr(ii)*field_Y(ix+1,iy+1);

		(*F)(ii,1) += IONS->wxl(ii)*IONS->wyl(ii)*field_Y(ix-1,iy-1);
		(*F)(ii,1) += IONS->wxl(ii)*IONS->wyr(ii)*field_Y(ix-1,iy+1);

		// z component
		(*F)(ii,2) += IONS->wxl(ii)*IONS->wyc(ii)*field_Z(ix-1,iy);
		(*F)(ii,2) += IONS->wxc(ii)*IONS->wyc(ii)*field_Z(ix,iy);
		(*F)(ii,2) += IONS->wxr(ii)*IONS->wyc(ii)*field_Z(ix+1,iy);

		(*F)(ii,2) += IONS->wxc(ii)*IONS->wyl(ii)*field_Z(ix,iy-1);
		(*F)(ii,2) += IONS->wxc(ii)*IONS->wyr(ii)*field_Z(ix,iy+1);

		(*F)(ii,2) += IONS->wxr(ii)*IONS->wyl(ii)*field_Z(ix+1,iy-1);
		(*F)(ii,2) += IONS->wxr(ii)*IONS->wyr(ii)*field_Z(ix+1,iy+1);

		(*F)(ii,2) += IONS->wxl(ii)*IONS->wyl(ii)*field_Z(ix-1,iy-1);
		(*F)(ii,2) += IONS->wxl(ii)*IONS->wyr(ii)*field_Z(ix-1,iy+1);
	}//End of the parallel region

}


void PIC::interpolateElectromagneticFields(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, twoDimensional::fields * EB, arma::mat * E, arma::mat * B){
	interpolateVectorField(params, IONS, &EB->E, E);
	interpolateVectorField(params, IONS, &EB->B, B);
}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT){
	MPI_Recvvfield_vec(params, &EB->E);
	MPI_Recvvfield_vec(params, &EB->B);

	fillGhosts(EB);

	for(int ii=0; ii<IONS->size(); ii++){//structure to iterate over all the ion species.
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
			arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

			interpolateElectromagneticFields(params, &IONS->at(ii), EB, &Ep, &Bp);

			IONS->at(ii).E = Ep;
			IONS->at(ii).B = Bp;

			//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
			int NSP = IONS->at(ii).NSP;
			double A = IONS->at(ii).Q*DT/IONS->at(ii).M; // A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)

			//#pragma omp parallel default(none) shared(IONS, params,Ep, Bp) firstprivate(ii, A, NSP, F_C_DS)
			{
				double gp;
				double sigma;
				double us;
				double s;
                                        double omega_ci;
                                        double rL_i;
                                        double dBx;
                                        double magMom;
                                        double Vpersq;
                                        double Vper;
                                        double sinPhi;
                                        double cosPhi;
                                        double UUx;
                                        double UUy;
                                        double UUz;
                                        double d1;
                                        double d2;
                                        double d3;
                                        double Term;
                                        
                                     
                                        
				arma::rowvec U = arma::zeros<rowvec>(3);
				arma::rowvec VxB = arma::zeros<rowvec>(3);
				arma::rowvec tau = arma::zeros<rowvec>(3);
				arma::rowvec up = arma::zeros<rowvec>(3);
				arma::rowvec t = arma::zeros<rowvec>(3);
				arma::rowvec upxt = arma::zeros<rowvec>(3);

				//#pragma omp for
				for(int ip=0;ip<NSP;ip++){
                                        
                                                  dBx = Bp(ip,1)/(-0.5*(params->BGP.Rphi0)); //for Phi_f = 0
                                                  Vpersq = IONS->at(ii).V(ip,1)*IONS->at(ii).V(ip,1)+IONS->at(ii).V(ip,2)*IONS->at(ii).V(ip,2);
                                                  Vper = sqrt(Vpersq);
                                                  omega_ci = (fabs(IONS->at(ii).Q)*Bp(ip,0)/IONS->at(ii).M);
                                                  rL_i = fabs(Vper/omega_ci);
                                                  UUx = arma::as_scalar(IONS->at(ii).V(ip,0));
                                                  UUy = arma::as_scalar(IONS->at(ii).V(ip,1));
                                                  UUz = arma::as_scalar(IONS->at(ii).V(ip,2));
                                                  
                                                  cosPhi = -(UUz/Vper);
                                                  sinPhi = +(UUy/Vper);
                                                 
                                                  Bp(ip,1) = -0.5*(params->BGP.Rphi0*sqrt(params->BGP.Bo/Bp(ip,0))+rL_i*cosPhi)*dBx;
                                                  Bp(ip,2) = -0.5*(rL_i*sinPhi)*dBx;
                                                  
                                                  VxB = arma::cross(IONS->at(ii).V.row(ip), Bp.row(ip));
                                                  
                                                  //Overwriting the x-component of VxB to simulate mirror force
                                                  //magMom = 0.5*IONS->at(ii).M*Vpersq/Bp(ip,0);
                                                  //VxB(0) = -(magMom/(fabs(IONS->at(ii).Q)))*dBx;
                                                  //VxB(1) = -UUx*(-0.5*rL_i*sinPhi*dBx) + UUz*Bp(ip,0);
                                                  //VxB(2) = +UUx*(-0.5*rL_i*cosPhi*dBx) - UUy*Bp(ip,0);
                                                  
                                                  
                                                  //To check KE conservation
                                                  //Term = dot(VxB,IONS->at(ii).V.row(ip));
                                                  //Term *= CS->velocity*CS->velocity*CS->bField;
                                                  //cout<<"The change in KE is " <<Term<<endl;
                                                  
					IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
					U = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

					U += 0.5*A*(Ep.row(ip) + VxB); // U_hs = U_L + 0.5*a*(E + cross(V, B)); % Half step for velocity
					tau = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
					up = U + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
					gp = sqrt( 1.0 + dot(up, up)/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
					sigma = gp*gp - dot(tau, tau); // sigma = gammap^2 - tau*tau';
					us = dot(up, tau)/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
					IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma + sqrt( sigma*sigma + 4.0*( dot(tau, tau) + us*us ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
					t = tau/IONS->at(ii).g(ip); 			// t = tau/gamma;
					s = 1.0/( 1.0 + dot(t, t) ); // s = 1/(1 + t*t'); % variable 's' in paper

					upxt = arma::cross(up, t);

					U = s*( up + dot(up, t)*t+ upxt ); 	// U_L = s*(up + (up*t')*t + cross(up, t));
					IONS->at(ii).V.row(ip) = U/IONS->at(ii).g(ip);	// V = U_L/gamma;

				}
			} // End of parallel region

			extrapolateIonVelocity(params, &IONS->at(ii));
		}

		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.X);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.Y);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.Z);

		for (int jj=0;jj<params->filtersPerIterationIons;jj++)
			smooth(&IONS->at(ii).nv, params->smoothingParameter);

		// Densities at various time levels are sent to fields processes
		PIC::MPI_SendVec(params, &IONS->at(ii).nv.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv.Z);

		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.Z);

		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.Z);
	}//structure to iterate over all the ion species.

	// Ghosts cells might be set to zero if needed, but before saving to HDF5 ghost cells need to be filled again.

}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT){
	MPI_Recvvfield_mat(params, &EB->E);
	MPI_Recvvfield_mat(params, &EB->B);

	fillGhosts(EB);

	for(int ii=0; ii<IONS->size(); ii++){//structure to iterate over all the ion species.
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
			arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

			interpolateElectromagneticFields(params, &IONS->at(ii), EB, &Ep, &Bp);

			IONS->at(ii).E = Ep;
			IONS->at(ii).B = Bp;

			//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
			int NSP = IONS->at(ii).NSP;
			double A = IONS->at(ii).Q*DT/IONS->at(ii).M; // A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)

			#pragma omp parallel default(none) shared(IONS, Ep, Bp) firstprivate(ii, A, NSP, F_C_DS)
			{
				double gp;
				double sigma;
				double us;
				double s;
				arma::rowvec U = arma::zeros<rowvec>(3);
				arma::rowvec VxB = arma::zeros<rowvec>(3);
				arma::rowvec tau = arma::zeros<rowvec>(3);
				arma::rowvec up = arma::zeros<rowvec>(3);
				arma::rowvec t = arma::zeros<rowvec>(3);
				arma::rowvec upxt = arma::zeros<rowvec>(3);

				#pragma omp for
				for(int ip=0;ip<NSP;ip++){
					VxB = arma::cross(IONS->at(ii).V.row(ip), Bp.row(ip));

					IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
					U = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

					U += 0.5*A*(Ep.row(ip) + VxB); // U_hs = U_L + 0.5*a*(E + cross(V, B)); % Half step for velocity
					tau = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
					up = U + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
					gp = sqrt( 1.0 + dot(up, up)/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
					sigma = gp*gp - dot(tau, tau); // sigma = gammap^2 - tau*tau';
					us = dot(up, tau)/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
					IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma + sqrt( sigma*sigma + 4.0*( dot(tau, tau) + us*us ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
					t = tau/IONS->at(ii).g(ip); 			// t = tau/gamma;
					s = 1.0/( 1.0 + dot(t, t) ); // s = 1/(1 + t*t'); % variable 's' in paper

					upxt = arma::cross(up, t);

					U = s*( up + dot(up, t)*t+ upxt ); 	// U_L = s*(up + (up*t')*t + cross(up, t));
					IONS->at(ii).V.row(ip) = U/IONS->at(ii).g(ip);	// V = U_L/gamma;

				}
			} // End of parallel region

			extrapolateIonVelocity(params, &IONS->at(ii));
		}

		PIC::MPI_ReduceMat(params, &IONS->at(ii).nv.X);
		PIC::MPI_ReduceMat(params, &IONS->at(ii).nv.Y);
		PIC::MPI_ReduceMat(params, &IONS->at(ii).nv.Z);

		for (int jj=0;jj<params->filtersPerIterationIons;jj++)
			smooth(&IONS->at(ii).nv, params->smoothingParameter);

		// Densities at various time levels are sent to fields processes
		PIC::MPI_SendMat(params, &IONS->at(ii).nv.X);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv.Y);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv.Z);

		PIC::MPI_SendMat(params, &IONS->at(ii).nv_.X);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv_.Y);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv_.Z);

		PIC::MPI_SendMat(params, &IONS->at(ii).nv__.X);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv__.Y);
		PIC::MPI_SendMat(params, &IONS->at(ii).nv__.Z);
	}//structure to iterate over all the ion species.
}



void PIC::advanceIonsPosition(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT){
    arma::vec x = {1.0, 0.0, 0.0};
    arma::vec y = {0.0, 1.0, 0.0};
    arma::vec z = {0.0, 0.0, 1.0};

    arma::vec b1; // Unitary vector along B field
    arma::vec b2; // Unitary vector perpendicular to b1
    arma::vec b3; // Unitary vector perpendicular to b1 and b2
    
	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			int NSP(IONS->at(ii).NSP);
                             

			#pragma omp parallel default(none) shared(params, IONS, x, y, z,std::cout) firstprivate(DT, NSP, ii, b1, b2, b3)
			{
                                        int pc = 0;
                                        double ec = 0;
				#pragma omp for
				for(int ip=0; ip<NSP; ip++)
                                                        {
					IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);

                                                       if((IONS->at(ii).X(ip,0) < 0)||(IONS->at(ii).X(ip,0) > params->mesh.LX))
                                                       {
                                                       //Record events
                                                        pc += 1; //Increase the leaking particles by one
                                                        ec += 0.5*IONS->at(ii).M*dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip));//Increase the leaking particles KE
                                                        
                                                        
                                                        //Variables for random number generator
                                                        arma::vec R = randu(1);
                                                        arma_rng::set_seed_random();
                                                        arma::vec phi = 2.0*M_PI*randu<vec>(1);

                                                        // Gaussian distribution in space for particle position:
                                                        double Xcenter = params->mesh.LX/2;
                                                        double sigmaX  = params->mesh.LX/10;
                                                        double Xnew = Xcenter  + (sigmaX)*sqrt( -2*log(R(0)) )*cos(phi(0));
                                                        double dLX = abs(Xnew - Xcenter);
                                                        
                                                        
                                                        while(dLX > params->mesh.LX/2){
                                                             std::cout<<"Out of bound X= "<< Xnew;
                                                             arma_rng::set_seed_random();
                                                             R = randu(1);
                                                             phi = 2.0*M_PI*randu<vec>(1);
                                                             
                                                             Xnew = Xcenter  + (sigmaX)*sqrt( -2*log(R(0)) )*cos(phi(0));
                                                             dLX  = abs(Xnew - Xcenter);
                                                             std::cout<< "Out of bound corrected X= " << Xnew;
                                                        }
                                                        
                                                        IONS->at(ii).X(ip,0) = Xnew;
                                                        
                                                        //Box Muller in velocity space
                                                        arma_rng::set_seed_random();
                                                        R = randu(1);
                                                        phi = 2.0*M_PI*randu<vec>(1);

                                                        arma::vec V2 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % cos(phi);
                                                        arma::vec V3 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % sin(phi);

                                                        arma_rng::set_seed_random();
                                                        phi = 2.0*M_PI*randu<vec>(1);

                                                        arma::vec V1 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % sin(phi);

                                                        // Creating magnetic field unit vectors: 
                                                        //Unit vectors have to take care of non-unifoprm B-field - To be done later

                                                        b1 = {params->BGP.Bx, params->BGP.By, params->BGP.Bz};
                                                        b1 = arma::normalise(b1);

                                                        if (arma::dot(b1,y) < PRO_ZERO){
                                                            b2 = arma::cross(b1,y);
                                                        }else{
                                                            b2 = arma::cross(b1,z);
                                                        }

                                                        // Unitary vector perpendicular to b1 and b2
                                                        b3 = arma::cross(b1,b2);

                                                        IONS->at(ii).V(ip,0) = V1(0)*dot(b1,x) + V2(0)*dot(b2,x) + V3(0)*dot(b3,x);
                                                        IONS->at(ii).V(ip,1) = V1(0)*dot(b1,y) + V2(0)*dot(b2,y) + V3(0)*dot(b3,y);
                                                        IONS->at(ii).V(ip,2) = V1(0)*dot(b1,z) + V2(0)*dot(b2,z) + V3(0)*dot(b3,z);

                                                        IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 - dot(IONS->at(ii).V.row(ip),IONS->at(ii).V.row(ip))/(F_C*F_C) );
                                                        IONS->at(ii).mu(ip) = 0.5*IONS->at(ii).g(ip)*IONS->at(ii).g(ip)*IONS->at(ii).M*( V2(0)*V2(0) + V3(0)*V3(0) )/params->BGP.Bo;
                                                        IONS->at(ii).Ppar(ip) = IONS->at(ii).g(ip)*IONS->at(ii).M*V1(0);
                                                        IONS->at(ii).avg_mu = mean(IONS->at(ii).mu);



				}
                                      
                              
				}
                              #pragma omp critical 
                              
                              IONS->at(ii).pCount += pc;
                              IONS->at(ii).eCount += ec;
                              
			}//End of the parallel region

			PIC::assignCell(params, EB, &IONS->at(ii));

			//Once the ions have been pushed,  we extrapolate the density at the node grids.
			extrapolateIonDensity(params, &IONS->at(ii));
		}

		PIC::MPI_ReduceVec(params, &IONS->at(ii).n);

		for (int jj=0; jj<params->filtersPerIterationIons; jj++)
			smooth(&IONS->at(ii).n, params->smoothingParameter);

		// Densities at various time levels are sent to fields processes
		PIC::MPI_SendVec(params, &IONS->at(ii).n);
		PIC::MPI_SendVec(params, &IONS->at(ii).n_);
		PIC::MPI_SendVec(params, &IONS->at(ii).n__);
		PIC::MPI_SendVec(params, &IONS->at(ii).n___);
	} // structure to iterate over all the ion species.
}


void PIC::advanceIonsPosition(const simulationParameters * params,  twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT){
	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			int NSP(IONS->at(ii).NSP);

			#pragma omp parallel default(none) shared(params, IONS) firstprivate(DT, NSP, ii)
			{
				#pragma omp for
				for(int ip=0; ip<NSP; ip++){
					IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0); // x
					IONS->at(ii).X(ip,1) += DT*IONS->at(ii).V(ip,1); // y

	                IONS->at(ii).X(ip,0) = fmod(IONS->at(ii).X(ip,0), params->mesh.LX); // Periodic condition along x-axis
					IONS->at(ii).X(ip,1) = fmod(IONS->at(ii).X(ip,1), params->mesh.LY); // Periodic condition along y-axis

	                if(IONS->at(ii).X(ip,0) < 0)
	        			IONS->at(ii).X(ip,0) += params->mesh.LX;

					if(IONS->at(ii).X(ip,1) < 0)
		        		IONS->at(ii).X(ip,1) += params->mesh.LY;
				}
			}//End of the parallel region

			PIC::assignCell(params, EB, &IONS->at(ii));

			extrapolateIonDensity(params, &IONS->at(ii));//Once the ions have been pushed,  we extrapolate the density at the node grids.
		}

		PIC::MPI_ReduceMat(params, &IONS->at(ii).n);

		for (int jj=0; jj<params->filtersPerIterationIons; jj++)
			smooth(&IONS->at(ii).n, params->smoothingParameter);

		// Densities at various time levels are sent to fields processes
		PIC::MPI_SendMat(params, &IONS->at(ii).n);
		PIC::MPI_SendMat(params, &IONS->at(ii).n_);
		PIC::MPI_SendMat(params, &IONS->at(ii).n__);
		PIC::MPI_SendMat(params, &IONS->at(ii).n___);
	}//structure to iterate over all the ion species.
}


// template class PIC<oneDimensional::ionSpecies, oneDimensional::fields>;
// template class PIC<twoDimensional::ionSpecies, twoDimensional::fields>;
