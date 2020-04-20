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
	arma::mat recvbuf = zeros(m->n_rows,m->n_cols);

	MPI_Allreduce(m->memptr(), recvbuf.memptr(), m->n_elem, MPI_DOUBLE, MPI_SUM, params->mpi.MPI_TOPO);

	*m = recvbuf;
}


void PIC::MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));
	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	//Allgather for x-component
	sendbuf = vfield->X.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	vfield->X.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;

	//Allgather for y-component
	sendbuf = vfield->Y.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	vfield->Y.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;

	//Allgather for z-component
	sendbuf = vfield->Z.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	vfield->Z.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
}


void PIC::MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield){
	unsigned int irow = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART))*params->mesh.NX_PER_MPI + 1;
	unsigned int frow = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)) + 1)*params->mesh.NX_PER_MPI;
	unsigned int icol = *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1)*params->mesh.NX_PER_MPI + 1;
	unsigned int fcol = ( *(params->mpi.MPI_CART_COORDS.at(params->mpi.MPI_DOMAIN_NUMBER_CART)+1) + 1)*params->mesh.NX_PER_MPI;

	arma::vec recvbuf = zeros(params->mesh.NX_IN_SIM*params->mesh.NY_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI);

	//Allgather for x-component
	sendbuf = vectorise(vfield->X.submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NX_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;
		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NX_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NX_PER_MPI;

		vfield->X.submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}

	recvbuf.zeros();

	//Allgather for y-component
	sendbuf = vectorise(vfield->Y.submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NX_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;
		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NX_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NX_PER_MPI;

		vfield->Y.submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}

	recvbuf.zeros();

	//Allgather for x-component
	sendbuf = vectorise(vfield->Z.submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_NODES_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NX_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;
		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NX_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NX_PER_MPI;

		vfield->Z.submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}
}


void PIC::MPI_Allgathervec(const simulationParameters * params, arma::vec * field){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));
	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	//Allgather for x-component
	sendbuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
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

//! Function to interpolate electromagnetic fields from staggered grid to non-staggered grid.

//! Linear interpolation is used to calculate the values of the fields in a non-staggered grid.
void PIC::computeFieldsOnNonStaggeredGrid(oneDimensional::fields * F, oneDimensional::fields * G){
	int NX = F->E.X.n_elem;

	G->E.X.subvec(1,NX-2) = 0.5*( F->E.X.subvec(1,NX-2) + F->E.X.subvec(0,NX-3) );
	G->E.Y.subvec(1,NX-2) = F->E.Y.subvec(1,NX-2);
	G->E.Z.subvec(1,NX-2) = F->E.Z.subvec(1,NX-2);

	G->B.X.subvec(1,NX-2) = F->B.X.subvec(1,NX-2);
	G->B.Y.subvec(1,NX-2) = 0.5*( F->B.Y.subvec(1,NX-2) + F->B.Y.subvec(0,NX-3) );
	G->B.Z.subvec(1,NX-2) = 0.5*( F->B.Z.subvec(1,NX-2) + F->B.Z.subvec(0,NX-3) );
}



//! Function to interpolate electromagnetic fields from staggered grid to non-staggered grid.

//! Bilinear interpolation is used to calculate the values of the fields in a non-staggered grid.
void PIC::computeFieldsOnNonStaggeredGrid(twoDimensional::fields * F, twoDimensional::fields * G){
	int NX = F->E.X.n_rows;
	int NY = F->E.X.n_cols;

	G->E.X.submat(1,1,NX-2,NY-2) = 0.5*( F->E.X.submat(0,1,NX-3,NY-2) + F->E.X.submat(1,1,NX-2,NY-2) );
	G->E.Y.submat(1,1,NX-2,NY-2) = 0.5*( F->E.Y.submat(1,0,NX-2,NY-3) + F->E.Y.submat(1,1,NX-2,NY-2) );
	G->E.Z.submat(1,1,NX-2,NY-2) = F->E.Z.submat(1,1,NX-2,NY-2);

	G->B.X.submat(1,1,NX-2,NY-2) = 0.5*( F->B.X.submat(1,0,NX-2,NY-3) + F->B.X.submat(1,1,NX-2,NY-2) );
	G->B.Y.submat(1,1,NX-2,NY-2) = 0.5*( F->B.Y.submat(0,1,NX-3,NY-2) + F->B.Y.submat(1,1,NX-2,NY-2) );
	G->B.Z.submat(1,1,NX-2,NY-2) = 0.25*( F->B.Z.submat(0,0,NX-3,NY-3) + F->B.Z.submat(1,0,NX-2,NY-3) + F->B.Z.submat(0,1,NX-3,NY-2) + F->B.Z.submat(1,1,NX-2,NY-2) );
}



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


void PIC::assignCell(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
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

	arma::vec X = zeros(NSP);
	uvec LOGIC;

	IONS->wxc.zeros();
	IONS->wxl.zeros();
	IONS->wxr.zeros();

	#pragma omp parallel shared(IONS, X, NSP, LOGIC)
	{
	#pragma omp for
	for(int ii=0; ii<NSP; ii++)
		IONS->mn(ii) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(IONS->mn(ii) != params->mesh.NX_IN_SIM){
			X(ii) = IONS->X(ii,0) - params->mesh.nodes.X(IONS->mn(ii));
		}else{
			X(ii) = IONS->X(ii,0) - params->mesh.LX;
		}
	}

	#pragma omp single
	{
	LOGIC = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
	X = abs(X);
	}

	#pragma omp for
	for(int ii=0; ii<NSP; ii++)
		IONS->wxc(ii) = 0.75 - (X(ii)/params->mesh.DX)*(X(ii)/params->mesh.DX);

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(LOGIC(ii) == 1){
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX);
		}else{
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX);
		}
	}

	}//End of the parallel region

    #ifdef CHECKS_ON
	if(!IONS->mn.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
}


void PIC::assignCell(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
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

	// cout << "MPI: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " | NSP: " << NSP << " | RNSP: " << IONS->wxc.n_elem << endl;
	// cout << "MPI: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " | " << IONS->wxc.n_elem << " | " << IONS->wxr.n_elem << " | " << IONS->wxl.n_elem << " | " << IONS->wyc.n_elem << " | " << IONS->wyr.n_elem << " | " << IONS->wyl.n_elem << endl;
	// cout << "MPI: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " | " << IONS->mn.n_rows << " | " << IONS->X.n_rows << endl;

	arma::vec X = zeros(NSP);
	arma::vec Y = zeros(NSP);
	uvec LOGIC_X;
	uvec LOGIC_Y;

	IONS->wxc.zeros();
	IONS->wxl.zeros();
	IONS->wxr.zeros();

	IONS->wyc.zeros();
	IONS->wyl.zeros();
	IONS->wyr.zeros();

	#pragma omp parallel shared(IONS, X, Y, NSP, LOGIC_X, LOGIC_Y)
	{
	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		IONS->mn(ii,0) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);
		IONS->mn(ii,1) = floor((IONS->X(ii,1) + 0.5*params->mesh.DY)/params->mesh.DY);
	}

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(IONS->mn(ii,0) != params->mesh.NX_IN_SIM){
			X(ii) = IONS->X(ii,0) - params->mesh.nodes.X(IONS->mn(ii,0));
		}else{
			X(ii) = IONS->X(ii,0) - params->mesh.LX;
		}

		if(IONS->mn(ii,1) != params->mesh.NY_IN_SIM){
			Y(ii) = IONS->X(ii,1) - params->mesh.nodes.Y(IONS->mn(ii,1));
		}else{
			Y(ii) = IONS->X(ii,1) - params->mesh.LY;
		}
	}

	#pragma omp single
	{
	LOGIC_X = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
	X = abs(X);

	LOGIC_Y = ( Y > 0 );//If , aux > 0, then the particle is on the right of the meshnode
	Y = abs(Y);
	}

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		IONS->wxc(ii) = 0.75 - (X(ii)/params->mesh.DX)*(X(ii)/params->mesh.DX);
		IONS->wyc(ii) = 0.75 - (Y(ii)/params->mesh.DY)*(Y(ii)/params->mesh.DY);
	}

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(LOGIC_X(ii) == 1){
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX);
		}else{
			IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX - X(ii))/params->mesh.DX);
			IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX)*(1.5 - (params->mesh.DX + X(ii))/params->mesh.DX);
		}

		if(LOGIC_Y(ii) == 1){
			IONS->wyl(ii) = 0.5*(1.5 - (params->mesh.DY + Y(ii))/params->mesh.DY)*(1.5 - (params->mesh.DY + Y(ii))/params->mesh.DY);
			IONS->wyr(ii) = 0.5*(1.5 - (params->mesh.DY - Y(ii))/params->mesh.DY)*(1.5 - (params->mesh.DY - Y(ii))/params->mesh.DY);
		}else{
			IONS->wyl(ii) = 0.5*(1.5 - (params->mesh.DY - Y(ii))/params->mesh.DY)*(1.5 - (params->mesh.DY - Y(ii))/params->mesh.DY);
			IONS->wyr(ii) = 0.5*(1.5 - (params->mesh.DY + Y(ii))/params->mesh.DY)*(1.5 - (params->mesh.DY + Y(ii))/params->mesh.DY);
		}
	}

	}//End of the parallel region

    #ifdef CHECKS_ON
	if(!IONS->mn.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
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

	vfield_vec nv;
	nv.zeros(params->mesh.NX_IN_SIM + 4);

	#pragma omp parallel shared(params, IONS) firstprivate(nv)
	{
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

	vfield_mat nv;
	nv.zeros(params->mesh.NX_IN_SIM + 4, params->mesh.NY_IN_SIM + 4);

	#pragma omp parallel shared(params, IONS) firstprivate(nv)
	{
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


void PIC::test(const simulationParameters * params){
	arma::mat m  = zeros(10,9);

	for (int ii=0; ii<m.n_rows; ii++){
		for (int jj=0; jj<m.n_cols; jj++){
			m(ii,jj) = ii*m.n_cols + jj + 1;
		}
	}

	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
		m.print("m");

	MPI_Barrier(MPI_COMM_WORLD);

	include4GhostsContributions(&m);

	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
		m.print("Ghosts");

	fillGhosts(&m);

	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
		m.print("PBC");

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Abort(MPI_COMM_WORLD,-1000);
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
	arma::vec n = zeros(params->mesh.NX_IN_SIM + 4); // Four ghosht cells considereds

	IONS->n.zeros(); // Setting to zero the ion density.

	#pragma omp parallel shared(params, IONS, NSP) firstprivate(n)
	{
		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->mn(ii) + 2;

			n(ix-1) += IONS->wxl(ii);
			n(ix) 	+= IONS->wxc(ii);
			n(ix+1) += IONS->wxr(ii);
		}

		include4GhostsContributions(&n);

		#pragma omp critical (update_density)
		{
		IONS->n.subvec(1,params->mesh.NX_IN_SIM) += n.subvec(2,params->mesh.NX_IN_SIM + 1);
		}

	}//End of the parallel region

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
	arma::mat n = zeros(params->mesh.NX_IN_SIM + 4, params->mesh.NY_IN_SIM + 4); // Four ghosht cells considereds

	IONS->n.zeros(); // Setting to zero the ion density.

	#pragma omp parallel shared(params, IONS, NSP) firstprivate(n)
	{
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
	#pragma omp parallel for shared(NSP, params, IONS, F, field_X, field_Y, field_Z)
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
	#pragma omp parallel for shared(NSP, params, IONS, F, field_X, field_Y, field_Z)
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
	MPI_Allgathervfield_vec(params, &EB->E);
	MPI_Allgathervfield_vec(params, &EB->B);

	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	fillGhosts(EB);
	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	oneDimensional::fields EB_(params->mesh.NX_IN_SIM + 2);

	computeFieldsOnNonStaggeredGrid(EB,&EB_);

	fillGhosts(&EB_);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
		arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

		interpolateElectromagneticFields(params, &IONS->at(ii), &EB_, &Ep, &Bp);

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

		extrapolateIonVelocity(params, &IONS->at(ii));

		// Reduce operation for including contribution of bulk velocity of all MPIs
		PIC::MPI_AllreduceVec(params, &IONS->at(ii).nv.X);
		PIC::MPI_AllreduceVec(params, &IONS->at(ii).nv.Y);
		PIC::MPI_AllreduceVec(params, &IONS->at(ii).nv.Z);


		for (int jj=0;jj<params->filtersPerIterationIons;jj++)
			smooth(&IONS->at(ii).nv, params->smoothingParameter);

	}//structure to iterate over all the ion species.

	// Ghosts cells might be set to zero if needed, but before saving to HDF5 ghost cells need to be filled again.

}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT){
	MPI_Allgathervfield_mat(params, &EB->E);
	MPI_Allgathervfield_mat(params, &EB->B);

	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	fillGhosts(EB);
	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	twoDimensional::fields EB_(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);

	computeFieldsOnNonStaggeredGrid(EB,&EB_);

	fillGhosts(&EB_);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		arma::mat Ep = zeros(IONS->at(ii).NSP, 3);
		arma::mat Bp = zeros(IONS->at(ii).NSP, 3);

		interpolateElectromagneticFields(params, &IONS->at(ii), &EB_, &Ep, &Bp);

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

		extrapolateIonVelocity(params, &IONS->at(ii));

		// Reduce operation for including contribution of bulk velocity of all MPIs
		PIC::MPI_AllreduceMat(params, &IONS->at(ii).nv.X);
		PIC::MPI_AllreduceMat(params, &IONS->at(ii).nv.Y);
		PIC::MPI_AllreduceMat(params, &IONS->at(ii).nv.Z);


		for (int jj=0;jj<params->filtersPerIterationIons;jj++)
			smooth(&IONS->at(ii).nv, params->smoothingParameter);

	}//structure to iterate over all the ion species.
}


void PIC::advanceIonsPosition(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, const double DT){

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(params, IONS, ii) firstprivate(DT, NSP)
		{
			#pragma omp for
			for(int ip=0; ip<NSP; ip++){
				IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);

                IONS->at(ii).X(ip,0) = fmod(IONS->at(ii).X(ip,0), params->mesh.LX);//x

                if(IONS->at(ii).X(ip,0) < 0)
        			IONS->at(ii).X(ip,0) += params->mesh.LX;
			}
		}//End of the parallel region

		PIC::assignCell(params, &IONS->at(ii));

		//Once the ions have been pushed,  we extrapolate the density at the node grids.
		extrapolateIonDensity(params, &IONS->at(ii));

		PIC::MPI_AllreduceVec(params, &IONS->at(ii).n);

		for (int jj=0; jj<params->filtersPerIterationIons; jj++)
			smooth(&IONS->at(ii).n, params->smoothingParameter);

	}//structure to iterate over all the ion species.
}


void PIC::advanceIonsPosition(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, const double DT){

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(params, IONS, ii) firstprivate(DT, NSP)
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

		PIC::assignCell(params, &IONS->at(ii));

		extrapolateIonDensity(params, &IONS->at(ii));//Once the ions have been pushed,  we extrapolate the density at the node grids.

		PIC::MPI_AllreduceMat(params, &IONS->at(ii).n);

		for (int jj=0; jj<params->filtersPerIterationIons; jj++)
			smooth(&IONS->at(ii).n, params->smoothingParameter);

	}//structure to iterate over all the ion species.
}


// template class PIC<oneDimensional::ionSpecies, oneDimensional::fields>;
// template class PIC<twoDimensional::ionSpecies, twoDimensional::fields>;
