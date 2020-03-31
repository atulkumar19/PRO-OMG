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

template <class IT, class FT> PIC<IT,FT>::PIC(){

}


template <class IT, class FT> void PIC<IT,FT>::MPI_AllreduceVec(const simulationParameters * params, arma::vec * v){
	arma::vec recvbuf = zeros(v->n_elem);

	MPI_Allreduce(v->memptr(), recvbuf.memptr(), v->n_elem, MPI_DOUBLE, MPI_SUM, params->mpi.MPI_TOPO);

	*v = recvbuf;
}


template <class IT, class FT> void PIC<IT,FT>::MPI_AllreduceMat(const simulationParameters * params, arma::mat * m){
	arma::mat recvbuf = zeros(m->n_rows,m->n_cols);

	MPI_Allreduce(m->memptr(), recvbuf.memptr(), m->n_elem, MPI_DOUBLE, MPI_SUM, params->mpi.MPI_TOPO);

	*m = recvbuf;
}


template <class IT, class FT> void PIC<IT,FT>::MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * field){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));
	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	//Allgather for x-component
	sendbuf = field->X.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->X.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;

	//Allgather for y-component
	sendbuf = field->Y.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->Y.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;

	//Allgather for z-component
	sendbuf = field->Z.subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->Z.subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
}


template <class IT, class FT> void PIC<IT,FT>::MPI_Allgathervec(const simulationParameters * params, arma::vec * field){
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
template <class IT, class FT> void PIC<IT,FT>::include4GhostsContributions(arma::vec * v){
	int N = v->n_elem;

	v->subvec(2,3) += v->subvec(N-2,N-1);
	v->subvec(N-4,N-3) += v->subvec(0,1);
}


template <class IT, class FT> void PIC<IT,FT>::include4GhostsContributions(arma::mat * m){
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

template <class IT, class FT> void PIC<IT,FT>::smooth(arma::vec * v, double as){
	int NX(v->n_elem);
	arma::vec b = zeros(NX);
	double wc(0.75); 	// center weight
	double ws(0.125);	// sides weight

	//Step 1: Averaging process
	b.subvec(1, NX-2) = v->subvec(1, NX-2);

	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1, NX-2) = (1.0 - as)*v->subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


template <class IT, class FT> void PIC<IT,FT>::smooth(arma::mat * m, double as){
	int NX(m->n_rows);
	int NY(m->n_cols);
	arma::mat b = zeros(NX,NY);
	double wc(9.0/16.0);
	double ws(3.0/32.0);
	double wcr(1.0/64.0);

	// Step 1: Averaging
	b.submat(1,1,NX-2,NY-2) = m->submat(1,1,NX-2,NY-2);

	forwardPBC_2D(&b);

	b.submat(1,1,NX-2,NY-2) = wc*b.submat(1,1,NX-2,NY-2) + \
								ws*b.submat(2,1,NX-1,NY-2) + ws*b.submat(0,1,NX-3,NY-2) + \
								ws*b.submat(1,2,NX-2,NY-1) + ws*b.submat(1,0,NX-2,NY-3) + \
								wcr*b.submat(2,2,NX-1,NY-1) + wcr*b.submat(0,2,NX-3,NY-1) + \
								wcr*b.submat(0,0,NX-3,NY-3) + wcr*b.submat(2,0,NX-1,NY-3);

	// Step 2: Averaged weighted variable estimation
	m->submat(1,1,NX-2,NY-2) = (1.0 - as)*m->submat(1,1,NX-2,NY-2) + as*b.submat(1,1,NX-2,NY-2);

}


template <class IT, class FT> void PIC<IT,FT>::smooth(vfield_vec * vf, double as){
	int NX(vf->X.n_elem);
	arma::vec b = zeros(NX);
	double wc(0.75);
	double ws(0.125);//weights

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->X.subvec(1, NX-2);

	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(1, NX-2) = (1-as)*vf->X.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Y.subvec(1, NX-2);

	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(1, NX-2) = (1-as)*vf->Y.subvec(1, NX-2) + as*b.subvec(1, NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1, NX-2) = vf->Z.subvec(1, NX-2);

	forwardPBC_1D(&b);

	b.subvec(1, NX-2) = wc*b.subvec(1, NX-2) + ws*b.subvec(2, NX-1) + ws*b.subvec(0, NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(1, NX-2) = (1-as)*vf->Z.subvec(1, NX-2) + as*b.subvec(1, NX-2);
}


template <class IT, class FT> void PIC<IT,FT>::smooth(vfield_mat * vf, double as){

}

// * * * Smoothing * * *


template <class IT, class FT> void PIC<IT,FT>::assignCell(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
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
		IONS->meshNode(ii) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(IONS->meshNode(ii) != params->mesh.NX_IN_SIM){
			X(ii) = IONS->X(ii,0) - params->mesh.nodes.X(IONS->meshNode(ii));
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
	if(!IONS->meshNode.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
}


template <class IT, class FT> void PIC<IT,FT>::assignCell(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
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
	// cout << "MPI: " << params->mpi.MPI_DOMAIN_NUMBER_CART << " | " << IONS->meshNode.n_rows << " | " << IONS->X.n_rows << endl;

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
		IONS->meshNode(ii,0) = floor((IONS->X(ii,0) + 0.5*params->mesh.DX)/params->mesh.DX);
		IONS->meshNode(ii,1) = floor((IONS->X(ii,1) + 0.5*params->mesh.DY)/params->mesh.DY);
	}

	#pragma omp for
	for(int ii=0; ii<NSP; ii++){
		if(IONS->meshNode(ii,0) != params->mesh.NX_IN_SIM){
			X(ii) = IONS->X(ii,0) - params->mesh.nodes.X(IONS->meshNode(ii,0));
		}else{
			X(ii) = IONS->X(ii,0) - params->mesh.LX;
		}

		if(IONS->meshNode(ii,1) != params->mesh.NY_IN_SIM){
			Y(ii) = IONS->X(ii,1) - params->mesh.nodes.Y(IONS->meshNode(ii,1));
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
	if(!IONS->meshNode.is_finite()){
		MPI_Abort(params->mpi.MPI_TOPO, -108);
	}
    #endif
}


template <class IT, class FT> void PIC<IT,FT>::crossProduct(const arma::mat * A, const arma::mat * B, arma::mat * AxB){
	if(A->n_elem != B->n_elem){
		cerr<<"\nERROR: The number of elements of A and B, unable to calculate AxB.\n";
		exit(1);
	}

	AxB->set_size(A->n_rows, 3);//Here we set up the size of the matrix AxB.

	AxB->col(0) = A->col(1)%B->col(2) - A->col(2)%B->col(1);//(AxB)_x
	AxB->col(1) = A->col(2)%B->col(0) - A->col(0)%B->col(2);//(AxB)_y
	AxB->col(2) = A->col(0)%B->col(1) - A->col(1)%B->col(0);//(AxB)_z
}


template <class IT, class FT> void PIC<IT,FT>::eiv(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NC(params->mesh.NX_IN_SIM + 2);//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);
	IONS->nv.zeros(); // Setting to zero the ions' bulk velocity

	vfield_vec nv;
	nv.zeros(params->mesh.NX_IN_SIM + 4);

	#pragma omp parallel shared(params, IONS, NC) firstprivate(nv)
	{
		#pragma omp for
		for(int ii=0; ii<NSP; ii++){
			int ix = IONS->meshNode(ii) + 2;

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


template <class IT, class FT> void PIC<IT,FT>::extrapolateIonVelocity(const simulationParameters * params, oneDimensional::ionSpecies * IONS){

	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	eiv(params, IONS);
}


template <class IT, class FT> void PIC<IT,FT>::test(const simulationParameters * params){
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

	forwardPBC_2D(&m);

	if (params->mpi.MPI_DOMAIN_NUMBER == 0)
		m.print("PBC");

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Abort(MPI_COMM_WORLD,-1000);
}


template <class IT, class FT> void PIC<IT,FT>::eid(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
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
			int ix = IONS->meshNode(ii) + 2;

			n(ix-1) += IONS->wxl(ii);
			n(ix) += IONS->wxc(ii);
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


template <class IT, class FT> void PIC<IT,FT>::eid(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
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
			int ix = IONS->meshNode(ii,0) + 2;
			int iy = IONS->meshNode(ii,1) + 2;

			n(ix-1,iy) += IONS->wxl(ii)*IONS->wyc(ii);
			n(ix,iy) += IONS->wxc(ii)*IONS->wyc(ii);
			n(ix+1,iy) += IONS->wxr(ii)*IONS->wyc(ii);

			n(ix,iy-1) += IONS->wxc(ii)*IONS->wyl(ii);
			n(ix,iy+1) += IONS->wxc(ii)*IONS->wyr(ii);

			n(ix+1,iy-1) += IONS->wxr(ii)*IONS->wyl(ii);
			n(ix+1,iy+1) += IONS->wxr(ii)*IONS->wyr(ii);

			n(ix-1,iy-1) += IONS->wxl(ii)*IONS->wyl(ii);
			n(ix-1,iy+1) += IONS->wxl(ii)*IONS->wyr(ii);
		}

		include4GhostsContributions(&n);

		#pragma omp critical (update_density)
		{
		IONS->n.submat(1,1,params->mesh.NX_IN_SIM,params->mesh.NY_IN_SIM) += n.submat(2,2,params->mesh.NX_IN_SIM+1,params->mesh.NY_IN_SIM+1);
		}

	}//End of the parallel region

	IONS->n *= IONS->NCP/(params->mesh.DX*params->mesh.DY);
}


template <class IT, class FT> void PIC<IT,FT>::extrapolateIonDensity(const simulationParameters * params, oneDimensional::ionSpecies * IONS){
	// First, the particle density of time steps (it - 1) is kept in n_, (it - 2) is kept in n__,
	// and at the time iteration (it - 3) in n___.

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	// Then, the ions' properties are distributed in the grid following the charge deposition of the Triangular
	// Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	eid(params, IONS);
}


template <class IT, class FT> void PIC<IT,FT>::extrapolateIonDensity(const simulationParameters * params, twoDimensional::ionSpecies * IONS){
	// First, the particle density of time steps (it - 1) is kept in n_, (it - 2) is kept in n__,
	// and at the time iteration (it - 3) in n___.

	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	// Then, the ions' properties are distributed in the grid following the charge deposition of the Triangular
	// Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	eid(params, IONS);
}


template <class IT, class FT> void PIC<IT,FT>::interpolateVectorField(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, vfield_vec * emf, arma::mat * F){
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NX =  params->mesh.NX_IN_SIM + 2;//Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for shared(NX, NSP, params, IONS, emf, F)
	for(int ii=0; ii<NSP; ii++){
		int ix = IONS->meshNode(ii) + 1;

		if(ix == (NX-1)){//For the particles on the right side boundary.
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


template <class IT, class FT> void PIC<IT,FT>::interpolateElectromagneticFields(const simulationParameters * params, const oneDimensional::ionSpecies * IONS, oneDimensional::fields * EB, arma::mat * E, arma::mat * B){
	interpolateVectorField(params, IONS, &EB->E, E);
	interpolateVectorField(params, IONS, &EB->B, B);
}


template <class IT, class FT> void PIC<IT,FT>::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT){
	MPI_Allgathervfield_vec(params, &EB->E);
	MPI_Allgathervfield_vec(params, &EB->B);

	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	oneDimensional::fields EB_(NX); // Electromagnetic fields computed in a single grid, where the densities and bulk velocities are known

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

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	// restoreVector(&EB->E.X);
	// restoreVector(&EB->E.Y);
	// restoreVector(&EB->E.Z);

	// restoreVector(&EB->B.X);
	// restoreVector(&EB->B.Y);
	// restoreVector(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

}


template <class IT, class FT> void PIC<IT,FT>::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT){
	/*
	MPI_AllgatherField(params, &EB->E);
	MPI_AllgatherField(params, &EB->B);

	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	// The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	oneDimensional::fields EB_(NX); // Electromagnetic fields computed in a single grid, where the densities and bulk velocities are known

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
	*/

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	// restoreVector(&EB->E.X);
	// restoreVector(&EB->E.Y);
	// restoreVector(&EB->E.Z);

	// restoreVector(&EB->B.X);
	// restoreVector(&EB->B.Y);
	// restoreVector(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

}


template <class IT, class FT> void PIC<IT,FT>::advanceIonsPosition(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, const double DT){

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(params, IONS, ii) firstprivate(DT, NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
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


template <class IT, class FT> void PIC<IT,FT>::advanceIonsPosition(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, const double DT){

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(params, IONS, ii) firstprivate(DT, NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);
				IONS->at(ii).X(ip,1) += DT*IONS->at(ii).V(ip,1);

                IONS->at(ii).X(ip,0) = fmod(IONS->at(ii).X(ip,0), params->mesh.LX); // Periodic condition along x-axis
				IONS->at(ii).X(ip,1) = fmod(IONS->at(ii).X(ip,1), params->mesh.LY); // Periodic condition along x-axis

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


template class PIC<oneDimensional::ionSpecies, oneDimensional::fields>;
template class PIC<twoDimensional::ionSpecies, twoDimensional::fields>;
