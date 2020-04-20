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

EMF_SOLVER::EMF_SOLVER(const simulationParameters * params, characteristicScales * CS){
	NX_S = params->mesh.NX_PER_MPI + 2;
	NX_T = params->mesh.NX_IN_SIM + 2;
	NX_R = params->mesh.NX_IN_SIM;

	NY_S = params->mesh.NY_PER_MPI + 2;
	NY_T = params->mesh.NY_IN_SIM + 2;
	NY_R = params->mesh.NY_IN_SIM;

	if (params->dimensionality == 1){
		n_cs = CS->length*CS->density;

		V1D._n.zeros(NX_S);
		V1D.n.zeros(NX_S);
		V1D.n_.zeros(NX_S);
		V1D.n__.zeros(NX_S);

		V1D.U.zeros(NX_S);
		V1D.U_.zeros(NX_S);
		V1D.U__.zeros(NX_S);
	}else if (params->dimensionality == 2){

	}

}


void EMF_SOLVER::MPI_AllgatherField(const simulationParameters * params, arma::vec * field){

	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	arma::vec recvBuf(params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->mesh.NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->mesh.NX_PER_MPI);

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for x-component
	sendBuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->subvec(1, params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);
}


void EMF_SOLVER::MPI_AllgatherField(const simulationParameters * params, vfield_vec * field){

	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	arma::vec recvBuf(params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->mesh.NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->mesh.NX_PER_MPI);

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for x-component
	sendBuf = field->X.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->X.subvec(1, params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for y-component
	sendBuf = field->Y.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->Y.subvec(1, params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for z-component
	sendBuf = field->Z.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->Z.subvec(1, params->mesh.NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, vfield_vec * field){

	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	double sendBuf;
	double recvBuf;

	sendBuf = field->X(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,0,&recvBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,0,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->X(iIndex-1) = recvBuf;

	sendBuf = field->X(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,1,&recvBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,1,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->X(fIndex+1) = recvBuf;

	sendBuf = field->Y(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,2,&recvBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,2,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->Y(iIndex-1) = recvBuf;

	sendBuf = field->Y(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,3,&recvBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,3,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->Y(fIndex+1) = recvBuf;

	sendBuf = field->Z(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,4,&recvBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,4,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->Z(iIndex-1) = recvBuf;

	sendBuf = field->Z(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,5,&recvBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,5,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	field->Z(fIndex+1) = recvBuf;


	MPI_Barrier(params->mpi.MPI_TOPO);
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params, arma::vec * field){

	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	double sendBuf;
	double recvBuf;

	sendBuf = (*field)(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,0,&recvBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,0,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	(*field)(iIndex-1) = recvBuf;

	sendBuf = (*field)(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.LEFT_MPI_DOMAIN_NUMBER_CART,1,&recvBuf,1,MPI_DOUBLE,params->mpi.RIGHT_MPI_DOMAIN_NUMBER_CART,1,params->mpi.MPI_TOPO,MPI_STATUS_IGNORE);
	(*field)(fIndex+1) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);
}


// * * * Smoothing * * *
void EMF_SOLVER::smooth(arma::vec * v, double as){
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


void EMF_SOLVER::smooth(arma::mat * m, double as){
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


void EMF_SOLVER::equilibrium(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, oneDimensional::fields * EB){

}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params, oneDimensional::fields * EB){//This function calculates -culr(EB->E)
	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));


	//There is not x-component of curl(B)
	EB->B.X.fill(0);

	//y-component
	EB->B.Y.subvec(iIndex,fIndex) =   ( EB->E.Z.subvec(iIndex+1,fIndex+1) - EB->E.Z.subvec(iIndex,fIndex) )/params->mesh.DX;

	//z-component
	EB->B.Z.subvec(iIndex,fIndex) = - ( EB->E.Y.subvec(iIndex+1,fIndex+1) - EB->E.Y.subvec(iIndex,fIndex) )/params->mesh.DX;
}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params, twoDimensional::fields * EB){

}


void EMF_SOLVER::advanceBField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS){
	//Using the RK4 scheme to advance B.
	//B^(N+1) = B^(N) + dt( K1^(N) + 2*V1D.K2^(N) + 2*K3^(N) + K4^(N) )/6
	dt = params->DT/((double)params->numberOfRKIterations);

	for(int RKit=0; RKit<params->numberOfRKIterations; RKit++){ // Runge-Kutta iterations

		V1D.K1 = *EB; 									// The value of the fields at the time level (N-1/2)
		advanceEField(params, &V1D.K1, IONS);			// E1 (using B^(N-1/2))
		FaradaysLaw(params, &V1D.K1);					// K1

		V1D.K2.B.X = EB->B.X;							// B^(N-1/2) + 0.5*dt*K1
		V1D.K2.B.Y = EB->B.Y + (0.5*dt)*V1D.K1.B.Y;
		V1D.K2.B.Z = EB->B.Z + (0.5*dt)*V1D.K1.B.Z;
		V1D.K2.E = EB->E;
		advanceEField(params, &V1D.K2, IONS);			// E2 (using B^(N-1/2) + 0.5*dt*K1)
		FaradaysLaw(params, &V1D.K2);					// K2

		V1D.K3.B.X = EB->B.X;							// B^(N-1/2) + 0.5*dt*K2
		V1D.K3.B.Y = EB->B.Y + (0.5*dt)*V1D.K2.B.Y;
		V1D.K3.B.Z = EB->B.Z + (0.5*dt)*V1D.K2.B.Z;
		V1D.K3.E = EB->E;
		advanceEField(params, &V1D.K3, IONS);			// E3 (using B^(N-1/2) + 0.5*dt*K2)
		FaradaysLaw(params, &V1D.K3);					// K3

		V1D.K4.B.X = EB->B.X;							// B^(N-1/2) + dt*K2
		V1D.K4.B.Y = EB->B.Y + dt*V1D.K3.B.Y;
		V1D.K4.B.Z = EB->B.Z + dt*V1D.K3.B.Z;
		V1D.K4.E = EB->E;
		advanceEField(params, &V1D.K4, IONS);			// E4 (using B^(N-1/2) + dt*K3)
		FaradaysLaw(params, &V1D.K4);					// K4

		EB->B += (dt/6)*( V1D.K1.B + 2*V1D.K2.B + 2*V1D.K3.B + V1D.K4.B );
	} // Runge-Kutta iterations



	if (params->includeElectronInertia){
		MPI_AllgatherField(params, &EB->B);

		// arma::vec n
		arma::mat M(NX_R, NX_R, fill::zeros);		// Matrix to be inverted to calculate the actual magnetic field
		arma::vec S(NX_R, fill::zeros);				// Modified magnetic field
		double de; 									// Electron skin depth

		de = (F_C_DS/F_E_DS)*sqrt( F_ME_DS*F_EPSILON_DS/params->BGP.ne );

		// Elements for first node in mesh
		M(0,0) = 1.0 + 2.0*pow(de,2.0)/pow(params->mesh.DX,2.0);
		M(0,1) = -pow(de,2.0)/pow(params->mesh.DX,2.0);


		// Elements for last node in mesh
		M(NX_R-1,NX_R-1) = 1.0 + 2.0*pow(de,2.0)/pow(params->mesh.DX,2.0);
		M(NX_R-1,NX_R-2) = -pow(de,2.0)/pow(params->mesh.DX,2.0);

		for(int ii=1; ii<NX_R-1; ii++){
			M(ii,ii-1) = -pow(de,2.0)/pow(params->mesh.DX,2.0);
			M(ii,ii) = 1.0 + 2.0*pow(de,2.0)/pow(params->mesh.DX,2.0);
			M(ii,ii+1) = -pow(de,2.0)/pow(params->mesh.DX,2.0);
		}

		// x-component of magnetic field
		S = EB->B.X.subvec(1,NX_T-2);

		EB->B.X.subvec(1,NX_T-2) = arma::solve(M,S);

		// y-component of magnetic field
		S = EB->B.Y.subvec(1,NX_T-2);

		EB->B.Y.subvec(1,NX_T-2) = arma::solve(M,S);

		// z-component of magnetic field
		S = EB->B.Z.subvec(1,NX_T-2);

		EB->B.Z.subvec(1,NX_T-2) = arma::solve(M,S);
	}


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

	smooth(&EB->B,params->smoothingParameter);

	EB->b_ = EB->b;

	MPI_passGhosts(params,&EB->B);
	MPI_passGhosts(params,&EB->b_);

	EB->_B.X.subvec(1,NX_T-2) = sqrt( EB->B.X.subvec(1,NX_T-2) % EB->B.X.subvec(1,NX_T-2) \
					+ 0.25*( ( EB->B.Y.subvec(1,NX_T-2) + EB->B.Y.subvec(0,NX_T-3) ) % ( EB->B.Y.subvec(1,NX_T-2) + EB->B.Y.subvec(0,NX_T-3) ) ) \
					+ 0.25*( ( EB->B.Z.subvec(1,NX_T-2) + EB->B.Z.subvec(0,NX_T-3) ) % ( EB->B.Z.subvec(1,NX_T-2) + EB->B.Z.subvec(0,NX_T-3) ) ) );

	EB->_B.Y.subvec(1,NX_T-2) = sqrt( 0.25*( ( EB->B.X.subvec(1,NX_T-2) + EB->B.X.subvec(0,NX_T-3) ) % ( EB->B.X.subvec(1,NX_T-2) + EB->B.X.subvec(0,NX_T-3) ) ) \
					+ EB->B.Y.subvec(1,NX_T-2) % EB->B.Y.subvec(1,NX_T-2) + EB->B.Z.subvec(1,NX_T-2) % EB->B.Z.subvec(1,NX_T-2) );

	EB->_B.Z.subvec(1,NX_T-2) = sqrt( 0.25*( ( EB->B.X.subvec(1,NX_T-2) + EB->B.X.subvec(0,NX_T-3) ) % ( EB->B.X.subvec(1,NX_T-2) + EB->B.X.subvec(0,NX_T-3) ) ) \
					+ EB->B.Y.subvec(1,NX_T-2) % EB->B.Y.subvec(1,NX_T-2) + EB->B.Z.subvec(1,NX_T-2) % EB->B.Z.subvec(1,NX_T-2) );

	EB->b = EB->B/EB->_B;


	MPI_passGhosts(params,&EB->b);
	MPI_passGhosts(params,&EB->_B);
}


void EMF_SOLVER::advanceBField(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS){

}


void EMF_SOLVER::advanceEField(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS){
	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	// Indices of subdomain
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART + 1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART + 1));

	// stting to zero number density and bulk velocity vectors
	V1D.n.zeros();
	V1D.U.zeros();

	for(int ii=0; ii<params->numberOfParticleSpecies; ii++){
		fillGhosts(&IONS->at(ii).n);
		fillGhosts(&IONS->at(ii).n_);
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

	V1D.n /= n_cs;//Normalized density

	// We compute the curl(B).
	vfield_vec curlB(NX_T);

	EB->E.zeros();

	// x-component

	// The Y and Z components of curl(B) are computed and linear interpolation used to compute its values at the Ex-nodes.
	curlB.Y.subvec(iIndex,fIndex) = - 0.5*( EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
	curlB.Z.subvec(iIndex,fIndex) =   0.5*( EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

	// Number density at Ex-nodes. Linear interpolation used.
	V1D.n_interp = 0.5*( V1D.n.subvec(1,NX_S-2) + V1D.n.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y.subvec(iIndex,fIndex) % EB->B.Z.subvec(iIndex,fIndex) - curlB.Z.subvec(iIndex,fIndex) % EB->B.Y.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D.n_interp );

	// Uy is interpolated to Ex-nodes
	V1D.U_interp = 0.5*( V1D.U.Y.subvec(1,NX_S-2) + V1D.U.Y.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) += - V1D.U_interp % EB->B.Z.subvec(iIndex,fIndex);

	// Uz is interpolated to Ex-nodes
	V1D.U_interp = 0.5*( V1D.U.Z.subvec(1,NX_S-2) + V1D.U.Z.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) +=   V1D.U_interp % EB->B.Y.subvec(iIndex,fIndex);

	// The partial derivative dn/dx is computed using centered finite differences with grid size DX/2
	V1D.dndx = ( V1D.n.subvec(2,NX_S-1) - V1D.n.subvec(1,NX_S-2) )/params->mesh.DX;

	EB->E.X.subvec(iIndex,fIndex) += - ( params->BGP.Te/F_E_DS )*V1D.dndx/V1D.n_interp;

	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFD with DX/2

		EB->E.X.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*0.5*( V1D.U.X.subvec(1,NX_S-2) + V1D.U.X.subvec(2,NX_S-1) ) % ( (V1D.U.X.subvec(2,NX_S-1) - V1D.U.X.subvec(1,NX_S-2))/params->mesh.DX );
	}

	curlB.zeros();


	// y-component

	// The Y and Z components of curl(B) are computed directly at Ey-nodes.
	curlB.Y.subvec(iIndex,fIndex) = -( EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
	curlB.Z.subvec(iIndex,fIndex) =  ( EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

	MPI_passGhosts(params,&curlB.Y);

	MPI_passGhosts(params,&curlB.Z);

	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D.n.subvec(1,NX_S-2) );

	// Bz is interpolated to Ey-nodes
	V1D.B_interp = 0.5*( EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1) );

	EB->E.Y.subvec(iIndex,fIndex) +=   V1D.U.X.subvec(1,NX_S-2) % V1D.B_interp;

	EB->E.Y.subvec(iIndex,fIndex) += - V1D.U.Z.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);

	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFDs with DX
		EB->E.Y.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*V1D.U.X.subvec(1,NX_S-2) % ( 0.5*(V1D.U.Y.subvec(2,NX_S-1) - V1D.U.Y.subvec(0,NX_S-3))/params->mesh.DX );

		EB->E.Y.subvec(iIndex,fIndex) += (F_ME_DS/F_E_DS)*( V1D.U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*V1D.n.subvec(1,NX_S-2)) ) % ( 0.5*(curlB.Y.subvec(iIndex+1,fIndex+1) - curlB.Y.subvec(iIndex-1,fIndex-1))/params->mesh.DX );

		EB->E.Y.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*( V1D.U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*V1D.n.subvec(1,NX_S-2) % V1D.n.subvec(1,NX_S-2)) ) % ( 0.5*(V1D.n.subvec(2,NX_S-1) - V1D.n.subvec(0,NX_S-3))/params->mesh.DX ) % curlB.Y.subvec(iIndex,fIndex);
	}


	// z-component

	// The y-component of Curl(B) has been calculated above at the right places. Note that Ey-nodes and Ez-nodes are the same.

	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*V1D.n.subvec(1,NX_S-2));

	// By is interpolated to Ez-nodes
	V1D.B_interp = 0.5*( EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1) );

	EB->E.Z.subvec(iIndex,fIndex) += - V1D.U.X.subvec(1,NX_S-2) % V1D.B_interp;

	EB->E.Z.subvec(iIndex,fIndex) +=   V1D.U.Y.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);


	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFDs with DX
		EB->E.Z.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*V1D.U.X.subvec(1,NX_S-2) % ( 0.5*(V1D.U.Z.subvec(2,NX_S-1) - V1D.U.Z.subvec(0,NX_S-3))/params->mesh.DX );

		EB->E.Z.subvec(iIndex,fIndex) += (F_ME_DS/F_E_DS)*( V1D.U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*V1D.n.subvec(1,NX_S-2)) ) % ( 0.5*(curlB.Z.subvec(iIndex+1,fIndex+1) - curlB.Z.subvec(iIndex-1,fIndex-1))/params->mesh.DX );
	}


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

	smooth(&EB->E, params->smoothingParameter);
}


/* In this function the different ionSpecies vectors represent the following:
	+ IONS__: Ions' variables at time level "l - 3/2"
	+ IONS_: Ions' variables at time level "l - 1/2"
	+ IONS: Ions' variables at time level "l + 1/2"
*/
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const int BAE){
	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	V1D._n.zeros();
	V1D.n.zeros();
	V1D.U.zeros();

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
			fillGhosts(&IONS->at(ii).n);
			fillGhosts(&IONS->at(ii).n_);
			fillGhosts(&IONS->at(ii).nv.X);
			fillGhosts(&IONS->at(ii).nv.Y);
			fillGhosts(&IONS->at(ii).nv.Z);

			// Electron density at time level "l + 1"
			V1D._n += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1);

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

	V1D._n /= n_cs;
	V1D.n /= n_cs;//Dimensionless density


	V1D.n_.zeros();
	V1D.U_.zeros();

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		fillGhosts(&IONS->at(ii).n_);
		fillGhosts(&IONS->at(ii).n__);
		fillGhosts(&IONS->at(ii).nv_.X);
		fillGhosts(&IONS->at(ii).nv_.Y);
		fillGhosts(&IONS->at(ii).nv_.Z);

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

	V1D.n_ /= n_cs;//Dimensionless density


	if(BAE == 1){
		V1D.n__.zeros();
		V1D.U__.zeros();

		for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
			fillGhosts(&IONS->at(ii).n__);
			fillGhosts(&IONS->at(ii).nv__.X);
			fillGhosts(&IONS->at(ii).nv__.Y);
			fillGhosts(&IONS->at(ii).nv__.Z);

			// Ions density at time level "l - 3/2"
			// n(l-3/2) = ( n(l-1) + n(l-2) )/2
			V1D.n__ += 0.5*IONS->at(ii).Z*( IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) + IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) );

			//sum_k[ Z_k*n_k*u_k ]
			V1D.U__.X += IONS->at(ii).Z*IONS->at(ii).nv__.X.subvec(iIndex-1,fIndex+1);
			V1D.U__.Y += IONS->at(ii).Z*IONS->at(ii).nv__.Y.subvec(iIndex-1,fIndex+1);
			V1D.U__.Z += IONS->at(ii).Z*IONS->at(ii).nv__.Z.subvec(iIndex-1,fIndex+1);
		}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

		V1D.U__.X /= V1D.n__;
		V1D.U__.Y /= V1D.n__;
		V1D.U__.Z /= V1D.n__;

		V1D.n__ /= n_cs;//Dimensionless density

		//Here we use the velocity extrapolation U^(N+1) = 2*U^(N+1/2) - 1.5*U^(N-1/2) + 0.5*U^(N-3/2)
		V1D.V = 2.0*V1D.U - 1.5*V1D.U_ + 0.5*V1D.U__;
	}else{
		//Here we use the velocity extrapolation U^(N+1) = 1.5*U^(N+1/2) - 0.5*U^(N-1/2)
		V1D.V = 1.5*V1D.U - 0.5*V1D.U_;
	}

	// We compute the curl(B).
	vfield_vec curlB(NX_T);

	EB->E.zeros();

	// x-component

	// The Y and Z components of curl(B) are computed and linear interpolation used to compute its values at the Ex-nodes.
	curlB.Y.subvec(iIndex,fIndex) = - 0.5*( EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
	curlB.Z.subvec(iIndex,fIndex) =   0.5*( EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

	// Number density at Ex-nodes. Linear interpolation used.
	V1D.n_interp = 0.5*( V1D._n.subvec(1,NX_S-2) + V1D._n.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y.subvec(iIndex,fIndex) % EB->B.Z.subvec(iIndex,fIndex) - curlB.Z.subvec(iIndex,fIndex) % EB->B.Y.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D.n_interp );

	// Uy is interpolated to Ex-nodes
	V1D.U_interp = 0.5*( V1D.V.Y.subvec(1,NX_S-2) + V1D.V.Y.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) += - V1D.U_interp % EB->B.Z.subvec(iIndex,fIndex);

	// Uz is interpolated to Ex-nodes
	V1D.U_interp = 0.5*( V1D.V.Z.subvec(1,NX_S-2) + V1D.V.Z.subvec(2,NX_S-1) );

	EB->E.X.subvec(iIndex,fIndex) +=   V1D.U_interp % EB->B.Y.subvec(iIndex,fIndex);

	// The partial derivative dn/dx is computed using centered finite differences with grid size DX/2
	V1D.dndx = ( V1D._n.subvec(2,NX_S-1) - V1D._n.subvec(1,NX_S-2) )/params->mesh.DX;

	EB->E.X.subvec(iIndex,fIndex) += - (params->BGP.Te/F_E_DS)*V1D.dndx/V1D.n_interp;


	curlB.zeros();


	// y-component

	// The Y and Z components of curl(B) are computed directly at Ey-nodes.
	curlB.Y.subvec(iIndex,fIndex) = - ( EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;
	curlB.Z.subvec(iIndex,fIndex) =   ( EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/params->mesh.DX;

	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*V1D._n.subvec(1,NX_S-2) );

	// Bz is interpolated to Ey-nodes
	V1D.B_interp = 0.5*( EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1) );

	EB->E.Y.subvec(iIndex,fIndex) +=   V1D.V.X.subvec(1,NX_S-2) % V1D.B_interp;

	EB->E.Y.subvec(iIndex,fIndex) += - V1D.V.Z.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);


	// z-component

	// The y-component of Curl(B) has been calculated above at the right places. Note that Ey-nodes and Ez-nodes are the same.

	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*V1D._n.subvec(1,NX_S-2));

	// By is interpolated to Ez-nodes
	V1D.B_interp = 0.5*( EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1) );

	EB->E.Z.subvec(iIndex,fIndex) += - V1D.V.X.subvec(1,NX_S-2) % V1D.B_interp;

	EB->E.Z.subvec(iIndex,fIndex) +=   V1D.V.Y.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);


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

	smooth(&EB->E, params->smoothingParameter);
}



void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const int BAE){

}
