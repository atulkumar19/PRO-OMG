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

#include "emf.h"

EMF_SOLVER::EMF_SOLVER(const simulationParameters * params, characteristicScales * CS){
	n_cs = CS->length*CS->density;
	NX_S = params->NX_PER_MPI + 2;
	NX_T = params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS + 2;
	NX_R = params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS;

	ne.zeros(NX_S);
	n.zeros(NX_S);
	n_.zeros(NX_S);
	n__.zeros(NX_S);

	U.zeros(NX_S);
	U_.zeros(NX_S);
	U__.zeros(NX_S);

	Ui.zeros(NX_S);
	Ui_.zeros(NX_S);
	Ui__.zeros(NX_S);
}


void EMF_SOLVER::MPI_AllgatherField(const simulationParameters * params, arma::vec * field){

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	arma::vec recvBuf(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->NX_PER_MPI);

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for x-component
	sendBuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);
}


void EMF_SOLVER::MPI_AllgatherField(const simulationParameters * params, vfield_vec * field){

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	arma::vec recvBuf(params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS);
	arma::vec sendBuf(params->NX_PER_MPI);

	MPI_ARMA_VEC chunk(params->NX_PER_MPI);

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for x-component
	sendBuf = field->X.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->X.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for y-component
	sendBuf = field->Y.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->Y.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);

	//Allgather for z-component
	sendBuf = field->Z.subvec(iIndex, fIndex);
	MPI_Allgather(sendBuf.memptr(), 1, chunk.type, recvBuf.memptr(), 1, chunk.type, params->mpi.MPI_TOPO);
	field->Z.subvec(1, params->NX_PER_MPI*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	MPI_Barrier(params->mpi.MPI_TOPO);
}


void EMF_SOLVER::MPI_passGhosts(const simulationParameters * params,vfield_vec * field){

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

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

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

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


void EMF_SOLVER::smooth_TOS(const simulationParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);

	arma::vec b = zeros(NX_S);

	double w0(23.0/48.0), w1(0.25), w2(1.0/96.0);//weights

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART + 1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART + 1));

	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(2,NX_S-3) = w0*b.subvec(2,NX_S-3) + w1*b.subvec(3,NX_S-2) + w1*b.subvec(1,NX_S-4) + w2*b.subvec(4,NX_S-1) + w2*b.subvec(0,NX_S-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(2,NX_S-3);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(2,NX_S-3) = w0*b.subvec(2,NX_S-3) + w1*b.subvec(3,NX_S-2) + w1*b.subvec(1,NX_S-4) + w2*b.subvec(4,NX_S-1) + w2*b.subvec(0,NX_S-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(2,NX_S-3);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(2,NX_S-3) = w0*b.subvec(2,NX_S-3) + w1*b.subvec(3,NX_S-2) + w1*b.subvec(1,NX_S-4) + w2*b.subvec(4,NX_S-1) + w2*b.subvec(0,NX_S-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(2,NX_S-3);

	b.fill(0);
}


void EMF_SOLVER::smooth_TOS(const simulationParameters * params,vfield_mat * vf,double as){

}


void EMF_SOLVER::smooth_TSC(const simulationParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);

	arma::vec b = zeros(NX_S);

	double w0(0.75), w1(0.125);//weights

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART + 1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART + 1));


	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2)+ w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2) + w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Z.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2) + w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(iIndex,fIndex) = (1.0 - as)*vf->Z.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);
}


void EMF_SOLVER::smooth_TSC(const simulationParameters * params,vfield_mat * vf,double as){

}


void EMF_SOLVER::smooth(const simulationParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);

	arma::vec b = zeros(NX_S);

	double w0(0.5), w1(0.25);//weights

	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));


	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2)+ w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2) + w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Z.subvec(iIndex-1,fIndex+1);
	b.subvec(1,NX_S-2) = w0*b.subvec(1,NX_S-2) + w1*b.subvec(2,NX_S-1) + w1*b.subvec(0,NX_S-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(iIndex,fIndex) = (1.0 - as)*vf->Z.subvec(iIndex,fIndex) + as*b.subvec(1,NX_S-2);
}


void EMF_SOLVER::smooth(const simulationParameters * params,vfield_mat * vf,double as){

}


void EMF_SOLVER::equilibrium(const simulationParameters * params,vector<ionSpecies> * IONS,fields * EB){

}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params,const meshParams * mesh,fields * EB){//This function calculates -culr(EB->E)
	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));


	//There is not x-component of curl(B)
	EB->B.X.fill(0);

	//y-component
//	EB->B.Y.subvec(1,NX-2) = ( EB->E.Z.subvec(2,NX-1) - EB->E.Z.subvec(1,NX-2) )/mesh->DX;
	EB->B.Y.subvec(iIndex,fIndex) = ( EB->E.Z.subvec(iIndex+1,fIndex+1) - EB->E.Z.subvec(iIndex,fIndex) )/mesh->DX;

	//z-component
//	EB->B.Z.subvec(1,NX-2) = - ( EB->E.Y.subvec(2,NX-1) - EB->E.Y.subvec(1,NX-2) )/mesh->DX;
	EB->B.Z.subvec(iIndex,fIndex) = - ( EB->E.Y.subvec(iIndex+1,fIndex+1) - EB->E.Y.subvec(iIndex,fIndex) )/mesh->DX;
}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params,const meshParams * mesh,twoDimensional::fields * EB){

}


void EMF_SOLVER::FaradaysLaw(const simulationParameters * params,const meshParams * mesh,threeDimensional::fields * EB){//This function calculates -culr(EB->E)

}


void EMF_SOLVER::advanceBField(const simulationParameters * params,const meshParams * mesh,fields * EB,vector<ionSpecies> * IONS){
	//Using the RK4 scheme to advance B.
	//B^(N+1) = B^(N) + dt( K1^(N) + 2*K2^(N) + 2*K3^(N) + K4^(N) )/6
	dt = params->DT/((double)params->numberOfRKIterations);

	// if(params->mpi.MPI_DOMAIN_NUMBER_CART == 0)
		// EB->B.Z.print("B");

	for(int RKit=0; RKit<params->numberOfRKIterations; RKit++){//Runge-Kutta iterations

		K1 = *EB;//The value of the fields at the time level (N-1/2)
		advanceEField(params,mesh,&K1,IONS);//E1 (using B^(N-1/2))
		FaradaysLaw(params,mesh,&K1);//K1

		K2.B.X = EB->B.X;//B^(N-1/2) + 0.5*dt*K1
		K2.B.Y = EB->B.Y + (0.5*dt)*K1.B.Y;
		K2.B.Z = EB->B.Z + (0.5*dt)*K1.B.Z;
		K2.E = EB->E;
		advanceEField(params,mesh,&K2,IONS);//E2 (using B^(N-1/2) + 0.5*dt*K1)
		FaradaysLaw(params,mesh,&K2);//K2

		K3.B.X = EB->B.X;//B^(N-1/2) + 0.5*dt*K2
		K3.B.Y = EB->B.Y + (0.5*dt)*K2.B.Y;
		K3.B.Z = EB->B.Z + (0.5*dt)*K2.B.Z;
		K3.E = EB->E;
		advanceEField(params,mesh,&K3,IONS);//E3 (using B^(N-1/2) + 0.5*dt*K2)
		FaradaysLaw(params,mesh,&K3);//K3

		K4.B.X = EB->B.X;//B^(N-1/2) + dt*K2
		K4.B.Y = EB->B.Y + dt*K3.B.Y;
		K4.B.Z = EB->B.Z + dt*K3.B.Z;
		K4.E = EB->E;
		advanceEField(params,mesh,&K4,IONS);//E4 (using B^(N-1/2) + dt*K3)
		FaradaysLaw(params,mesh,&K4);//K4

		EB->B += (dt/6)*( K1.B + 2*K2.B + 2*K3.B + K4.B );
	}//Runge-Kutta iterations



	if (params->includeElectronInertia){
		MPI_AllgatherField(params, &EB->B);

		// arma::vec n
		arma::mat M(NX_R, NX_R, fill::zeros);	// Matrix to be inverted to calculate the actual magnetic field
		arma::vec S(NX_R, fill::zeros);			// Modified magnetic field
		double de; 									// Electron skin depth

		de = (F_C_DS/F_E_DS)*sqrt( F_ME_DS*F_EPSILON_DS/params->ne );

		// Elements for first node in mesh
		M(0,0) = 1.0 + 2.0*pow(de,2.0)/pow(mesh->DX,2.0);
		M(0,1) = -pow(de,2.0)/pow(mesh->DX,2.0);


		// Elements for last node in mesh
		M(NX_R-1,NX_R-1) = 1.0 + 2.0*pow(de,2.0)/pow(mesh->DX,2.0);
		M(NX_R-1,NX_R-2) = -pow(de,2.0)/pow(mesh->DX,2.0);

		for(int ii=1; ii<NX_R-1; ii++){
			M(ii,ii-1) = -pow(de,2.0)/pow(mesh->DX,2.0);
			M(ii,ii) = 1.0 + 2.0*pow(de,2.0)/pow(mesh->DX,2.0);
			M(ii,ii+1) = -pow(de,2.0)/pow(mesh->DX,2.0);
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
		cout << "ERROR: Non finite values in Bx" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->B.Y.is_finite()){
		cout << "ERROR: Non finite values in By" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->B.Z.is_finite()){
		cout << "ERROR: Non finite values in Bz" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}
#endif


	switch (params->weightingScheme){
		case(0):{
				smooth_TOS(params,&EB->B,params->smoothingParameter);//Just added!
				break;
				}
		case(1):{
				smooth_TSC(params,&EB->B,params->smoothingParameter);//Just added!
				break;
				}
		case(2):{
				smooth(params,&EB->B,params->smoothingParameter);//Just added!
				break;
				}
		default:{
				smooth_TSC(params,&EB->B,params->smoothingParameter);//Just added!
				}
	}

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


#ifdef ONED
void EMF_SOLVER::aef_1D(const simulationParameters * params,const meshParams * mesh,oneDimensional::fields * EB,vector<ionSpecies> * IONS){

	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART + 1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART + 1));

	// n and U are armadillo vectors initialized in the EMF_SOLVER class constructor.
	n.zeros();
	U.zeros();

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		forwardPBC_1D(&IONS->at(ii).n);
		forwardPBC_1D(&IONS->at(ii).n_);
		forwardPBC_1D(&IONS->at(ii).nv.X);
		forwardPBC_1D(&IONS->at(ii).nv.Y);
		forwardPBC_1D(&IONS->at(ii).nv.Z);

		// Ions density at time level "l + 1/2"
		// n(l+1/2) = ( n(l+1) + n(l) )/2
		n += 0.5*IONS->at(ii).Z*(IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1) + IONS->at(ii).n_.subvec(iIndex - 1, fIndex + 1));
		// n += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1);

		// Ions bulk velocity at time level "l + 1/2"
		//sum_k[ Z_k*n_k*u_k ]
		U.X += IONS->at(ii).Z*IONS->at(ii).nv.X.subvec(iIndex - 1, fIndex + 1);
		U.Y += IONS->at(ii).Z*IONS->at(ii).nv.Y.subvec(iIndex - 1, fIndex + 1);
		U.Z += IONS->at(ii).Z*IONS->at(ii).nv.Z.subvec(iIndex - 1, fIndex + 1);
	}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

	U.X /= n;
	U.Y /= n;
	U.Z /= n;

	n /= n_cs;//Normalized density

	//Definitions

	vfield_vec curlB(NX_T);

	EB->E.fill(0);

	// NOTE: In the following, the indices range per subdomain goes as follows: (iIndex,fIndex) = (1,NX-2).

	// x-component

	curlB.Y.subvec(iIndex,fIndex) = -0.5*( EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/mesh->DX;

	curlB.Z.subvec(iIndex,fIndex) = 0.5*( EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/mesh->DX;

	EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y.subvec(iIndex,fIndex) % EB->B.Z.subvec(iIndex,fIndex) - curlB.Z.subvec(iIndex,fIndex) % EB->B.Y.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*( 0.5*( n.subvec(1,NX_S-2) + n.subvec(2,NX_S-1) ) ) );

	EB->E.X.subvec(iIndex,fIndex) += - 0.5*( U.Y.subvec(1,NX_S-2) + U.Y.subvec(2,NX_S-1) ) % EB->B.Z.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += 0.5*( U.Z.subvec(1,NX_S-2) + U.Z.subvec(2,NX_S-1) ) % EB->B.Y.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += - (params->BGP.Te/F_E_DS)*( (n.subvec(2,NX_S-1) - n.subvec(1,NX_S-2))/mesh->DX )/(0.5*( n.subvec(1,NX_S-2) + n.subvec(2,NX_S-1) ) );

	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFD with DX/2

		EB->E.X.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*0.5*( U.X.subvec(1,NX_S-2) + U.X.subvec(2,NX_S-1) ) % ( (U.X.subvec(2,NX_S-1) - U.X.subvec(1,NX_S-2))/mesh->DX );
	}


	curlB.fill(0);


	//y-component

	curlB.Y.subvec(iIndex,fIndex) = -(EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX;//curl(B)z(i)

	curlB.Z.subvec(iIndex,fIndex) = (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX;//curl(B)z(i)

	MPI_passGhosts(params,&curlB.Y);

	MPI_passGhosts(params,&curlB.Z);

	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*n.subvec(1,NX_S-2));

	EB->E.Y.subvec(iIndex,fIndex) += - U.Z.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);

	EB->E.Y.subvec(iIndex,fIndex) += U.X.subvec(1,NX_S-2) % ( 0.5*(EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1)) );

	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFDs with DX
		EB->E.Y.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*U.X.subvec(1,NX_S-2) % ( 0.5*(U.Y.subvec(2,NX_S-1) - U.Y.subvec(0,NX_S-3))/mesh->DX );

		EB->E.Y.subvec(iIndex,fIndex) += (F_ME_DS/F_E_DS)*( U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*n.subvec(1,NX_S-2)) ) % ( 0.5*(curlB.Y.subvec(iIndex+1,fIndex+1) - curlB.Y.subvec(iIndex-1,fIndex-1))/mesh->DX );

		EB->E.Y.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*( U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*n.subvec(1,NX_S-2) % n.subvec(1,NX_S-2)) ) % ( 0.5*(n.subvec(2,NX_S-1) - n.subvec(0,NX_S-3))/mesh->DX ) % curlB.Y.subvec(iIndex,fIndex);
	}


	//z-component

	// The y-component of Curl(B) has been calculated above.

	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*n.subvec(1,NX_S-2));

	EB->E.Z.subvec(iIndex,fIndex) += - U.X.subvec(1,NX_S-2) % ( 0.5*(EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1)) );

	EB->E.Z.subvec(iIndex,fIndex) += U.Y.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);


	// Including electron inertia term
	if (params->includeElectronInertia){
		// CFDs with DX
		EB->E.Z.subvec(iIndex,fIndex) -= (F_ME_DS/F_E_DS)*U.X.subvec(1,NX_S-2) % ( 0.5*(U.Z.subvec(2,NX_S-1) - U.Z.subvec(0,NX_S-3))/mesh->DX );

		EB->E.Z.subvec(iIndex,fIndex) += (F_ME_DS/F_E_DS)*( U.X.subvec(1,NX_S-2)/(F_MU_DS*F_E_DS*n.subvec(1,NX_S-2)) ) % ( 0.5*(curlB.Z.subvec(iIndex+1,fIndex+1) - curlB.Z.subvec(iIndex-1,fIndex-1))/mesh->DX );
	}


#ifdef CHECKS_ON
	if(!EB->E.X.is_finite()){
		cout << "ERROR: Non finite values in Ex" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->E.Y.is_finite()){
		cout << "ERROR: Non finite values in Ey" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->E.Z.is_finite()){
		cout << "ERROR: Non finite values in Ez" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}
#endif

	switch (params->weightingScheme){
		case(0):{
				smooth_TOS(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		case(1):{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		case(2):{
				smooth(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		default:{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				}
	}
}
#endif


#ifdef TWOD
void EMF_SOLVER::aef_2D(const simulationParameters * params,const meshParams * mesh,twoDimensional::fields * EB,vector<ionSpecies> * IONS){

}
#endif


#ifdef THREED
void EMF_SOLVER::aef_3D(const simulationParameters * params,const meshParams * mesh,threeDimensional::fields * EB,vector<ionSpecies> * IONS){

}
#endif


void EMF_SOLVER::advanceEField(const simulationParameters * params,const meshParams * mesh,fields * EB,vector<ionSpecies> * IONS){

	//The ions' density and flow velocities are stored in the integer nodes,we'll use mean values of these quantities in order to calculate the electric field in the staggered grid.

	#ifdef ONED
	aef_1D(params,mesh,EB,IONS);
	#endif

	#ifdef TWOD
	aef_2D(params,mesh,EB,IONS);
	#endif

	#ifdef THREED
	aef_3D(params,mesh,EB,IONS);
	#endif

}


#ifdef ONED
/* In this function the different ionSpecies vectors represent the following:
	+ IONS__: Ions' variables at time level "l - 3/2"
	+ IONS_: Ions' variables at time level "l - 1/2"
	+ IONS: Ions' variables at time level "l + 1/2"
*/
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const simulationParameters * params, const meshParams * mesh,\
														oneDimensional::fields * EB, vector<ionSpecies> * IONS, const int BAE){

	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	ne.zeros();
	n.zeros();
	U.zeros();

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
			forwardPBC_1D(&IONS->at(ii).n);
			forwardPBC_1D(&IONS->at(ii).n_);
			forwardPBC_1D(&IONS->at(ii).nv.X);
			forwardPBC_1D(&IONS->at(ii).nv.Y);
			forwardPBC_1D(&IONS->at(ii).nv.Z);

			// Electron density at time level "l + 1"
			ne += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1);

			// Ions density at time level "l + 1/2"
			// n(l+1/2) = ( n(l+1) + n(l) )/2
			n += 0.5*IONS->at(ii).Z*(IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1) + IONS->at(ii).n_.subvec(iIndex - 1, fIndex + 1));
			// n += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex - 1, fIndex + 1);

			// Ions bulk velocity at time level "l + 1/2"
			//sum_k[ Z_k*n_k*u_k ]
			U.X += IONS->at(ii).Z*IONS->at(ii).nv.X.subvec(iIndex - 1, fIndex + 1);
			U.Y += IONS->at(ii).Z*IONS->at(ii).nv.Y.subvec(iIndex - 1, fIndex + 1);
			U.Z += IONS->at(ii).Z*IONS->at(ii).nv.Z.subvec(iIndex - 1, fIndex + 1);
	}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

	U.X /= n;
	U.Y /= n;
	U.Z /= n;

	ne /= n_cs;
	n /= n_cs;//Dimensionless density


	n_.zeros();
	U_.zeros();

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		forwardPBC_1D(&IONS->at(ii).n_);
		forwardPBC_1D(&IONS->at(ii).n__);
		forwardPBC_1D(&IONS->at(ii).nv_.X);
		forwardPBC_1D(&IONS->at(ii).nv_.Y);
		forwardPBC_1D(&IONS->at(ii).nv_.Z);

		// Ions density at time level "l - 1/2"
		// n(l-1/2) = ( n(l) + n(l-1) )/2
		n_ += 0.5*IONS->at(ii).Z*(IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) + IONS->at(ii).n_.subvec(iIndex - 1,fIndex + 1));

		// Ions bulk velocity at time level "l - 1/2"
		//sum_k[ Z_k*n_k*u_k ]
		U_.X += IONS->at(ii).Z*IONS->at(ii).nv_.X.subvec(iIndex-1,fIndex+1);
		U_.Y += IONS->at(ii).Z*IONS->at(ii).nv_.Y.subvec(iIndex-1,fIndex+1);
		U_.Z += IONS->at(ii).Z*IONS->at(ii).nv_.Z.subvec(iIndex-1,fIndex+1);
	}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

	U_.X /= n_;
	U_.Y /= n_;
	U_.Z /= n_;

	n_ /= n_cs;//Dimensionless density


	if(BAE == 1){
		n__.zeros();
		U__.fill(0.0);

		for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
			forwardPBC_1D(&IONS->at(ii).n__);
			forwardPBC_1D(&IONS->at(ii).nv__.X);
			forwardPBC_1D(&IONS->at(ii).nv__.Y);
			forwardPBC_1D(&IONS->at(ii).nv__.Z);

			// Ions density at time level "l - 3/2"
			// n(l-3/2) = ( n(l-1) + n(l-2) )/2
			n__ += 0.5*IONS->at(ii).Z*(IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1) + IONS->at(ii).n__.subvec(iIndex - 1,fIndex + 1));

			//sum_k[ Z_k*n_k*u_k ]
			U__.X += IONS->at(ii).Z*IONS->at(ii).nv__.X.subvec(iIndex-1,fIndex+1);
			U__.Y += IONS->at(ii).Z*IONS->at(ii).nv__.Y.subvec(iIndex-1,fIndex+1);
			U__.Z += IONS->at(ii).Z*IONS->at(ii).nv__.Z.subvec(iIndex-1,fIndex+1);
		}//This density is not normalized (n =/= n/n_cs) but it is dimensionless.

		U__.X /= n__;
		U__.Y /= n__;
		U__.Z /= n__;

		n__ /= n_cs;//Dimensionless density

		//Here we use the velocity extrapolation U^(N+1) = 2*U^(N+1/2) - 1.5*U^(N-1/2) + 0.5*U^(N-3/2)
		V = 2.0*U - 1.5*U_ + 0.5*U__;
	}else{
		//Here we use the velocity extrapolation U^(N+1) = 1.5*U^(N+1/2) - 0.5*U^(N-1/2)
		V = 1.5*U - 0.5*U_;
	}


	vfield_vec curlB(NX_T);

	EB->E.fill(0);

	//x-component

	curlB.Y.subvec(iIndex,fIndex) = -0.5*( EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex-1,fIndex-1) )/mesh->DX;

	curlB.Z.subvec(iIndex,fIndex) = 0.5*( EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex-1,fIndex-1) )/mesh->DX;

	EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y.subvec(iIndex,fIndex) % EB->B.Z.subvec(iIndex,fIndex) - curlB.Z.subvec(iIndex,fIndex) % EB->B.Y.subvec(iIndex,fIndex) )/( F_MU_DS*F_E_DS*( 0.5*( ne.subvec(1,NX_S-2) + ne.subvec(2,NX_S-1) ) ) );

	EB->E.X.subvec(iIndex,fIndex) += - 0.5*( V.Y.subvec(1,NX_S-2) + V.Y.subvec(2,NX_S-1) ) % EB->B.Z.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += 0.5*( V.Z.subvec(1,NX_S-2) + V.Z.subvec(2,NX_S-1) ) % EB->B.Y.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += - (params->BGP.Te/F_E_DS)*( (ne.subvec(2,NX_S-1) - ne.subvec(1,NX_S-2))/mesh->DX )/(0.5*( ne.subvec(1,NX_S-2) + ne.subvec(2,NX_S-1) ) );


	curlB.fill(0);


	//y-component

	curlB.Z.subvec(iIndex,fIndex) = (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX;//curl(B)z(i)

	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*ne.subvec(1,NX_S-2));

	EB->E.Y.subvec(iIndex,fIndex) += - V.Z.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);

	EB->E.Y.subvec(iIndex,fIndex) += V.X.subvec(1,NX_S-2) % ( 0.5*(EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1)) );


	//z-component

	curlB.Y.subvec(iIndex,fIndex) = - (EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX ;//curl(B)y(i)

	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y.subvec(iIndex,fIndex) % EB->B.X.subvec(iIndex,fIndex) )/(F_MU_DS*F_E_DS*ne.subvec(1,NX_S-2));

	EB->E.Z.subvec(iIndex,fIndex) += - V.X.subvec(1,NX_S-2) % ( 0.5*(EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1)) );

	EB->E.Z.subvec(iIndex,fIndex) += V.Y.subvec(1,NX_S-2) % EB->B.X.subvec(iIndex,fIndex);


#ifdef CHECKS_ON
	if(!EB->E.X.is_finite()){
		cout << "ERROR: Non finite values in Ex" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->E.Y.is_finite()){
		cout << "ERROR: Non finite values in Ey" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}else if(!EB->E.Z.is_finite()){
		cout << "ERROR: Non finite values in Ez" << endl;
		MPI_Abort(params->mpi.MPI_TOPO, -2);
	}
#endif

		switch (params->weightingScheme){
			case(0):{
					smooth_TOS(params,&EB->E,params->smoothingParameter);//Just added!
					break;
					}
			case(1):{
					smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
					break;
					}
			case(2):{
					smooth(params,&EB->E,params->smoothingParameter);//Just added!
					break;
					}
			default:{
					smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
					}
		}


	}
#endif


#ifdef TWOD
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const simulationParameters * params,const meshParams * mesh,twoDimensional::fields * EB,vector<ionSpecies> * IONS,const int BAE){

}
#endif


#ifdef THREED
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const simulationParameters * params,const meshParams * mesh,threeDimensional::fields * EB,vector<ionSpecies> * IONS,const int BAE){

}
#endif
