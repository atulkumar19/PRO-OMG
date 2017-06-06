#include "emf.h"

void EMF_SOLVER::MPI_passGhosts(const inputParameters * params,vfield_vec * field){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	double sendBuf, recvBuf;

	sendBuf = field->X(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.rRank,0,&recvBuf,1,MPI_DOUBLE,params->mpi.lRank,0,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->X(iIndex-1) = recvBuf;

	sendBuf = field->X(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.lRank,1,&recvBuf,1,MPI_DOUBLE,params->mpi.rRank,1,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->X(fIndex+1) = recvBuf;

	sendBuf = field->Y(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.rRank,2,&recvBuf,1,MPI_DOUBLE,params->mpi.lRank,2,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->Y(iIndex-1) = recvBuf;

	sendBuf = field->Y(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.lRank,3,&recvBuf,1,MPI_DOUBLE,params->mpi.rRank,3,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->Y(fIndex+1) = recvBuf;

	sendBuf = field->Z(fIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.rRank,4,&recvBuf,1,MPI_DOUBLE,params->mpi.lRank,4,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->Z(iIndex-1) = recvBuf;

	sendBuf = field->Z(iIndex);
	MPI_Sendrecv(&sendBuf,1,MPI_DOUBLE,params->mpi.lRank,5,&recvBuf,1,MPI_DOUBLE,params->mpi.rRank,5,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
	field->Z(fIndex+1) = recvBuf;

	
	MPI_Barrier(params->mpi.mpi_topo);
}


void EMF_SOLVER::smooth_TOS(const inputParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);
	int dim = params->meshDim(0) + 2;
	vec b = zeros(dim);
	double w0(23.0/48.0), w1(0.25), w2(1.0/96.0);//weights

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart + 1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart + 1));

	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(2,dim-3) = w0*b.subvec(2,dim-3) + w1*b.subvec(3,dim-2) + w1*b.subvec(1,dim-4) + w2*b.subvec(4,dim-1) + w2*b.subvec(0,dim-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(2,dim-3);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(2,dim-3) = w0*b.subvec(2,dim-3) + w1*b.subvec(3,dim-2) + w1*b.subvec(1,dim-4) + w2*b.subvec(4,dim-1) + w2*b.subvec(0,dim-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(2,dim-3);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(2,dim-3) = w0*b.subvec(2,dim-3) + w1*b.subvec(3,dim-2) + w1*b.subvec(1,dim-4) + w2*b.subvec(4,dim-1) + w2*b.subvec(0,dim-5);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(2,dim-3);

	b.fill(0);
}

void EMF_SOLVER::smooth_TOS(const inputParameters * params,vfield_mat * vf,double as){

}

void EMF_SOLVER::smooth_TSC(const inputParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);
	int dim = params->meshDim(0) + 2;
	vec b = zeros(dim);
	double w0(0.75), w1(0.125);//weights

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart + 1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart + 1));


	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2)+ w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2) + w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Z.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2) + w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(iIndex,fIndex) = (1.0 - as)*vf->Z.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);
}

void EMF_SOLVER::smooth_TSC(const inputParameters * params,vfield_mat * vf,double as){

}


void EMF_SOLVER::smooth(const inputParameters * params,vfield_vec * vf,double as){
	MPI_passGhosts(params,vf);
	int dim = params->meshDim(0) + 2;
	vec b = zeros(dim);
	double w0(0.5), w1(0.25);//weights

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));


	//Step 1: Averaging process
	b = vf->X.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2)+ w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(iIndex,fIndex) = (1.0 - as)*vf->X.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Y.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2) + w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(iIndex,fIndex) = (1.0 - as)*vf->Y.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);

	b.fill(0);

	//Step 1: Averaging process
	b = vf->Z.subvec(iIndex-1,fIndex+1);
	b.subvec(1,dim-2) = w0*b.subvec(1,dim-2) + w1*b.subvec(2,dim-1) + w1*b.subvec(0,dim-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(iIndex,fIndex) = (1.0 - as)*vf->Z.subvec(iIndex,fIndex) + as*b.subvec(1,dim-2);
}


void EMF_SOLVER::smooth(const inputParameters * params,vfield_mat * vf,double as){

}

void EMF_SOLVER::equilibrium(const inputParameters * params,vector<ionSpecies> * IONS,emf * EB,characteristicScales * CS){

}

void EMF_SOLVER::curlE(const inputParameters * params,const meshGeometry * mesh,emf * EB){//This function calculates -culr(EB->E)

	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);

	//Definitions
	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));


	//There is not x-component of curl(B)
	EB->B.X.fill(0);

	//y-component
//	EB->B.Y.subvec(1,NX-2) = ( EB->E.Z.subvec(2,NX-1) - EB->E.Z.subvec(1,NX-2) )/mesh->DX;
	EB->B.Y.subvec(iIndex,fIndex) = ( EB->E.Z.subvec(iIndex+1,fIndex+1) - EB->E.Z.subvec(iIndex,fIndex) )/mesh->DX;
	
	//z-component
//	EB->B.Z.subvec(1,NX-2) = - ( EB->E.Y.subvec(2,NX-1) - EB->E.Y.subvec(1,NX-2) )/mesh->DX;
	EB->B.Z.subvec(iIndex,fIndex) = - ( EB->E.Y.subvec(iIndex+1,fIndex+1) - EB->E.Y.subvec(iIndex,fIndex) )/mesh->DX;
	
}

void EMF_SOLVER::curlE(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB){

}

void EMF_SOLVER::curlE(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB){//This function calculates -culr(EB->E)


}


void EMF_SOLVER::advanceBField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,characteristicScales * CS){

	//Using the RK4 scheme to advance B.
	//B^(N+1) = B^(N) + dt( K1^(N) + 2*K2^(N) + 2*K3^(N) + K4^(N) )/6
	dt = params->DT/((double)params->numberOfRKIterations);

	for(int RKit=0; RKit<params->numberOfRKIterations; RKit++){//Runge-Kutta iterations


		K1 = *EB;//The value of the emf at the time level (N)
		advanceEField(params,mesh,&K1,IONS,CS);//E1 (using B^(N))
		curlE(params,mesh,&K1);//K1

		K2.B.X = EB->B.X;//B^(N) + 0.5*dt*K1
		K2.B.Y = EB->B.Y + (0.5*dt)*K1.B.Y;
		K2.B.Z = EB->B.Z + (0.5*dt)*K1.B.Z;
		K2.E = EB->E;
		advanceEField(params,mesh,&K2,IONS,CS);//E2 (using B^(N) + 0.5*dt*K1)
		curlE(params,mesh,&K2);//K2
	
		K3.B.X = EB->B.X;//B^(N) + 0.5*dt*K2
		K3.B.Y = EB->B.Y + (0.5*dt)*K2.B.Y;
		K3.B.Z = EB->B.Z + (0.5*dt)*K2.B.Z;
		K3.E = EB->E;
		advanceEField(params,mesh,&K3,IONS,CS);//E3 (using B^(N) + 0.5*dt*K2)
		curlE(params,mesh,&K3);//K3

		K4.B.X = EB->B.X;//B^(N) + dt*K2
		K4.B.Y = EB->B.Y + dt*K3.B.Y;
		K4.B.Z = EB->B.Z + dt*K3.B.Z;
		K4.E = EB->E;
		advanceEField(params,mesh,&K4,IONS,CS);//E4 (using B^(N) + dt*K3)
		curlE(params,mesh,&K4);//K4

		EB->B += (dt/6)*( K1.B + 2*K2.B + 2*K3.B + K4.B );

	}//Runge-Kutta iterations

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
		case(3):{
				smooth_TOS(params,&EB->B,params->smoothingParameter);//Just added!
				break;
				}
		case(4):{
				smooth_TSC(params,&EB->B,params->smoothingParameter);//Just added!
				break;
				}
		default:{
				smooth_TSC(params,&EB->B,params->smoothingParameter);//Just added!
				}
	}

	MPI_passGhosts(params,&EB->B);
}

#ifdef ONED
void EMF_SOLVER::aef_1D(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS){

	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);
	
	//Definitions
	int dim(params->meshDim(0) + 2); //dimension of temporal vectors taking into account ghost cells.
	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	double MU = F_MU*( CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass );//dimensionless permeability.
	double e0(F_E/CS->charge);//dimensionless electron charge.
	double n_ch = CS->length*CS->density;//Dimensionless characteristic density.

	vec n = zeros(dim);//Density n = ne = sum_k[ Z_k*n_k ]
	vfield_vec U;//Ion's bulk velocity of the specific population under study (already dimensionless).
	U.zeros(dim);

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		forwardPBC_1D(&IONS->at(ii).n);
		forwardPBC_1D(&IONS->at(ii).nv.X);
		forwardPBC_1D(&IONS->at(ii).nv.Y);
		forwardPBC_1D(&IONS->at(ii).nv.Z);
		
		n += IONS->at(ii).Z*IONS->at(ii).n.subvec(iIndex-1,fIndex+1) \
			+ IONS->at(ii).Z*(CS->length*CS->density)*IONS->at(ii).BGP.BG_n;
			
		//sum_k[ Z_k*n_k*u_k ]
		U.X += IONS->at(ii).Z*IONS->at(ii).nv.X.subvec(iIndex-1,fIndex+1) \
			+ IONS->at(ii).Z*(CS->length*CS->density)*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UX;
		U.Y += IONS->at(ii).Z*IONS->at(ii).nv.Y.subvec(iIndex-1,fIndex+1) \
			+ IONS->at(ii).Z*(CS->length*CS->density)*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UY;
		U.Z += IONS->at(ii).Z*IONS->at(ii).nv.Z.subvec(iIndex-1,fIndex+1) \
			+ IONS->at(ii).Z*(CS->length*CS->density)*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UZ;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	U.X /= n;
	U.Y /= n;
	U.Z /= n;

	n /= n_ch;//Normalized density

	//Definitions
	
	vfield_vec curlB(dim-2);

	EB->E.fill(0);

	//x-component
//(iIndex,fIndex) = (1,NX-2)
	curlB.Y = 0.5*( - (EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX );//curl(B)y(i)

	curlB.Y += 0.5*( - (EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex,fIndex))/mesh->DX );//curl(B)y(i+1)

//	MPI_Barrier(params->mpi.mpi_topo);
//	cout << "Message from process\t" << params->mpi.rank_cart << '\t' << EB->E.X.n_elem << '\t' << EB->E.Y.n_elem << '\n';

	EB->E.X.subvec(iIndex,fIndex) = ( curlB.Y % EB->B.Z.subvec(iIndex,fIndex) )/( MU*e0*( 0.5*( n.subvec(1,dim-2)+ n.subvec(2,dim-1) ) ) );


	curlB.Z = 0.5*( (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX );//curl(B)z(i)

	curlB.Z += 0.5*( (EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex,fIndex))/mesh->DX );//curl(B)z(i+1)

	EB->E.X.subvec(iIndex,fIndex) += - ( curlB.Z % EB->B.Y.subvec(iIndex,fIndex) )/( MU*e0*( 0.5*( n.subvec(1,dim-2) + n.subvec(2,dim-1) ) ) );


	curlB.fill(0);


	EB->E.X.subvec(iIndex,fIndex) += - 0.5*( U.Y.subvec(1,dim-2) + U.Y.subvec(2,dim-1) ) % EB->B.Z.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += 0.5*( U.Z.subvec(1,dim-2) + U.Z.subvec(2,dim-1) ) % EB->B.Y.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += - (params->BGP.backgroundTemperature/e0)*( (n.subvec(2,dim-1) - n.subvec(1,dim-2))/mesh->DX )/(0.5*( n.subvec(1,dim-2) + n.subvec(2,dim-1) ) );


	curlB.Y = - (EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX ;//curl(B)y(i)

	curlB.Z = (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX;//curl(B)z(i)


	//y-component


	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z % EB->B.X.subvec(iIndex,fIndex) )/(MU*e0*n.subvec(1,dim-2));

	EB->E.Y.subvec(iIndex,fIndex) += - U.Z.subvec(1,dim-2) % EB->B.X.subvec(iIndex,fIndex);

	EB->E.Y.subvec(iIndex,fIndex) += U.X.subvec(1,dim-2) % ( 0.5*(EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1)) );


	//z-component


	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y % EB->B.X.subvec(iIndex,fIndex) )/(MU*e0*n.subvec(1,dim-2));

	EB->E.Z.subvec(iIndex,fIndex) += - U.X.subvec(1,dim-2) % ( 0.5*(EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1)) );

	EB->E.Z.subvec(iIndex,fIndex) += U.Y.subvec(1,dim-2) % EB->B.X.subvec(iIndex,fIndex);


#ifdef CHECKS_ON
	if(!EB->E.X.is_finite()){
		std::ofstream ofs ("errors/aef_1D.txt",std::ofstream::out);
		ofs << "\nIn Ex!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Y.is_finite()){
		std::ofstream ofs ("errors/aef_1D.txt",std::ofstream::out);
		ofs << "\nIn Ey!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Z.is_finite()){
		std::ofstream ofs ("errors/aef_1D.txt",std::ofstream::out);
		ofs << "\nIn Ez!\n";
		ofs.close();
		exit(0);
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
		case(3):{
				smooth_TOS(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		case(4):{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		default:{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				}
	}
}
#endif

#ifdef TWOD
void EMF_SOLVER::aef_2D(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS){

}
#endif

#ifdef THREED
void EMF_SOLVER::aef_3D(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS,characteristicScales * CS){

	//Definitions	
	int NX(EB->E.X.n_rows),NY(EB->E.X.n_cols),NZ(EB->E.X.n_slices);

	double MU(0); //dimensionless permeability.
	MU = F_MU*( CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass );
	double e0(F_E/CS->charge);//dimensionless electron charge.

	cube n = zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Density n = ne = sum_k[ Z_k*n_k ]
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			n += IONS->at(ii).Z*IONS->at(ii).n + IONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS->at(ii).BGP.BG_n;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.


	vfield_cube U;//Ion's flow velocity of the specific population under study (already dimensionless).
	U.zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Flow velocity along the x-direction
	
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//sum_k[ Z_k*n_k*u_k ]
			U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS->at(ii).Z*IONS->at(ii).nv.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UX;
			U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS->at(ii).Z*IONS->at(ii).nv.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UY;
			U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS->at(ii).Z*IONS->at(ii).nv.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS->at(ii).BGP.BG_n*IONS->at(ii).BGP.BG_UZ;
	}//The density still have units here.

	U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = U.X.subcube(1,1,1,NX-2,NY-2,NZ-2)/n.subcube(1,1,1,NX-2,NY-2,NZ-2);
	U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2)/n.subcube(1,1,1,NX-2,NY-2,NZ-2);
	U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2)/n.subcube(1,1,1,NX-2,NY-2,NZ-2);

	double n_ch(0);//Dimensionless characteristic density.
	n_ch = (CS->length*CS->length*CS->length)*CS->density;
	n = n/n_ch;//Normalized density

	//Definitions
	
	forwardPBC_3D(&EB->B.X);
	forwardPBC_3D(&EB->B.Y);
	forwardPBC_3D(&EB->B.Z);

	forwardPBC_3D(&n);
	forwardPBC_3D(&U.X);
	forwardPBC_3D(&U.Y);
	forwardPBC_3D(&U.Z);

	vfield_cube curlB;
	curlB.zeros(mesh->dim(0),mesh->dim(1),mesh->dim(2));

	EB->E.fill(0);

	//x-component

	curlB.Y = 0.25*( (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX );//curl(B)y(i,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.X.subcube(2,1,0,NX-1,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX );//curl(B)y(i+1,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DX );//curl(B)y(i,j-1,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(2,0,1,NX-1,NY-3,NZ-2) - EB->B.X.subcube(2,0,0,NX-1,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(2,0,1,NX-1,NY-3,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DX );//curl(B)y(i+1,j-1,k)


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = ( curlB.Y % ( 0.5*( EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) ) ) )/( MU*e0*( 0.5*( n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(2,1,1,NX-1,NY-2,NZ-2) ) ) );


	curlB.Z = 0.25*( (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY );//curl(B)z(i,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.X.subcube(2,0,1,NX-1,NY-3,NZ-2))/mesh->DY );//curl(B)z(i+1,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DY );//curl(B)z(i,j,k-1)

	curlB.Z += 0.25*( (EB->B.Y.subcube(2,1,0,NX-1,NY-2,NZ-3) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(2,1,0,NX-1,NY-2,NZ-3) - EB->B.X.subcube(2,0,0,NX-1,NY-3,NZ-3))/mesh->DY );//curl(B)z(i+1,j,k-1)


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - ( curlB.Z % ( 0.5*( EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) ) ) )/( MU*e0*( 0.5*( n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(2,1,1,NX-1,NY-2,NZ-2) ) ) );


	curlB.Y.fill(0);
	curlB.Z.fill(0);


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Y.subcube(2,1,1,NX-1,NY-2,NZ-2) ) % ( 0.5*(EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2)) );

	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Z.subcube(2,1,1,NX-1,NY-2,NZ-2) )% ( 0.5*( EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) ) );

	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (n.subcube(2,1,1,NX-1,NY-2,NZ-2) - n.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX )/(0.5*( n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(2,1,1,NX-1,NY-2,NZ-2) ) );

	//y-component

	curlB.Z = 0.25*( (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY );//curl(B)z(i,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Y.subcube(0,2,1,NX-3,NY-1,NZ-2))/mesh->DX - (EB->B.X.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY );//curl(B)z(i,j+1,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DY );//curl(B)z(i,j,k-1)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,2,0,NX-2,NY-1,NZ-3) - EB->B.Y.subcube(0,2,0,NX-3,NY-1,NZ-3))/mesh->DX - (EB->B.X.subcube(1,2,0,NX-2,NY-1,NZ-3) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DY );//curl(B)z(i,j+1,k-1)


	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) =  curlB.Z % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3)) )/(MU*e0*( 0.5*(n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,2,1,NX-2,NY-1,NZ-2)) ));


	curlB.X = 0.25*( (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Y.subcube(1,2,0,NX-2,NY-1,NZ-3))/mesh->DZ );//curl(B)x(i,j+1,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i-1,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,2,1,NX-3,NY-1,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,2,1,NX-3,NY-1,NZ-2) - EB->B.Y.subcube(0,2,0,NX-3,NY-1,NZ-3))/mesh->DZ );//curl(B)x(i-1,j+1,k)


	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - curlB.X % ( 0.5*(EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2)) )/(MU*e0*( 0.5*(n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,2,1,NX-2,NY-1,NZ-2)) ));


	curlB.Z.fill(0);
	curlB.X.fill(0);
	

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Z.subcube(1,2,1,NX-2,NY-1,NZ-2) ) % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3)) );

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.X.subcube(1,2,1,NX-2,NY-1,NZ-2) ) % ( 0.5*( EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) ) );

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (n.subcube(1,2,1,NX-2,NY-1,NZ-2) - n.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY )/(0.5*( n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,2,1,NX-2,NY-1,NZ-2) ) );


	//z-component

	curlB.X = 0.25*( (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Z.subcube(1,0,2,NX-2,NY-3,NZ-1))/mesh->DY - (EB->B.Y.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ );//curl(B)x(i,j,k+1)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i-1,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,2,NX-3,NY-2,NZ-1) - EB->B.Z.subcube(0,0,2,NX-3,NY-3,NZ-1))/mesh->DY - (EB->B.Y.subcube(0,1,2,NX-3,NY-2,NZ-1) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DZ );//curl(B)x(i-1,j,k+1)


	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) =  curlB.X % ( 0.5*(EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2)) )/(MU*e0*( 0.5*(n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,1,2,NX-2,NY-2,NZ-1)) ));


	curlB.Y = 0.25*( (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX );//curl(B)y(i,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ - (EB->B.Z.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Z.subcube(0,1,2,NX-3,NY-2,NZ-1))/mesh->DX );//curl(B)y(i,j,k+1)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DX );//curl(B)y(i,j-1,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,2,NX-2,NY-3,NZ-1) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DZ - (EB->B.Z.subcube(1,0,2,NX-2,NY-3,NZ-1) - EB->B.Z.subcube(0,0,2,NX-3,NY-3,NZ-1))/mesh->DX );//curl(B)y(i,j-1,k+1)


	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - curlB.Y % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2)) )/(MU*e0*( 0.5*(n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,1,2,NX-2,NY-2,NZ-1)) ));


	curlB.Z.fill(0);
	curlB.X.fill(0);
	

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.X.subcube(1,1,2,NX-2,NY-2,NZ-1) ) % ( 0.5*(EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2)) );

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Y.subcube(1,1,2,NX-2,NY-2,NZ-1) ) % ( 0.5*( EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) ) );

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (n.subcube(1,1,2,NX-2,NY-2,NZ-1) - n.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ )/(0.5*( n.subcube(1,1,1,NX-2,NY-2,NZ-2) + n.subcube(1,1,2,NX-2,NY-2,NZ-1) ) );


	if(!EB->E.X.is_finite()){
		std::ofstream ofs ("errors/aef_3D.txt",std::ofstream::out);
		ofs << "\nIn Ex!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Y.is_finite()){
		std::ofstream ofs ("errors/aef_3D.txt",std::ofstream::out);
		ofs << "\nIn Ey!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Z.is_finite()){
		std::ofstream ofs ("errors/aef_3D.txt",std::ofstream::out);
		ofs << "\nIn Ez!\n";
		ofs.close();
		exit(0);
	}

	restoreCube(&EB->B.X);
	restoreCube(&EB->B.Y);
	restoreCube(&EB->B.Z);

}
#endif

void EMF_SOLVER::advanceEField(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,characteristicScales * CS){

	//The ions' density and flow velocities are stored in the integer nodes,we'll use mean values of these quantities in order to calculate the electric field in the staggered grid.

	#ifdef ONED
	aef_1D(params,mesh,EB,IONS,CS);
	#endif

	#ifdef TWOD
	aef_2D(params,mesh,EB,IONS,CS);
	#endif

	#ifdef THREED
	aef_3D(params,mesh,EB,IONS,CS);
	#endif

}

#ifdef ONED
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,oneDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE){

	MPI_passGhosts(params,&EB->E);
	MPI_passGhosts(params,&EB->B);
	
	//Definitions
	int dim(params->meshDim(0) + 2); //dimension of temporal vectors taking into account ghost cells.
	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	double MU = F_MU*( CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass );//dimensionless permeability.
	double e0(F_E/CS->charge);//dimensionless electron charge.
	double n_ch = CS->length*CS->density;//Dimensionless characteristic density.

	vec nNew = zeros(dim);//Density n = ne = sum_k[ Z_k*n_k ]
	vfield_vec newU;//Ion's flow velocity of the specific population under study (already dimensionless).
	newU.zeros(dim);//Flow velocity along the x-direction

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			forwardPBC_1D(&newIONS->at(ii).n);
			forwardPBC_1D(&newIONS->at(ii).nv.X);
			forwardPBC_1D(&newIONS->at(ii).nv.Y);
			forwardPBC_1D(&newIONS->at(ii).nv.Z);

			nNew += newIONS->at(ii).Z*newIONS->at(ii).n.subvec(iIndex-1,fIndex+1) \
					+ newIONS->at(ii).Z*(CS->length*CS->density)*newIONS->at(ii).BGP.BG_n;

			//sum_k[ Z_k*n_k*u_k ]
			newU.X += newIONS->at(ii).Z*newIONS->at(ii).nv.X.subvec(iIndex-1,fIndex+1) \
					+ newIONS->at(ii).Z*(CS->length*CS->density)*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UX;
			newU.Y += newIONS->at(ii).Z*newIONS->at(ii).nv.Y.subvec(iIndex-1,fIndex+1) \
					+ newIONS->at(ii).Z*(CS->length*CS->density)*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UY;
			newU.Z += newIONS->at(ii).Z*newIONS->at(ii).nv.Z.subvec(iIndex-1,fIndex+1) \
					+ newIONS->at(ii).Z*(CS->length*CS->density)*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UZ;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	newU.X /= nNew;
	newU.Y /= nNew;
	newU.Z /= nNew;

	nNew /= n_ch;//Dimensionless density

	vec nOld = zeros(dim);//Density n = ne = sum_k[ Z_k*n_k ]
	vfield_vec oldU;//Ion's flow velocity of the specific population under study (already dimensionless).
	oldU.zeros(dim);//Flow velocity along the x-direction

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		forwardPBC_1D(&oldIONS->at(ii).n);
		forwardPBC_1D(&oldIONS->at(ii).nv.X);
		forwardPBC_1D(&oldIONS->at(ii).nv.Y);
		forwardPBC_1D(&oldIONS->at(ii).nv.Z);


		nOld += oldIONS->at(ii).Z*oldIONS->at(ii).n.subvec(iIndex-1,fIndex+1) \
				+ oldIONS->at(ii).Z*(CS->length*CS->density)*oldIONS->at(ii).BGP.BG_n;

		//sum_k[ Z_k*n_k*u_k ]
		oldU.X += oldIONS->at(ii).Z*oldIONS->at(ii).nv.X.subvec(iIndex-1,fIndex+1) \
				+ oldIONS->at(ii).Z*(CS->length*CS->density)*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UX;
		oldU.Y += oldIONS->at(ii).Z*oldIONS->at(ii).nv.Y.subvec(iIndex-1,fIndex+1) \
				+ oldIONS->at(ii).Z*(CS->length*CS->density)*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UY;
		oldU.Z += oldIONS->at(ii).Z*oldIONS->at(ii).nv.Z.subvec(iIndex-1,fIndex+1) \
				+ oldIONS->at(ii).Z*(CS->length*CS->density)*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UZ;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	oldU.X /= nOld;
	oldU.Y /= nOld;
	oldU.Z /= nOld;

	nOld /= n_ch;//Dimensionless density

	/* Bashford-Adams extrapolation variables */
	vec n_BAE = zeros(dim);
	vfield_vec U_BAE(dim);//Ion's flow velocity of the specific population under study (already dimensionless).

	if(BAE == 1){
		U_BAE.fill(0);

		for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			forwardPBC_1D(&IONS_BAE->at(ii).n);
			forwardPBC_1D(&IONS_BAE->at(ii).nv.X);
			forwardPBC_1D(&IONS_BAE->at(ii).nv.Y);
			forwardPBC_1D(&IONS_BAE->at(ii).nv.Z);

			n_BAE += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).n.subvec(iIndex-1,fIndex+1) \
					+ IONS_BAE->at(ii).Z*(CS->length*CS->density)*IONS_BAE->at(ii).BGP.BG_n;

			//sum_k[ Z_k*n_k*u_k ]
			U_BAE.X += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.X.subvec(iIndex-1,fIndex+1) \
					+ IONS_BAE->at(ii).Z*(CS->length*CS->density)*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UX;
			U_BAE.Y += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.Y.subvec(iIndex-1,fIndex+1) \
					+ IONS_BAE->at(ii).Z*(CS->length*CS->density)*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UY;
			U_BAE.Z += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.Z.subvec(iIndex-1,fIndex+1) \
					+ IONS_BAE->at(ii).Z*(CS->length*CS->density)*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UZ;
		}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

		U_BAE.X /= n_BAE;
		U_BAE.Y /= n_BAE;
		U_BAE.Z /= n_BAE;

		n_BAE /= n_ch;//Dimensionless density
	}


	//Definitions

	vfield_vec U;

	if(BAE == 1){//Here we use the velocity extrapolation V^(N+1) = 2*V^(N+1/2) - 1.5*V^(N-1/2) + 0.5*V^(N-3/2)
		U = 2.0*newU - 1.5*oldU + 0.5*U_BAE;
	}else{//Here we use the velocity extrapolation V^(N+1) = 1.5*V^(N+1/2) - 0.5*V^(N-1/2)
		U = 1.5*newU - 0.5*oldU;
	}
	
	vfield_vec curlB;
	curlB.zeros(dim-2);

	EB->E.fill(0);

	//x-component

	curlB.Y = 0.5*( - (EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX );//curl(B)y(i)

	curlB.Y += 0.5*( - (EB->B.Z.subvec(iIndex+1,fIndex+1) - EB->B.Z.subvec(iIndex,fIndex))/mesh->DX );//curl(B)y(i+1)


	EB->E.X.subvec(iIndex,fIndex)  = ( curlB.Y % EB->B.Z.subvec(iIndex,fIndex)  )/( MU*e0*( 0.5*( nNew.subvec(1,dim-2) + nNew.subvec(2,dim-1) ) ) );


	curlB.Z = 0.5*( (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX );//curl(B)z(i)

	curlB.Z += 0.5*( (EB->B.Y.subvec(iIndex+1,fIndex+1) - EB->B.Y.subvec(iIndex,fIndex))/mesh->DX );//curl(B)z(i+1)


	EB->E.X.subvec(iIndex,fIndex) += - ( curlB.Z % EB->B.Y.subvec(iIndex,fIndex) )/( MU*e0*( 0.5*( nNew.subvec(1,dim-2) + nNew.subvec(2,dim-1) ) ) );


	curlB.fill(0);


	EB->E.X.subvec(iIndex,fIndex) += - 0.5*( U.Y.subvec(1,dim-2) + U.Y.subvec(2,dim-1) ) % EB->B.Z.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += 0.5*( U.Z.subvec(1,dim-2) + U.Z.subvec(2,dim-1) ) % EB->B.Y.subvec(iIndex,fIndex);

	EB->E.X.subvec(iIndex,fIndex) += - (params->BGP.backgroundTemperature/e0)*( (nNew.subvec(2,dim-1) \
									- nNew.subvec(1,dim-2))/mesh->DX )/(0.5*( nNew.subvec(1,dim-2) + nNew.subvec(2,dim-1) ) );


	curlB.Y = - (EB->B.Z.subvec(iIndex,fIndex) - EB->B.Z.subvec(iIndex-1,fIndex-1))/mesh->DX ;//curl(B)y(i)

	curlB.Z = (EB->B.Y.subvec(iIndex,fIndex) - EB->B.Y.subvec(iIndex-1,fIndex-1))/mesh->DX;//curl(B)z(i)


	//y-component


	EB->E.Y.subvec(iIndex,fIndex) = ( curlB.Z % EB->B.X.subvec(iIndex,fIndex) )/(MU*e0*nNew.subvec(1,dim-2));

	EB->E.Y.subvec(iIndex,fIndex) += - U.Z.subvec(1,dim-2) % EB->B.X.subvec(iIndex,fIndex);

	EB->E.Y.subvec(iIndex,fIndex) += U.X.subvec(1,dim-2) % ( 0.5*(EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1)) );


	//z-component


	EB->E.Z.subvec(iIndex,fIndex) = - ( curlB.Y % EB->B.X.subvec(iIndex,fIndex) )/(MU*e0*nNew.subvec(1,dim-2));

	EB->E.Z.subvec(iIndex,fIndex) += - U.X.subvec(1,dim-2) % ( 0.5*(EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1)) );

	EB->E.Z.subvec(iIndex,fIndex) += U.Y.subvec(1,dim-2) % EB->B.X.subvec(iIndex,fIndex);


#ifdef CHECKS_ON
	if(!EB->E.X.is_finite()){
		std::ofstream ofs ("errors/aefwve_1D.txt",std::ofstream::out);
		ofs << "\nIn Ex!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Y.is_finite()){
		std::ofstream ofs ("errors/aefwve_1D.txt",std::ofstream::out);
		ofs << "\nIn Ey!\n";
		ofs.close();
		exit(0);
	}else if(!EB->E.Z.is_finite()){
		std::ofstream ofs ("errors/aefwve_1D.txt",std::ofstream::out);
		ofs << "\nIn Ez!\n";
		ofs.close();
		exit(0);
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
		case(3):{
				smooth_TOS(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		case(4):{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				break;
				}
		default:{
				smooth_TSC(params,&EB->E,params->smoothingParameter);//Just added!
				}
	}

}
#endif

#ifdef TWOD
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,twoDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE){

}
#endif

#ifdef THREED
void EMF_SOLVER::advanceEFieldWithVelocityExtrapolation(const inputParameters * params,const meshGeometry * mesh,threeDimensional::electromagneticFields * EB,vector<ionSpecies> * IONS_BAE,vector<ionSpecies> * oldIONS,vector<ionSpecies> * newIONS,characteristicScales * CS,const int BAE){

	//Definitions	
	int NX(EB->E.X.n_rows),NY(EB->E.X.n_cols),NZ(EB->E.X.n_slices);

	double MU(0); //dimensionless permeability.
	MU = F_MU*( CS->density*pow(CS->charge*CS->velocity*CS->time,2)/CS->mass );
	double e0(F_E/CS->charge);//dimensionless electron charge.

	cube nNew = zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Density n = ne = sum_k[ Z_k*n_k ]
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		nNew += newIONS->at(ii).Z*newIONS->at(ii).n + (CS->length*CS->length*CS->length)*CS->density*newIONS->at(ii).BGP.BG_n;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	vfield_cube newU;//Ion's flow velocity of the specific population under study (already dimensionless).
	newU.zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);
	
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//sum_k[ Z_k*n_k*u_k ]
		newU.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += newIONS->at(ii).Z*newIONS->at(ii).nv.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + newIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UX;
		newU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += newIONS->at(ii).Z*newIONS->at(ii).nv.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + newIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UY;
		newU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += newIONS->at(ii).Z*newIONS->at(ii).nv.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + newIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*newIONS->at(ii).BGP.BG_n*newIONS->at(ii).BGP.BG_UZ;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	newU.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = newU.X.subcube(1,1,1,NX-2,NY-2,NZ-2)/nNew.subcube(1,1,1,NX-2,NY-2,NZ-2);
	newU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = newU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2)/nNew.subcube(1,1,1,NX-2,NY-2,NZ-2);
	newU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = newU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2)/nNew.subcube(1,1,1,NX-2,NY-2,NZ-2);

	double n_ch(0);//Dimensionless characteristic density.
	n_ch = (CS->length*CS->length*CS->length)*CS->density;
	nNew = nNew/n_ch;//Dimensionless density

	cube nOld = zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Density n = ne = sum_k[ Z_k*n_k ]
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		nOld += oldIONS->at(ii).Z*oldIONS->at(ii).n + oldIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*oldIONS->at(ii).BGP.BG_n;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.


	vfield_cube oldU;//Ion's flow velocity of the specific population under study (already dimensionless).
	oldU.zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);
	for(int ii=0;ii<params->numberOfIonSpecies;ii++){//sum_k[ Z_k*n_k*u_k ]
		oldU.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += oldIONS->at(ii).Z*oldIONS->at(ii).nv.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + oldIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UX;
		oldU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += oldIONS->at(ii).Z*oldIONS->at(ii).nv.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + oldIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UX;
		oldU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += oldIONS->at(ii).Z*oldIONS->at(ii).nv.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + oldIONS->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*oldIONS->at(ii).BGP.BG_n*oldIONS->at(ii).BGP.BG_UX;
	}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

	oldU.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = oldU.X.subcube(1,1,1,NX-2,NY-2,NZ-2)/nOld.subcube(1,1,1,NX-2,NY-2,NZ-2);
	oldU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = oldU.Y.subcube(1,1,1,NX-2,NY-2,NZ-2)/nOld.subcube(1,1,1,NX-2,NY-2,NZ-2);
	oldU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = oldU.Z.subcube(1,1,1,NX-2,NY-2,NZ-2)/nOld.subcube(1,1,1,NX-2,NY-2,NZ-2);

	nOld = nOld/n_ch;//Dimensionless density

	/* Bashford-Adams extrapolation variables */
	cube n_BAE = zeros(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Density n = ne = sum_k[ Z_k*n_k ]
	vfield_cube U_BAE(mesh->dim(0)+2,mesh->dim(1)+2,mesh->dim(2)+2);//Ion's flow velocity of the specific population under study (already dimensionless).

	if(BAE == 1){

		for(int ii=0;ii<IONS_BAE->size();ii++){
			if(IONS_BAE->at(ii).SPECIES != 0){//If the ions are not tracers then...	
				n_BAE += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).n + IONS_BAE->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS_BAE->at(ii).BGP.BG_n;
			}
		}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

		U_BAE.fill(0);
	
		for(int ii=0;ii<IONS_BAE->size();ii++){//sum_k[ Z_k*n_k*u_k ]
			if(IONS_BAE->at(ii).SPECIES != 0){//If the ions are not tracers then...	
				U_BAE.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS_BAE->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UX;
				U_BAE.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS_BAE->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UX;
				U_BAE.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += IONS_BAE->at(ii).Z*IONS_BAE->at(ii).nv.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + IONS_BAE->at(ii).Z*(CS->length*CS->length*CS->length)*CS->density*IONS_BAE->at(ii).BGP.BG_n*IONS_BAE->at(ii).BGP.BG_UX;
			}
		}//This density is not normalized (n =/= n/n_ch) but it is dimensionless.

		U_BAE.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = U_BAE.X.subcube(1,1,1,NX-2,NY-2,NZ-2)/n_BAE.subcube(1,1,1,NX-2,NY-2,NZ-2);
		U_BAE.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = U_BAE.Y.subcube(1,1,1,NX-2,NY-2,NZ-2)/n_BAE.subcube(1,1,1,NX-2,NY-2,NZ-2);
		U_BAE.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = U_BAE.Z.subcube(1,1,1,NX-2,NY-2,NZ-2)/n_BAE.subcube(1,1,1,NX-2,NY-2,NZ-2);

		n_BAE = n_BAE/n_ch;//Dimensionless density
	}

	//Definitions

	vfield_cube U;

	if(BAE == 1){//Here we use the velocity extrapolation V^(N+1) = 2*V^(N+1/2) - 1.5*V^(N-1/2) + 0.5*V^(N-3/2)
		U = 2.0*newU - 1.5*oldU + 0.5*U_BAE;
	}else{//Here we use the velocity extrapolation V^(N+1) = 1.5*V^(N+1/2) - 0.5*V^(N-1/2)
		U = 1.5*newU - 0.5*oldU;
	}

/*
	if(!U.X.is_finite()){
		cout << "There is,at least,one NaN in U.X\n";
	}else if(!U.Y.is_finite()){
		cout << "There is,at least,one NaN in U.Y\n";
	}else if(!U.Z.is_finite()){
		cout << "There is,at least,one NaN in U.Z\n";
	}
*/
	
	forwardPBC_3D(&EB->B.X);
	forwardPBC_3D(&EB->B.Y);
	forwardPBC_3D(&EB->B.Z);

	forwardPBC_3D(&nNew);
	forwardPBC_3D(&U.X);
	forwardPBC_3D(&U.Y);
	forwardPBC_3D(&U.Z);

	vfield_cube curlB;
	curlB.zeros(mesh->dim(0),mesh->dim(1),mesh->dim(2));

	EB->E.fill(0);

	//x-component

	curlB.Y = 0.25*( (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX );//curl(B)y(i,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.X.subcube(2,1,0,NX-1,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX );//curl(B)y(i+1,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DX );//curl(B)y(i,j-1,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(2,0,1,NX-1,NY-3,NZ-2) - EB->B.X.subcube(2,0,0,NX-1,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(2,0,1,NX-1,NY-3,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DX );//curl(B)y(i+1,j-1,k)


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = ( curlB.Y % ( 0.5*( EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) ) ) )/( MU*e0*( 0.5*( nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(2,1,1,NX-1,NY-2,NZ-2) ) ) );


	curlB.Z = 0.25*( (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY );//curl(B)z(i,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(2,1,1,NX-1,NY-2,NZ-2) - EB->B.X.subcube(2,0,1,NX-1,NY-3,NZ-2))/mesh->DY );//curl(B)z(i+1,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DY );//curl(B)z(i,j,k-1)

	curlB.Z += 0.25*( (EB->B.Y.subcube(2,1,0,NX-1,NY-2,NZ-3) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(2,1,0,NX-1,NY-2,NZ-3) - EB->B.X.subcube(2,0,0,NX-1,NY-3,NZ-3))/mesh->DY );//curl(B)z(i+1,j,k-1)


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - ( curlB.Z % ( 0.5*( EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) ) ) )/( MU*e0*( 0.5*( nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(2,1,1,NX-1,NY-2,NZ-2) ) ) );


	curlB.Y.fill(0);
	curlB.Z.fill(0);


	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Y.subcube(2,1,1,NX-1,NY-2,NZ-2) ) % ( 0.5*(EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2)) );

	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Z.subcube(2,1,1,NX-1,NY-2,NZ-2) )% ( 0.5*( EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) ) );

	EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (nNew.subcube(2,1,1,NX-1,NY-2,NZ-2) - nNew.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DX )/(0.5*( nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(2,1,1,NX-1,NY-2,NZ-2) ) );

	//y-component

	curlB.Z = 0.25*( (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX - (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY );//curl(B)z(i,j,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Y.subcube(0,2,1,NX-3,NY-1,NZ-2))/mesh->DX - (EB->B.X.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY );//curl(B)z(i,j+1,k)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DX - (EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DY );//curl(B)z(i,j,k-1)

	curlB.Z += 0.25*( (EB->B.Y.subcube(1,2,0,NX-2,NY-1,NZ-3) - EB->B.Y.subcube(0,2,0,NX-3,NY-1,NZ-3))/mesh->DX - (EB->B.X.subcube(1,2,0,NX-2,NY-1,NZ-3) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DY );//curl(B)z(i,j+1,k-1)


	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) =  curlB.Z % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3)) )/(MU*e0*( 0.5*(nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,2,1,NX-2,NY-1,NZ-2)) ));


	curlB.X = 0.25*( (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,2,1,NX-2,NY-1,NZ-2) - EB->B.Y.subcube(1,2,0,NX-2,NY-1,NZ-3))/mesh->DZ );//curl(B)x(i,j+1,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i-1,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,2,1,NX-3,NY-1,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,2,1,NX-3,NY-1,NZ-2) - EB->B.Y.subcube(0,2,0,NX-3,NY-1,NZ-3))/mesh->DZ );//curl(B)x(i-1,j+1,k)


	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - curlB.X % ( 0.5*(EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2)) )/(MU*e0*( 0.5*(nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,2,1,NX-2,NY-1,NZ-2)) ));


	curlB.Z.fill(0);
	curlB.X.fill(0);
	

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Z.subcube(1,2,1,NX-2,NY-1,NZ-2) ) % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3)) );

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.X.subcube(1,2,1,NX-2,NY-1,NZ-2) ) % ( 0.5*( EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) ) );

	EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (nNew.subcube(1,2,1,NX-2,NY-1,NZ-2) - nNew.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DY )/(0.5*( nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,2,1,NX-2,NY-1,NZ-2) ) );


	//z-component

	curlB.X = 0.25*( (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Z.subcube(1,0,2,NX-2,NY-3,NZ-1))/mesh->DY - (EB->B.Y.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ );//curl(B)x(i,j,k+1)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DY - (EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2) - EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3))/mesh->DZ );//curl(B)x(i-1,j,k)

	curlB.X += 0.25*( (EB->B.Z.subcube(0,1,2,NX-3,NY-2,NZ-1) - EB->B.Z.subcube(0,0,2,NX-3,NY-3,NZ-1))/mesh->DY - (EB->B.Y.subcube(0,1,2,NX-3,NY-2,NZ-1) - EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DZ );//curl(B)x(i-1,j,k+1)


	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) =  curlB.X % ( 0.5*(EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2)) )/(MU*e0*( 0.5*(nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,1,2,NX-2,NY-2,NZ-1)) ));


	curlB.Y = 0.25*( (EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) - EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2))/mesh->DX );//curl(B)y(i,j,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ - (EB->B.Z.subcube(1,1,2,NX-2,NY-2,NZ-1) - EB->B.Z.subcube(0,1,2,NX-3,NY-2,NZ-1))/mesh->DX );//curl(B)y(i,j,k+1)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3))/mesh->DZ - (EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) - EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2))/mesh->DX );//curl(B)y(i,j-1,k)

	curlB.Y += 0.25*( (EB->B.X.subcube(1,0,2,NX-2,NY-3,NZ-1) - EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2))/mesh->DZ - (EB->B.Z.subcube(1,0,2,NX-2,NY-3,NZ-1) - EB->B.Z.subcube(0,0,2,NX-3,NY-3,NZ-1))/mesh->DX );//curl(B)y(i,j-1,k+1)


	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - curlB.Y % ( 0.5*(EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2)) )/(MU*e0*( 0.5*(nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,1,2,NX-2,NY-2,NZ-1)) ));


	curlB.Z.fill(0);
	curlB.X.fill(0);
	

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - 0.5*( U.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.X.subcube(1,1,2,NX-2,NY-2,NZ-1) ) % ( 0.5*(EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2)) );

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += 0.5*( U.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + U.Y.subcube(1,1,2,NX-2,NY-2,NZ-1) ) % ( 0.5*( EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) ) );

	EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) += - (params->BGP.backgroundTemperature/e0)*( (nNew.subcube(1,1,2,NX-2,NY-2,NZ-1) - nNew.subcube(1,1,1,NX-2,NY-2,NZ-2))/mesh->DZ )/(0.5*( nNew.subcube(1,1,1,NX-2,NY-2,NZ-2) + nNew.subcube(1,1,2,NX-2,NY-2,NZ-1) ) );


	if(!EB->E.X.is_finite()){
		std::ofstream ofs ("errors/aefwve_3D.txt",std::ofstream::out);
		ofs << "\nIn Ex!\n";
		ofs.close();
		exit(1);
	}else if(!EB->E.Y.is_finite()){
		std::ofstream ofs ("errors/aefwve_3D.txt",std::ofstream::out);
		ofs << "\nIn Ey!\n";
		ofs.close();
		exit(1);
	}else if(!EB->E.Z.is_finite()){
		std::ofstream ofs ("errors/aefwve_3D.txt",std::ofstream::out);
		ofs << "\nIn Ez!\n";
		ofs.close();
		exit(1);
	}

	restoreCube(&EB->B.X);
	restoreCube(&EB->B.Y);
	restoreCube(&EB->B.Z);

}
#endif

