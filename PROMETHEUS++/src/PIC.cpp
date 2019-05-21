#include "PIC.h"

PIC::PIC(){
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

}

void PIC::MPI_BcastDensity(const inputParameters * params,ionSpecies * ions){

	vec nSend = zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);
	vec nRecv = zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);
	nSend = ions->n;

	MPI_ARMA_VEC mpi_n(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);

	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(nSend.memptr(),1,mpi_n.type,params->mpi.rRank,0,nRecv.memptr(),1,mpi_n.type,params->mpi.lRank,0,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
		ions->n += nRecv;
		nSend = nRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::MPI_BcastBulkVelocity(const inputParameters * params,ionSpecies * ions){

	vec bufSend = zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);
	vec bufRecv = zeros(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);

	MPI_ARMA_VEC mpi_n(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS+2);


	//x-component
	bufSend = ions->nv.X;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(),1,mpi_n.type,params->mpi.rRank,0,bufRecv.memptr(),1,mpi_n.type,params->mpi.lRank,0,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
		ions->nv.X += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//x-component
	bufSend = ions->nv.Y;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(),1,mpi_n.type,params->mpi.rRank,0,bufRecv.memptr(),1,mpi_n.type,params->mpi.lRank,0,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
		ions->nv.Y += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	//x-component
	bufSend = ions->nv.Z;
	for(int ii=0;ii<params->mpi.NUMBER_MPI_DOMAINS-1;ii++){
		MPI_Sendrecv(bufSend.memptr(),1,mpi_n.type,params->mpi.rRank,0,bufRecv.memptr(),1,mpi_n.type,params->mpi.lRank,0,params->mpi.mpi_topo,MPI_STATUS_IGNORE);
		ions->nv.Z += bufRecv;
		bufSend = bufRecv;
	}

	MPI_Barrier(params->mpi.mpi_topo);
}


void PIC::MPI_AllgatherField(const inputParameters * params,vfield_vec * field){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));
	vec recvBuf(params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS);
	vec sendBuf(params->meshDim(0));

	MPI_ARMA_VEC chunk(params->meshDim(0));

	//Allgather for x-component
	sendBuf = field->X.subvec(iIndex,fIndex);
	MPI_Allgather(sendBuf.memptr(),1,chunk.type,recvBuf.memptr(),1,chunk.type,params->mpi.mpi_topo);
	field->X.subvec(1,params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	//Allgather for y-component
	sendBuf = field->Y.subvec(iIndex,fIndex);
	MPI_Allgather(sendBuf.memptr(),1,chunk.type,recvBuf.memptr(),1,chunk.type,params->mpi.mpi_topo);
	field->Y.subvec(1,params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;

	//Allgather for z-component
	sendBuf = field->Z.subvec(iIndex,fIndex);
	MPI_Allgather(sendBuf.memptr(),1,chunk.type,recvBuf.memptr(),1,chunk.type,params->mpi.mpi_topo);
	field->Z.subvec(1,params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS) = recvBuf;
}


void PIC::smooth_TOS(vec * v,double as){

	//Step 1: Averaging process
	int NX(v->n_elem), N(v->n_elem + 2);
	vec b = zeros(NX+2);
	double w0(23.0/48.0), w1(0.25), w2(1.0/96.0);//weights

	b.subvec(2,N-3) = v->subvec(1,NX-2);

	forwardPBC_1D_TOS(&b);

	b.subvec(2,N-3) = w0*b.subvec(2,N-3) + w1*b.subvec(3,N-2) + w1*b.subvec(1,N-4) + w2*b.subvec(4,N-1) + w2*b.subvec(0,N-5);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1,NX-2) = (1-as)*v->subvec(1,NX-2) + as*b.subvec(2,N-3);
}


void PIC::smooth_TOS(vfield_vec * vf,double as){

	int NX(vf->X.n_elem), N(vf->X.n_elem + 2);
	vec b = zeros(NX+2);
	double w0(23.0/48.0), w1(0.25), w2(1.0/96.0);//weights

	//Step 1: Averaging process
	b.subvec(2,N-3) = vf->X.subvec(1,NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2,N-3) = w0*b.subvec(2,N-3) + w1*b.subvec(3,N-2) + w1*b.subvec(1,N-4) + w2*b.subvec(4,N-1) + w2*b.subvec(0,N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->X.subvec(1,NX-2) = (1-as)*vf->X.subvec(1,NX-2) + as*b.subvec(2,N-3);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(2,N-3) = vf->Y.subvec(1,NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2,N-3) = w0*b.subvec(2,N-3) + w1*b.subvec(3,N-2) + w1*b.subvec(1,N-4) + w2*b.subvec(4,N-1) + w2*b.subvec(0,N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->Y.subvec(1,NX-2) = (1-as)*vf->Y.subvec(1,NX-2) + as*b.subvec(2,N-3);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(2,N-3) = vf->Z.subvec(1,NX-2);
	forwardPBC_1D_TOS(&b);
	b.subvec(2,N-3) = w0*b.subvec(2,N-3) + w1*b.subvec(3,N-2) + w1*b.subvec(1,N-4) + w2*b.subvec(4,N-1) + w2*b.subvec(0,N-5);

	//Step 2: Averaged weighted variable estimation.
	vf->Z.subvec(1,NX-2) = (1-as)*vf->Z.subvec(1,NX-2) + as*b.subvec(2,N-3);
}


void PIC::smooth_TOS(vfield_mat * vf,double as){

}


void PIC::smooth_TSC(vec * v,double as){

	//Step 1: Averaging process

	int NX(v->n_elem);
	vec b = zeros(NX);
	double w0(0.75), w1(0.125);//weights

	b.subvec(1,NX-2) = v->subvec(1,NX-2);
	forwardPBC_1D(&b);

	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1,NX-2) = (1-as)*v->subvec(1,NX-2) + as*b.subvec(1,NX-2);
}


void PIC::smooth_TSC(vfield_vec * vf,double as){

	int NX(vf->X.n_elem);
	vec b = zeros(NX);
	double w0(0.75), w1(0.125);//weights


	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->X.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(1,NX-2) = (1-as)*vf->X.subvec(1,NX-2) + as*b.subvec(1,NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->Y.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(1,NX-2) = (1-as)*vf->Y.subvec(1,NX-2) + as*b.subvec(1,NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->Z.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(1,NX-2) = (1-as)*vf->Z.subvec(1,NX-2) + as*b.subvec(1,NX-2);
}


void PIC::smooth_TSC(vfield_mat * vf,double as){

}


void PIC::smooth(vec * v,double as){

	//Step 1: Averaging process

	int NX(v->n_elem);
	vec b = zeros(NX);
	double w0(0.5), w1(0.25);//weights

	b.subvec(1,NX-2) = v->subvec(1,NX-2);
	forwardPBC_1D(&b);

	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted variable estimation.
	v->subvec(1,NX-2) = (1-as)*v->subvec(1,NX-2) + as*b.subvec(1,NX-2);
}


void PIC::smooth(vfield_vec * vf,double as){

	int NX(vf->X.n_elem);
	vec b = zeros(NX);
	double w0(0.5), w1(0.25);//weights


	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->X.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->X.subvec(1,NX-2) = (1-as)*vf->X.subvec(1,NX-2) + as*b.subvec(1,NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->Y.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Y.subvec(1,NX-2) = (1-as)*vf->Y.subvec(1,NX-2) + as*b.subvec(1,NX-2);

	b.fill(0);

	//Step 1: Averaging process
	b.subvec(1,NX-2) = vf->Z.subvec(1,NX-2);
	forwardPBC_1D(&b);
	b.subvec(1,NX-2) = w0*b.subvec(1,NX-2) + w1*b.subvec(2,NX-1) + w1*b.subvec(0,NX-3);

	//Step 2: Averaged weighted vector field estimation.
	vf->Z.subvec(1,NX-2) = (1-as)*vf->Z.subvec(1,NX-2) + as*b.subvec(1,NX-2);
}


void PIC::smooth(vfield_mat * vf,double as){

}


void PIC::assingCell_TOS(const inputParameters * params,const meshGeometry * mesh, ionSpecies * ions, int dim){
	//This function assings the particles to the closest mesh node depending in their position and
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
			int NSP(ions->NSP);//number of superparticles
			int NC(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);//number of grid cells
			vec X;
			uvec LOGIC;

			#pragma omp parallel shared(mesh,ions,X,LOGIC) private(ii) firstprivate(NSP,NC)
			{
				#pragma omp for
				for(ii=0;ii<NSP;ii++)
					ions->meshNode(ii) = floor((ions->X(ii,0) + 0.5*mesh->DX)/mesh->DX);

				#pragma omp single
				{
					ions->wxc = zeros(NSP);
					ions->wxl = zeros(NSP);
					ions->wxr = zeros(NSP);
					ions->wxll = zeros(NSP);
					ions->wxrr = zeros(NSP);
					X = zeros(NSP);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(ions->meshNode(ii) != NC){
						X(ii) = ions->X(ii,0) - mesh->nodes.X(ions->meshNode(ii));
					}else{
						X(ii) =  ions->X(ii,0) - (mesh->nodes.X(NC-1) + mesh->DX);
					}
				}

				#pragma omp single
				{
					LOGIC = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
					X = abs(X);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					ions->wxc(ii) = CC1 - CC2*(X(ii)/mesh->DX)*(X(ii)/mesh->DX);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(LOGIC(ii) == 1){
						ions->wxl(ii) = CC3*((mesh->DX + X(ii))/mesh->DX - 1)*((mesh->DX + X(ii))/mesh->DX - CC4)*((mesh->DX + X(ii))/mesh->DX + CC5) + CC2;
						ions->wxr(ii) = CC3*((mesh->DX - X(ii))/mesh->DX - 1)*((mesh->DX - X(ii))/mesh->DX - CC4)*((mesh->DX - X(ii))/mesh->DX + CC5) + CC2;
						ions->wxll(ii) = CC6*(CC7 - (2*mesh->DX + X(ii))/mesh->DX)*((2 - (2*mesh->DX + X(ii))/mesh->DX)*(2 - (2*mesh->DX + X(ii))/mesh->DX) + CC8) - CC6;
						ions->wxrr(ii) = CC6*(CC7 - (2*mesh->DX - X(ii))/mesh->DX)*((2 - (2*mesh->DX - X(ii))/mesh->DX)*(2 - (2*mesh->DX - X(ii))/mesh->DX) + CC8) - CC6;
					}else{
						ions->wxl(ii) = CC3*((mesh->DX - X(ii))/mesh->DX - 1)*((mesh->DX - X(ii))/mesh->DX - CC4)*((mesh->DX - X(ii))/mesh->DX + CC5) + CC2;
						ions->wxr(ii) = CC3*((mesh->DX + X(ii))/mesh->DX - 1)*((mesh->DX + X(ii))/mesh->DX - CC4)*((mesh->DX + X(ii))/mesh->DX + CC5) + CC2;
						ions->wxll(ii) = CC6*(CC7 - (2*mesh->DX - X(ii))/mesh->DX)*((2 - (2*mesh->DX - X(ii))/mesh->DX)*(2 - (2*mesh->DX - X(ii))/mesh->DX) + CC8) - CC6;
						ions->wxrr(ii) = CC6*(CC7 - (2*mesh->DX + X(ii))/mesh->DX)*((2 - (2*mesh->DX + X(ii))/mesh->DX)*(2 - (2*mesh->DX + X(ii))/mesh->DX) + CC8) - CC6;
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
			std::ofstream ofs("errors/assingCell_TSC.txt", std::ofstream::out);
			ofs << "assingCell_TSC: Introduce a valid option!\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!ions->meshNode.is_finite()){
		std::ofstream ofs("errors/assingCell_TSC.txt", std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::assingCell_TSC(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions,int dim){
	//This function assings the particles to the closest mesh node depending in their position and
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
			int NSP(ions->NSP);//number of superparticles
			int NC;//number of grid cells
			vec X;
			uvec LOGIC;
			#pragma omp parallel shared(mesh,ions,X,NSP,LOGIC) private(ii,NC)
			{
				#pragma omp for
				for(ii=0;ii<NSP;ii++)
					//ions->meshNode(ii,0) = floor((ions->X(ii,0) + 0.5*mesh->DX)/mesh->DX);
					ions->meshNode(ii) = floor((ions->X(ii,0) + 0.5*mesh->DX)/mesh->DX);

				NC = mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS;

				#pragma omp single
				{
					ions->wxc = zeros(NSP);
					ions->wxl = zeros(NSP);
					ions->wxr = zeros(NSP);
					X = zeros(NSP);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					//if(ions->meshNode(ii,0) != NC){
					if(ions->meshNode(ii) != NC){
						//X(ii) = ions->X(ii,0) - mesh->nodes.X(ions->meshNode(ii,0));
						X(ii) = ions->X(ii,0) - mesh->nodes.X(ions->meshNode(ii));
					}else{
						X(ii) = ions->X(ii,0) - (mesh->nodes.X(NC-1) + mesh->DX);
					}
				}

				#pragma omp single
				{
					LOGIC = ( X > 0 );//If , aux > 0, then the particle is on the right of the meshnode
					X = abs(X);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					ions->wxc(ii) = 0.75 - (X(ii)/mesh->DX)*(X(ii)/mesh->DX);
				}

				#pragma omp for
				for(ii=0;ii<NSP;ii++){
					if(LOGIC(ii) == 1){
						ions->wxl(ii) = 0.5*(1.5 - (mesh->DX + X(ii))/mesh->DX)*(1.5 - (mesh->DX + X(ii))/mesh->DX);
						ions->wxr(ii) = 0.5*(1.5 - (mesh->DX - X(ii))/mesh->DX)*(1.5 - (mesh->DX - X(ii))/mesh->DX);
					}else{
						ions->wxl(ii) = 0.5*(1.5 - (mesh->DX - X(ii))/mesh->DX)*(1.5 - (mesh->DX - X(ii))/mesh->DX);
						ions->wxr(ii) = 0.5*(1.5 - (mesh->DX + X(ii))/mesh->DX)*(1.5 - (mesh->DX + X(ii))/mesh->DX);
					}
				}

			}//End of the parallel region
			break;
		}
		case(2):{
			ions->meshNode.col(0) = floor((ions->X.col(0) + 0.5*mesh->DX)/mesh->DX);
			ions->meshNode.col(1) = floor((ions->X.col(1) + 0.5*mesh->DX)/mesh->DX);
			break;
		}
		case(3):{
			ions->meshNode.col(0) = floor((ions->X.col(0) + 0.5*mesh->DX)/mesh->DX);
			ions->meshNode.col(1) = floor((ions->X.col(1) + 0.5*mesh->DX)/mesh->DX);
			ions->meshNode.col(2) = floor((ions->X.col(2) + 0.5*mesh->DX)/mesh->DX);
			break;
		}
		default:{
			std::ofstream ofs("errors/assingCell_TSC.txt",std::ofstream::out);
			ofs << "assingCell_TSC: Introduce a valid option!\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!ions->meshNode.is_finite()){
		std::ofstream ofs("errors/assingCell_TSC.txt",std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::assingCell(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions,int dim){

	switch (dim){
		case(1):{
			/*Periodic boundary condition*/
			double aux(0);
			for(int ii=0;ii<ions->NSP;ii++){
				aux = floor(ions->X(ii,0)/mesh->DX);
				if(aux == mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS){
					ions->meshNode(ii) = 0;
					ions->X(ii) = 0;
				}else{
					ions->meshNode(ii) = aux;
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
			std::ofstream ofs("errors/assingCell.txt", std::ofstream::out);
			ofs << "ERROR: Invalid option.\n";
			ofs.close();
			exit(1);
		}
	}

    #ifdef CHECKS_ON
	if(!ions->meshNode.is_finite()){
		std::ofstream ofs("errors/assingCell.txt", std::ofstream::out);
		ofs << "ERROR: Non-finite value for the particle's index.\n";
		ofs.close();
		exit(1);
	}
    #endif
}


void PIC::crossProduct(const mat * A,const mat * B,mat * AxB){
	if(A->n_elem != B->n_elem){
		cerr<<"\nERROR: The number of elements of A and B, unable to calculate AxB.\n";
		exit(1);
	}

	AxB->set_size(A->n_rows,3);//Here we set up the size of the matrix AxB.

	AxB->col(0) = A->col(1)%B->col(2) - A->col(2)%B->col(1);//(AxB)_x
	AxB->col(1) = A->col(2)%B->col(0) - A->col(0)%B->col(2);//(AxB)_y
	AxB->col(2) = A->col(0)%B->col(1) - A->col(1)%B->col(0);//(AxB)_z
}


#ifdef ONED
void PIC::eivTOS_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

	int NC(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);//Mesh size along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);
	vfield_vec nv;

	ions->nv.zeros(NC);//Setting to zero the ions' bulk velocity
	nv.zeros(NC);

	#pragma omp parallel shared(mesh,ions) firstprivate(NC,nv) private(ii)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = ions->meshNode(ii) + 1;
			if(ix == (NC-2)){//For the particles on the right side boundary.
				nv.X(NC-4) += ions->wxll(ii)*ions->V(ii,0);
				nv.X(NC-3) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(NC-2) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(NC-1) += ions->wxr(ii)*ions->V(ii,0);
				nv.X(0) += ions->wxrr(ii)*ions->V(ii,0);

				nv.Y(NC-4) += ions->wxll(ii)*ions->V(ii,1);
				nv.Y(NC-3) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(NC-2) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(NC-1) += ions->wxr(ii)*ions->V(ii,1);
				nv.Y(0) += ions->wxrr(ii)*ions->V(ii,1);

				nv.Z(NC-4) += ions->wxll(ii)*ions->V(ii,2);
				nv.Z(NC-3) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(NC-2) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(NC-1) += ions->wxr(ii)*ions->V(ii,2);
				nv.Z(0) += ions->wxrr(ii)*ions->V(ii,2);
			}else if(ix == (NC-1)){//For the particles on the right side boundary.
				nv.X(NC-3) += ions->wxll(ii)*ions->V(ii,0);
				nv.X(NC-2) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(NC-1) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(2) += ions->wxr(ii)*ions->V(ii,0);
				nv.X(3) += ions->wxrr(ii)*ions->V(ii,0);

				nv.Y(NC-3) += ions->wxll(ii)*ions->V(ii,1);
				nv.Y(NC-2) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(NC-1) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(2) += ions->wxr(ii)*ions->V(ii,1);
				nv.Y(3) += ions->wxrr(ii)*ions->V(ii,1);

				nv.Z(NC-3) += ions->wxll(ii)*ions->V(ii,2);
				nv.Z(NC-2) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(NC-1) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(2) += ions->wxr(ii)*ions->V(ii,2);
				nv.Z(3) += ions->wxrr(ii)*ions->V(ii,2);
			}else if(ix == 1){
				nv.X(NC-1) += ions->wxll(ii)*ions->V(ii,0);
				nv.X(0) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(ix) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(ix+1) += ions->wxr(ii)*ions->V(ii,0);
				nv.X(ix+2) += ions->wxrr(ii)*ions->V(ii,0);

				nv.Y(NC-1) += ions->wxll(ii)*ions->V(ii,1);
				nv.Y(0) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(ix) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(ix+1) += ions->wxr(ii)*ions->V(ii,1);
				nv.Y(ix+2) += ions->wxrr(ii)*ions->V(ii,1);

				nv.Z(NC-1) += ions->wxll(ii)*ions->V(ii,2);
				nv.Z(0) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(ix) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(ix+1) += ions->wxr(ii)*ions->V(ii,2);
				nv.Z(ix+2) += ions->wxrr(ii)*ions->V(ii,2);
			}else{
				nv.X(ix-2) += ions->wxll(ii)*ions->V(ii,0);
				nv.X(ix-1) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(ix) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(ix+1) += ions->wxr(ii)*ions->V(ii,0);
				nv.X(ix+2) += ions->wxrr(ii)*ions->V(ii,0);

				nv.Y(ix-2) += ions->wxll(ii)*ions->V(ii,1);
				nv.Y(ix-1) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(ix) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(ix+1) += ions->wxr(ii)*ions->V(ii,1);
				nv.Y(ix+2) += ions->wxrr(ii)*ions->V(ii,1);

				nv.Z(ix-2) += ions->wxll(ii)*ions->V(ii,2);
				nv.Z(ix-1) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(ix) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(ix+1) += ions->wxr(ii)*ions->V(ii,2);
				nv.Z(ix+2) += ions->wxrr(ii)*ions->V(ii,2);
			}
		}

		#pragma omp critical (update_bulk_velocity)
		{
		ions->nv += nv;
		}

	}//End of the parallel region

	backwardPBC_1D(&ions->nv.X);
	backwardPBC_1D(&ions->nv.Y);
	backwardPBC_1D(&ions->nv.Z);

	ions->nv *= ions->NCP/mesh->DX;

}
#endif


#ifdef ONED
void PIC::eivTSC_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NC(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);//Mesh size along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);
	vfield_vec nv;
	ions->nv.zeros(NC);//Setting to zero the ions' bulk velocity

	#pragma omp parallel shared(mesh,ions) private(ii,NC,nv)
	{
		NC = mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2;
		nv.zeros(NC);
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = ions->meshNode(ii) + 1;
			if(ix == (NC-1)){//For the particles on the right side boundary.
				nv.X(NC-2) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(NC-1) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(2) += ions->wxr(ii)*ions->V(ii,0);

				nv.Y(NC-2) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(NC-1) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(2) += ions->wxr(ii)*ions->V(ii,1);

				nv.Z(NC-2) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(NC-1) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(2) += ions->wxr(ii)*ions->V(ii,2);
			}else if(ix != (NC-1)){
				nv.X(ix-1) += ions->wxl(ii)*ions->V(ii,0);
				nv.X(ix) += ions->wxc(ii)*ions->V(ii,0);
				nv.X(ix+1) += ions->wxr(ii)*ions->V(ii,0);

				nv.Y(ix-1) += ions->wxl(ii)*ions->V(ii,1);
				nv.Y(ix) += ions->wxc(ii)*ions->V(ii,1);
				nv.Y(ix+1) += ions->wxr(ii)*ions->V(ii,1);

				nv.Z(ix-1) += ions->wxl(ii)*ions->V(ii,2);
				nv.Z(ix) += ions->wxc(ii)*ions->V(ii,2);
				nv.Z(ix+1) += ions->wxr(ii)*ions->V(ii,2);
			}
		}

		#pragma omp critical (update_bulk_velocity)
		{
		ions->nv += nv;
		}

	}//End of the parallel region

	backwardPBC_1D(&ions->nv.X);
	backwardPBC_1D(&ions->nv.Y);
	backwardPBC_1D(&ions->nv.Z);

	ions->nv *= ions->NCP/mesh->DX;

}
#endif


#ifdef TWOD
void PIC::eivTSC_2D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

}
#endif


#ifdef THREED
void PIC::eivTSC_3D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

}
#endif


void PIC::extrapolateIonVelocity(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

	ions->nv__ = ions->nv_;
	ions->nv_ = ions->nv;

	switch (params->weightingScheme){
		case(0):{
				#ifdef ONED
					eivTOS_1D(params,mesh,ions);
				#endif
				break;
				}
		case(1):{
				#ifdef ONED
					eivTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eivTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eivTSC_3D(params,mesh,ions);
				#endif
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				#ifdef ONED
					eivTOS_1D(params,mesh,ions);
				#endif
				break;
				}
		case(4):{
				#ifdef ONED
					eivTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eivTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eivTSC_3D(params,mesh,ions);
				#endif
				break;
				}
		default:{
				#ifdef ONED
					eivTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eivTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eivTSC_3D(params,mesh,ions);
				#endif
				}
	}
}


#ifdef ONED
void PIC::eidTOS_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){
	int NC(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);//Grid cells along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);
	vec n;

	ions->n = zeros(NC);//Setting to zero the ion density.
	n.zeros(NC);

	#pragma omp parallel shared(mesh,ions) private(ii) firstprivate(NSP,NC,n)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = ions->meshNode(ii) + 1;
			if(ix == (NC-2)){//For the particles on the right side boundary.
				n(NC-4) += ions->wxll(ii);
				n(NC-3) += ions->wxl(ii);
				n(NC-2) += ions->wxc(ii);
				n(NC-1) += ions->wxr(ii);
				n(0) += ions->wxrr(ii);
			}else if(ix == (NC-1)){//For the particles on the right side boundary.
				n(NC-3) += ions->wxll(ii);
				n(NC-2) += ions->wxl(ii);
				n(NC-1) += ions->wxc(ii);
				n(2) += ions->wxr(ii);
				n(3) += ions->wxrr(ii);
			}else if(ix == 1){
				n(NC-1) += ions->wxll(ii);
				n(0) += ions->wxl(ii);
				n(ix) += ions->wxc(ii);
				n(ix+1) += ions->wxr(ii);
				n(ix+2) += ions->wxrr(ii);
			}else{
				n(ix-2) += ions->wxll(ii);
				n(ix-1) += ions->wxl(ii);
				n(ix) += ions->wxc(ii);
				n(ix+1) += ions->wxr(ii);
				n(ix+2) += ions->wxrr(ii);
			}
		}

		#pragma omp critical (update_density)
		{
		ions->n += n;
		}

	}//End of the parallel region

	backwardPBC_1D(&ions->n);
	ions->n *= ions->NCP/mesh->DX;
}
#endif


#ifdef ONED
void PIC::eidTSC_1D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NC(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2);//Grid cells along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);
	vec n;

	ions->n = zeros(NC);//Setting to zero the ion density.
	n.zeros(NC);

	#pragma omp parallel shared(mesh,ions) private(ii) firstprivate(NSP,NC,n)
	{
		#pragma omp for
		for(ii=0;ii<NSP;ii++){
			int ix = ions->meshNode(ii) + 1;

			if(ix == (NC-1)){//For the particles on the right side boundary.
				n(NC-2) += ions->wxl(ii);
				n(NC-1) += ions->wxc(ii);
				n(2) += ions->wxr(ii);
			}else if(ix != (NC-1)){
				n(ix-1) += ions->wxl(ii);
				n(ix) += ions->wxc(ii);
				n(ix+1) += ions->wxr(ii);
			}
		}

		#pragma omp critical (update_density)
		{
		ions->n += n;
		}

	}//End of the parallel region

	backwardPBC_1D(&ions->n);
	ions->n *= ions->NCP/mesh->DX;

}
#endif


#ifdef TWOD
void PIC::eidTSC_2D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

}
#endif


#ifdef THREED
void PIC::eidTSC_3D(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

}
#endif


void PIC::extrapolateIonDensity(const inputParameters * params,const meshGeometry * mesh,ionSpecies * ions){

	ions->n___ = ions->n__;
	ions->n__ = ions->n_;
	ions->n_ = ions->n;

	switch (params->weightingScheme){
		case(0):{
				#ifdef ONED
					eidTOS_1D(params,mesh,ions);
				#endif
				break;
				}
		case(1):{
				#ifdef ONED
					eidTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eidTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eidTSC_3D(params,mesh,ions);
				#endif
				break;
				}
		case(2):{
				exit(0);
				break;
				}
		case(3):{
				#ifdef ONED
					eidTOS_1D(params,mesh,ions);
				#endif
				break;
				}
		case(4):{
				#ifdef ONED
					eidTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eidTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eidTSC_3D(params,mesh,ions);
				#endif
				break;
				}
		default:{
				#ifdef ONED
					eidTSC_1D(params,mesh,ions);
				#endif

				#ifdef TWOD
					eidTSC_2D(params,mesh,ions);
				#endif

				#ifdef THREED
					eidTSC_3D(params,mesh,ions);
				#endif
				}
	}
}


#ifdef ONED
void PIC::EMF_TOS_1D(const inputParameters * params,const ionSpecies * ions,vfield_vec * EMF,mat * F){

	//wxc = 23/48 - (x/H)^2/4
	//wxr = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxrr = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12
	//wxl = (abs(x)/H - 1)*(abs(x)/H - 5/2)*(abs(x)/H + 1/2)/6 + 1/4
	//wxll = [7/2 - abs(x)/H]*[(2 - abs(x)/H)^2 + 3/4]/12 -1/12

	int N =  params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2;//Mesh size along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);

	//Contrary to what may be thought, F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.

	#pragma omp parallel for private(ii) shared(ions,EMF,F) firstprivate(N,NSP)
	for(ii=0;ii<NSP;ii++){
		int ix = ions->meshNode(ii) + 1;
		if(ix == (N-2)){//For the particles on the right side boundary.
			(*F)(ii,0) += ions->wxll(ii)*EMF->X(N-4);
			(*F)(ii,1) += ions->wxll(ii)*EMF->Y(N-4);
			(*F)(ii,2) += ions->wxll(ii)*EMF->Z(N-4);

			(*F)(ii,0) += ions->wxl(ii)*EMF->X(N-3);
			(*F)(ii,1) += ions->wxl(ii)*EMF->Y(N-3);
			(*F)(ii,2) += ions->wxl(ii)*EMF->Z(N-3);

			(*F)(ii,0) += ions->wxc(ii)*EMF->X(N-2);
			(*F)(ii,1) += ions->wxc(ii)*EMF->Y(N-2);
			(*F)(ii,2) += ions->wxc(ii)*EMF->Z(N-2);

			(*F)(ii,0) += ions->wxr(ii)*EMF->X(N-1);
			(*F)(ii,1) += ions->wxr(ii)*EMF->Y(N-1);
			(*F)(ii,2) += ions->wxr(ii)*EMF->Z(N-1);

			(*F)(ii,0) += ions->wxrr(ii)*EMF->X(0);
			(*F)(ii,1) += ions->wxrr(ii)*EMF->Y(0);
			(*F)(ii,2) += ions->wxrr(ii)*EMF->Z(0);
		}else if(ix == (N-1)){//For the particles on the right side boundary.
			(*F)(ii,0) += ions->wxll(ii)*EMF->X(N-3);
			(*F)(ii,1) += ions->wxll(ii)*EMF->Y(N-3);
			(*F)(ii,2) += ions->wxll(ii)*EMF->Z(N-3);

			(*F)(ii,0) += ions->wxl(ii)*EMF->X(N-2);
			(*F)(ii,1) += ions->wxl(ii)*EMF->Y(N-2);
			(*F)(ii,2) += ions->wxl(ii)*EMF->Z(N-2);

			(*F)(ii,0) += ions->wxc(ii)*EMF->X(N-1);
			(*F)(ii,1) += ions->wxc(ii)*EMF->Y(N-1);
			(*F)(ii,2) += ions->wxc(ii)*EMF->Z(N-1);

			(*F)(ii,0) += ions->wxr(ii)*EMF->X(2);
			(*F)(ii,1) += ions->wxr(ii)*EMF->Y(2);
			(*F)(ii,2) += ions->wxr(ii)*EMF->Z(2);

			(*F)(ii,0) += ions->wxrr(ii)*EMF->X(3);
			(*F)(ii,1) += ions->wxrr(ii)*EMF->Y(3);
			(*F)(ii,2) += ions->wxrr(ii)*EMF->Z(3);
		}else if(ix == 1){
			(*F)(ii,0) += ions->wxll(ii)*EMF->X(N-1);
			(*F)(ii,1) += ions->wxll(ii)*EMF->Y(N-1);
			(*F)(ii,2) += ions->wxll(ii)*EMF->Z(N-1);

			(*F)(ii,0) += ions->wxl(ii)*EMF->X(0);
			(*F)(ii,1) += ions->wxl(ii)*EMF->Y(0);
			(*F)(ii,2) += ions->wxl(ii)*EMF->Z(0);

			(*F)(ii,0) += ions->wxc(ii)*EMF->X(ix);
			(*F)(ii,1) += ions->wxc(ii)*EMF->Y(ix);
			(*F)(ii,2) += ions->wxc(ii)*EMF->Z(ix);

			(*F)(ii,0) += ions->wxr(ii)*EMF->X(ix+1);
			(*F)(ii,1) += ions->wxr(ii)*EMF->Y(ix+1);
			(*F)(ii,2) += ions->wxr(ii)*EMF->Z(ix+1);

			(*F)(ii,0) += ions->wxrr(ii)*EMF->X(ix+2);
			(*F)(ii,1) += ions->wxrr(ii)*EMF->Y(ix+2);
			(*F)(ii,2) += ions->wxrr(ii)*EMF->Z(ix+2);
		}else{
			(*F)(ii,0) += ions->wxll(ii)*EMF->X(ix-2);
			(*F)(ii,1) += ions->wxll(ii)*EMF->Y(ix-2);
			(*F)(ii,2) += ions->wxll(ii)*EMF->Z(ix-2);

			(*F)(ii,0) += ions->wxl(ii)*EMF->X(ix-1);
			(*F)(ii,1) += ions->wxl(ii)*EMF->Y(ix-1);
			(*F)(ii,2) += ions->wxl(ii)*EMF->Z(ix-1);

			(*F)(ii,0) += ions->wxc(ii)*EMF->X(ix);
			(*F)(ii,1) += ions->wxc(ii)*EMF->Y(ix);
			(*F)(ii,2) += ions->wxc(ii)*EMF->Z(ix);

			(*F)(ii,0) += ions->wxr(ii)*EMF->X(ix+1);
			(*F)(ii,1) += ions->wxr(ii)*EMF->Y(ix+1);
			(*F)(ii,2) += ions->wxr(ii)*EMF->Z(ix+1);

			(*F)(ii,0) += ions->wxrr(ii)*EMF->X(ix+2);
			(*F)(ii,1) += ions->wxrr(ii)*EMF->Y(ix+2);
			(*F)(ii,2) += ions->wxrr(ii)*EMF->Z(ix+2);
		}
	}//End of the parallel region
}
#endif


#ifdef ONED
void PIC::EMF_TSC_1D(const inputParameters * params,const ionSpecies * ions,vfield_vec * emf,mat * F){

	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int N =  params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS + 2;//Mesh size along the X axis (considering the gosht cell)
	int NSP(ions->NSP);
	int ii(0);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.

	#pragma omp parallel for private(ii) shared(N,NSP,params,ions,emf,F)
	for(ii=0;ii<NSP;ii++){
		int ix = ions->meshNode(ii) + 1;
		if(ix == (N-1)){//For the particles on the right side boundary.
			(*F)(ii,0) += ions->wxl(ii)*emf->X(N-2);
			(*F)(ii,1) += ions->wxl(ii)*emf->Y(N-2);
			(*F)(ii,2) += ions->wxl(ii)*emf->Z(N-2);

			(*F)(ii,0) += ions->wxc(ii)*emf->X(N-1);
			(*F)(ii,1) += ions->wxc(ii)*emf->Y(N-1);
			(*F)(ii,2) += ions->wxc(ii)*emf->Z(N-1);

			(*F)(ii,0) += ions->wxr(ii)*emf->X(2);
			(*F)(ii,1) += ions->wxr(ii)*emf->Y(2);
			(*F)(ii,2) += ions->wxr(ii)*emf->Z(2);
		}else if(ix != (N-1)){
			(*F)(ii,0) += ions->wxl(ii)*emf->X(ix-1);
			(*F)(ii,1) += ions->wxl(ii)*emf->Y(ix-1);
			(*F)(ii,2) += ions->wxl(ii)*emf->Z(ix-1);

			(*F)(ii,0) += ions->wxc(ii)*emf->X(ix);
			(*F)(ii,1) += ions->wxc(ii)*emf->Y(ix);
			(*F)(ii,2) += ions->wxc(ii)*emf->Z(ix);

			(*F)(ii,0) += ions->wxr(ii)*emf->X(ix+1);
			(*F)(ii,1) += ions->wxr(ii)*emf->Y(ix+1);
			(*F)(ii,2) += ions->wxr(ii)*emf->Z(ix+1);
		}
	}//End of the parallel region

}
#endif


#ifdef TWOD
void PIC::EMF_TSC_2D(const meshGeometry * mesh,const ionSpecies * ions,vfield_cube * emf,mat * F){

}
#endif


#ifdef THREED
void PIC::EMF_TSC_3D(const meshGeometry * mesh,const ionSpecies * ions,vfield_cube * emf,mat * F){

	for(int ii=0;ii<ions->NSP;ii++){//Iterating over the particles

		double wx(0), wy(0), wz(0);
		double w1(0), w2(0), w3(0), w4(0), w5(0), w6(0), w7(0), w8(0);
		int ix(ions->meshNode(ii,0)+1),iy(ions->meshNode(ii,1)+1),iz(ions->meshNode(ii,2)+1);

		wx = 1 - (ions->X(ii,0) - mesh->nodes.X(ions->meshNode(ii,0)))/mesh->DX;//
		wy = 1 - (ions->X(ii,1) - mesh->nodes.Y(ions->meshNode(ii,1)))/mesh->DY;//
		wz = 1 - (ions->X(ii,2) - mesh->nodes.Z(ions->meshNode(ii,2)))/mesh->DZ;//

		wx = (wx<0) ? 0 : wx;
		wy = (wy<0) ? 0 : wy;
		wz = (wz<0) ? 0 : wz;

		w1 = wx*wy*wz;
		(*F)(ii,0) += w1*emf->X(ix,iy,iz);//(i,j,k)
		(*F)(ii,1) += w1*emf->Y(ix,iy,iz);//(i,j,k)
		(*F)(ii,2) += w1*emf->Z(ix,iy,iz);//(i,j,k)
		w2 = (1-wx)*wy*wz;
		(*F)(ii,0) += w2*emf->X(ix+1,iy,iz);//(i+1,j,k)
		(*F)(ii,1) += w2*emf->Y(ix+1,iy,iz);//(i+1,j,k)
		(*F)(ii,2) += w2*emf->Z(ix+1,iy,iz);//(i+1,j,k)
		w3 = wx*(1-wy)*wz;
		(*F)(ii,0) += w3*emf->X(ix,iy+1,iz);//(i,j+1,k)
		(*F)(ii,1) += w3*emf->Y(ix,iy+1,iz);//(i,j+1,k)
		(*F)(ii,2) += w3*emf->Z(ix,iy+1,iz);//(i,j+1,k)
		w4 = (1-wx)*(1-wy)*wz;
		(*F)(ii,0) += w4*emf->X(ix+1,iy+1,iz);//(i+1,j+1,k)
		(*F)(ii,1) += w4*emf->Y(ix+1,iy+1,iz);//(i+1,j+1,k)
		(*F)(ii,2) += w4*emf->Z(ix+1,iy+1,iz);//(i+1,j+1,k)
		w5 = wx*wy*(1-wz);
		(*F)(ii,0) += w5*emf->X(ix,iy,iz+1);//(i,j,k+1)
		(*F)(ii,1) += w5*emf->Y(ix,iy,iz+1);//(i,j,k+1)
		(*F)(ii,2) += w5*emf->Z(ix,iy,iz+1);//(i,j,k+1)
		w6 = (1-wx)*wy*(1-wz);
		(*F)(ii,0) += w6*emf->X(ix+1,iy,iz+1);//(i+1,j,k+1)
		(*F)(ii,1) += w6*emf->Y(ix+1,iy,iz+1);//(i+1,j,k+1)
		(*F)(ii,2) += w6*emf->Z(ix+1,iy,iz+1);//(i+1,j,k+1)
		w7 = wx*(1-wy)*(1-wz);
		(*F)(ii,0) += w7*emf->X(ix,iy+1,iz+1);//(i,j+1,k+1)
		(*F)(ii,1) += w7*emf->Y(ix,iy+1,iz+1);//(i,j+1,k+1)
		(*F)(ii,2) += w7*emf->Z(ix,iy+1,iz+1);//(i,j+1,k+1)
		w8 = (1-wx)*(1-wy)*(1-wz);
		(*F)(ii,0) += w8*emf->X(ix+1,iy+1,iz+1);//(i+1,j+1,k+1)
		(*F)(ii,1) += w8*emf->Y(ix+1,iy+1,iz+1);//(i+1,j+1,k+1)
		(*F)(ii,2) += w8*emf->Z(ix+1,iy+1,iz+1);//(i+1,j+1,k+1)

	}//Iterating over the particles
}
#endif


#ifdef ONED
void PIC::aiv_GC_1D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){


	MPI_AllgatherField(params,&EB->E);
	MPI_AllgatherField(params,&EB->B);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	emf emf_nodes;
	emf_nodes.zeros(NX);

	emf_nodes.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	emf_nodes.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	emf_nodes.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	emf_nodes.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	emf_nodes.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	emf_nodes.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	forwardPBC_1D(&emf_nodes.E.X);
	forwardPBC_1D(&emf_nodes.E.Y);
	forwardPBC_1D(&emf_nodes.E.Z);

	forwardPBC_1D(&emf_nodes.B.X);
	forwardPBC_1D(&emf_nodes.B.Y);
	forwardPBC_1D(&emf_nodes.B.Z);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		mat Ep = zeros(IONS->at(ii).NSP,3);
		mat Bp = zeros(IONS->at(ii).NSP,3);

		switch (params->weightingScheme){
			case(0):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(1):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(2):{
					exit(0);
					break;
					}
			case(3):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(4):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			default:{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					}
		}

		//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		int NSP(IONS->at(ii).NSP);
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)
		vec gp(IONS->at(ii).NSP);
		vec sigma(IONS->at(ii).NSP);
		vec us(IONS->at(ii).NSP);
		vec s(IONS->at(ii).NSP);
		mat U(IONS->at(ii).NSP,3);
		mat VxB(IONS->at(ii).NSP,3);
		mat tau(IONS->at(ii).NSP,3);
		mat up(IONS->at(ii).NSP,3);
		mat t(IONS->at(ii).NSP,3);
		mat upxt(IONS->at(ii).NSP,3);

		crossProduct(&IONS->at(ii).V,&Bp,&VxB);//VxB

		#pragma omp parallel shared(IONS,U,gp,sigma,us,s,VxB,tau,up,t,upxt) firstprivate(A,NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip),IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
				U.row(ip) = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

				U.row(ip) += 0.5*A*(Ep.row(ip) + VxB.row(ip)); // U_hs = U_L + 0.5*a*(E + cross(V,B)); % Half step for velocity
				tau.row(ip) = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
				up.row(ip) = U.row(ip) + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
				gp(ip) = sqrt( 1.0 + dot(up.row(ip),up.row(ip))/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
				sigma(ip) = gp(ip)*gp(ip) - dot(tau.row(ip),tau.row(ip)); // sigma = gammap^2 - tau*tau';
				us(ip) = dot(up.row(ip),tau.row(ip))/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
				IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma(ip) + sqrt( sigma(ip)*sigma(ip) + 4.0*( dot(tau.row(ip),tau.row(ip)) + us(ip)*us(ip) ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
				t.row(ip) = tau.row(ip)/IONS->at(ii).g(ip); 			// t = tau/gamma;
				s(ip) = 1.0/( 1.0 + dot(t.row(ip),t.row(ip)) ); // s = 1/(1 + t*t'); % variable 's' in paper
			}

			#pragma omp critical
			crossProduct(&up,&t,&upxt);

			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				U.row(ip) = s(ip)*( up.row(ip) + dot(up.row(ip),t.row(ip))*t.row(ip)+ upxt.row(ip) ); 	// U_L = s*(up + (up*t')*t + cross(up,t));
				IONS->at(ii).V.row(ip) = U.row(ip)/IONS->at(ii).g(ip);	// V = U_L/gamma;
			}
		} // End of parallel region

		extrapolateIonVelocity(params,mesh,&IONS->at(ii));

		MPI_BcastBulkVelocity(params,&IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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
						smooth(&IONS->at(ii).nv,params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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


#ifdef ONED
void PIC::aiv_Vay_1D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){


	MPI_AllgatherField(params,&EB->E);
	MPI_AllgatherField(params,&EB->B);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	emf emf_nodes;
	emf_nodes.zeros(NX);

	emf_nodes.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	emf_nodes.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	emf_nodes.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	emf_nodes.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	emf_nodes.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	emf_nodes.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	forwardPBC_1D(&emf_nodes.E.X);
	forwardPBC_1D(&emf_nodes.E.Y);
	forwardPBC_1D(&emf_nodes.E.Z);

	forwardPBC_1D(&emf_nodes.B.X);
	forwardPBC_1D(&emf_nodes.B.Y);
	forwardPBC_1D(&emf_nodes.B.Z);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		mat Ep = zeros(IONS->at(ii).NSP,3);
		mat Bp = zeros(IONS->at(ii).NSP,3);

		switch (params->weightingScheme){
			case(0):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(1):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(2):{
					exit(0);
					break;
					}
			case(3):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(4):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			default:{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					}
		}

		//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		int NSP(IONS->at(ii).NSP);
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)
		vec gp(IONS->at(ii).NSP);
		vec sigma(IONS->at(ii).NSP);
		vec us(IONS->at(ii).NSP);
		vec s(IONS->at(ii).NSP);
		mat U(IONS->at(ii).NSP,3);
		mat VxB(IONS->at(ii).NSP,3);
		mat tau(IONS->at(ii).NSP,3);
		mat up(IONS->at(ii).NSP,3);
		mat t(IONS->at(ii).NSP,3);
		mat upxt(IONS->at(ii).NSP,3);

		crossProduct(&IONS->at(ii).V,&Bp,&VxB);//VxB

		#pragma omp parallel shared(IONS,U,gp,sigma,us,s,VxB,tau,up,t,upxt) firstprivate(A,NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip),IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
				U.row(ip) = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

				U.row(ip) += 0.5*A*(Ep.row(ip) + VxB.row(ip)); // U_hs = U_L + 0.5*a*(E + cross(V,B)); % Half step for velocity
				tau.row(ip) = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
				up.row(ip) = U.row(ip) + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
				gp(ip) = sqrt( 1.0 + dot(up.row(ip),up.row(ip))/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
				sigma(ip) = gp(ip)*gp(ip) - dot(tau.row(ip),tau.row(ip)); // sigma = gammap^2 - tau*tau';
				us(ip) = dot(up.row(ip),tau.row(ip))/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
				IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma(ip) + sqrt( sigma(ip)*sigma(ip) + 4.0*( dot(tau.row(ip),tau.row(ip)) + us(ip)*us(ip) ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
				t.row(ip) = tau.row(ip)/IONS->at(ii).g(ip); 			// t = tau/gamma;
				s(ip) = 1.0/( 1.0 + dot(t.row(ip),t.row(ip)) ); // s = 1/(1 + t*t'); % variable 's' in paper
			}

			#pragma omp critical
			crossProduct(&up,&t,&upxt);

			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				U.row(ip) = s(ip)*( up.row(ip) + dot(up.row(ip),t.row(ip))*t.row(ip)+ upxt.row(ip) ); 	// U_L = s*(up + (up*t')*t + cross(up,t));
				IONS->at(ii).V.row(ip) = U.row(ip)/IONS->at(ii).g(ip);	// V = U_L/gamma;
			}
		} // End of parallel region

/**
		IONS->g = 1.0/sqrt( 1.0 -  sum(IONS->at(ii).V % IONS->at(ii).V,1)/(F_C_DS*F_C_DS) );

		U = IONS->at(ii).V;
		U.each_col() %= g;

		crossProduct(&IONS->at(ii).V,&Bp,&VxB);//V\times B

		U += 0.5*A*(Ep + VxB); // U_hs = U_L + 0.5*a*(E + cross(V,B)); % Half step for velocity
		tau = 0.5*A*Bp; // tau = 0.5*q*dt*B/m;
		up = U + 0.5*A*Ep; // up = U_hs + 0.5*a*E;
		gp = sqrt(1.0 + sum(up % up,1)); // gammap = sqrt(1 + up*up');
		sigma = gp % gp - sum(tau % tau,1); // sigma = gammap^2 - tau*tau';
		us = sum(up % tau, 1); // us = up*tau'; % variable 'u^*' in paper
		g = sqrt(0.5)*sqrt( sigma + sqrt( sigma % sigma + 4.0*(sum(tau % tau,1) + us % us) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
		t = tau; 			// t = tau/gamma;
		t.each_col() /= g; 	// t = tau/gamma;
		s = 1.0/( 1.0 + sum(t % t,1) ); // s = 1/(1 + t*t'); % variable 's' in paper

		crossProduct(&up,&t,&upxt);

		U.col(0) = s % (up.col(0) + sum(up % t,1) % t.col(0) + upxt.col(0)); 	// U_L = s*(up + (up*t')*t + cross(up,t));
		U.col(1) = s % (up.col(1) + sum(up % t,1) % t.col(1) + upxt.col(1)); 	// U_L = s*(up + (up*t')*t + cross(up,t));
		U.col(2) = s % (up.col(2) + sum(up % t,1) % t.col(2) + upxt.col(2)); 	// U_L = s*(up + (up*t')*t + cross(up,t));

		//IONS->at(ii).g = g;
		IONS->at(ii).V = U;				// V = U_L/gamma;
		IONS->at(ii).V.each_col() /= g; // V = U_L/gamma;
**/

		extrapolateIonVelocity(params,mesh,&IONS->at(ii));

		MPI_BcastBulkVelocity(params,&IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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
						smooth(&IONS->at(ii).nv,params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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
void PIC::aiv_Vay_2D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

}
#endif


#ifdef THREED
void PIC::aiv_Vay_3D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

}
#endif


#ifdef ONED
void PIC::aiv_Boris_1D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){


	MPI_AllgatherField(params,&EB->E);
	MPI_AllgatherField(params,&EB->B);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_elem);

	emf emf_nodes;
	emf_nodes.zeros(NX);

	emf_nodes.E.X.subvec(1,NX-2) = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	emf_nodes.E.Y.subvec(1,NX-2) = EB->E.Y.subvec(1,NX-2);
	emf_nodes.E.Z.subvec(1,NX-2) = EB->E.Z.subvec(1,NX-2);

	emf_nodes.B.X.subvec(1,NX-2) = EB->B.X.subvec(1,NX-2);
	emf_nodes.B.Y.subvec(1,NX-2) = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) );
	emf_nodes.B.Z.subvec(1,NX-2) = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) );

	forwardPBC_1D(&emf_nodes.E.X);
	forwardPBC_1D(&emf_nodes.E.Y);
	forwardPBC_1D(&emf_nodes.E.Z);

	forwardPBC_1D(&emf_nodes.B.X);
	forwardPBC_1D(&emf_nodes.B.Y);
	forwardPBC_1D(&emf_nodes.B.Z);

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		mat Ep = zeros(IONS->at(ii).NSP,3);
		mat Bp = zeros(IONS->at(ii).NSP,3);

		switch (params->weightingScheme){
			case(0):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(1):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(2):{
					exit(0);
					break;
					}
			case(3):{
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TOS_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			case(4):{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					break;
					}
			default:{
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.E,&Ep);
					EMF_TSC_1D(params,&IONS->at(ii),&emf_nodes.B,&Bp);
					}
		}

		//Once the electric and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' velocity. (Q*NCP/M*NCP=Q/M)
		vec C1, C2, C3, C4, BB, VB, EB;
		mat ExB, VxB;

		#pragma omp parallel sections shared(IONS,BB,VB,EB,ExB,VxB,Ep,Bp)
		{
			#pragma omp section
			BB = sum(Bp % Bp,1);//B\dotB evaluated at each particle position.
			#pragma omp section
			VB = sum(IONS->at(ii).V % Bp,1);//V\dotB
			#pragma omp section
			EB = sum(Ep % Bp,1);//E\dotB
			#pragma omp section
			crossProduct(&Ep,&Bp,&ExB);//E\times B
			#pragma omp section
			crossProduct(&IONS->at(ii).V,&Bp,&VxB);//V\times B
		}//end of the parallel region

		C1 = ( 1.0 - (A*A)*BB/4.0 )/( 1.0 + (A*A)*BB/4.0 );
		C2 = A/( 1.0 + (A*A)*BB/4.0 );
		C3 = ((A*A)/2.0)/( 1.0 + (A*A)*BB/4.0 );
		C4 = ((A*A)/4.0)/( 1.0 + (A*A)*BB/4.0 );

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(IONS,BB,VB,EB,ExB,VxB,Ep,Bp,C1,C2,C3,C4) firstprivate(NSP)
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
		}//End of the parallel region

		extrapolateIonVelocity(params,mesh,&IONS->at(ii));

		MPI_BcastBulkVelocity(params,&IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).nv, params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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
						smooth(&IONS->at(ii).nv,params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).nv,params->smoothingParameter);
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
void PIC::aiv_Boris_2D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

}
#endif


#ifdef THREED
void PIC::aiv_Boris_3D(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	forwardPBC_3D(&EB->E.X);
	forwardPBC_3D(&EB->E.Y);
	forwardPBC_3D(&EB->E.Z);

	forwardPBC_3D(&EB->B.X);
	forwardPBC_3D(&EB->B.Y);
	forwardPBC_3D(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	int NX(EB->E.X.n_rows),NY(EB->E.Y.n_cols),NZ(EB->E.Z.n_slices);

	emf emf_nodes;
	emf_nodes.zeros(NX,NY,NZ);

	emf_nodes.E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.5*( EB->E.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->E.X.subcube(0,1,1,NX-3,NY-2,NZ-2) );
	emf_nodes.E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.5*( EB->E.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->E.Y.subcube(1,0,1,NX-2,NY-3,NZ-2) );
	emf_nodes.E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.5*( EB->E.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->E.Z.subcube(1,1,0,NX-2,NY-2,NZ-3) );

	emf_nodes.B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.25*( EB->B.X.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.X.subcube(1,0,1,NX-2,NY-3,NZ-2) ) + 0.25*( EB->B.X.subcube(1,1,0,NX-2,NY-2,NZ-3) + EB->B.X.subcube(1,0,0,NX-2,NY-3,NZ-3) );

	emf_nodes.B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.25*( EB->B.Y.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Y.subcube(0,1,1,NX-3,NY-2,NZ-2) ) + 0.25*( EB->B.Y.subcube(1,1,0,NX-2,NY-2,NZ-3) + EB->B.Y.subcube(0,1,0,NX-3,NY-2,NZ-3) );

	emf_nodes.B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) = 0.25*( EB->B.Z.subcube(1,1,1,NX-2,NY-2,NZ-2) + EB->B.Z.subcube(1,0,1,NX-2,NY-3,NZ-2) ) + 0.25*( EB->B.Z.subcube(0,1,1,NX-3,NY-2,NZ-2) + EB->B.Z.subcube(0,0,1,NX-3,NY-3,NZ-2) );

	forwardPBC_3D(&emf_nodes.E.X);
	forwardPBC_3D(&emf_nodes.E.Y);
	forwardPBC_3D(&emf_nodes.E.Z);

	forwardPBC_3D(&emf_nodes.B.X);
	forwardPBC_3D(&emf_nodes.B.Y);
	forwardPBC_3D(&emf_nodes.B.Z);

	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.
	restoreCube(&EB->E.X);
	restoreCube(&EB->E.Y);
	restoreCube(&EB->E.Z);

	restoreCube(&EB->B.X);
	restoreCube(&EB->B.Y);
	restoreCube(&EB->B.Z);
	//The electric and magntic fields in EB are defined in their staggered positions, not in the vertex nodes.

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.

		mat Ep = zeros(IONS->at(ii).NSP,3);
		mat Bp = zeros(IONS->at(ii).NSP,3);

		EMF_TSC_3D(mesh,&IONS->at(ii),&emf_nodes.E,&Ep);
		EMF_TSC_3D(mesh,&IONS->at(ii),&emf_nodes.B,&Bp);


		//Once the electrostatic and magnetic fields have been interpolated to the ions' positions we advance the ions' velocities.
		double A(IONS->at(ii).Q*DT/IONS->at(ii).M);//A = \alpha in the dimensionless equation for the ions' V. (Q*NCP/M*NCP=Q/M)
		vec C1, C2, C3, C4, BB, VB, EB;
		mat ExB, VxB;

		BB = sum(Bp % Bp,1);//B\dotB evaluated at each particle position.
		VB = sum(IONS->at(ii).V % Bp,1);//V\dotB
		EB = sum(Ep % Bp,1);//E\dotB
		crossProduct(&Ep,&Bp,&ExB);//E\times B
		crossProduct(&IONS->at(ii).V,&Bp,&VxB);//V\times B

		C1 = ( 1 - pow(A,2)*BB/4 )/( 1 + pow(A,2)*BB/4 );
		C2 = A/( 1 + pow(A,2)*BB/4 );
		C3 = (pow(A,2)/2)/( 1 + pow(A,2)*BB/4 );
		C4 = (pow(A,3)/4)/( 1 + pow(A,2)*BB/4 );

		IONS->at(ii).V.col(0) = C1 % IONS->at(ii).V.col(0);
		IONS->at(ii).V.col(0) += C2 % ( Ep.col(0) + VxB.col(0) );
		IONS->at(ii).V.col(0) += C3 % ( ExB.col(0) + VB % Bp.col(0) );
		IONS->at(ii).V.col(0) += C4 % ( EB % Bp.col(0) );

		IONS->at(ii).V.col(1) = C1 % IONS->at(ii).V.col(1);
		IONS->at(ii).V.col(1) += C2 % ( Ep.col(1) + VxB.col(1) );
		IONS->at(ii).V.col(1) += C3 % ( ExB.col(1) + VB % Bp.col(1) );
		IONS->at(ii).V.col(1) += C4 % ( EB % Bp.col(1) );

		IONS->at(ii).V.col(2) = C1 % IONS->at(ii).V.col(2);
		IONS->at(ii).V.col(2) += C2 % ( Ep.col(2) + VxB.col(2) );
		IONS->at(ii).V.col(2) += C3 % ( ExB.col(2) + VB % Bp.col(2) );
		IONS->at(ii).V.col(2) += C4 % ( EB % Bp.col(2) );

		extrapolateIonVelocity(params,mesh,&IONS->at(ii));

	}//structure to iterate over all the ion species.
}
#endif


void PIC::aip_1D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT){

	double lx = mesh->DX*mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS;//

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		int NSP(IONS->at(ii).NSP);
		#pragma omp parallel shared(IONS) firstprivate(DT,lx,NSP)
		{
			#pragma omp for
			for(int ip=0;ip<NSP;ip++){
				IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);

                IONS->at(ii).X(ip,0) = fmod(IONS->at(ii).X(ip,0),lx);//x

                if(IONS->at(ii).X(ip,0) < 0)
        			IONS->at(ii).X(ip,0) += lx;
			}
		}//End of the parallel region

		switch (params->weightingScheme){
			case(0):{
					assingCell_TOS(params,mesh,&IONS->at(ii),1);
					break;
					}
			case(1):{
					assingCell_TSC(params,mesh,&IONS->at(ii),1);
					break;
					}
			case(2):{
					assingCell(params,mesh,&IONS->at(ii),1);
					break;
					}
			case(3):{
					assingCell_TOS(params,mesh,&IONS->at(ii),1);
					break;
					}
			case(4):{
					assingCell_TSC(params,mesh,&IONS->at(ii),1);
					break;
					}
			default:{
					assingCell_TSC(params,mesh,&IONS->at(ii),1);
					}
		}

		extrapolateIonDensity(params,mesh,&IONS->at(ii));//Once the ions have been pushed, we extrapolate the density at the node grids.

		MPI_BcastDensity(params,&IONS->at(ii));

		switch (params->weightingScheme){
			case(0):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TOS(&IONS->at(ii).n,params->smoothingParameter);
					break;
					}
			case(1):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth_TSC(&IONS->at(ii).n,params->smoothingParameter);
					break;
					}
			case(2):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n,params->smoothingParameter);
					break;
					}
			case(3):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n,params->smoothingParameter);
					break;
					}
			case(4):{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n,params->smoothingParameter);
					break;
					}
			default:{
					for(int jj=0;jj<params->filtersPerIterationIons;jj++)
						smooth(&IONS->at(ii).n,params->smoothingParameter);
					}
		}

	}//structure to iterate over all the ion species.
}


void PIC::aip_2D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT){

	double lx,ly;
	lx = mesh->nodes.X(mesh->dim(0)-1) + mesh->DX;//
	ly = mesh->nodes.Y(mesh->dim(1)-1) + mesh->DY;//

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)


		IONS->at(ii).X.col(0) += DT*IONS->at(ii).V.col(0);//x-component
		IONS->at(ii).X.col(1) += DT*IONS->at(ii).V.col(1);//y-component

		for(int jj=0;jj<IONS->at(ii).NSP;jj++){//Periodic boundary condition for the ions
			IONS->at(ii).X(jj,0) = fmod(IONS->at(ii).X(jj,0),lx);//x
			if(IONS->at(ii).X(jj,0) < 0)
				IONS->at(ii).X(jj,0) += lx;

			IONS->at(ii).X(jj,1) = fmod(IONS->at(ii).X(jj,1),ly);//y
			if(IONS->at(ii).X(jj,1) < 0)
				IONS->at(ii).X(jj,1) += ly;
		}//Periodic boundary condition for the ions

		assingCell_TSC(params,mesh,&IONS->at(ii),2);

		extrapolateIonDensity(params,mesh,&IONS->at(ii));//Once the ions have been pushed, we extrapolate the density at the node grids.

	}//structure to iterate over all the ion species.
}


void PIC::aip_3D(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT){

	double lx, ly, lz;
	lx = mesh->nodes.X(mesh->dim(0)-1) + mesh->DX;//
	ly = mesh->nodes.Y(mesh->dim(1)-1) + mesh->DY;//
	lz = mesh->nodes.Z(mesh->dim(2)-1) + mesh->DZ;//

	for(int ii=0;ii<IONS->size();ii++){//structure to iterate over all the ion species.
		//X^(N+1) = X^(N) + DT*V^(N+1/2)

		#ifdef THREED
		IONS->at(ii).X.col(0) += DT*IONS->at(ii).V.col(0);//x-component
		IONS->at(ii).X.col(1) += DT*IONS->at(ii).V.col(1);//y-component
		IONS->at(ii).X.col(2) += DT*IONS->at(ii).V.col(2);//z-component
		#endif

		#ifdef TWOD
		IONS->at(ii).X.col(0) += DT*IONS->at(ii).V.col(0);//x-component
		IONS->at(ii).X.col(1) += DT*IONS->at(ii).V.col(1);//y-component
		#endif

		#ifdef ONED
		IONS->at(ii).X.col(0) += DT*IONS->at(ii).V.col(0);//x-component
		#endif


		#ifdef THREED
		for(int jj=0;jj<IONS->at(ii).NSP;jj++){//Periodic boundary condition for the ions
			IONS->at(ii).X(jj,0) = fmod(IONS->at(ii).X(jj,0),lx);//x
			if(IONS->at(ii).X(jj,0) < 0)
				IONS->at(ii).X(jj,0) += lx;

			IONS->at(ii).X(jj,1) = fmod(IONS->at(ii).X(jj,1),ly);//y
			if(IONS->at(ii).X(jj,1) < 0)
				IONS->at(ii).X(jj,1) += ly;

			IONS->at(ii).X(jj,2) = fmod(IONS->at(ii).X(jj,2),lz);//z
			if(IONS->at(ii).X(jj,2) < 0)
				IONS->at(ii).X(jj,2) += lz;
		}//Periodic boundary condition for the ions
		#endif

		#ifdef TWOD
		for(int jj=0;jj<IONS->at(ii).NSP;jj++){//Periodic boundary condition for the ions
			IONS->at(ii).X(jj,0) = fmod(IONS->at(ii).X(jj,0),lx);//x
			if(IONS->at(ii).X(jj,0) < 0)
				IONS->at(ii).X(jj,0) += lx;

			IONS->at(ii).X(jj,1) = fmod(IONS->at(ii).X(jj,1),ly);//y
			if(IONS->at(ii).X(jj,1) < 0)
				IONS->at(ii).X(jj,1) += ly;
		}//Periodic boundary condition for the ions
		#endif

		#ifdef ONED
		for(int jj=0;jj<IONS->at(ii).NSP;jj++){//Periodic boundary condition for the ions
			IONS->at(ii).X(jj,0) = fmod(IONS->at(ii).X(jj,0),lx);//x
			if(IONS->at(ii).X(jj,0) < 0)
				IONS->at(ii).X(jj,0) += lx;
		}//Periodic boundary condition for the ions
		#endif

		assingCell_TSC(params,mesh,&IONS->at(ii),3);

		extrapolateIonDensity(params,mesh,&IONS->at(ii));//Once the ions have been pushed, we extrapolate the density at the node grids.

	}//structure to iterate over all the ion species.
}


void PIC::advanceIonsVelocity(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

	//cout << "Status: Advancing the ions' velocity...\n";

	#ifdef ONED
	switch (params->particleIntegrator){
		case(1):{
				aiv_Boris_1D(params,CS,mesh,EB,IONS,DT);
				break;
				}
		case(2):{
				aiv_Vay_1D(params,CS,mesh,EB,IONS,DT);
				break;
				}
		case(3):{
				exit(0);
				break;
				}
		default:{
				aiv_Vay_1D(params,CS,mesh,EB,IONS,DT);
				}
	}
	#endif

	#ifdef TWOD
		aiv_Boris_2D(params,CS,mesh,EB,IONS,DT);
	#endif

	#ifdef THREED
		aiv_Boris_3D(params,CS,mesh,EB,IONS,DT);
	#endif
}

void PIC::advanceIonsPosition(const inputParameters * params,const meshGeometry * mesh,vector<ionSpecies> * IONS,const double DT){

	//cout << "Status: Advancing the ions' position...\n";

	#ifdef ONED
		aip_1D(params,mesh,IONS,DT);
	#endif

	#ifdef TWOD
		aip_2D(params,mesh,IONS,DT);
	#endif

	#ifdef THREED
		aip_3D(params,mesh,IONS,DT);
	#endif
}

void PIC::advanceGCIons(const inputParameters * params,const characteristicScales * CS,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS,const double DT){

	vector<ionSpecies> IONS_RK = *IONS;



}
