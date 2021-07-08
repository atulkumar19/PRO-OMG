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



PIC::PIC()
{
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


void PIC::MPI_ReduceVec(const simulationParameters * params, arma::vec * v)
{
    // Create receive buffer:
    // ======================
    arma::vec recvbuf = zeros(v->n_elem);

    // Reduce the vector at the root process of particles:
    // ======================================================
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        MPI_Reduce(v->memptr(), recvbuf.memptr(), v->n_elem, MPI_DOUBLE, MPI_SUM, 0, params->mpi.COMM);

        if (params->mpi.IS_PARTICLES_ROOT)
        {
             *v = recvbuf;
        }
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
	//int N = v->n_elem;

	//v->subvec(N-2,N-1) = v->subvec(2,3);
	//v->subvec(0,1) = v->subvec(N-4,N-3);
}

void PIC::include4GhostsContributions(arma::vec * v){
	//int N = v->n_elem;

	//v->subvec(2,3) += v->subvec(N-2,N-1);
	//v->subvec(N-4,N-3) += v->subvec(0,1);

	int N = v->n_elem;

	v->subvec(0,1)     = v->subvec(2,3);
	v->subvec(N-2,N-1) = v->subvec(N-4,N-3);
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
void PIC::smooth(arma::vec * v, double as)
{
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


void PIC::smooth(arma::mat * m, double as)
{
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


void PIC::smooth(vfield_vec * vf, double as)
{
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}


void PIC::smooth(vfield_mat * vf, double as)
{
	smooth(&vf->X,as); // x component
	smooth(&vf->Y,as); // y component
	smooth(&vf->Z,as); // z component
}
// * * * Smoothing * * *


void PIC::assignCell(const simulationParameters * params,  oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
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
    for(int ii=0; ii<NSP; ii++)
    {
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

            if(LOGIC)
            {
                IONS->wxl(ii) = 0.5*(1.5 - (params->mesh.DX + X)/params->mesh.DX)*(1.5 - (params->mesh.DX + X)/params->mesh.DX);
                IONS->wxr(ii) = 0.5*(1.5 - (params->mesh.DX - X)/params->mesh.DX)*(1.5 - (params->mesh.DX - X)/params->mesh.DX);
            }
            else
            {
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


void PIC::assignCell(const simulationParameters * params,  twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{}


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


void PIC::interpolateVectorField(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, vfield_mat * field, arma::mat * F)
{}


void PIC::interpolateElectromagneticFields(const simulationParameters * params, const twoDimensional::ionSpecies * IONS, twoDimensional::fields * EB, arma::mat * E, arma::mat * B)
{}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT)
{
	MPI_Recvvfield_vec(params, &EB->E);
	MPI_Recvvfield_vec(params, &EB->B);

	fillGhosts(EB);

	// Iterate over all the ion species:
	for(int ii=0; ii<IONS->size(); ii++)
	{
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
		{
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
				arma::rowvec U = arma::zeros<rowvec>(3);
				arma::rowvec VxB = arma::zeros<rowvec>(3);
				arma::rowvec tau = arma::zeros<rowvec>(3);
				arma::rowvec up = arma::zeros<rowvec>(3);
				arma::rowvec t = arma::zeros<rowvec>(3);
				arma::rowvec upxt = arma::zeros<rowvec>(3);

				//#pragma omp for
				for(int ip=0;ip<NSP;ip++)
				{
					// Magnetic field gradient:
					double dBx = Bp(ip,1)/(-0.5*(params->geometry.r2));

					// Calculate larmour radius:
					double Vper = sqrt(IONS->at(ii).V(ip,1)*IONS->at(ii).V(ip,1)+IONS->at(ii).V(ip,2)*IONS->at(ii).V(ip,2));
					double omega_ci = (fabs(IONS->at(ii).Q)*Bp(ip,0)/IONS->at(ii).M);
					double rL_i = fabs(Vper/omega_ci);

					// Calculate gyro-phase terms:
					double UUy = arma::as_scalar(IONS->at(ii).V(ip,1));
					double UUz = arma::as_scalar(IONS->at(ii).V(ip,2));
					double cosPhi = -(UUz/Vper);
					double sinPhi = +(UUy/Vper);

					// Particle defined magnetic field:
					Bp(ip,1) = -0.5*(params->geometry.r2*sqrt(params->em_IC.BX/Bp(ip,0)) + rL_i*cosPhi)*dBx;
					Bp(ip,2) = -0.5*(rL_i*sinPhi)*dBx;

					// VxB term:
					VxB = arma::cross(IONS->at(ii).V.row(ip), Bp.row(ip));

					// Calculate new V:
					IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 -  dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip))/(F_C_DS*F_C_DS) );
					U = IONS->at(ii).g(ip)*IONS->at(ii).V.row(ip);

					U += 0.5*A*(Ep.row(ip) + VxB); // U_hs = U_L + 0.5*a*(E + cross(V, B)); % Half step for velocity
					tau = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
					up = U + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
					double gp = sqrt( 1.0 + dot(up, up)/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
					double sigma = gp*gp - dot(tau, tau); // sigma = gammap^2 - tau*tau';
					double us = dot(up, tau)/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
					IONS->at(ii).g(ip) = sqrt(0.5)*sqrt( sigma + sqrt( sigma*sigma + 4.0*( dot(tau, tau) + us*us ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
					t = tau/IONS->at(ii).g(ip); 			// t = tau/gamma;
					double s = 1.0/( 1.0 + dot(t, t) ); // s = 1/(1 + t*t'); % variable 's' in paper

					upxt = arma::cross(up, t);

					U = s*( up + dot(up, t)*t+ upxt ); 	// U_L = s*(up + (up*t')*t + cross(up, t));
					IONS->at(ii).V.row(ip) = U/IONS->at(ii).g(ip);	// V = U_L/gamma;
				}
			} // End of parallel region
		}
	}//structure to iterate over all the ion species.
}


void PIC::advanceIonsVelocity(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT)
{}



void PIC::advanceIonsPosition(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS, const double DT)
{
    // Iterate over all ion species:
    // =============================
    for(int ii=0;ii<IONS->size();ii++)
    {
        // Advance particle position:
        //X^(N+1) = X^(N) + DT*V^(N+1/2)
        if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            // Number of computational particles per process:
            int NSP(IONS->at(ii).NSP);

            #pragma omp parallel default(none) shared(params, IONS) firstprivate(DT, NSP, ii)
            {
                #pragma omp for
                for(int ip=0; ip<NSP; ip++)
                    {
                        // Advance position:
                        IONS->at(ii).X(ip,0) += DT*IONS->at(ii).V(ip,0);

                    } // pragma omp for
            }//End of the parallel region
        } // IF particle sentinel
    } // Iterate over all ion species
}


void PIC::advanceIonsPosition(const simulationParameters * params,  twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS, const double DT)
{}

void PIC::extrapolateIonsMoments(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
	// Iterate over all ion species:
    // =============================
    for(int ii=0;ii<IONS->size();ii++)
    {
        // Assign cell and calculate partial ion moments:
        if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            // Assign cell:
            PIC::assignCell(params, EB, &IONS->at(ii));

            //Calculate partial moments:
			calculateIonMoments(params, EB, &IONS->at(ii));
        }

        // Reduce IONS moments to PARTICLE ROOT:
        // =====================================
        PIC::MPI_ReduceVec(params, &IONS->at(ii).n);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.X);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.Y);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).nv.Z);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).P11);
		PIC::MPI_ReduceVec(params, &IONS->at(ii).P22);

		// Broadcast ion moments to all PARTICLE processes:
		// ================================================
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
		{
			MPI_Bcast(IONS->at(ii).n.memptr(), IONS->at(ii).n.size(), MPI_DOUBLE, 0, params->mpi.COMM);
			MPI_Bcast(IONS->at(ii).nv.X.memptr(), IONS->at(ii).nv.X.size(), MPI_DOUBLE, 0, params->mpi.COMM);
			MPI_Bcast(IONS->at(ii).nv.Y.memptr(), IONS->at(ii).nv.Y.size(), MPI_DOUBLE, 0, params->mpi.COMM);
			MPI_Bcast(IONS->at(ii).nv.Z.memptr(), IONS->at(ii).nv.Z.size(), MPI_DOUBLE, 0, params->mpi.COMM);
			MPI_Bcast(IONS->at(ii).P11.memptr(), IONS->at(ii).P11.size(), MPI_DOUBLE, 0, params->mpi.COMM);
			MPI_Bcast(IONS->at(ii).P22.memptr(), IONS->at(ii).P22.size(), MPI_DOUBLE, 0, params->mpi.COMM);
		}

        // Apply smoothing:
        // ===============
        for (int jj=0; jj<params->filtersPerIterationIons; jj++)
        {
          smooth(&IONS->at(ii).n, params->smoothingParameter);
		  smooth(&IONS->at(ii).nv, params->smoothingParameter);
		  smooth(&IONS->at(ii).P11, params->smoothingParameter);
		  smooth(&IONS->at(ii).P22, params->smoothingParameter);
        }

		// Calculate derived ion moments: Tpar_m, Tper_m:
		// ====================================================
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
		{
			MPI_Barrier(params->mpi.COMM);
			calculateDerivedIonMoments(params, &IONS->at(ii));
		}

        // 0th and 1st moments at various time levels are sent to fields processes:
        // =============================================================
        PIC::MPI_SendVec(params, &IONS->at(ii).n);
        PIC::MPI_SendVec(params, &IONS->at(ii).n_);
        PIC::MPI_SendVec(params, &IONS->at(ii).n__);
        PIC::MPI_SendVec(params, &IONS->at(ii).n___);

		PIC::MPI_SendVec(params, &IONS->at(ii).nv.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv.Z);

		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv_.Z);

		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.X);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.Y);
		PIC::MPI_SendVec(params, &IONS->at(ii).nv__.Z);
	}
}

void PIC::extrapolateIonsMoments(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}


// template class PIC<oneDimensional::ionSpecies, oneDimensional::fields>;
// template class PIC<twoDimensional::ionSpecies, twoDimensional::fields>;


// calculateIonMoments:

void PIC::calculateIonMoments(const simulationParameters * params, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
	// Ion density:
	IONS->n___ = IONS->n__;
	IONS->n__ = IONS->n_;
	IONS->n_ = IONS->n;

	// Ion flux:
	IONS->nv__ = IONS->nv_;
	IONS->nv_ = IONS->nv;

	// Calculate ion moments:
	eim(params, EB, IONS);
}

void PIC::eim(const simulationParameters * params, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NSP(IONS->NSP);

	// Clearing content of ion moments:
	// ===============================
	IONS->n.zeros();
	IONS->nv.zeros();
	IONS->P11.zeros();
	IONS->P22.zeros();

	#pragma omp parallel default(none) shared(params, IONS) firstprivate(NSP)
	{
		// Create private moments:
		// ======================
		double Ma(IONS->M);
		arma::vec n = zeros(params->mesh.NX_IN_SIM + 4);
		vfield_vec nv(params->mesh.NX_IN_SIM + 4);
		arma::vec P11 = zeros(params->mesh.NX_IN_SIM + 4);
		arma::vec P22 = zeros(params->mesh.NX_IN_SIM + 4);

		// Assemble moments:
		// =================
		#pragma omp for
		for(int ii=0; ii<NSP; ii++)
		{
			// Nearest grid point:
			int ix = IONS->mn(ii) + 2;

			// Perpendicular velocity:
			//double V_perp = sqrt( pow(IONS->V(ii,1),2) + pow(IONS->V(ii,2),2));
			double Vy = IONS->V(ii,1);
			//double Vz = IONS->V(ii,2);

			// Particle weight:
			double a = IONS->a(ii);

			// Density:
			n(ix-1) += IONS->wxl(ii)*a;
			n(ix)   += IONS->wxc(ii)*a;
			n(ix+1) += IONS->wxr(ii)*a;

			// Particle flux density:
			nv.X(ix-1) 	+= IONS->wxl(ii)*a*IONS->V(ii,0);
			nv.X(ix) 	+= IONS->wxc(ii)*a*IONS->V(ii,0);
			nv.X(ix+1) 	+= IONS->wxr(ii)*a*IONS->V(ii,0);

			nv.Y(ix-1) 	+= IONS->wxl(ii)*a*IONS->V(ii,1);
			nv.Y(ix) 	+= IONS->wxc(ii)*a*IONS->V(ii,1);
			nv.Y(ix+1) 	+= IONS->wxr(ii)*a*IONS->V(ii,1);

			nv.Z(ix-1) 	+= IONS->wxl(ii)*a*IONS->V(ii,2);
			nv.Z(ix) 	+= IONS->wxc(ii)*a*IONS->V(ii,2);
			nv.Z(ix+1) 	+= IONS->wxr(ii)*a*IONS->V(ii,2);

			// Stress tensor P11:
			P11(ix-1) += IONS->wxl(ii)*a*Ma*pow(IONS->V(ii,0),2);
			P11(ix)   += IONS->wxc(ii)*a*Ma*pow(IONS->V(ii,0),2);
			P11(ix+1) += IONS->wxr(ii)*a*Ma*pow(IONS->V(ii,0),2);

			// Stress tensor P22:
			P22(ix-1) += IONS->wxl(ii)*a*Ma*pow(Vy,2);
			P22(ix)   += IONS->wxc(ii)*a*Ma*pow(Vy,2);
			P22(ix+1) += IONS->wxr(ii)*a*Ma*pow(Vy,2);

			// Stress tensor P33:
			//P33(ix-1) += IONS->wxl(ii)*a*Ma*pow(Vz,2);
			//P33(ix)   += IONS->wxc(ii)*a*Ma*pow(Vz,2);
			//P33(ix+1) += IONS->wxr(ii)*a*Ma*pow(Vz,2);
		}

		// Ghost contributions:
		// ====================
		include4GhostsContributions(&n);
		include4GhostsContributions(&nv.X);
		include4GhostsContributions(&nv.Y);
		include4GhostsContributions(&nv.Z);
		include4GhostsContributions(&P11);
		include4GhostsContributions(&P22);

		// Reduce partial moments from each thread:
		// ========================================
		#pragma omp critical (update_ion_moments)
		{
			IONS->n.subvec(1,params->mesh.NX_IN_SIM) += n.subvec(2,params->mesh.NX_IN_SIM + 1);
			IONS->nv.X.subvec(1,params->mesh.NX_IN_SIM) += nv.X.subvec(2,params->mesh.NX_IN_SIM + 1);
			IONS->nv.Y.subvec(1,params->mesh.NX_IN_SIM) += nv.Y.subvec(2,params->mesh.NX_IN_SIM + 1);
			IONS->nv.Z.subvec(1,params->mesh.NX_IN_SIM) += nv.Z.subvec(2,params->mesh.NX_IN_SIM + 1);
			IONS->P11.subvec(1,params->mesh.NX_IN_SIM) += P11.subvec(2,params->mesh.NX_IN_SIM + 1);
			IONS->P22.subvec(1,params->mesh.NX_IN_SIM) += P22.subvec(2,params->mesh.NX_IN_SIM + 1);
		}

	}//End of the parallel region

	// Calculate compression factor:
	arma::vec compressionFactor = (EB->B.X.subvec(1,params->mesh.NX_IN_SIM)/params->em_IC.BX)/params->geometry.A_0;

	// Apply magnetic compression:
	IONS->n.subvec(1,params->mesh.NX_IN_SIM) = IONS->n.subvec(1,params->mesh.NX_IN_SIM)% compressionFactor;
	IONS->nv.X.subvec(1,params->mesh.NX_IN_SIM) = IONS->nv.X.subvec(1,params->mesh.NX_IN_SIM)%compressionFactor;
	//IONS->nv.Y.subvec(1,params->mesh.NX_IN_SIM) = IONS->nv.Y.subvec(1,params->mesh.NX_IN_SIM) % compressionFactor;
	//IONS->nv.Z.subvec(1,params->mesh.NX_IN_SIM) = IONS->nv.Z.subvec(1,params->mesh.NX_IN_SIM) % compressionFactor;
	IONS->P11.subvec(1,params->mesh.NX_IN_SIM) = IONS->P11.subvec(1,params->mesh.NX_IN_SIM)% compressionFactor;
	IONS->P22.subvec(1,params->mesh.NX_IN_SIM) = IONS->P22.subvec(1,params->mesh.NX_IN_SIM)% compressionFactor;

	// Scale:
	IONS->n *= IONS->NCP/params->mesh.DX;
	IONS->nv *= IONS->NCP/params->mesh.DX;
	IONS->P11 *= IONS->NCP/params->mesh.DX;
	IONS->P22 *= IONS->NCP/params->mesh.DX;

}

void PIC::calculateIonMoments(const simulationParameters * params, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{}

void PIC::eim(const simulationParameters * params, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{}

void PIC::calculateDerivedIonMoments(const simulationParameters * params, oneDimensional::ionSpecies * IONS)
{
	double Ma(IONS->M);

	// Ion pressures:
	arma::vec Ppar = IONS->P11 - (Ma*IONS->nv.X % IONS->nv.X/IONS->n);
	arma::vec Pper = IONS->P22; // We have neglected perp drift kinetic energy

	// Ion temperatures:
	IONS->Tpar_m = Ppar/(F_E_DS*IONS->n);
	IONS->Tper_m = Pper/(F_E_DS*IONS->n);
}

void PIC::calculateDerivedIonMoments(const simulationParameters * params, twoDimensional::ionSpecies * IONS)
{}

void PIC::interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec field, arma::vec * F)
{
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

	field_X.subvec(1,NX-2) = field;

	fill4Ghosts(&field_X);

	//Contrary to what may be thought,F is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for default(none) shared(params, IONS, F, field_X) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++)
	{
		int ix = IONS->mn(ii) + 2;

		(*F)(ii) += IONS->wxl(ii)*field_X(ix-1);
		(*F)(ii) += IONS->wxc(ii)*field_X(ix);
		(*F)(ii) += IONS->wxr(ii)*field_X(ix+1);

	}//End of the parallel region
}

void PIC::interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat field, arma::mat * F)
{}
