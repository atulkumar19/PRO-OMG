#include "collisionOperator.h"

using namespace std;

collisionOperator::collisionOperator()
{}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, oneDimensional::ionSpecies * IONS)
{
        // Plasma density:
        &IONS->n_p.zeros();
        interpolateScalarField(params, IONS, &IONS->n, &IONS->n_p);

        // Parallel temperature:
        &IONS->Tpar_p.zeros();
        interpolateScalarField(params, IONS, &IONS->Tpar_m, &IONS->Tpar_p);

        // Perpendicular temperature:
        &IONS->Tper_p.zeros();
        interpolateScalarField(params, IONS, &IONS->Tper_m, &IONS->Tper_p);

        // Parallel drift velocity:
        &IONS->U_p.X.zeros();
        interpolateScalarField(params, IONS, &IONS->U_m.X, &IONS->U_p.X);
}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, twoDimensional::ionSpecies * IONS)
{}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Interpolate ion moments:
        // ========================
        for(int ii=0;ii<IONS->size();ii++)
        {
            interpolateIonMoments(params,&IONS->at(ii));
        }

        // Apply operator:
        // ===============
        for(int ii=0;ii<IONS->size();ii++)
        {
            mcOperator(params,CS,&IONS->at(ii))
        }

    }
}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{}

void collisionOperator::interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * field, arma::vec * F)
{
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NX =  params->mesh.NX_IN_SIM + 4; //Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);

	arma::vec field_X = zeros(NX);

	field_X.subvec(1,NX-2) = *field;

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

void collisionOperator::interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * field, arma::mat * F)
{}

void collisionOperator::fill4Ghosts(arma::vec * v)
{
	int N = v->n_elem;

	v->subvec(N-2,N-1) = v->subvec(2,3);
	v->subvec(0,1) = v->subvec(N-4,N-3);
}

void cartesian2Spherical(const double * wx, const double * wy, const double * wz, double * w, double * xi, double * phi)
{
    // code
}

void collisionOperator::mcOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS)
{
    int NSP(IONS->NSP);

    #pragma omp parallel for default(none) shared(params, IONS) firstprivate(NSP)
	for(int pp=0; pp<NSP; pp++)
	{
        // The code below needs to be generalized to allow multiple ion species
        // and thus the concept of center of mass needs to be incorporated:

        // Particle "a" conditions:
        // =======================
        // Velocity:
        double vxa = IONS->V(pp,1)*CS->velocity;
        double vya = IONS->V(pp,2)*CS->velocity;
        double vza = IONS->V(pp,3)*CS->velocity;

        // Parameters:
        double Ma = IONS->M*CS->mass;
        double Za = IONS->Z;

        // Drift velocity:
        Uxa = IONS->U_p.X(pp);

        // Convert to center of mass frame:
        double wxa = vxa - Uxa;
        double wya = vya;
        double wza = vza;

        // Convert velocity from cartesian to spherical coordinate system:
        // ==============================================================
        double w(0.0);
        double xi(0.0);
        double phi(0.0);

        cartesian2Spherical(&wxa, &wya, &wza, &w, &xi, &phi);


        // Loop over background species "b":
        // ===============================
        int num_species = IONS->size() + 1;
        for (int ss=0,ss<num_species,ss++)
        {
            // Select "b" conditions:

            // Ar

        }

	}
}
