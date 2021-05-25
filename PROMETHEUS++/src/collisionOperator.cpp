#include "collisionOperator.h"

using namespace std;

collisionOperator::collisionOperator()
{}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Interpolate ion moments:
        // ========================
        //for(int ii=0;ii<IONS->size();ii++)
        //{
        //    interpolateIonMoments(params,&IONS->at(ii));
        //}

        // Apply operator:
        // ===============
        //for(int ii=0;ii<IONS->size();ii++)
        //{
        //    mcOperator(params,CS,&IONS->at(ii));
        //}

    }
}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, int a, int b)
{
    /*
        // Plasma density:
        &IONS->n_p.zeros();
        interpolateScalarField(params, IONS, &IONS->n, &IONS->n_p);

        // Parallel drift velocity:
        &IONS->nv_p.zeros();
        interpolateScalarField(params, IONS, &IONS->nv.X, &IONS->nv_p);

        // Parallel temperature:
        &IONS->Tpar_p.zeros();
        interpolateScalarField(params, IONS, &IONS->Tpar_m, &IONS->Tpar_p);

        // Perpendicular temperature:
        &IONS->Tper_p.zeros();
        interpolateScalarField(params, IONS, &IONS->Tper_m, &IONS->Tper_p);
        */

        //  Number of computational particles:
    	int NSP(IONS->at(a).NSP);

    	// Create particle-defined quantities:
    	arma::vec n_p    = zeros(NSP,1);
    	arma::vec nv_p   = zeros(NSP,1);
    	arma::vec Tpar_p = zeros(NSP,1);
    	arma::vec Tper_p = zeros(NSP,1);

    	// Create mesh defined quantities:
    	arma::vec n_m    = IONS->at(b).n;
    	arma::vec nv_m   = IONS->at(b).nv.X;
    	arma::vec Tpar_m = IONS->at(b).Tpar_m;
    	arma::vec Tper_m = IONS->at(b).Tper_m;

    	// Interpolate:
    	interpolateScalarField(params, &IONS->at(a), &n_m   , &n_p   );
    	interpolateScalarField(params, &IONS->at(a), &nv_m  , &nv_p  );
    	interpolateScalarField(params, &IONS->at(a), &Tpar_m, &Tpar_p);
    	interpolateScalarField(params, &IONS->at(a), &Tper_m, &Tper_p);

    	// Assign values:
    	IONS->at(a).n_p    = n_p;
    	IONS->at(a).nv_p   = nv_p;
    	IONS->at(a).Tpar_p = Tpar_p;
    	IONS->at(a).Tper_p = Tper_p;
}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, int a, int b)
{}

void collisionOperator::interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, arma::vec * F_p)
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

	arma::vec F = zeros(NX);

	F.subvec(1,NX-2) = *F_m;

	fill4Ghosts(&F);

	//Contrary to what may be thought,F_p is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for default(none) shared(params, IONS, F_p, F) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++)
	{
		int ix = IONS->mn(ii) + 2;

		(*F_p)(ii) += IONS->wxl(ii)*F(ix-1);
		(*F_p)(ii) += IONS->wxc(ii)*F(ix);
		(*F_p)(ii) += IONS->wxr(ii)*F(ix+1);

	}//End of the parallel region
}

void collisionOperator::interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * F_m, arma::mat * F_p)
{}

void collisionOperator::fill4Ghosts(arma::vec * v)
{
	int N = v->n_elem;

	v->subvec(N-2,N-1) = v->subvec(2,3);
	v->subvec(0,1) = v->subvec(N-4,N-3);
}

void collisionOperator::mcOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS)
{
    int NSP(IONS->NSP);

    #pragma omp parallel default(none) shared(params, CS, IONS) firstprivate(NSP)
    {
        #pragma omp for
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
            double Uxa = IONS->nv_p(pp)/IONS->n_p(pp);

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
            int num_species = params->numberOfParticleSpecies; /*IONS->size() + 1;*/
            for (int ss=0;ss < num_species;ss++)
            {
                // Select "b" conditions:

                // Ar

            }
        }
	}
}

void collisionOperator::cartesian2Spherical(double * wx, double * wy, double * wz, double * w, double * xi, double * phi)
{
    *w = sqrt( pow(*wx,2) + pow(*wy,2) + pow(*wz,2) );
    *xi = 0.0;
    *phi = 0.0;
}
