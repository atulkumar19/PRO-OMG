#include <math.h>
#include "particleBoundaryConditions.h"

using namespace std;

PARTICLE_BC::PARTICLE_BC()
{}

void PARTICLE_BC::MPI_AllreduceDouble(const simulationParameters * params, double * v)
{
    double recvbuf = 0;

    MPI_Allreduce(v, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, params->mpi.COMM);

    *v = recvbuf;
}

void PARTICLE_BC::checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles is "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Ion mass:
            // =========
            double Ma = IONS->at(aa).M;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(params, IONS, aa, CS, std::cout) firstprivate(NSP,Ma)
            for(int ii=0; ii<NSP; ii++)
            {
                // left boundary:
                if (IONS->at(aa).X(ii,0) < 0)
                {
                    // Particle flag:
                    IONS->at(aa).f1(ii) = 1;

                    // Particle kinetic energy:
                    double KE = 0.5*Ma*dot(IONS->at(aa).V.row(ii), IONS->at(aa).V.row(ii));
                    IONS->at(aa).dE1(ii) = KE;
                }

                // Right boundary:
                if (IONS->at(aa).X(ii,0) > params->mesh.LX)
                {
                    // Particle flag:
                    IONS->at(aa).f2(ii) = 1;

                    // Particle kinetic energy:
                    double KE = 0.5*Ma*dot(IONS->at(aa).V.row(ii), IONS->at(aa).V.row(ii));
                    IONS->at(aa).dE2(ii) = KE;
                }

            } // Particle loop

        } // species loop

    } // Particle MPI guard

}

void PARTICLE_BC::checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void PARTICLE_BC::calculateParticleWeight(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Simulation time step:
    // =====================
    double DT = params->DT*CS->time;

    // Iterate over all ion species:
    // =============================
    for(int ss=0;ss<IONS->size();ss++)
    {
        if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            // Number of computational particles per process:
            int NSP(IONS->at(ss).NSP);

            // Super particle conversion factor:
            double alpha = IONS->at(ss).NCP;

            // Computational particle leak rate:
            double S1 = 0;
            double S2 = 0;

            #pragma omp parallel default(none) shared(S1, S2, params, IONS, ss, std::cout) firstprivate(NSP)
            {
                // Computational particle accumulators:
                double S1_private = 0;
                double S2_private = 0;

                #pragma omp for
                for(int ii=0; ii<NSP; ii++)
                {
                    S1_private += IONS->at(ss).f1(ii);
                    S2_private += IONS->at(ss).f2(ii);
                }

                #pragma omp critical
                S1 += S1_private;
                S2 += S2_private;

            } // pragma omp parallel

            // AllReduce S1 and S2 over all particle MPIs:
            MPI_AllreduceDouble(params,&S1);
            MPI_AllreduceDouble(params,&S2);

            // Accumulate computational particles:
            IONS->at(ss).p_BC.S1 += S1;
            IONS->at(ss).p_BC.S2 += S2;

            // Accumulate fueling rate:
            double G = IONS->at(ss).p_BC.G;
            IONS->at(ss).p_BC.GSUM += G;

            // Minimum number of computational particles to trigger fueling:
            double S_min = 3;

            if ( (IONS->at(ss).p_BC.S1 + IONS->at(ss).p_BC.S2) >= S_min )
            {
                // Total number of computational particles leaked:
                double S_total  = IONS->at(ss).p_BC.S1 + IONS->at(ss).p_BC.S2;

                // Calculate computational particle leak rate:
                double uN_total = (alpha/DT)*S_total;

                // Calculate particle weight:
                double GSUM  = IONS->at(ss).p_BC.GSUM;
                double a_new = GSUM/uN_total;

                /*
                if (params->mpi.IS_PARTICLES_ROOT)
                {
                    cout << "S_total:" << S_total << endl;
                    cout << "uN_total:" << uN_total << endl;
                    cout << "a_new:" << a_new << endl;
                    cout << "GSUM:" << GSUM << endl;
                }
                */

                if (a_new > 100)
                {
                    if (params->mpi.IS_PARTICLES_ROOT)
                    {
                        cout << "S_total:" << S_total << endl;
                        cout << "uN_total:" << uN_total << endl;
                        cout << "a_new:" << a_new << endl;
                        cout << "GSUM:" << GSUM << endl;
                    }
                    a_new = 1;
                }
                IONS->at(ss).p_BC.a_new = a_new;

                // Reset accumulators:
                IONS->at(ss).p_BC.S1   = 0;
                IONS->at(ss).p_BC.S2   = 0;
                IONS->at(ss).p_BC.GSUM = 0;
            }

        } // Particle MPIs
    } //  Species
}

void PARTICLE_BC::calculateParticleWeight(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void PARTICLE_BC::applyParticleReinjection(const simulationParameters * params,const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{

    // Iterate over all ion species:
    // =============================
    for(int ss=0;ss<IONS->size();ss++)
    {
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            // Number of computational particles per process:
            int NSP(IONS->at(ss).NSP);

			#pragma omp parallel default(none) shared(params, CS, EB, IONS, std::cout) firstprivate(NSP, ss)
            {
                // Leak diagnostics:
                int pc = 0;
                double ec = 0;

                #pragma omp for
                for(int ii=0; ii<NSP; ii++)
				{
					if ( IONS->at(ss).f1(ii) == 1 || IONS->at(ss).f2(ii) == 1 )
					{
                        //Record events:
                        // =============
                        pc += 1; // Increase the leaking particles by one
                        ec += 0.5*IONS->at(ss).M*dot(IONS->at(ss).V.row(ii), IONS->at(ss).V.row(ii)); // Increase the leaking particles KE

						// Re-inject particle:
                        // ===================
						particleReinjection(ii, params, CS, EB,&IONS->at(ss));

                        // Reset injection flag:
                        // =====================
						IONS->at(ss).f1(ii) = 0;
						IONS->at(ss).f2(ii) = 0;

					} // flag guard
				} // pragma omp for

				#pragma omp critical
                // Accumulate leak diagnotics:
                IONS->at(ss).pCount += pc;
                IONS->at(ss).eCount += ec;

			} // pragma omp parallel
		} // Particle MPIs
	} //  Species
}

void PARTICLE_BC::applyParticleReinjection(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void PARTICLE_BC::particleReinjection(int ii, const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
    // Particle weight:
    // =========================================================================
    IONS->a(ii) = IONS->p_BC.a_new;

    // Particle position:
    // =========================================================================
    // Variables for random number generator:
    arma::vec R = randu(1);
    arma_rng::set_seed_random();
    arma::vec phi = 2.0*M_PI*randu<vec>(1);

    // Gaussian distribution in space:
    double Xcenter = params->mesh.LX/2;
    double sigmaX  = params->mesh.LX/10;
    double Xnew = Xcenter  + (sigmaX)*sqrt( -2*log(R(0)) )*cos(phi(0));
    double dLX = abs(Xnew - Xcenter);

    while(dLX > params->mesh.LX/2)
    {
         std::cout<<"Out of bound X= "<< Xnew;
         arma_rng::set_seed_random();
         R = randu(1);
         phi = 2.0*M_PI*randu<vec>(1);

         Xnew = Xcenter  + (sigmaX)*sqrt( -2*log(R(0)) )*cos(phi(0));
         dLX  = abs(Xnew - Xcenter);
         std::cout<< "Out of bound corrected X= " << Xnew;
    }

    IONS->X(ii,0) = Xnew;

    // Particle velocity:
    // =========================================================================
    arma_rng::set_seed_random();
    R = randu(1);
    phi = 2.0*M_PI*randu<vec>(1);

    arma::vec V2 = IONS->VTper*sqrt( -log(1.0 - R) ) % cos(phi);
    arma::vec V3 = IONS->VTper*sqrt( -log(1.0 - R) ) % sin(phi);

    arma_rng::set_seed_random();
    phi = 2.0*M_PI*randu<vec>(1);

    arma::vec V1 = IONS->VTper*sqrt( -log(1.0 - R) ) % sin(phi);

    // Cartesian unit vectors:
    // ===========================
    arma::vec x = params->mesh.e_x;
    arma::vec y = params->mesh.e_y;
    arma::vec z = params->mesh.e_z;

    // B field PIC interpolation needed at this point

    // Field-alinged unit vectors:
    // ===========================
    arma::vec b1; // Unitary vector along B field
    arma::vec b2; // Unitary vector perpendicular to b1
    arma::vec b3; // Unitary vector perpendicular to b1 and b2

    b1 = {params->BGP.Bx, params->BGP.By, params->BGP.Bz};
    b1 = arma::normalise(b1);

    if (arma::dot(b1,y) < PRO_ZERO)
    {
        b2 = arma::cross(b1,y);
    }
    else
    {
        b2 = arma::cross(b1,z);
    }

    // Unitary vector perpendicular to b1 and b2
    b3 = arma::cross(b1,b2);

    IONS->V(ii,0) = V1(0)*dot(b1,x) + V2(0)*dot(b2,x) + V3(0)*dot(b3,x);
    IONS->V(ii,1) = V1(0)*dot(b1,y) + V2(0)*dot(b2,y) + V3(0)*dot(b3,y);
    IONS->V(ii,2) = V1(0)*dot(b1,z) + V2(0)*dot(b2,z) + V3(0)*dot(b3,z);

    IONS->g(ii) = 1.0/sqrt( 1.0 - dot(IONS->V.row(ii),IONS->V.row(ii))/(F_C*F_C) );
    IONS->mu(ii) = 0.5*IONS->g(ii)*IONS->g(ii)*IONS->M*( V2(0)*V2(0) + V3(0)*V3(0) )/params->BGP.Bo;
    IONS->Ppar(ii) = IONS->g(ii)*IONS->M*V1(0);
    IONS->avg_mu = mean(IONS->mu);
}

void PARTICLE_BC::particleReinjection(int ii, const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{}
