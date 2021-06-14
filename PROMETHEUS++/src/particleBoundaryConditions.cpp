#include <math.h>
#include "particleBoundaryConditions.h"

using namespace std;

PARTICLE_BC::PARTICLE_BC()
{}

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

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(params, IONS, aa, CS) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                // left boundary:
                if (IONS->at(aa).X(ii,0) < 0)
                {
                    // Particle flag:
                    IONS->at(aa).f1 = 1;

                    // Particle kinetic energy:
                    double Ma = IONS->at(aa).M;
                    double KE = 0.5*Ma*dot(IONS->at(aa).V.row(ii), IONS->at(aa).V.row(ii));
                    IONS->at(aa).dE1 = KE;
                }

                // Right boundary:
                if (IONS->at(aa).X(ii,0) > params->mesh.LX)
                {
                    // Particle flag:
                    IONS->at(aa).f2 = 1;

                    // Particle kinetic energy:
                    double Ma = IONS->at(aa).M;
                    double KE = 0.5*Ma*dot(IONS->at(aa).V.row(ii), IONS->at(aa).V.row(ii));
                    IONS->at(aa).dE2 = KE;
                }

            } // Particle loop

        }

    } // close MPI

}

void PARTICLE_BC::checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void PARTICLE_BC::applyParticleReinjection(const simulationParameters * params,const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Cartesian unit vectors:
    // =======================
    arma::vec x = {1.0, 0.0, 0.0};
    arma::vec y = {0.0, 1.0, 0.0};
    arma::vec z = {0.0, 0.0, 1.0};

    // Field alinged unit vectors:
    // ===========================
    arma::vec b1; // Unitary vector along B field
    arma::vec b2; // Unitary vector perpendicular to b1
    arma::vec b3; // Unitary vector perpendicular to b1 and b2

    // Iterate over all ion species:
    // =============================
    for(int ii=0;ii<IONS->size();ii++)
    {
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
        {
            // Number of computational particles per process:
            int NSP(IONS->at(ii).NSP);

			#pragma omp parallel default(none) shared(params, CS, EB, IONS, x, y, z, std::cout) firstprivate(NSP, ii, b1, b2, b3)
            {
                // Leak diagnostics:
                int pc = 0;
                double ec = 0;

                #pragma omp for
                for(int ip=0; ip<NSP; ip++)
				{
					if ( IONS->at(ii).f1(ip) == 1 || IONS->at(ii).f2(ip) == 1 )
					{
						// Reset injection flag:
						IONS->at(ii).f1(ip) = 0;
						IONS->at(ii).f2(ip) = 0;

                        //Record events:
                        pc += 1; // Increase the leaking particles by one
                        ec += 0.5*IONS->at(ii).M*dot(IONS->at(ii).V.row(ip), IONS->at(ii).V.row(ip)); // Increase the leaking particles KE

						// Re-inject particle:
						// particleReinjection(ip, params, CS, EB,&IONS->at(ii));

                        // Variables for random number generator:
                        arma::vec R = randu(1);
                        arma_rng::set_seed_random();
                        arma::vec phi = 2.0*M_PI*randu<vec>(1);

                        // Gaussian distribution in space for particle position:
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

                        IONS->at(ii).X(ip,0) = Xnew;

                        // Box Muller in velocity space:
                        arma_rng::set_seed_random();
                        R = randu(1);
                        phi = 2.0*M_PI*randu<vec>(1);

                        arma::vec V2 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % cos(phi);
                        arma::vec V3 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % sin(phi);

                        arma_rng::set_seed_random();
                        phi = 2.0*M_PI*randu<vec>(1);

                        arma::vec V1 = IONS->at(ii).VTper*sqrt( -log(1.0 - R) ) % sin(phi);

                        // Creating magnetic field unit vectors:
                        // Unit vectors have to take care of non-unifoprm B-field - To be done later

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

                        IONS->at(ii).V(ip,0) = V1(0)*dot(b1,x) + V2(0)*dot(b2,x) + V3(0)*dot(b3,x);
                        IONS->at(ii).V(ip,1) = V1(0)*dot(b1,y) + V2(0)*dot(b2,y) + V3(0)*dot(b3,y);
                        IONS->at(ii).V(ip,2) = V1(0)*dot(b1,z) + V2(0)*dot(b2,z) + V3(0)*dot(b3,z);

                        IONS->at(ii).g(ip) = 1.0/sqrt( 1.0 - dot(IONS->at(ii).V.row(ip),IONS->at(ii).V.row(ip))/(F_C*F_C) );
                        IONS->at(ii).mu(ip) = 0.5*IONS->at(ii).g(ip)*IONS->at(ii).g(ip)*IONS->at(ii).M*( V2(0)*V2(0) + V3(0)*V3(0) )/params->BGP.Bo;
                        IONS->at(ii).Ppar(ip) = IONS->at(ii).g(ip)*IONS->at(ii).M*V1(0);
                        IONS->at(ii).avg_mu = mean(IONS->at(ii).mu);
					}
				} // pragma omp for

				#pragma omp critical
                // Accumulate leak diagnotics:
                IONS->at(ii).pCount += pc;
                IONS->at(ii).eCount += ec;

			} // pragma omp parallel
		} // Particle MPIs
	} //  Species
}

void PARTICLE_BC::applyParticleReinjection(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void PARTICLE_BC::particleReinjection(int ip, const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS)
{
    // Variables for random number generator:
    arma::vec R = randu(1);
    arma_rng::set_seed_random();
    arma::vec phi = 2.0*M_PI*randu<vec>(1);

    // Gaussian distribution in space for particle position:
    double Xcenter = params->mesh.LX/2;
    double sigmaX  = params->mesh.LX/10;
    double Xnew = Xcenter  + (sigmaX)*sqrt( -2*log(R(0)) )*cos(phi(0));
    double dLX = abs(Xnew - Xcenter);

    // Check that Xnew is within bounds:
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

    IONS->X(ip,0) = Xnew;


    // 2- Generate random number:
    // 3- Extract BC variables from IONS->at(ii).p_BC
    // 4- Apply box-muller in physical space:
    // 	4.1 - Calculate candidate xnew and make sure it is within bounds
    // 	4.2 - Assign new particle position
    // 5- Apply box-muller in velocity space:
    // 	5.1 - Calculate new V
    //	5.2 - Project to field aligned coordinate
    //	5.3 - Calculate derived quantities
}

void PARTICLE_BC::particleReinjection(int ip, const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS)
{}
