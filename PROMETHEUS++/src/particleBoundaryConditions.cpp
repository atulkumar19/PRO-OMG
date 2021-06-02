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
