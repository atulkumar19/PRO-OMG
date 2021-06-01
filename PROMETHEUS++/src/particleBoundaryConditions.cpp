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

            // ==================================
            #pragma omp parallel for default(none) shared(params, IONS, aa, CS) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                // Check left boundary:
                if (IONS->at(aa).X(ii,0) < 0)
                {
                    IONS->at(aa).f1 = 1;
                    // IONS->at(aa).dE1 = kinetic energy;
                }

                if (IONS->at(aa).X(ii,0) > params->mesh.LX)
                {
                    IONS->at(aa).f2 = 1;
                    // IONS->at(aa).dE2 = kinetic energy;
                }
            }

        }

    } // close MPI

}

void PARTICLE_BC::checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}
