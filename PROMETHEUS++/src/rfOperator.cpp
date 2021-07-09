#include <math.h>
#include "rfOperator.h"

using namespace std;

rfOperator::rfOperator()
{}

void rfOperator::checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        // RF boundary locations:
        // ======================
        double x1 = params->RF.x1;
        double x2 = params->RF.x2;

        // Loop over all ion species:
        // ==========================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles is "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(x1, x2, params, IONS, aa, CS, std::cout) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                double X = IONS->at(aa).X(ii,0);

                if ( (X > x1) && (X < x2) )
                {
                    IONS->at(aa).f3(ii) = 1;
                }

            } // Particle loop

        } // Species loop

    } // Particle MPI guard

}

void rfOperator::checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::calculateErf(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Apply unit electric field:
    // ==========================
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        // RF electric field to apply:
        // ======================
        double E_RF = 1/CS->eField;

        // Radius of flux surface:
        // =======================
        double R0 = params->geometry.r2*CS->length;

        // Reference magnetic field:
        // =========================
        double BX0 = params->em_IC.BX;

        // RF parameters:
        // ==============
        double kper = params->RF.kper;
        double kpar = params->RF.kpar;
        double w    = params->RF.freq*2*M_PI;

        // Current time:
        // ============
        // double t = params->currentTime*CS->time;

        // Loop over all ion species:
        // ==========================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles in "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Mass of "aa" species:
            // =====================
            double Ma = IONS->at(aa).M;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(Ma, params, IONS, EB, aa, CS, std::cout) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                if ( IONS->at(aa).f3(ii) == 1 )
                {
                    // Interpolate magnetic field at particle position:
                    // ================================================
                    // BX:
                    double BXp;
                    // interpolateField(ii,&EB->B.X,&BXp); ******* need to be created in private function
                    IONS->at(aa).B(ii,0) = BXp;

                    // BY:
                    double BYp;
                    // interpolateField(ii,&EB->B.Y,&BYp);
                    IONS->at(aa).B(ii,1) = BYp;

                    // RF phase term:
                    // ==============
                    // double cosPhi = ? see PIC.cpp
                    // double rho = ?*CS->length; see PIC.cpp [m]
                    // double r = R0*sqrt(BX0/BXp) + rho*cosPhi;  [m]
                    // IONS->at(aa).p_RF.phase(ii) = kper*r + kpar*x - w*t;

                    // dont forget to initialize and RF.phase with NSP numner of particles
                    // And to create RF.phase in IONS

                    // Advance ion velocity due to RF electric field:
                    // ==============================================
                    // arma::vec V = zeros(1,3);
                    // advanceVelocity(ii,params,&IONS->at(aa),&V,E_RF,params->DT);

                    // Calculate udE3:
                    // ===============
                    // IONS->at(aa).RF.udE3(ii) = 0.5*Ma*dot(V)*CS->energy

                } // f3 guard

            } // Particle loop

        } // Species Loop

    } // Particle MPI guard

    // Calculate uE3:
    // ==============
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
    }

    // Calculate Erf:
    // ==============
    // params->RF.Erf = sqrt(params->Prf/uE3_total) [V/m]


    // Descroptopm of procedure:

    //  - MPI particle sentinel
    //    double E_RF = 1/CS->Efield;;
    //      - Loop over all Species
    //        {
    //          double Ma = IONS->at(aa).M;
    //          - Loop over all Particles using openMP
    //            if (f3(ii) == 1)
    //            {
    //              - interpolateField(ii,EB->B.X,IONS->at(aa).Bp(ii,0))
    //              - interpolateField(ii,EB->B.Y,IONS->at(aa).Bp(ii,1))
    //              - Calculate particle RF phase factor and assign to IONS->at(aa).RF.phase(ii)
    //              - advanceVelocity(ii,params,IONS->at(aa),arma::vec V,E_RF,params->DT); Advance ion velocity with unit Erf
    //                  - Do not advance IONS->at(aa).V
    //              - Calculate udE3 and assign to IONS->at(aa).RF.udE3(ii) = 0.5*Ma*dot(V)*CS->energy; [J]
    //            }
    //        }

    // Calculate uE3:
    // ==============
    //  - MPI particle sentinel
    //    double uE3_total = 0;
    //    double DT = params->DT;
    //    {
    //      - Loop over all Species
    //        double alpha = IONS->at(aa).NCP;
    //        {
    //          IONS->at(aa).RF.uE3 = 0;
    //          - Loop over all Particles
    //           if (f3(ii) == 1)
    //           {
    //             double a    = *IONS->at(aa).a(ii);
    //             double udE3 = *IONS->at(aa).RF.udE3(ii);
    //             IONS->at(aa).RF.uE3 += (alpha/DT)*a*udE3; [W]
    //           }
    //          MPI_AllreduceDouble on IONS->at(aa).RF.uE3
    //        }
    //        uE3_total += IONS->at(aa).RF.uE3; [W]
    //     }



    // To do list:
    // 0- Create time vector and keep track of current time in params.currentTime
    // 1- In initialize.cpp read RF parameters into varianles in params
    // 2- In initiazlie.cpp give dimensions and intialize IONS->at(aa).RF.phase etc, see initializeParticlesArrays
    // 3-  Create RF structure in IONS to hold particle-defined quantities:
    //      3.1 - rfphaseterm, udE3, uE3, f3
    // 4- Create/copy MPI_Allreduce from particleBoundaryConditions.cpp
    // 5- Create advanceVelocity private method, they are single particle operations
    // 6- Create interpolateField private method, single particle operations
    // 7 - RF.phase, udE3, uE3 needs to be initialized and given dimensions (1,NSP): IONS->RF.phase.zeros(IONS->NSP);

}

void rfOperator::calculateErf(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Particle MPI sentinel
    //      loop over all Species
    //      double Ma = IONS->at(aa).M;
    //      loop over all particles
    //      if (f3 == 1)
    //      {
    //          arma::vec V(1,3);
    //          double E_RF = params->RF.E_RF/CS->Efield; [normalized]
    //          advanceVelocity(ii,IONS->at(aa),V,params->RF.E_RF,params->DT)
    //          IONS->at(aa).V.row(ii) = V;
    //          IONS->at(aa).RF.dE3(ii) = 0.5*Ma*dot(V)*CS->energy; [J]
    //          IONS->at(aa).f3(ii) = 0;
    //      }
}

void rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    // Particle MPI sentinel
    //      params->RF.E3 = 0;
    //      double DT = params->DT;
    //      loop over all Species
    //      {
    //          double alpha = IONS->at(aa).NCP;
    //          IONS->at(aa).RF.E3 = 0;
    //          loop over all particles
    //          if (f3 == 1)
    //          {
    //              double a   = IONS->at(aa).a(ii);
    //              double dE3 = IONS->at(aa).RF.dE3(ii); [J]
    //              IONS->at(aa).RF.E3 += (alpha/DT)*a*dE3; [W]
    //          }
    //          MPI_AllreduceDouble on IONS->at(aa).RF.E3
    //        }
    //        params->RF.E3 += IONS->at(aa).RF.E3; [W]
}

void rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}
