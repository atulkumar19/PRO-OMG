#include <math.h>
#include "rfOperator.h"

using namespace std;

rfOperator::rfOperator()
{}

rfOperator::checkResonanceCondition(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    // PARTICLE MPI sentinel
    //  loop over all Species
    //      loop over all Particles
    //          flag f3 and assign to IONS->at(ss).f3(ii)
}

rfOperator::checkResonanceCondition(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}

rfOperator::calculateErf(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Apply unit electric field:
    //  - MPI particle sentinel
    //    double E_RF = 1;
    //      - Loop over all Species
    //        {
    //          double Ma = IONS->at(ss).M;
    //          - Loop over all Particles using openMP
    //            if (f3(ii) == 1)
    //            {
    //              - interpolateField(ii,EB->B.X,IONS->at(ss).Bp(ii,0))
    //              - interpolateField(ii,EB->B.Y,IONS->at(ss).Bp(ii,1))
    //              - Calculate particle RF phase factor and assign to IONS->at(ss).RF.phase(ii)
    //              - advanceVelocity(ii,params,IONS->at(ss),arma::vec V,E_RF); Advance ion velocity with unit Erf
    //                  - Do not advance IONS->at(ss).V
    //              - Calculate udE3 and assign to IONS->at(ss).RF.udE3(ii) = 0.5*Ma*dot(V);
    //            }
    //        }

    // Calculate uE3 (unit P_RF)
    //  - MPI particle sentinel
    //    double udE3_total = 0;
    //    {
    //      - Loop over all Species
    //        {
    //          - Loop over all Particles
    //           if (f3(ii) == 1)
    //           {
    //             Accumulate udE3 to obtain IONS->at(ss).RF.uE3
    //           }
    //          MPI_AllreduceDouble on IONS->at(ss).RF.uE3
    //        }
    //        udE3_total += IONS->at(ss).RF.uE3;
    //     }

    // Calculate E_rf:
    // Erf = sqrt(params->P_RF/udE3_total)
    // assign to params->RF.E_RF

    // To do list:
    // 0- Create time vector and keep track of current time in params.currentTime
    // 1- Add RF parameters to input file: freq, P_rf, kpar, kper, xRes1, xRes2, pHandedness
    // 2- Create rfParams structure in params to hold input file Data
    //      2.1 - Create E_rf to hold current value of E_RF based on P_rf
    // 3-  Create RF structure in IONS to hold particle-defined quantities:
    //      3.1 - rfphaseterm, udE3, uE3, f3
    // 4- Create/copy MPI_Allreduce from particleBoundaryConditions.cpp
    // 5- Create advanceVelocity private method, they are single particle operations
    // 6- Create interpolateField private method, single particle operations
    // 7- Update input file in PRO++ files

}

rfOperator::calculateErf(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // Particle MPI sentinel
    //      loop over all Species
    //      {
    //
    //          Loop over all particles
    //      }
}
}

rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{

}

rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}
