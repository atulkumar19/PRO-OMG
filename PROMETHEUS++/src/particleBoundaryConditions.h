#ifndef H_PARTICLEBOUNDARYCONDITIONS
#define H_PARTICLEBOUNDARYCONDITIONS

#include <iostream>
#include <cmath>
#include <vector>

#include "armadillo"
#include "structures.h"
#include "types.h"
#include "mpi_main.h"
#include "omp.h"

using namespace std;
using namespace arma;

class PARTICLE_BC
{
private:

public:
    PARTICLE_BC();
    void checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);
    void checkBoundaryAndFlag(const simulationParameters * params,const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

};

#endif
