#ifndef H_COLLISIONOPERATOR
#define H_COLLISIONOPERATOR

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

class collisionOperator
{  
    public:
       
    // Main method:
    void ApplyCollisionOperator(const characteristicScales * CS, const simulationParameters * params, const oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS); 
    
};

#endif