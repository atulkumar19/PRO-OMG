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
    void interpolateIonMoments(const simulationParameters * params, oneDimensional::ionSpecies * IONS);
    void interpolateIonMoments(const simulationParameters * params, twoDimensional::ionSpecies * IONS);
    void interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * field, arma::vec * F);
    void interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * field, arma::mat * F);
    void fill4Ghosts(arma::vec * v);
    void mcOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS);
    void cartesian2Spherical(const double * wx, const double * wy, const double * wz, double * w, double * xi, double * phi);

    public:
    collisionOperator();
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);
};

#endif
