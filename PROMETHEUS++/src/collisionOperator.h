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
    void interpolateIonMoments(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, int a, int b);
    void interpolateIonMoments(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, int a, int b);
    void interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, arma::vec * F_p);
    void interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * F_m, arma::mat * F_p);
    void fill4Ghosts(arma::vec * v);
    void cartesian2Spherical(double * wx, double * wy, double * wz, double * w, double * xi, double * phi);
    void mcOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS);

    public:
    collisionOperator();
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);
};

#endif
