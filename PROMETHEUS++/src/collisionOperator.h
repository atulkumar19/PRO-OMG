#ifndef H_COLLISIONOPERATOR
#define H_COLLISIONOPERATOR

#include <iostream>
#include <cmath>
#include <vector>
#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"
#include "structures.h"
#include "types.h"
#include "mpi_main.h"
#include "omp.h"

using namespace std;
using namespace arma;

class collisionOperator
{
private:
    void interpolateIonMoments(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, int a, int b);
    void interpolateIonMoments(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, int a, int b);
    void interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, arma::vec * F_p);
    void interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * F_m, arma::mat * F_p);
    void fill4Ghosts(arma::vec * v);

    void cartesian2Spherical(double * wx, double * wy, double * wz, double * w, double * xi, double * phi);
    void Spherical2Cartesian(double * w, double * xi, double * phi, double * wx, double * wy, double * wz);

    void u_CollisionOperator(double * w, double xab, double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt);
    void xi_CollisionOperator(double * xi, double xab, double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt);
    void phi_CollisionOperator(double * phi, double xi, double xab, double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt);

    double nu_E(double xab, double nb, double Tb, double Mb, double Zb, double Za, double Ma, int energyOperatorModel);
    double nu_D(double xab, double nb, double Tb, double Mb, double Zb, double Za, double Ma);
    double nu_ab0(double nb, double Tb, double Mb, double Zb, double Za, double Ma);
    double logA(double nb, double Tb);
    double Gb(double xab);
    double erfp(double xab);
    double erfpp(double xab);
    double E_nuE_d_nu_E_dE(double xab);

public:
    collisionOperator();
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
    void ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);
};

#endif
