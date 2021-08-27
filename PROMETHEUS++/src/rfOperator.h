#ifndef H_RFOPERATOR
#define H_RFOPERATOR

#include <iostream>
#include <cmath>
#include <vector>
#define ARMA_ALLOW_FAKE_GCC
#include "armadillo"
#include "structures.h"
#include "types.h"
#include "boundaryConditions.h"
#include "mpi_main.h"
#include "omp.h"

using namespace std;
using namespace arma;

class rfOperator
{
private:
  void applyUnitEField(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);
  void applyUnitEField(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);

  void calculateUnitPrf(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS);
  void calculateUnitPrf(const simulationParameters * params, const characteristicScales * CS, twoDimensional::ionSpecies * IONS);

  void interpolateScalarField(int ii,const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, double * F_p);
  void interpolateScalarField(int ii,const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::vec * F_m, double * F_p);

  void advanceVelocity(int ii, const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::rowvec * V, double E_RF, double DT);
  void advanceVelocity(int ii, const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::rowvec * V, double E_RF, double DT);

  void fill4Ghosts(arma::vec * v);

  void MPI_AllreduceDouble(const simulationParameters * params, double * v);

public:
  rfOperator();
  void checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
  void checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);

  void calculateErf(simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);
  void calculateErf(simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

  void applyRfOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);
  void applyRfOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

  void calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
  void calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);
};

#endif
