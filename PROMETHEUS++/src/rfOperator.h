#ifndef H_RFOPERATOR
#define H_RFOPERATOR

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

class rfOperator
{
private:
  void applyUnitEField(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, oneDimensional::ionSpecies * IONS);
  void applyUnitEField(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, twoDimensional::ionSpecies * IONS);

  void calculateUnitPrf(const simulationParameters * params, const characteristicScales * CS, oneDimensional::ionSpecies * IONS);
  void calculateUnitPrf(const simulationParameters * params, const characteristicScales * CS, twoDimensional::ionSpecies * IONS);

public:
  rfOperator();
  void checkResonanceCondition(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
  void checkResonanceCondition(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);

  void calculateErf(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);
  void calculateErf(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

  void applyRfOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS);
  void applyRfOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS);

  void calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS);
  void calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS);
};

#endif
