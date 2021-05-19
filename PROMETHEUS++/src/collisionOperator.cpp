#include "collisionOperator.h"

collisionOperator::collisionOperator()
{}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // code
}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    // code
}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{}
