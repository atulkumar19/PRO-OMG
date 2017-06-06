#ifndef H_BOUNDARYCONDITIONS
#define H_BOUNDARYCONDITIONS

#include <iostream>
#include <cmath>
#include <vector>

#include "armadillo"
#include "structures.h"

using namespace std;
using namespace arma;

//We can use different namespaces in order to include this operations for different boundary conditions

void restoreVector(vec * C);

void restoreMatrix(mat * C);

void restoreCube(cube * C);



void backwardPBC_1D(vec * C);//Apply periodic boundary conditions. The variable is pushed from the ghost nodes towards the real nodes.

void backwardPBC_2D(mat * C);//Apply periodic boundary conditions. The variable is pushed from the ghost nodes towards the real nodes.

void backwardPBC_3D(cube * C);//Apply periodic boundary conditions. The variable is pushed from the ghost nodes towards the real nodes.


void forwardPBC_1D_TOS(vec * C);

void forwardPBC_1D(vec * C);

void forwardPBC_2D(mat * C);

void forwardPBC_3D(cube * C);//Apply periodic boundary conditions. The variable is pushed from the real nodes towards the ghost nodes.


#endif
