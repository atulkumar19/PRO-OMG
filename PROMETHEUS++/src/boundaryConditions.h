// COPYRIGHT 2015-2019 LEOPOLDO CARBAJAL

/*	This file is part of PROMETHEUS++.

    PROMETHEUS++ is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    any later version.

    PROMETHEUS++ is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PROMETHEUS++.  If not, see <https://www.gnu.org/licenses/>.
*/

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
