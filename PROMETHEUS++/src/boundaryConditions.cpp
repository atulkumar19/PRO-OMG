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

#include "boundaryConditions.h"

void backwardPBC_1D(vec * C){
	double NX(C->n_elem);

	(*C)(NX-2) += (*C)(0);//X=0 face.
	(*C)(0) = 0;

	(*C)(1) += (*C)(NX-1);//X=0 face.
	(*C)(NX-1) = 0;

}


void backwardPBC_2D(mat * C){


}



void fillGhosts(vec * C){
	double NX(C->n_elem);

	(*C)(0) = (*C)(NX-2);//X=0 face.
	(*C)(NX-1) = (*C)(1);//X=NX-1 face.

}


void fillGhosts(oneDimensional::fields * F){
	fillGhosts(&F->E.X);
	fillGhosts(&F->E.Y);
	fillGhosts(&F->E.Z);

	fillGhosts(&F->B.X);
	fillGhosts(&F->B.Y);
	fillGhosts(&F->B.Z);
}


void fillGhosts(mat * C){
	double NX(C->n_rows);
	double NY(C->n_cols);

	C->submat(0,1,0,NY-2) = C->submat(NX-2,1,NX-2,NY-2); // Left ghost cells along x-axis
	C->submat(NX-1,1,NX-1,NY-2) = C->submat(1,1,1,NY-2); // Right ghost cells along x-axis

	C->submat(1,0,NX-2,0) = C->submat(1,NY-2,NX-2,NY-2); // Left ghost cells along y-axis
	C->submat(1,NY-1,NX-2,NY-1) = C->submat(1,1,NX-2,1); // Right ghost cells along x-axis

	// Corners
	(*C)(0,0) = (*C)(NX-2,NY-2);
	(*C)(NX-1,0) = (*C)(1,NY-2);
	(*C)(0,NY-1) = (*C)(NX-2,1);
	(*C)(NX-1,NY-1) = (*C)(1,1);
}


void fillGhosts(twoDimensional::fields * F){
	fillGhosts(&F->E.X);
	fillGhosts(&F->E.Y);
	fillGhosts(&F->E.Z);

	fillGhosts(&F->B.X);
	fillGhosts(&F->B.Y);
	fillGhosts(&F->B.Z);
}


void setGhostsToZero(vec * C){

	double NX(C->n_elem);

	(*C)(0) = 0;//X=0 face.
	(*C)(NX-1) = 0;//X=NX-1 face.

}

void setGhostsToZero(mat * C){


}
