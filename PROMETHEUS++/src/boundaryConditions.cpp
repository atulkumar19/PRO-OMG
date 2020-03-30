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


void backwardPBC_3D(cube * C){
	double NX(C->n_rows), NY(C->n_cols), NZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,NZ-2,NX-2,NY-2,NZ-2) += C->subcube(1,1,0,NX-2,NY-2,0);//Z=0 face
	C->subcube(1,1,0,NX-2,NY-2,0).fill(0);
	C->subcube(1,1,1,NX-2,NY-2,1) += C->subcube(1,1,NZ-1,NX-2,NY-2,NZ-1);//Z=NZ-1 face
	C->subcube(1,1,NZ-1,NX-2,NY-2,NZ-1).fill(0);
	C->subcube(NX-2,1,1,NX-2,NY-2,NZ-2) += C->subcube(0,1,1,0,NY-2,NZ-2);//X=0 face.
	C->subcube(0,1,1,0,NY-2,NZ-2).fill(0);
	C->subcube(1,1,1,1,NY-2,NZ-2) += C->subcube(NX-1,1,1,NX-1,NY-2,NZ-2);//X=NX-1 face.
	C->subcube(NX-1,1,1,NX-1,NY-2,NZ-2).fill(0);
	C->subcube(1,NY-2,1,NX-2,NY-2,NZ-2) += C->subcube(1,0,1,NX-2,0,NZ-2);//Y=0 face.
	C->subcube(1,0,1,NX-2,0,NZ-2).fill(0);
	C->subcube(1,1,1,NX-2,1,NZ-2) += C->subcube(1,NY-1,1,NX-2,NY-1,NZ-2);//Y=NY-1 face.
	C->subcube(1,NY-1,1,NX-2,NY-1,NZ-2).fill(0);

	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,NY-2,NZ-2,NX-2,NY-2,NZ-2) += C->subcube(1,0,0,NX-2,0,0);//(1)(2)->(8)(7)
	C->subcube(1,0,0,NX-2,0,0).fill(0);
	C->subcube(1,1,1,NX-2,1,1) += C->subcube(1,NY-1,NZ-1,NX-2,NY-1,NZ-1);//(1)(2)<-(8)(7)
	C->subcube(1,NY-1,NZ-1,NX-2,NY-1,NZ-1).fill(0);

	C->subcube(1,1,NZ-2,NX-2,1,NZ-2) += C->subcube(1,NY-1,0,NX-2,NY-1,0);//(4)(3)->(5)(6)
	C->subcube(1,NY-1,0,NX-2,NY-1,0).fill(0);
	C->subcube(1,NY-2,1,NX-2,NY-2,1) += C->subcube(1,0,NZ-1,NX-2,0,NZ-1);//(4)(3)<-(5)(6)
	C->subcube(1,0,NZ-1,NX-2,0,NZ-1).fill(0);

	C->subcube(NX-2,1,NZ-2,NX-2,NY-2,NZ-2) += C->subcube(0,1,0,0,NY-2,0);//(1)(4)->(6)(7)
	C->subcube(0,1,0,0,NY-2,0).fill(0);
	C->subcube(1,1,1,1,NY-2,1) += C->subcube(NX-1,1,NZ-1,NX-1,NY-2,NZ-1);//(1)(4)<-(6)(7)
	C->subcube(NX-1,1,NZ-1,NX-1,NY-2,NZ-1).fill(0);

	C->subcube(1,1,NZ-2,1,NY-2,NZ-2) += C->subcube(NX-1,1,0,NX-1,NY-2,0);//(2)(3)->(5)(8)
	C->subcube(NX-1,1,0,NX-1,NY-2,0).fill(0);
	C->subcube(NX-2,1,1,NX-2,NY-2,1) += C->subcube(0,1,NZ-1,0,NY-2,NZ-1);//(2)(3)<-(5)(8)
	C->subcube(0,1,NZ-1,0,NY-2,NZ-1).fill(0);

	C->subcube(NX-2,NY-2,1,NX-2,NY-2,NZ-2) += C->subcube(0,0,1,0,0,NZ-2);//(1)(5)->(3)(7)
	C->subcube(0,0,1,0,0,NZ-2).fill(0);
	C->subcube(1,1,1,1,1,NZ-2) += C->subcube(NX-1,NY-1,1,NX-1,NY-1,NZ-2);//(1)(5)<-(3)(7)
	C->subcube(NX-1,NY-1,1,NX-1,NY-1,NZ-2).fill(0);

	C->subcube(1,NY-2,1,1,NY-2,NZ-2) += C->subcube(NX-1,0,1,NX-1,0,NZ-2);//(2)(6)->(4)(8)
	C->subcube(NX-1,0,1,NX-1,0,NZ-2).fill(0);
	C->subcube(NX-2,1,1,NX-2,1,NZ-2) += C->subcube(0,NY-1,1,0,NY-1,NZ-2);//(2)(6)<-(4)(8)
	C->subcube(0,NY-1,1,0,NY-1,NZ-2).fill(0);

	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(1,1,1) += (*C)(NX-1,NY-1,NZ-1);//(1)<-(7)
	(*C)(NX-1,NY-1,NZ-1) = 0;
	(*C)(NX-2,1,1) += (*C)(0,NY-1,NZ-1);//(2)<-(8)
	(*C)(0,NY-1,NZ-1) = 0;
	(*C)(NX-2,NY-2,1) += (*C)(0,0,NZ-1);//(3)<-(5)
	(*C)(0,0,NZ-1) = 0;
	(*C)(1,NY-2,1) += (*C)(NX-1,0,NZ-1);//(4)<-(6)
	(*C)(NX-1,0,NZ-1) = 0;
	(*C)(1,1,NZ-2) += (*C)(NX-1,NY-1,0);//(5)<-(3)
	(*C)(NX-1,NY-1,0) = 0;
	(*C)(NX-2,1,NZ-2) += (*C)(0,NY-1,0);//(6)<-(4)
	(*C)(0,NY-1,0) = 0;
	(*C)(NX-2,NY-2,NZ-2) += (*C)(0,0,0);//(7)<-(1)
	(*C)(0,0,0) = 0;
	(*C)(1,NY-2,NZ-2) += (*C)(NX-1,0,0);//(8)<-(2)
	(*C)(NX-1,0,0) = 0;
}


void forwardPBC_1D_TOS(vec * C){
	double NX(C->n_elem);

	(*C)(1) = (*C)(NX-3);
	(*C)(0) = (*C)(NX-4);


	(*C)(NX-2) = (*C)(2);
	(*C)(NX-1) = (*C)(3);
}


void forwardPBC_1D(vec * C){
	double NX(C->n_elem);

	(*C)(0) = (*C)(NX-2);//X=0 face.
	(*C)(NX-1) = (*C)(1);//X=NX-1 face.

}


void forwardPBC_2D(mat * C){
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

void forwardPBC_3D(cube * C){
	double NX(C->n_rows), NY(C->n_cols), NZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,0,NX-2,NY-2,0) = C->subcube(1,1,NZ-2,NX-2,NY-2,NZ-2);//Z=0 face
	C->subcube(1,1,NZ-1,NX-2,NY-2,NZ-1) = C->subcube(1,1,1,NX-2,NY-2,1);//Z=NZ-1 face
	C->subcube(0,1,1,0,NY-2,NZ-2) = C->subcube(NX-2,1,1,NX-2,NY-2,NZ-2);//X=0 face.
	C->subcube(NX-1,1,1,NX-1,NY-2,NZ-2) = C->subcube(1,1,1,1,NY-2,NZ-2);//X=NX-1 face.
	C->subcube(1,0,1,NX-2,0,NZ-2) = C->subcube(1,NY-2,1,NX-2,NY-2,NZ-2);//Y=0 face.
	C->subcube(1,NY-1,1,NX-2,NY-1,NZ-2) = C->subcube(1,1,1,NX-2,1,NZ-2);//Y=NY-1 face.


	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,0,0,NX-2,0,0) = C->subcube(1,NY-2,NZ-2,NX-2,NY-2,NZ-2);//(1)(2)<-(8)(7)
	C->subcube(1,NY-1,NZ-1,NX-2,NY-1,NZ-1) = C->subcube(1,1,1,NX-2,1,1);//(1)(2)->(8)(7)

	C->subcube(1,NY-1,0,NX-2,NY-1,0) = C->subcube(1,1,NZ-2,NX-2,1,NZ-2);//(4)(3)<-(5)(6)
	C->subcube(1,0,NZ-1,NX-2,0,NZ-1) = C->subcube(1,NY-2,1,NX-2,NY-2,1);//(4)(3)->(5)(6)

	C->subcube(0,1,0,0,NY-2,0) = C->subcube(NX-2,1,NZ-2,NX-2,NY-2,NZ-2);//(1)(4)<-(6)(7)
	C->subcube(NX-1,1,NZ-1,NX-1,NY-2,NZ-1) = C->subcube(1,1,1,1,NY-2,1);//(1)(4)->(6)(7)

	C->subcube(NX-1,1,0,NX-1,NY-2,0) = C->subcube(1,1,NZ-2,1,NY-2,NZ-2);//(2)(3)<-(5)(8)
	C->subcube(0,1,NZ-1,0,NY-2,NZ-1) = C->subcube(NX-2,1,1,NX-2,NY-2,1);//(2)(3)->(5)(8)

	C->subcube(0,0,1,0,0,NZ-2) = C->subcube(NX-2,NY-2,1,NX-2,NY-2,NZ-2);//(1)(5)<-(3)(7)
	C->subcube(NX-1,NY-1,1,NX-1,NY-1,NZ-2) = C->subcube(1,1,1,1,1,NZ-2);//(1)(5)->(3)(7)

	C->subcube(NX-1,0,1,NX-1,0,NZ-2) = C->subcube(1,NY-2,1,1,NY-2,NZ-2);//(2)(6)<-(4)(8)
	C->subcube(0,NY-1,1,0,NY-1,NZ-2) = C->subcube(NX-2,1,1,NX-2,1,NZ-2);//(2)(6)->(4)(8)


	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(NX-1,NY-1,NZ-1) = (*C)(1,1,1);//(1)->(7)
	(*C)(0,NY-1,NZ-1) = (*C)(NX-2,1,1);//(2)->(8)
	(*C)(0,0,NZ-1) = (*C)(NX-2,NY-2,1);//(3)->(5)
	(*C)(NX-1,0,NZ-1) = (*C)(1,NY-2,1);//(4)->(6)
	(*C)(NX-1,NY-1,0) = (*C)(1,1,NZ-2);//(5)->(3)
	(*C)(0,NY-1,0) = (*C)(NX-2,1,NZ-2);//(6)->(4)
	(*C)(0,0,0) = (*C)(NX-2,NY-2,NZ-2);//(7)->(1)
	(*C)(NX-1,0,0) = (*C)(1,NY-2,NZ-2);//(8)->(2)

}


void restoreVector(vec * C){

	double NX(C->n_elem);

	(*C)(0) = 0;//X=0 face.
	(*C)(NX-1) = 0;//X=NX-1 face.

}

void restoreMatrix(mat * C){


}

void restoreCube(cube * C){

	double NX(C->n_rows), NY(C->n_cols), NZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,0,NX-2,NY-2,0).fill(0);//Z=0 face
	C->subcube(1,1,NZ-1,NX-2,NY-2,NZ-1).fill(0);//Z=NZ-1 face
	C->subcube(0,1,1,0,NY-2,NZ-2).fill(0);//X=0 face.
	C->subcube(NX-1,1,1,NX-1,NY-2,NZ-2).fill(0);//X=NX-1 face.
	C->subcube(1,0,1,NX-2,0,NZ-2).fill(0);//Y=0 face.
	C->subcube(1,NY-1,1,NX-2,NY-1,NZ-2).fill(0);//Y=NY-1 face.


	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,0,0,NX-2,0,0).fill(0);//(1)(2)<-(8)(7)
	C->subcube(1,NY-1,NZ-1,NX-2,NY-1,NZ-1).fill(0);//(1)(2)->(8)(7)

	C->subcube(1,NY-1,0,NX-2,NY-1,0).fill(0);//(4)(3)<-(5)(6)
	C->subcube(1,0,NZ-1,NX-2,0,NZ-1).fill(0);//(4)(3)->(5)(6)

	C->subcube(0,1,0,0,NY-2,0).fill(0);//(1)(4)<-(6)(7)
	C->subcube(NX-1,1,NZ-1,NX-1,NY-2,NZ-1).fill(0);//(1)(4)->(6)(7)

	C->subcube(NX-1,1,0,NX-1,NY-2,0).fill(0);//(2)(3)<-(5)(8)
	C->subcube(0,1,NZ-1,0,NY-2,NZ-1).fill(0);//(2)(3)->(5)(8)

	C->subcube(0,0,1,0,0,NZ-2).fill(0);//(1)(5)<-(3)(7)
	C->subcube(NX-1,NY-1,1,NX-1,NY-1,NZ-2).fill(0);//(1)(5)->(3)(7)

	C->subcube(NX-1,0,1,NX-1,0,NZ-2).fill(0);//(2)(6)<-(4)(8)
	C->subcube(0,NY-1,1,0,NY-1,NZ-2).fill(0);//(2)(6)->(4)(8)


	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(NX-1,NY-1,NZ-1) = 0;//(1)->(7)
	(*C)(0,NY-1,NZ-1) = 0;//(2)->(8)
	(*C)(0,0,NZ-1) = 0;//(3)->(5)
	(*C)(NX-1,0,NZ-1) = 0;//(4)->(6)
	(*C)(NX-1,NY-1,0) = 0;//(5)->(3)
	(*C)(0,NY-1,0) = 0;//(6)->(4)
	(*C)(0,0,0) = 0;//(7)->(1)
	(*C)(NX-1,0,0) = 0;//(8)->(2)

}
