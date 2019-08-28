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

	double dimX(C->n_elem);

	(*C)(dimX-2) += (*C)(0);//X=0 face.
	(*C)(0) = 0;

	(*C)(1) += (*C)(dimX-1);//X=0 face.
	(*C)(dimX-1) = 0;

}


void backwardPBC_2D(mat * C){


}

void backwardPBC_3D(cube * C){

	double dimX(C->n_rows), dimY(C->n_cols), dimZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,dimZ-2,dimX-2,dimY-2,dimZ-2) += C->subcube(1,1,0,dimX-2,dimY-2,0);//Z=0 face
	C->subcube(1,1,0,dimX-2,dimY-2,0).fill(0);
	C->subcube(1,1,1,dimX-2,dimY-2,1) += C->subcube(1,1,dimZ-1,dimX-2,dimY-2,dimZ-1);//Z=dimZ-1 face
	C->subcube(1,1,dimZ-1,dimX-2,dimY-2,dimZ-1).fill(0);
	C->subcube(dimX-2,1,1,dimX-2,dimY-2,dimZ-2) += C->subcube(0,1,1,0,dimY-2,dimZ-2);//X=0 face.
	C->subcube(0,1,1,0,dimY-2,dimZ-2).fill(0);
	C->subcube(1,1,1,1,dimY-2,dimZ-2) += C->subcube(dimX-1,1,1,dimX-1,dimY-2,dimZ-2);//X=dimX-1 face.
	C->subcube(dimX-1,1,1,dimX-1,dimY-2,dimZ-2).fill(0);
	C->subcube(1,dimY-2,1,dimX-2,dimY-2,dimZ-2) += C->subcube(1,0,1,dimX-2,0,dimZ-2);//Y=0 face.
	C->subcube(1,0,1,dimX-2,0,dimZ-2).fill(0);
	C->subcube(1,1,1,dimX-2,1,dimZ-2) += C->subcube(1,dimY-1,1,dimX-2,dimY-1,dimZ-2);//Y=dimY-1 face.
	C->subcube(1,dimY-1,1,dimX-2,dimY-1,dimZ-2).fill(0);

	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,dimY-2,dimZ-2,dimX-2,dimY-2,dimZ-2) += C->subcube(1,0,0,dimX-2,0,0);//(1)(2)->(8)(7)
	C->subcube(1,0,0,dimX-2,0,0).fill(0);
	C->subcube(1,1,1,dimX-2,1,1) += C->subcube(1,dimY-1,dimZ-1,dimX-2,dimY-1,dimZ-1);//(1)(2)<-(8)(7)
	C->subcube(1,dimY-1,dimZ-1,dimX-2,dimY-1,dimZ-1).fill(0);

	C->subcube(1,1,dimZ-2,dimX-2,1,dimZ-2) += C->subcube(1,dimY-1,0,dimX-2,dimY-1,0);//(4)(3)->(5)(6)
	C->subcube(1,dimY-1,0,dimX-2,dimY-1,0).fill(0);
	C->subcube(1,dimY-2,1,dimX-2,dimY-2,1) += C->subcube(1,0,dimZ-1,dimX-2,0,dimZ-1);//(4)(3)<-(5)(6)
	C->subcube(1,0,dimZ-1,dimX-2,0,dimZ-1).fill(0);

	C->subcube(dimX-2,1,dimZ-2,dimX-2,dimY-2,dimZ-2) += C->subcube(0,1,0,0,dimY-2,0);//(1)(4)->(6)(7)
	C->subcube(0,1,0,0,dimY-2,0).fill(0);
	C->subcube(1,1,1,1,dimY-2,1) += C->subcube(dimX-1,1,dimZ-1,dimX-1,dimY-2,dimZ-1);//(1)(4)<-(6)(7)
	C->subcube(dimX-1,1,dimZ-1,dimX-1,dimY-2,dimZ-1).fill(0);

	C->subcube(1,1,dimZ-2,1,dimY-2,dimZ-2) += C->subcube(dimX-1,1,0,dimX-1,dimY-2,0);//(2)(3)->(5)(8)
	C->subcube(dimX-1,1,0,dimX-1,dimY-2,0).fill(0);
	C->subcube(dimX-2,1,1,dimX-2,dimY-2,1) += C->subcube(0,1,dimZ-1,0,dimY-2,dimZ-1);//(2)(3)<-(5)(8)
	C->subcube(0,1,dimZ-1,0,dimY-2,dimZ-1).fill(0);

	C->subcube(dimX-2,dimY-2,1,dimX-2,dimY-2,dimZ-2) += C->subcube(0,0,1,0,0,dimZ-2);//(1)(5)->(3)(7)
	C->subcube(0,0,1,0,0,dimZ-2).fill(0);
	C->subcube(1,1,1,1,1,dimZ-2) += C->subcube(dimX-1,dimY-1,1,dimX-1,dimY-1,dimZ-2);//(1)(5)<-(3)(7)
	C->subcube(dimX-1,dimY-1,1,dimX-1,dimY-1,dimZ-2).fill(0);

	C->subcube(1,dimY-2,1,1,dimY-2,dimZ-2) += C->subcube(dimX-1,0,1,dimX-1,0,dimZ-2);//(2)(6)->(4)(8)
	C->subcube(dimX-1,0,1,dimX-1,0,dimZ-2).fill(0);
	C->subcube(dimX-2,1,1,dimX-2,1,dimZ-2) += C->subcube(0,dimY-1,1,0,dimY-1,dimZ-2);//(2)(6)<-(4)(8)
	C->subcube(0,dimY-1,1,0,dimY-1,dimZ-2).fill(0);

	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(1,1,1) += (*C)(dimX-1,dimY-1,dimZ-1);//(1)<-(7)
	(*C)(dimX-1,dimY-1,dimZ-1) = 0;
	(*C)(dimX-2,1,1) += (*C)(0,dimY-1,dimZ-1);//(2)<-(8)
	(*C)(0,dimY-1,dimZ-1) = 0;
	(*C)(dimX-2,dimY-2,1) += (*C)(0,0,dimZ-1);//(3)<-(5)
	(*C)(0,0,dimZ-1) = 0;
	(*C)(1,dimY-2,1) += (*C)(dimX-1,0,dimZ-1);//(4)<-(6)
	(*C)(dimX-1,0,dimZ-1) = 0;
	(*C)(1,1,dimZ-2) += (*C)(dimX-1,dimY-1,0);//(5)<-(3)
	(*C)(dimX-1,dimY-1,0) = 0;
	(*C)(dimX-2,1,dimZ-2) += (*C)(0,dimY-1,0);//(6)<-(4)
	(*C)(0,dimY-1,0) = 0;
	(*C)(dimX-2,dimY-2,dimZ-2) += (*C)(0,0,0);//(7)<-(1)
	(*C)(0,0,0) = 0;
	(*C)(1,dimY-2,dimZ-2) += (*C)(dimX-1,0,0);//(8)<-(2)
	(*C)(dimX-1,0,0) = 0;

	/*	for(int kk=0;kk<dimZ;kk++){
	 for(int jj=0;jj<dimY;jj++){
	 for(int ii=0;ii<dimX;ii++)
	 N2 += (*C)(ii,jj,kk);
	 }
	 }

	 cout << "The total density is: " << scientific << N << '\n';
	 cout << "The FINAL density is: " << scientific << N2 << '\n';
	 */

}


void forwardPBC_1D_TOS(vec * C){
	double dimX(C->n_elem);

	(*C)(1) = (*C)(dimX-3);
	(*C)(0) = (*C)(dimX-4);


	(*C)(dimX-2) = (*C)(2);
	(*C)(dimX-1) = (*C)(3);
}


void forwardPBC_1D(vec * C){

	double dimX(C->n_elem);

	(*C)(0) = (*C)(dimX-2);//X=0 face.
	(*C)(dimX-1) = (*C)(1);//X=dimX-1 face.

}

void forwardPBC_2D(mat * C){



}

void forwardPBC_3D(cube * C){

	double dimX(C->n_rows), dimY(C->n_cols), dimZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,0,dimX-2,dimY-2,0) = C->subcube(1,1,dimZ-2,dimX-2,dimY-2,dimZ-2);//Z=0 face
	C->subcube(1,1,dimZ-1,dimX-2,dimY-2,dimZ-1) = C->subcube(1,1,1,dimX-2,dimY-2,1);//Z=dimZ-1 face
	C->subcube(0,1,1,0,dimY-2,dimZ-2) = C->subcube(dimX-2,1,1,dimX-2,dimY-2,dimZ-2);//X=0 face.
	C->subcube(dimX-1,1,1,dimX-1,dimY-2,dimZ-2) = C->subcube(1,1,1,1,dimY-2,dimZ-2);//X=dimX-1 face.
	C->subcube(1,0,1,dimX-2,0,dimZ-2) = C->subcube(1,dimY-2,1,dimX-2,dimY-2,dimZ-2);//Y=0 face.
	C->subcube(1,dimY-1,1,dimX-2,dimY-1,dimZ-2) = C->subcube(1,1,1,dimX-2,1,dimZ-2);//Y=dimY-1 face.


	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,0,0,dimX-2,0,0) = C->subcube(1,dimY-2,dimZ-2,dimX-2,dimY-2,dimZ-2);//(1)(2)<-(8)(7)
	C->subcube(1,dimY-1,dimZ-1,dimX-2,dimY-1,dimZ-1) = C->subcube(1,1,1,dimX-2,1,1);//(1)(2)->(8)(7)

	C->subcube(1,dimY-1,0,dimX-2,dimY-1,0) = C->subcube(1,1,dimZ-2,dimX-2,1,dimZ-2);//(4)(3)<-(5)(6)
	C->subcube(1,0,dimZ-1,dimX-2,0,dimZ-1) = C->subcube(1,dimY-2,1,dimX-2,dimY-2,1);//(4)(3)->(5)(6)

	C->subcube(0,1,0,0,dimY-2,0) = C->subcube(dimX-2,1,dimZ-2,dimX-2,dimY-2,dimZ-2);//(1)(4)<-(6)(7)
	C->subcube(dimX-1,1,dimZ-1,dimX-1,dimY-2,dimZ-1) = C->subcube(1,1,1,1,dimY-2,1);//(1)(4)->(6)(7)

	C->subcube(dimX-1,1,0,dimX-1,dimY-2,0) = C->subcube(1,1,dimZ-2,1,dimY-2,dimZ-2);//(2)(3)<-(5)(8)
	C->subcube(0,1,dimZ-1,0,dimY-2,dimZ-1) = C->subcube(dimX-2,1,1,dimX-2,dimY-2,1);//(2)(3)->(5)(8)

	C->subcube(0,0,1,0,0,dimZ-2) = C->subcube(dimX-2,dimY-2,1,dimX-2,dimY-2,dimZ-2);//(1)(5)<-(3)(7)
	C->subcube(dimX-1,dimY-1,1,dimX-1,dimY-1,dimZ-2) = C->subcube(1,1,1,1,1,dimZ-2);//(1)(5)->(3)(7)

	C->subcube(dimX-1,0,1,dimX-1,0,dimZ-2) = C->subcube(1,dimY-2,1,1,dimY-2,dimZ-2);//(2)(6)<-(4)(8)
	C->subcube(0,dimY-1,1,0,dimY-1,dimZ-2) = C->subcube(dimX-2,1,1,dimX-2,1,dimZ-2);//(2)(6)->(4)(8)


	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(dimX-1,dimY-1,dimZ-1) = (*C)(1,1,1);//(1)->(7)
	(*C)(0,dimY-1,dimZ-1) = (*C)(dimX-2,1,1);//(2)->(8)
	(*C)(0,0,dimZ-1) = (*C)(dimX-2,dimY-2,1);//(3)->(5)
	(*C)(dimX-1,0,dimZ-1) = (*C)(1,dimY-2,1);//(4)->(6)
	(*C)(dimX-1,dimY-1,0) = (*C)(1,1,dimZ-2);//(5)->(3)
	(*C)(0,dimY-1,0) = (*C)(dimX-2,1,dimZ-2);//(6)->(4)
	(*C)(0,0,0) = (*C)(dimX-2,dimY-2,dimZ-2);//(7)->(1)
	(*C)(dimX-1,0,0) = (*C)(1,dimY-2,dimZ-2);//(8)->(2)

}


void restoreVector(vec * C){

	double dimX(C->n_elem);

	(*C)(0) = 0;//X=0 face.
	(*C)(dimX-1) = 0;//X=dimX-1 face.

}

void restoreMatrix(mat * C){


}

void restoreCube(cube * C){

	double dimX(C->n_rows), dimY(C->n_cols), dimZ(C->n_slices);

	//faces of the cube
	C->subcube(1,1,0,dimX-2,dimY-2,0).fill(0);//Z=0 face
	C->subcube(1,1,dimZ-1,dimX-2,dimY-2,dimZ-1).fill(0);//Z=dimZ-1 face
	C->subcube(0,1,1,0,dimY-2,dimZ-2).fill(0);//X=0 face.
	C->subcube(dimX-1,1,1,dimX-1,dimY-2,dimZ-2).fill(0);//X=dimX-1 face.
	C->subcube(1,0,1,dimX-2,0,dimZ-2).fill(0);//Y=0 face.
	C->subcube(1,dimY-1,1,dimX-2,dimY-1,dimZ-2).fill(0);//Y=dimY-1 face.


	//edges (1)(2)<->(8)(7), (4)(3)<->(5)(6), (1)(4)<->(6)(7), (2)(3)<->(5)(8)
	C->subcube(1,0,0,dimX-2,0,0).fill(0);//(1)(2)<-(8)(7)
	C->subcube(1,dimY-1,dimZ-1,dimX-2,dimY-1,dimZ-1).fill(0);//(1)(2)->(8)(7)

	C->subcube(1,dimY-1,0,dimX-2,dimY-1,0).fill(0);//(4)(3)<-(5)(6)
	C->subcube(1,0,dimZ-1,dimX-2,0,dimZ-1).fill(0);//(4)(3)->(5)(6)

	C->subcube(0,1,0,0,dimY-2,0).fill(0);//(1)(4)<-(6)(7)
	C->subcube(dimX-1,1,dimZ-1,dimX-1,dimY-2,dimZ-1).fill(0);//(1)(4)->(6)(7)

	C->subcube(dimX-1,1,0,dimX-1,dimY-2,0).fill(0);//(2)(3)<-(5)(8)
	C->subcube(0,1,dimZ-1,0,dimY-2,dimZ-1).fill(0);//(2)(3)->(5)(8)

	C->subcube(0,0,1,0,0,dimZ-2).fill(0);//(1)(5)<-(3)(7)
	C->subcube(dimX-1,dimY-1,1,dimX-1,dimY-1,dimZ-2).fill(0);//(1)(5)->(3)(7)

	C->subcube(dimX-1,0,1,dimX-1,0,dimZ-2).fill(0);//(2)(6)<-(4)(8)
	C->subcube(0,dimY-1,1,0,dimY-1,dimZ-2).fill(0);//(2)(6)->(4)(8)


	//corners (1)<->(7), (2)<->(8), (3)<->(5), (4)<->(6)
	(*C)(dimX-1,dimY-1,dimZ-1) = 0;//(1)->(7)
	(*C)(0,dimY-1,dimZ-1) = 0;//(2)->(8)
	(*C)(0,0,dimZ-1) = 0;//(3)->(5)
	(*C)(dimX-1,0,dimZ-1) = 0;//(4)->(6)
	(*C)(dimX-1,dimY-1,0) = 0;//(5)->(3)
	(*C)(0,dimY-1,0) = 0;//(6)->(4)
	(*C)(0,0,0) = 0;//(7)->(1)
	(*C)(dimX-1,0,0) = 0;//(8)->(2)

}
