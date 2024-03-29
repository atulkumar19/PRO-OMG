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

#ifndef H_MPI_MAIN
#define H_MPI_MAIN

#include <iostream>
#include <vector>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <omp.h>
#include <cmath>

#include "structures.h"

#include "mpi.h"

using namespace std;
using namespace arma;

//! \class Class MPI_MAIN
/*!
 * \brief Main class of MPI routines.
 */
class MPI_MAIN{

private:


public:

	//! Constructor of MPI_MAIN
	MPI_MAIN(){};

	void createMPITopology(simulationParameters * params);

	void finalizeCommunications(simulationParameters * params);

};

class MPI_ARMA_VEC{

public:

	MPI_Datatype type;

	unsigned int size;

	MPI_ARMA_VEC(unsigned int SIZE){
		size = SIZE;
		MPI_Type_contiguous(SIZE,MPI_DOUBLE,&type);
		MPI_Type_commit(&type);
	};

	~MPI_ARMA_VEC() {MPI_Type_free(&type);}

};

class MPI_ARMA_IVEC{

public:

	MPI_Datatype type;

	unsigned int size;

	MPI_ARMA_IVEC(unsigned int SIZE){
		size = SIZE;
		MPI_Type_contiguous(SIZE,MPI_INT,&type);
		MPI_Type_commit(&type);
	};

	~MPI_ARMA_IVEC() {MPI_Type_free(&type);}

};

#endif
