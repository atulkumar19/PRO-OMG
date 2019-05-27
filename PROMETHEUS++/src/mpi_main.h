#ifndef H_MPI_MAIN
#define H_MPI_MAIN

#include <iostream>
#include <vector>
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

	void mpi_function(inputParameters * params);

	void createMPITopology(inputParameters * params);

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
