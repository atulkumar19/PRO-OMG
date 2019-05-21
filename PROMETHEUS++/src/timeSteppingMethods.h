#ifndef H_TIME_STEPPING_METHODS
#define H_TIME_STEPPING_METHODS

#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <ctime>

#include "structures.h"
#include "initialize.h"
#include "PIC.h"
#include "emf.h"
#include "generalFunctions.h"
#include "outputHDF5.h"

#include <omp.h>
#include "mpi_main.h"


class TIME_STEPPING_METHODS{
    double t1;						//
    double t2;
    double currentTime; 			// Current time in simulation.	//
    int outputIterator;			//

public:

    TIME_STEPPING_METHODS(inputParameters * params);

    void advanceFullOrbitIonsAndMasslessElectrons(inputParameters * params, meshGeometry * mesh, characteristicScales * CS, HDF * hdfObj, vector<ionSpecies> * IONS, emf * EB);

    void advanceGCIonsAndMasslessElectrons(inputParameters * params, meshGeometry * mesh, characteristicScales * CS, HDF * hdfObj, vector<ionSpecies> * IONS, emf * EB);

};


#endif
