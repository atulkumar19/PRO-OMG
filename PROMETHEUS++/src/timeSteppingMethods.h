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


template <class T, class Y> class TIME_STEPPING_METHODS{
    double t1;						//
    double t2;
    double currentTime; 			// Current time in simulation.	//
    int outputIterator;			//

public:

    TIME_STEPPING_METHODS(simulationParameters * params);

    void advanceFullOrbitIonsAndMasslessElectrons(simulationParameters * params, characteristicScales * CS, HDF<T,Y> * hdfObj, vector<T> * IONS, Y * EB);

    void advanceGCIonsAndMasslessElectrons(simulationParameters * params, characteristicScales * CS, HDF<T,Y> * hdfObj, vector<T> * IONS, Y * EB);

};


#endif
