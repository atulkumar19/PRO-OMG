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


/* 	ACKNOWLEDGEMENTS:
	+ Francisco Calderon
*/


/*
	This file provides a template for including mandatory tests when including or modifying features
	of PRO++.

	Please go through the file and look for commented sections with "//*** @devs" that give directions
	for including your own tests.

	NOTE: Deleting existing tests MUST NOT be deleted. Only addition of new tests is allowed.
*/

#include <iostream>
#include <vector>
#include <armadillo>
#include <cmath>
#include <ctime>

#include "structures.h"
#include "dev_test_2D.h"

//*** @devs Include your headers here.

#include <omp.h>
#include "mpi_main.h"

using namespace std;
using namespace arma;

int main(int argc, char* argv[]){
	MPI_Init(&argc, &argv);
	MPI_MAIN mpi_main;

	DEV_TEST_2D weon_2D;
	
	return(0);
}
