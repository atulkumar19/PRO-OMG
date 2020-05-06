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

#ifndef H_OUTPUTHDF5
#define H_OUTPUTHDF5

#ifdef OLD_HEADER_FILENAME
#include <iostream.h>
#else
#include <iostream>
#endif

#include <string>
#include <cmath>

#include <armadillo>
#include "structures.h"
#include "boundaryConditions.h"
#include "types.h"
#include "energyDiagnostic.h"

#include "H5Cpp.h"

using namespace H5;
using namespace std;
using namespace arma;

template <class IT, class FT> class HDF{

	#ifdef HDF5_DOUBLE
		#define HDF_TYPE PredType::NATIVE_DOUBLE
		#define CPP_TYPE double
	#elif defined HDF5_FLOAT
		#define HDF_TYPE PredType::NATIVE_FLOAT
		#define CPP_TYPE float
	#endif


	void MPI_Allgathervec(const simulationParameters * params, arma::vec * field);

	void MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield);

	void MPI_Allgathermat(const simulationParameters * params, arma::mat * field);

	void MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield);


	void saveToHDF5(H5File * file, string name, int * value);

	void saveToHDF5(H5File * file, string name, CPP_TYPE * value);

	void saveToHDF5(Group * group, string name, int * value);

	void saveToHDF5(Group * group, string name, CPP_TYPE * value);

	void saveToHDF5(H5File * file, string name, std::vector<int> * values);

	void saveToHDF5(H5File * file, string name, std::vector<CPP_TYPE> * values);

	void saveToHDF5(H5File * file, string name, arma::ivec * values);

	void saveToHDF5(Group * group, string name, arma::ivec * values);

	void saveToHDF5(H5File * file, string name, arma::vec * values);

	void saveToHDF5(Group * group, string name, arma::vec * values);

	void saveToHDF5(Group * group, string name, arma::fvec * values);

	void saveToHDF5(H5File * file, string name, arma::imat * values);

	void saveToHDF5(Group * group, string name, arma::imat * values);

	void saveToHDF5(Group * group, string name, arma::mat * values);

	void saveToHDF5(Group * group, string name, arma::fmat * values);


	void computeFieldsOnNonStaggeredGrid(oneDimensional::fields * F, oneDimensional::fields * G);

	void computeFieldsOnNonStaggeredGrid(twoDimensional::fields * F, twoDimensional::fields * G);


	void saveIonsVariables(const simulationParameters * params, const vector<oneDimensional::ionSpecies> * IONS, const characteristicScales * CS, const int it);

	void saveIonsVariables(const simulationParameters * params, const vector<twoDimensional::ionSpecies> * IONS, const characteristicScales * CS, const int it);


	void saveFieldsVariables(const simulationParameters * params, oneDimensional::fields * EB, const characteristicScales * CS, const int it);

	void saveFieldsVariables(const simulationParameters * params, twoDimensional::fields * EB, const characteristicScales * CS, const int it);

	void armaCastDoubleToFloat(vec * doubleVector, fvec * floatVector);


	void saveEnergy(const simulationParameters * params, const vector<IT> * IONS, FT * EB, const characteristicScales * CS, const int it);

public:

	HDF(simulationParameters * params, fundamentalScales * FS, vector<IT> * IONS);

	void saveOutputs(const simulationParameters * params, const vector<IT> * IONS, FT * EB, const characteristicScales * CS, const int it, double totalTime);
};


#endif
