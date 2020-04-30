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

#include "outputHDF5.h"


#ifdef HDF5_FLOAT
template <class IT, class FT> void HDF<IT,FT>::armaCastDoubleToFloat(vec * doubleVector, fvec * floatVector){
	int N_ELEM((*doubleVector).n_elem);
	int ii;
	(*floatVector).set_size(N_ELEM);

	#pragma omp parallel shared(doubleVector, floatVector) private(ii)
	{
		#pragma omp for
		for(ii=0;ii<N_ELEM;ii++){
			(*floatVector)(ii) = (float)(*doubleVector)(ii);
		}
	}//End of the parallel region


}
#endif


// Function to save a single integer value
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, int * value){
	H5std_string nameSpace( name );
	int data[1] = {*value};
	hsize_t dims[1] = {1};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( data, PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save a single CPP_TYPE value
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, CPP_TYPE * value){
	H5std_string nameSpace( name );
	CPP_TYPE data[1] = {*value};
	hsize_t dims[1] = {1};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( data, HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save a single integer value (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, int * value){
	H5std_string nameSpace( name );
	int data[1] = {*value};
	hsize_t dims[1] = {1};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( data, PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save a single CPP_TYPE value (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, CPP_TYPE * value){
	H5std_string nameSpace( name );
	CPP_TYPE data[1] = {*value};
	hsize_t dims[1] = {1};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( data, HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save a vector of int values
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, std::vector<int> * values){
	H5std_string nameSpace( name );
	unsigned long long int size = (unsigned long long int)values->size();

	int * data;
   	data = new int[size];
    std::copy(values->begin(), values->end(), data);

	hsize_t dims[1] = {size};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( data, PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save a vector of CPP_TYPE values
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, std::vector<CPP_TYPE> * values){
	H5std_string nameSpace( name );
	unsigned long long int size = (unsigned long long int)values->size();

	CPP_TYPE * data;
   	data = new CPP_TYPE[size];
    std::copy(values->begin(), values->end(), data);

	hsize_t dims[1] = {size};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( data, HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo ivec vector (to a HDF5 file)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, arma::ivec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( values->memptr(), PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo ivec vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::ivec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( values->memptr(), PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, arma::vec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::vec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo fvec vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::fvec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo imat vector (to a HDF5 file)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, arma::imat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( values->memptr(), PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo imat vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::imat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, PredType::NATIVE_INT, *dataspace ));

	dataset->write( values->memptr(), PredType::NATIVE_INT);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::mat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(Group * group, string name, arma::fmat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


//! Function to interpolate electromagnetic fields from staggered grid to non-staggered grid.

//! Linear interpolation is used to calculate the values of the fields in a non-staggered grid.
template <class IT, class FT> void HDF<IT,FT>::computeFieldsOnNonStaggeredGrid(oneDimensional::fields * F, oneDimensional::fields * G){
	int NX = F->E.X.n_elem;

	G->E.X.subvec(1,NX-2) = 0.5*( F->E.X.subvec(1,NX-2) + F->E.X.subvec(0,NX-3) );
	G->E.Y.subvec(1,NX-2) = F->E.Y.subvec(1,NX-2);
	G->E.Z.subvec(1,NX-2) = F->E.Z.subvec(1,NX-2);

	G->B.X.subvec(1,NX-2) = F->B.X.subvec(1,NX-2);
	G->B.Y.subvec(1,NX-2) = 0.5*( F->B.Y.subvec(1,NX-2) + F->B.Y.subvec(0,NX-3) );
	G->B.Z.subvec(1,NX-2) = 0.5*( F->B.Z.subvec(1,NX-2) + F->B.Z.subvec(0,NX-3) );
}



//! Function to interpolate electromagnetic fields from staggered grid to non-staggered grid.

//! Bilinear interpolation is used to calculate the values of the fields in a non-staggered grid.
template <class IT, class FT> void HDF<IT,FT>::computeFieldsOnNonStaggeredGrid(twoDimensional::fields * F, twoDimensional::fields * G){
	int NX = F->E.X.n_rows;
	int NY = F->E.X.n_cols;

	G->E.X.submat(1,1,NX-2,NY-2) = 0.5*( F->E.X.submat(0,1,NX-3,NY-2) + F->E.X.submat(1,1,NX-2,NY-2) );
	G->E.Y.submat(1,1,NX-2,NY-2) = 0.5*( F->E.Y.submat(1,0,NX-2,NY-3) + F->E.Y.submat(1,1,NX-2,NY-2) );
	G->E.Z.submat(1,1,NX-2,NY-2) = F->E.Z.submat(1,1,NX-2,NY-2);

	G->B.X.submat(1,1,NX-2,NY-2) = 0.5*( F->B.X.submat(1,0,NX-2,NY-3) + F->B.X.submat(1,1,NX-2,NY-2) );
	G->B.Y.submat(1,1,NX-2,NY-2) = 0.5*( F->B.Y.submat(0,1,NX-3,NY-2) + F->B.Y.submat(1,1,NX-2,NY-2) );
	G->B.Z.submat(1,1,NX-2,NY-2) = 0.25*( F->B.Z.submat(0,0,NX-3,NY-3) + F->B.Z.submat(1,0,NX-2,NY-3) + F->B.Z.submat(0,1,NX-3,NY-2) + F->B.Z.submat(1,1,NX-2,NY-2) );
}


//
// CLASS CONSTRUCTOR //
//
template <class IT, class FT> HDF<IT,FT>::HDF(simulationParameters * params, fundamentalScales * FS, vector<IT> *IONS){

	try{
		stringstream dn;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		string name;
		string path;

		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		path = params->PATH + "/HDF5/";
		name = path + "main_D"  + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		Exception::dontPrint();

		// Create a new file using the default property lists.
		H5File * outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );

		name = "dimensionality";
		saveToHDF5(outputFile, name, &params->dimensionality);
		name.clear();

		name = "smoothingParameter";
		cpp_type_value = params->smoothingParameter;
		saveToHDF5(outputFile, name, &cpp_type_value);
		name.clear();

		name = "numberOfRKIterations";
		saveToHDF5(outputFile, name, &params->numberOfRKIterations);
		name.clear();

		name = "filtersPerIterationFields";
		saveToHDF5(outputFile, name, &params->filtersPerIterationFields);
		name.clear();

		name = "numOfDomains";
		saveToHDF5(outputFile, name, &params->mpi.NUMBER_MPI_DOMAINS);
		name.clear();

		// Fundamental scales group
		Group * group_scales = new Group( outputFile->createGroup( "/scales" ) );

		name = "electronSkinDepth";
		cpp_type_value = FS->electronSkinDepth;
		saveToHDF5(group_scales, name, &cpp_type_value);
		name.clear();

		name = "electronGyroPeriod";
		cpp_type_value = FS->electronGyroPeriod;
		saveToHDF5(group_scales, name, &cpp_type_value);
		name.clear();

		name = "electronGyroRadius";
		cpp_type_value = FS->electronGyroRadius;
		saveToHDF5(group_scales, name, &cpp_type_value);
		name.clear();

		name = "ionGyroRadius";
		#ifdef HDF5_DOUBLE
		vec_values = zeros(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			vec_values(ss) = FS->ionGyroRadius[ss];
		saveToHDF5(group_scales, name, &vec_values);
		#elif defined HDF5_FLOAT
		fvec_values = zeros<fvec>(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			fvec_values(ss) = (float)FS->ionGyroRadius[ss];
		saveToHDF5(group_scales, name, &fvec_values);
		#endif
		name.clear();

		name = "ionGyroPeriod";
		#ifdef HDF5_DOUBLE
		vec_values = zeros(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			vec_values(ss) = FS->ionGyroPeriod[ss];
		saveToHDF5(group_scales, name, &vec_values);
		#elif defined HDF5_FLOAT
		fvec_values = zeros<fvec>(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			fvec_values(ss) = (float)FS->ionGyroPeriod[ss];
		saveToHDF5(group_scales, name, &fvec_values);
		#endif
		name.clear();

		name = "ionSkinDepth";
		#ifdef HDF5_DOUBLE
		vec_values = zeros(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			vec_values(ss) = FS->ionSkinDepth[ss];
		saveToHDF5(group_scales, name, &vec_values);
		#elif defined HDF5_FLOAT
		fvec_values = zeros<fvec>(params->numberOfParticleSpecies);
		for (int ss=0; ss<params->numberOfParticleSpecies; ss++)
			fvec_values(ss) = (float)FS->ionSkinDepth[ss];
		saveToHDF5(group_scales, name, &fvec_values);
		#endif
		name.clear();

		delete group_scales;

		//Geometry of the mesh
		Group * group_geo = new Group( outputFile->createGroup( "/geometry" ) );

		name = "SPLIT_DIRECTION";
		int_value = params->mesh.SPLIT_DIRECTION;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "DX";
		cpp_type_value = params->mesh.DX;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "DY";
		cpp_type_value = params->mesh.DY;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "DZ";
		cpp_type_value = params->mesh.DZ;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "LX";
		cpp_type_value = params->mesh.LX;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "LY";
		cpp_type_value = params->mesh.LY;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "LZ";
		cpp_type_value = params->mesh.LZ;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "NX";
		int_value = params->mesh.NX_PER_MPI;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NY";
		int_value = params->mesh.NY_PER_MPI;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NZ";
		int_value = params->mesh.NZ_PER_MPI;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NX_IN_SIM";
		int_value = params->mesh.NX_IN_SIM;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NY_IN_SIM";
		int_value = params->mesh.NY_IN_SIM;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NZ_IN_SIM";
		int_value = params->mesh.NZ_IN_SIM;
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		if (params->dimensionality == 1){
			name = "X_MPI_CART_COORD";
			int_value = params->mpi.MPI_CART_COORDS_1D[0];
			saveToHDF5(group_geo, name, &int_value);
			name.clear();
		}else{
			name = "X_MPI_CART_COORD";
			int_value = params->mpi.MPI_CART_COORDS_2D[0];
			saveToHDF5(group_geo, name, &int_value);
			name.clear();

			name = "Y_MPI_CART_COORD";
			int_value = params->mpi.MPI_CART_COORDS_2D[1];
			saveToHDF5(group_geo, name, &int_value);
			name.clear();
		}

		//Saving the x-axis coordinates
		name = "xAxis";

		#ifdef HDF5_DOUBLE
		vec_values = params->mesh.nodes.X;
		saveToHDF5(group_geo, name, &vec_values);
		#elif defined HDF5_FLOAT
		fvec_values = conv_to<fvec>::from(params->mesh.nodes.X);
		saveToHDF5(group_geo, name, &fvec_values);
		#endif

		name.clear();

		if (params->dimensionality == 2){
			name = "yAxis";

			#ifdef HDF5_DOUBLE
			vec_values = params->mesh.nodes.Y;
			saveToHDF5(group_geo, name, &vec_values);
			#elif defined HDF5_FLOAT
			fvec_values = conv_to<fvec>::from(params->mesh.nodes.Y);
			saveToHDF5(group_geo, name, &fvec_values);
			#endif

			name.clear();
		}

		delete group_geo;
		//Geometry of the mesh

		//Electron temperature
		name = "Te";
		cpp_type_value = params->BGP.Te;
		saveToHDF5(outputFile, name, &cpp_type_value);
		name.clear();

		//Ions
		Group * group_ions = new Group( outputFile->createGroup( "/ions" ) );

		name = "numberOfParticleSpecies";
		int_value = params->numberOfParticleSpecies;
		saveToHDF5(group_ions, name, &int_value);
		name.clear();

		name = "ne";
		cpp_type_value = (CPP_TYPE)params->BGP.ne;
		saveToHDF5(group_ions, name, &cpp_type_value);
		name.clear();

		for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "/ions/spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( outputFile->createGroup( name ) );
			name.clear();

			name = "Dn";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).Dn;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "NCP";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).NCP;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "NSP";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).NSP;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "NSP_OUT";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).nSupPartOutput;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "Tpar";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).Tpar;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "Tper";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).Tper;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "M";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).M;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "Q";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).Q;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "Z";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).Z;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			delete group_ionSpecies;
		}

		delete group_ions;
		//Ions

		//Electromagnetic fields
		name = "Bo";
		vector_values = {(CPP_TYPE)params->BGP.Bx, (CPP_TYPE)params->BGP.By, (CPP_TYPE)params->BGP.Bz};
		saveToHDF5(outputFile, name, &vector_values);
		name.clear();

		delete outputFile;

	}//End of try block

    // catch failure caused by the H5File operations
    catch( FileIException error ){
		error.printErrorStack();
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error ){
		error.printErrorStack();
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error ){
		error.printErrorStack();
    }

}


template <class IT, class FT> void HDF<IT,FT>::saveIonsVariables(const simulationParameters * params, const vector<oneDimensional::ionSpecies> * IONS, const characteristicScales * CS, const int it){

	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	try{
		string path;
		string name;
		stringstream iteration;
		stringstream dn;


		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::ivec ivec_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;

		iteration << it;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		path = params->PATH + "/HDF5/";
		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File * outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

		name = "/" + iteration.str();
		Group * group_iteration = new Group (outputFile->openGroup( name ));
		name.clear();

		//Ions
		name = "ions";
		Group * group_ions = new Group( group_iteration->createGroup( "ions" ) );
		name.clear();

		for(int ii=0; ii<IONS->size(); ii++){//Iterations over the ion species.
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( group_ions->createGroup( name ) );
			name.clear();

			for(int ov=0; ov<params->outputs_variables.size(); ov++){
				if(params->outputs_variables.at(ov) == "X"){

					//Saving the x-axis coordinates
					name = "X";
					#ifdef HDF5_DOUBLE
					vec_values = CS->length*IONS->at(ii).X.col(0);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->length*IONS->at(ii).X.col(0));
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "V"){

					name = "V";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).V;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from(CS->velocity*IONS->at(ii).V);
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Ep"){

					name = "E";
					#ifdef HDF5_DOUBLE
					mat_values = CS->eField*IONS->at(ii).E;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->eField*IONS->at(ii).E );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Bp"){

					name = "B";
					#ifdef HDF5_DOUBLE
					mat_values = CS->bField*IONS->at(ii).B;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->bField*IONS->at(ii).B );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "n"){

					//Saving ions species density
					name = "n";
					#ifdef HDF5_DOUBLE
					vec_values = IONS->at(ii).n.subvec(iIndex,fIndex)/CS->length;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(IONS->at(ii).n.subvec(iIndex,fIndex)/CS->length);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "g"){

					//Saving ions species density
					name = "g";
					#ifdef HDF5_DOUBLE
					vec_values = IONS->at(ii).g;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(IONS->at(ii).g);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "mu"){

					//Saving ions species density
					name = "mu";
					#ifdef HDF5_DOUBLE
					vec_values = CS->magneticMoment*IONS->at(ii).mu;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->magneticMoment*IONS->at(ii).mu);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Ppar"){

					//Saving ions species density
					name = "Ppar";
					#ifdef HDF5_DOUBLE
					vec_values = CS->momentum*IONS->at(ii).Ppar;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from( CS->momentum*IONS->at(ii).Ppar );
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "mn"){

					//Saving ions species density
					name = "mn";
					ivec_values = IONS->at(ii).mn;
					saveToHDF5(group_ionSpecies, name, &ivec_values);
					name.clear();

				}else if(params->outputs_variables.at(ov) == "U"){

					Group * group_bulkVelocity = new Group( group_ionSpecies->createGroup( "U" ) );

					//x-component species bulk velocity
					name = "x";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS->at(ii).nv.X.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.X.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex));
					saveToHDF5(group_bulkVelocity, name, &fvec_values);
					#endif
					name.clear();

					//x-component species bulk velocity
					name = "y";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS->at(ii).nv.Y.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.Y.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex));
					saveToHDF5(group_bulkVelocity, name, &fvec_values);
					#endif
					name.clear();

					//x-component species bulk velocity
					name = "z";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS->at(ii).nv.Z.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.Z.subvec(iIndex,fIndex)/IONS->at(ii).n.subvec(iIndex,fIndex));
					saveToHDF5(group_bulkVelocity, name, &fvec_values);
					#endif
					name.clear();

					delete group_bulkVelocity;
				}
			}

			delete group_ionSpecies;
		}//Iterations over the ion species.

		delete group_ions;
		//Ions*/

		delete group_iteration;

		delete outputFile;
	}//End of try block


    catch( FileIException error ){// catch failure caused by the H5File operations
		error.printErrorStack();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printErrorStack();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printErrorStack();
    }
}


template <class IT, class FT> void HDF<IT,FT>::saveIonsVariables(const simulationParameters * params, const vector<twoDimensional::ionSpecies> * IONS, const characteristicScales * CS, const int it){
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;
	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	try{
		string path;
		string name;
		stringstream iteration;
		stringstream dn;


		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::ivec ivec_values;
		arma::imat imat_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;

		iteration << it;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		path = params->PATH + "/HDF5/";
		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File * outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

		name = "/" + iteration.str();
		Group * group_iteration = new Group (outputFile->openGroup( name ));
		name.clear();

		//Ions
		name = "ions";
		Group * group_ions = new Group( group_iteration->createGroup( "ions" ) );
		name.clear();

		for(int ii=0; ii<IONS->size(); ii++){//Iterations over the ion species.
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( group_ions->createGroup( name ) );
			name.clear();

			for(int ov=0; ov<params->outputs_variables.size(); ov++){
				if(params->outputs_variables.at(ov) == "X"){

					//Saving the x-axis coordinates
					name = "X";
					#ifdef HDF5_DOUBLE
					mat_values = CS->length*IONS->at(ii).X.cols(0,1);
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->length*IONS->at(ii).X.cols(0,1) );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "V"){

					name = "V";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).V;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from(CS->velocity*IONS->at(ii).V);
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Ep"){

					name = "E";
					#ifdef HDF5_DOUBLE
					mat_values = CS->eField*IONS->at(ii).E;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->eField*IONS->at(ii).E );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Bp"){

					name = "B";
					#ifdef HDF5_DOUBLE
					mat_values = CS->bField*IONS->at(ii).B;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->bField*IONS->at(ii).B );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "n"){

					//Saving ions species density
					name = "n";
					#ifdef HDF5_DOUBLE
					mat_values = IONS->at(ii).n.submat(irow,icol,frow,fcol)/(CS->length*CS->length);
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( IONS->at(ii).n.submat(irow,icol,frow,fcol)/(CS->length*CS->length) );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "g"){

					name = "g";
					#ifdef HDF5_DOUBLE
					vec_values = IONS->at(ii).g;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from( IONS->at(ii).g );
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "mu"){

					name = "mu";
					#ifdef HDF5_DOUBLE
					vec_values = CS->magneticMoment*IONS->at(ii).mu;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from( CS->magneticMoment*IONS->at(ii).mu );
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Ppar"){

					name = "Ppar";
					#ifdef HDF5_DOUBLE
					vec_values = CS->momentum*IONS->at(ii).Ppar;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from( CS->momentum*IONS->at(ii).Ppar );
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "mn"){

					name = "mn";
					imat_values = IONS->at(ii).mn;
					saveToHDF5(group_ionSpecies, name, &imat_values);
					name.clear();

				}else if(params->outputs_variables.at(ov) == "U"){

					Group * group_bulkVelocity = new Group( group_ionSpecies->createGroup( "U" ) );

					//x-component species bulk velocity
					name = "x";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).nv.X.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol);
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->velocity*IONS->at(ii).nv.X.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol) );
					saveToHDF5(group_bulkVelocity, name, &fmat_values);
					#endif
					name.clear();

					//y-component species bulk velocity
					name = "y";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).nv.Y.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol);
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->velocity*IONS->at(ii).nv.Y.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol) );
					saveToHDF5(group_bulkVelocity, name, &fmat_values);
					#endif
					name.clear();

					//z-component species bulk velocity
					name = "z";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).nv.Z.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol);
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->velocity*IONS->at(ii).nv.Z.submat(irow,icol,frow,fcol)/IONS->at(ii).n.submat(irow,icol,frow,fcol) );
					saveToHDF5(group_bulkVelocity, name, &fmat_values);
					#endif
					name.clear();

					delete group_bulkVelocity;
				}
			}

			delete group_ionSpecies;
		}//Iterations over the ion species.

		delete group_ions;
		//Ions*/

		delete group_iteration;

		delete outputFile;
	}//End of try block

    catch( FileIException error ){// catch failure caused by the H5File operations
		error.printErrorStack();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printErrorStack();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printErrorStack();
	   }
}


template <class IT, class FT> void HDF<IT,FT>::saveFieldsVariables(const simulationParameters * params, oneDimensional::fields * EB, const characteristicScales * CS, const int it){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	fillGhosts(&EB->E.X);
	fillGhosts(&EB->E.Y);
	fillGhosts(&EB->E.Z);

	fillGhosts(&EB->B.X);
	fillGhosts(&EB->B.Y);
	fillGhosts(&EB->B.Z);

	oneDimensional::fields F(params->mesh.NX_IN_SIM + 2);

	computeFieldsOnNonStaggeredGrid(EB, &F);

	try{
		string name;
		string path;
		stringstream iteration;
		stringstream dn;

		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;


		iteration << it;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		path = params->PATH + "/HDF5/";
		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File * outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group * group_iteration = new Group (outputFile->openGroup( group_iteration_name ));

		Group * group_fields = new Group( group_iteration->createGroup( "fields" ) );//Electromagnetic fields

		for(int ov=0; ov<params->outputs_variables.size(); ov++){
			if(params->outputs_variables.at(ov) == "E"){
				Group * group_field = new Group( group_fields->createGroup( "E" ) );//Electric fields

				//x-component of electric field
				name = "x";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*F.E.X.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*F.E.X.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//y-component of electric field
				name = "y";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*F.E.Y.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*F.E.Y.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//z-component of electric field
				name = "z";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*F.E.Z.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*F.E.Z.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				delete group_field;
			}if(params->outputs_variables.at(ov) == "B"){
				Group * group_field = new Group( group_fields->createGroup( "B" ) );//Electric fields

				//x-component of magnetic field
				name = "x";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*F.B.X.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*F.B.X.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//y-component of magnetic field
				name = "y";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*F.B.Y.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*F.B.Y.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//z-component of magnetic field
				name = "z";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*F.B.Z.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*F.B.Z.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				delete group_field;
			}
		}

		delete group_fields;//Electromagnetic fields

		delete group_iteration;

		delete outputFile;
	}


    catch( FileIException error ){// catch failure caused by the H5File operations
		error.printErrorStack();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printErrorStack();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printErrorStack();
    }

	setGhostsToZero(&EB->E.X);
	setGhostsToZero(&EB->E.Y);
	setGhostsToZero(&EB->E.Z);

	setGhostsToZero(&EB->B.X);
	setGhostsToZero(&EB->B.Y);
	setGhostsToZero(&EB->B.Z);
}


template <class IT, class FT> void HDF<IT,FT>::saveFieldsVariables(const simulationParameters * params, twoDimensional::fields * EB, const characteristicScales * CS, const int it){
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;
	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	twoDimensional::fields F(params->mesh.NX_IN_SIM + 2, params->mesh.NY_IN_SIM + 2);

	computeFieldsOnNonStaggeredGrid(EB, &F);

	try{
		string name;
		string path;
		stringstream iteration;
		stringstream dn;

		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;


		iteration << it;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		path = params->PATH + "/HDF5/";
		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File * outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group * group_iteration = new Group (outputFile->openGroup( group_iteration_name ));

		Group * group_fields = new Group( group_iteration->createGroup( "fields" ) );//Electromagnetic fields

		for(int ov=0; ov<params->outputs_variables.size(); ov++){
			if(params->outputs_variables.at(ov) == "E"){
				Group * group_field = new Group( group_fields->createGroup( "E" ) );//Electric fields

				//x-component of electric field
				name = "x";
				#ifdef HDF5_DOUBLE
				mat_values = CS->eField*F.E.X.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->eField*F.E.X.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				//y-component of electric field
				name = "y";
				#ifdef HDF5_DOUBLE
				mat_values = CS->eField*F.E.Y.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->eField*F.E.Y.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				//z-component of electric field
				name = "z";
				#ifdef HDF5_DOUBLE
				mat_values = CS->eField*F.E.Z.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->eField*F.E.Z.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				delete group_field;
			}if(params->outputs_variables.at(ov) == "B"){
				Group * group_field = new Group( group_fields->createGroup( "B" ) );//Electric fields

				//x-component of magnetic field
				name = "x";
				#ifdef HDF5_DOUBLE
				mat_values = CS->bField*F.B.X.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->bField*F.B.X.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				//y-component of magnetic field
				name = "y";
				#ifdef HDF5_DOUBLE
				mat_values = CS->bField*F.B.Y.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->bField*F.B.Y.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				//z-component of magnetic field
				name = "z";
				#ifdef HDF5_DOUBLE
				mat_values = CS->bField*F.B.Z.submat(irow,icol,frow,fcol);
				saveToHDF5(group_ionSpecies, name, &mat_values);
				#elif defined HDF5_FLOAT
				fmat_values = conv_to<fmat>::from( CS->bField*F.B.Z.submat(irow,icol,frow,fcol) );
				saveToHDF5(group_field, name, &fmat_values);
				#endif
				name.clear();

				delete group_field;
			}
		}

		delete group_fields;//Electromagnetic fields

		delete group_iteration;

		delete outputFile;
	}


    catch( FileIException error ){// catch failure caused by the H5File operations
		error.printErrorStack();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printErrorStack();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printErrorStack();
    }

}


template <class IT, class FT> void HDF<IT,FT>::saveOutputs(const simulationParameters * params, const vector<IT> * IONS, FT * EB, const characteristicScales * CS, const int it, double totalTime){

	try{
		stringstream dn;
		dn << params->mpi.MPI_DOMAIN_NUMBER_CART;

		stringstream iteration;
		iteration << it;

		string name, path;
		path = params->PATH + "/HDF5/";

		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();


		H5File * outputFile;

		if(it == 0){
			outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );// Create a new file using the default property lists.
		}else{
			outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );// Create a new file using the default property lists.
		}

		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group * group_iteration = new Group( outputFile->createGroup( group_iteration_name ) );

		CPP_TYPE cpp_type_value;

		name = "time";
		cpp_type_value = (CPP_TYPE)totalTime;
		saveToHDF5(group_iteration, name, &cpp_type_value);
		name.clear();

		delete group_iteration;

		delete outputFile;
	}//End of try block

    catch( FileIException error ){// catch failure caused by the H5File operations
		error.printErrorStack();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printErrorStack();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printErrorStack();
    }

	saveIonsVariables(params, IONS, CS, it);

	saveFieldsVariables(params, EB, CS, it);

}


template class HDF<oneDimensional::ionSpecies, oneDimensional::fields>;
template class HDF<twoDimensional::ionSpecies, twoDimensional::fields>;
