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
void HDF::armaCastDoubleToFloat(vec * doubleVector, fvec * floatVector){
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
void saveToHDF5(H5File * file, string name, int * value){
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
void saveToHDF5(H5File * file, string name, CPP_TYPE * value){
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
void saveToHDF5(Group * group, string name, int * value){
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
void saveToHDF5(Group * group, string name, CPP_TYPE * value){
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
void saveToHDF5(H5File * file, string name, std::vector<int> * values){
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
void saveToHDF5(H5File * file, string name, std::vector<CPP_TYPE> * values){
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


// Function to save an Armadillo vec vector
void saveToHDF5(H5File * file, string name, arma::vec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(file->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
void saveToHDF5(Group * group, string name, arma::vec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo fvec vector (to a HDF5 group)
void saveToHDF5(Group * group, string name, arma::fvec * values){
	H5std_string nameSpace( name );

	hsize_t dims[1] = {(hsize_t)values->n_elem};
	DataSpace * dataspace = new DataSpace(1, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
void saveToHDF5(Group * group, string name, arma::mat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


// Function to save an Armadillo vec vector (to a HDF5 group)
void saveToHDF5(Group * group, string name, arma::fmat * values){
	H5std_string nameSpace( name );

	hsize_t dims[2] = {(hsize_t)values->n_cols, (hsize_t)values->n_rows};
	DataSpace * dataspace = new DataSpace(2, dims);
	DataSet * dataset = new DataSet(group->createDataSet( nameSpace, HDF_TYPE, *dataspace ));

	dataset->write( values->memptr(), HDF_TYPE);

	delete dataspace;
	delete dataset;
}


//Constructor of HDF5Obj class
HDF::HDF(inputParameters *params, meshGeometry *mesh, vector<ionSpecies> *IONS){

	try{
		stringstream dn;
		dn << params->mpi.rank_cart;

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

		name = "numOfDomains";
		saveToHDF5(outputFile, name, &params->mpi.NUMBER_MPI_DOMAINS);
		name.clear();


		//Geometry of the mesh
		Group * group_geo = new Group( outputFile->createGroup( "/geometry" ) );

		name = "DX";
		cpp_type_value = mesh->DX;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "DY";
		cpp_type_value = mesh->DY;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "DZ";
		cpp_type_value = mesh->DZ;
		saveToHDF5(group_geo, name, &cpp_type_value);
		name.clear();

		name = "NX";
		int_value = mesh->dim(0);
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NY";
		int_value = mesh->dim(1);
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		name = "NZ";
		int_value = mesh->dim(2);
		saveToHDF5(group_geo, name, &int_value);
		name.clear();

		//Saving the x-axis coordinates
		name = "xAxis";
#ifdef HDF5_DOUBLE
		vec_values = mesh->nodes.X;
		saveToHDF5(group_geo, name, &vec_values);
#elif defined HDF5_FLOAT
		fvec_values = conv_to<fvec>::from(mesh->nodes.X);
		saveToHDF5(group_geo, name, &fvec_values);
#endif
		name.clear();

		delete group_geo;
		//Geometry of the mesh

		//Energy of ions and electromagnetic fields
//		Group *group_energy = new Group( outputFile->createGroup( "/energy" ) );
//		delete group_energy;

		//Electron temperature
		name = "Te";
		cpp_type_value = params->BGP.Te;
		saveToHDF5(outputFile, name, &cpp_type_value);
		name.clear();

		//Ions
		Group * group_ions = new Group( outputFile->createGroup( "/ions" ) );

		name = "numberOfIonSpecies";
		int_value = params->numberOfIonSpecies;
		saveToHDF5(group_ions, name, &int_value);
		name.clear();

		name = "ne";
		cpp_type_value = (CPP_TYPE)params->ne;
		saveToHDF5(group_ions, name, &cpp_type_value);
		name.clear();

		for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "/ions/spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( outputFile->createGroup( name ) );
			name.clear();

			name = "Dn";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).BGP.Dn;
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
			cpp_type_value = (CPP_TYPE)IONS->at(ii).BGP.Tpar;
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			name = "Tper";
			cpp_type_value = (CPP_TYPE)IONS->at(ii).BGP.Tper;
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
		error.printError();
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error ){
		error.printError();
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error ){
		error.printError();
    }

}



#ifdef ONED
void HDF::siv_1D(const inputParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	try{
		string path;
		string name;
		stringstream iteration;
		stringstream dn;


		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;

		iteration << IT;
		dn << params->mpi.rank_cart;

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

		for(int ii=0;ii<IONS_OUT->size();ii++){//Iterations over the ion species.
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
					vec_values = CS->length*IONS_OUT->at(ii).X.col(0);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->length*IONS_OUT->at(ii).X.col(0));
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "V"){

					name = "V";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS_OUT->at(ii).V;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from(CS->velocity*IONS_OUT->at(ii).V);
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "n"){

					//Saving ions species density
					name = "n";
					#ifdef HDF5_DOUBLE
					vec_values = IONS_OUT->at(ii).n.subvec(iIndex,fIndex)/CS->length;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(IONS_OUT->at(ii).n.subvec(iIndex,fIndex)/CS->length);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "g"){

					//Saving ions species density
					name = "g";
					#ifdef HDF5_DOUBLE
					vec_values = IONS_OUT->at(ii).g;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(IONS_OUT->at(ii).g);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "mu"){

					//Saving ions species density
					name = "mu";
					#ifdef HDF5_DOUBLE
					vec_values = CS->magneticMoment*IONS_OUT->at(ii).mu;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->magneticMoment*IONS_OUT->at(ii).mu);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "Ppar"){

					//Saving ions species density
					name = "Ppar";
					#ifdef HDF5_DOUBLE
					vec_values = CS->momentum*IONS_OUT->at(ii).Ppar;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from( CS->momentum*IONS_OUT->at(ii).Ppar );
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();

				}else if(params->outputs_variables.at(ov) == "U"){

					Group * group_bulkVelocity = new Group( group_ionSpecies->createGroup( "U" ) );

					//x-component species bulk velocity
					name = "x";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS_OUT->at(ii).nv.X.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS_OUT->at(ii).nv.X.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex));
					saveToHDF5(group_bulkVelocity, name, &fvec_values);
					#endif
					name.clear();

					//x-component species bulk velocity
					name = "y";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS_OUT->at(ii).nv.Y.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS_OUT->at(ii).nv.Y.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex));
					saveToHDF5(group_bulkVelocity, name, &fvec_values);
					#endif
					name.clear();

					//x-component species bulk velocity
					name = "z";
					#ifdef HDF5_DOUBLE
					vec_values = CS->velocity*IONS_OUT->at(ii).nv.Z.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex);
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(CS->velocity*IONS_OUT->at(ii).nv.Z.subvec(iIndex,fIndex)/IONS_OUT->at(ii).n.subvec(iIndex,fIndex));
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
		error.printError();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printError();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printError();
    }

}
#endif

#ifdef TWOD
void HDF::siv_2D(const inputParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT){

}
#endif

#ifdef THREED
void HDF::siv_3D(const inputParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT){

}
#endif


void HDF::saveIonsVariables(const inputParameters * params, const vector<ionSpecies> * IONS_OUT, const characteristicScales * CS, const int IT){

	#ifdef ONED
		siv_1D(params, IONS_OUT, CS, IT);
	#endif

	#ifdef TWOD
		siv_2D(params, IONS_OUT, CS, IT);
	#endif

	#ifdef THREED
		siv_3D(params, IONS_OUT, CS, IT);
	#endif

}


void HDF::saveFieldsVariables(const inputParameters * params, oneDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	try{
		// forwardPBC_1D(&EB->E.X);
		// forwardPBC_1D(&EB->E.Y);
		// forwardPBC_1D(&EB->E.Z);

		// forwardPBC_1D(&EB->B.X);
		// forwardPBC_1D(&EB->B.Y);
		// forwardPBC_1D(&EB->B.Z);

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


		iteration << IT;
		dn << params->mpi.rank_cart;

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
				vec_values = 0.5*CS->eField*( EB->E.X.subvec(iIndex,fIndex) + EB->E.X.subvec(iIndex-1,fIndex-1) );
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( 0.5*CS->eField*( EB->E.X.subvec(iIndex,fIndex) + EB->E.X.subvec(iIndex-1,fIndex-1) ) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//y-component of electric field
				name = "y";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*EB->E.Y.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*EB->E.Y.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//z-component of electric field
				name = "z";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*EB->E.Z.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*EB->E.Z.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				delete group_field;
			}if(params->outputs_variables.at(ov) == "B"){
				Group * group_field = new Group( group_fields->createGroup( "B" ) );//Electric fields

				//x-component of magnetic field
				name = "x";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*EB->B.X.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*EB->B.X.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//y-component of magnetic field
				name = "y";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*( 0.5*( EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1) ) );
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*( 0.5*( EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1) ) ) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//z-component of magnetic field
				name = "z";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*( 0.5*( EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1) ) );
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*( 0.5*( EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1) ) ) );
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
		error.printError();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printError();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printError();
    }

	restoreVector(&EB->E.X);
	restoreVector(&EB->E.Y);
	restoreVector(&EB->E.Z);

	restoreVector(&EB->B.X);
	restoreVector(&EB->B.Y);
	restoreVector(&EB->B.Z);

}

void HDF::saveFieldsVariables(const inputParameters * params, twoDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT){

}

void HDF::saveFieldsVariables(const inputParameters * params, threeDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT){

}

void HDF::saveOutputs(const inputParameters * params, const vector<ionSpecies> * IONS_OUT, fields * EB, const characteristicScales * CS, const int IT, double totalTime){


	try{

		stringstream dn;
		dn << params->mpi.rank_cart;

		stringstream iteration;
		iteration << IT;

		string name, path;
		path = params->PATH + "/HDF5/";

		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();


		H5File * outputFile;

		if(IT == 0){
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
		error.printError();
    }

    catch( DataSetIException error ){// catch failure caused by the DataSet operations
		error.printError();
    }

    catch( DataSpaceIException error ){// catch failure caused by the DataSpace operations
		error.printError();
    }

	saveIonsVariables(params, IONS_OUT, CS, IT);

	saveFieldsVariables(params, EB, CS, IT);



}
