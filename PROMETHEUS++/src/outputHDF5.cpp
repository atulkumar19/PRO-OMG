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

template <class IT, class FT> void HDF<IT,FT>::MPI_Allgathervec(const simulationParameters * params, arma::vec * field)
{
	unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

	arma::vec recvbuf(params->mesh.NX_IN_SIM);
	arma::vec sendbuf(params->mesh.NX_PER_MPI);

	//Allgather for x-component
	sendbuf = field->subvec(iIndex, fIndex);
	MPI_Allgather(sendbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NX_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);
	field->subvec(1, params->mesh.NX_IN_SIM) = recvbuf;
}


template <class IT, class FT> void HDF<IT,FT>::MPI_Allgathervfield_vec(const simulationParameters * params, vfield_vec * vfield)
{
	MPI_Allgathervec(params, &vfield->X);
	MPI_Allgathervec(params, &vfield->Y);
	MPI_Allgathervec(params, &vfield->Z);
}


template <class IT, class FT> void HDF<IT,FT>::MPI_Allgathermat(const simulationParameters * params, arma::mat * field)
{
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;

	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

	arma::vec recvbuf = zeros(params->mesh.NX_IN_SIM*params->mesh.NY_IN_SIM);
	arma::vec sendbuf = zeros(params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI);

	//Allgather for x-component
	sendbuf = vectorise(field->submat(irow,icol,frow,fcol));
	MPI_Allgather(sendbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, recvbuf.memptr(), params->mesh.NUM_CELLS_PER_MPI, MPI_DOUBLE, params->mpi.MPI_TOPO);

	for (int mpis=0; mpis<params->mpi.NUMBER_MPI_DOMAINS; mpis++){
		unsigned int ie = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*mpis;
		unsigned int fe = params->mesh.NX_PER_MPI*params->mesh.NY_PER_MPI*(mpis+1) - 1;

		unsigned int ir = *(params->mpi.MPI_CART_COORDS.at(mpis))*params->mesh.NX_PER_MPI + 1;
		unsigned int fr = ( *(params->mpi.MPI_CART_COORDS.at(mpis)) + 1)*params->mesh.NX_PER_MPI;

		unsigned int ic = *(params->mpi.MPI_CART_COORDS.at(mpis)+1)*params->mesh.NY_PER_MPI + 1;
		unsigned int fc = ( *(params->mpi.MPI_CART_COORDS.at(mpis)+1) + 1)*params->mesh.NY_PER_MPI;

		field->submat(ir,ic,fr,fc) = reshape(recvbuf.subvec(ie,fe), params->mesh.NX_PER_MPI, params->mesh.NY_PER_MPI);
	}

}


template <class IT, class FT> void HDF<IT,FT>::MPI_Allgathervfield_mat(const simulationParameters * params, vfield_mat * vfield)
{
	MPI_Allgathermat(params, &vfield->X);
	MPI_Allgathermat(params, &vfield->Y);
	MPI_Allgathermat(params, &vfield->Z);
}


// Function to save a single integer value
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, int * value)
{
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
template <class IT, class FT> void HDF<IT,FT>::saveToHDF5(H5File * file, string name, CPP_TYPE * value)
{
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



//
// CLASS CONSTRUCTOR //
//
template <class IT, class FT> HDF<IT,FT>::HDF(simulationParameters * params, fundamentalScales * FS, vector<IT> *IONS)
{

    if (params->mpi.MPI_DOMAIN_NUMBER == 0)
        {
            try
            {
                string name;
                string path;

                int int_value;
                CPP_TYPE cpp_type_value;
                std::vector<CPP_TYPE> vector_values;

                arma::vec vec_values;
                arma::fvec fvec_values;

                path = params->PATH + "/HDF5/";
                name = path + "main.h5";
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

                name = "numOfMPIs";
                saveToHDF5(outputFile, name, &params->mpi.NUMBER_MPI_DOMAINS);
                name.clear();

                name = "numMPIsParticles";
                saveToHDF5(outputFile, name, &params->mpi.MPIS_PARTICLES);
                name.clear();


                name = "numMPIsFields";
                saveToHDF5(outputFile, name, &params->mpi.MPIS_FIELDS);
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
                cpp_type_value = params->f_IC.Te;
                saveToHDF5(outputFile, name, &cpp_type_value);
                name.clear();

                //Ions
                Group * group_ions = new Group( outputFile->createGroup( "/ions" ) );

                name = "numberOfParticleSpecies";
                int_value = params->numberOfParticleSpecies;
                saveToHDF5(group_ions, name, &int_value);
                name.clear();

                name = "ne";
                cpp_type_value = (CPP_TYPE)params->f_IC.ne;
                saveToHDF5(group_ions, name, &cpp_type_value);
                name.clear();

                for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
                        stringstream ionSpec;
                        ionSpec << (ii+1);
                        name = "/ions/spp_" + ionSpec.str();
                        Group * group_ionSpecies = new Group( outputFile->createGroup( name ) );
                        name.clear();

                        name = "densityFraction";
                        cpp_type_value = (CPP_TYPE)IONS->at(ii).p_IC.densityFraction;
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
                        cpp_type_value = (CPP_TYPE)IONS->at(ii).p_IC.Tpar;
                        saveToHDF5(group_ionSpecies, name, &cpp_type_value);
                        name.clear();

                        name = "Tper";
                        cpp_type_value = (CPP_TYPE)IONS->at(ii).p_IC.Tper;
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
                name = "BX_0";
                vector_values = {(CPP_TYPE)params->em_IC.BX, (CPP_TYPE)params->em_IC.BX, (CPP_TYPE)params->em_IC.BX};
                saveToHDF5(outputFile, name, &vector_values);
                name.clear();

                delete outputFile;

            }//End of try block

	    // catch failure caused by the H5File operations:
            // =============================================
	    catch( FileIException error )
            {
            	error.printErrorStack();
	    }

	    // catch failure caused by the DataSet operations:
            // ===============================================
	    catch( DataSetIException error )
            {
            	error.printErrorStack();
	    }

	    // catch failure caused by the DataSpace operations:
            // ================================================
	    catch( DataSpaceIException error )
            {
            	error.printErrorStack();
	    }
	}

}


template <class IT, class FT> void HDF<IT,FT>::saveIonsVariables(const simulationParameters * params, const vector<oneDimensional::ionSpecies> * IONS, const characteristicScales * CS, const Group * group_iteration){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	try{
		string name;

		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::ivec ivec_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;

		//Ions
		name = "ions";
		Group * group_ions = new Group( group_iteration->createGroup( "ions" ) );
		name.clear();

		//Iterations over the ion species.
		for(int ii=0; ii<IONS->size(); ii++)
		{
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( group_ions->createGroup( name ) );
			name.clear();

			for(int ov=0; ov<params->outputs_variables.size(); ov++)
			{
				if(params->outputs_variables.at(ov) == "X")
				{
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
				 }
				 if(params->outputs_variables.at(ov) == "a")
 				{
 					//Saving the x-axis coordinates
 					name = "a";
 					#ifdef HDF5_DOUBLE
 					vec_values = IONS->at(ii).a;
 					saveToHDF5(group_ionSpecies, name, &vec_values);
 					#elif defined HDF5_FLOAT
 					fvec_values = conv_to<fvec>::from(IONS->at(ii).a);
 					saveToHDF5(group_ionSpecies, name, &fvec_values);
 					#endif
 					name.clear();
 				 }
				 else if(params->outputs_variables.at(ov) == "pCount")
				 {
					name = "pCount";
					ivec_values = IONS->at(ii).pCount;
					saveToHDF5(group_ionSpecies, name, &ivec_values);
					name.clear();
				}
				else if(params->outputs_variables.at(ov) == "eCount")
				{
					//Saving ions species density
					name = "eCount";
					#ifdef HDF5_DOUBLE
					vec_values = (CS->velocity*CS->velocity*CS->mass)*IONS->at(ii).eCount;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from((CS->velocity*CS->velocity*CS->mass)*IONS->at(ii).eCount);
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();
                }
				else if(params->outputs_variables.at(ov) == "V")
				{
					name = "V";
					#ifdef HDF5_DOUBLE
					mat_values = CS->velocity*IONS->at(ii).V;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from(CS->velocity*IONS->at(ii).V);
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();
				}
				else if(params->outputs_variables.at(ov) == "Ep")
				{
					name = "E";
					#ifdef HDF5_DOUBLE
					mat_values = CS->eField*IONS->at(ii).E;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->eField*IONS->at(ii).E );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();
				}
				else if(params->outputs_variables.at(ov) == "Bp")
				{
					name = "Bp";
					#ifdef HDF5_DOUBLE
					mat_values = CS->bField*IONS->at(ii).B;
					saveToHDF5(group_ionSpecies, name, &mat_values);
					#elif defined HDF5_FLOAT
					fmat_values = conv_to<fmat>::from( CS->bField*IONS->at(ii).B );
					saveToHDF5(group_ionSpecies, name, &fmat_values);
					#endif
					name.clear();
				}
				if(params->outputs_variables.at(ov) == "np")
				{
					//Saving the x-axis coordinates
					name = "np";
					#ifdef HDF5_DOUBLE
					vec_values = IONS->at(ii).n_p/CS->length;
					saveToHDF5(group_ionSpecies, name, &vec_values);
					#elif defined HDF5_FLOAT
					fvec_values = conv_to<fvec>::from(IONS->at(ii).n_p)/CS->length;
					saveToHDF5(group_ionSpecies, name, &fvec_values);
					#endif
					name.clear();
				 }
				 if(params->outputs_variables.at(ov) == "nvp")
				 {
					 //Saving the "X" component of ion flux density at particle positions:
					 name = "nvp";
					 #ifdef HDF5_DOUBLE
					 vec_values = IONS->at(ii).nv_p*CS->velocity/CS->length;
					 saveToHDF5(group_ionSpecies, name, &vec_values);
					 #elif defined HDF5_FLOAT
					 fvec_values = conv_to<fvec>::from(IONS->at(ii).nv_p)*CS->velocity/CS->length;
					 saveToHDF5(group_ionSpecies, name, &fvec_values);
					 #endif
					 name.clear();
				  }
				  if(params->outputs_variables.at(ov) == "Tparp")
				  {
					  //Saving the "X" drift velocity at the particle positions:
					  name = "Tparp";
					  #ifdef HDF5_DOUBLE
					  vec_values = IONS->at(ii).Tpar_p*CS->temperature;
					  saveToHDF5(group_ionSpecies, name, &vec_values);
					  #elif defined HDF5_FLOAT
					  fvec_values = conv_to<fvec>::from(IONS->at(ii).Tpar_p)*CS->temperature;
					  saveToHDF5(group_ionSpecies, name, &fvec_values);
					  #endif
					  name.clear();
				   }
				   if(params->outputs_variables.at(ov) == "Tperp")
				   {
					   //Saving the "X" drift velocity at the particle positions:
					   name = "Tperp";
					   #ifdef HDF5_DOUBLE
					   vec_values = IONS->at(ii).Tper_p*CS->temperature;
					   saveToHDF5(group_ionSpecies, name, &vec_values);
					   #elif defined HDF5_FLOAT
					   fvec_values = conv_to<fvec>::from(IONS->at(ii).Tper_p)*CS->temperature;
					   saveToHDF5(group_ionSpecies, name, &fvec_values);
					   #endif
					   name.clear();
					}
				else if(params->outputs_variables.at(ov) == "n")
				{
					if (params->mpi.IS_PARTICLES_ROOT)
					{
						//Saving ions species density
						name = "n";
						#ifdef HDF5_DOUBLE
						vec_values = IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM)/CS->length;
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM)/CS->length);
						saveToHDF5(group_ionSpecies, name, &fvec_values);
						#endif
						name.clear();
					}

				}
				else if(params->outputs_variables.at(ov) == "Tpar")
				{
					if (params->mpi.IS_PARTICLES_ROOT)
					{
						//Saving ions species density
						name = "Tpar";
						#ifdef HDF5_DOUBLE
						vec_values = IONS->at(ii).Tpar_m.subvec(1,params->mesh.NX_IN_SIM)*CS->temperature;
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(IONS->at(ii).Tpar_m.subvec(1,params->mesh.NX_IN_SIM)*CS->temperature);
						saveToHDF5(group_ionSpecies, name, &fvec_values);
						#endif
						name.clear();
					}

				}
				else if(params->outputs_variables.at(ov) == "Tper")
				{
					if (params->mpi.IS_PARTICLES_ROOT)
					{
						//Saving ions species density
						name = "Tper";
						#ifdef HDF5_DOUBLE
						vec_values = IONS->at(ii).Tper_m.subvec(1,params->mesh.NX_IN_SIM)*CS->temperature;
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(IONS->at(ii).Tper_m.subvec(1,params->mesh.NX_IN_SIM)*CS->temperature);
						saveToHDF5(group_ionSpecies, name, &fvec_values);
						#endif
						name.clear();
					}

				}
				else if(params->outputs_variables.at(ov) == "g")
				{
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
				}
				else if(params->outputs_variables.at(ov) == "mu")
				{
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

				}
				else if(params->outputs_variables.at(ov) == "Ppar")
				{
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

				}
				else if(params->outputs_variables.at(ov) == "mn")
				{
					//Saving ions species density
					name = "mn";
					ivec_values = IONS->at(ii).mn;
					saveToHDF5(group_ionSpecies, name, &ivec_values);
					name.clear();

				}
				else if(params->outputs_variables.at(ov) == "U")
				{
					if (params->mpi.IS_PARTICLES_ROOT)
					{
						Group * group_bulkVelocity = new Group( group_ionSpecies->createGroup( "U" ) );

						//x-component species bulk velocity
						name = "x";
						#ifdef HDF5_DOUBLE
						vec_values = CS->velocity*IONS->at(ii).nv.X.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM);
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.X.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM));
						saveToHDF5(group_bulkVelocity, name, &fvec_values);
						#endif
						name.clear();

						//x-component species bulk velocity
						name = "y";
						#ifdef HDF5_DOUBLE
						vec_values = CS->velocity*IONS->at(ii).nv.Y.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM);
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.Y.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM));
						saveToHDF5(group_bulkVelocity, name, &fvec_values);
						#endif
						name.clear();

						//x-component species bulk velocity
						name = "z";
						#ifdef HDF5_DOUBLE
						vec_values = CS->velocity*IONS->at(ii).nv.Z.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM);
						saveToHDF5(group_ionSpecies, name, &vec_values);
						#elif defined HDF5_FLOAT
						fvec_values = conv_to<fvec>::from(CS->velocity*IONS->at(ii).nv.Z.subvec(1,params->mesh.NX_IN_SIM)/IONS->at(ii).n.subvec(1,params->mesh.NX_IN_SIM));
						saveToHDF5(group_bulkVelocity, name, &fvec_values);
						#endif
						name.clear();

						delete group_bulkVelocity;
					}
				}
			}

			delete group_ionSpecies;
		}//Iterations over the ion species.

		delete group_ions;
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


template <class IT, class FT> void HDF<IT,FT>::saveIonsVariables(const simulationParameters * params, const vector<twoDimensional::ionSpecies> * IONS, const characteristicScales * CS, const Group * group_iteration)
{}


template <class IT, class FT> void HDF<IT,FT>::saveFieldsVariables(const simulationParameters * params, oneDimensional::fields * EB, const characteristicScales * CS, const Group * group_iteration){
	unsigned int iIndex(params->mesh.NX_PER_MPI*params->mpi.MPI_DOMAIN_NUMBER_CART+1);
	unsigned int fIndex(params->mesh.NX_PER_MPI*(params->mpi.MPI_DOMAIN_NUMBER_CART+1));

	try{
		string name;

		int int_value;
		CPP_TYPE cpp_type_value;
		std::vector<CPP_TYPE> vector_values;

		arma::vec vec_values;
		arma::fvec fvec_values;

		arma::mat mat_values;
		arma::fmat fmat_values;

		Group * group_fields = new Group( group_iteration->createGroup( "fields" ) );//Electromagnetic fields

		for(int ov=0; ov<params->outputs_variables.size(); ov++){
			if(params->outputs_variables.at(ov) == "E"){
				Group * group_field = new Group( group_fields->createGroup( "E" ) );//Electric fields

				//x-component of electric field
				name = "x";
				#ifdef HDF5_DOUBLE
				vec_values = CS->eField*EB->E.X.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->eField*EB->E.X.subvec(iIndex,fIndex) );
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
				vec_values = CS->bField*EB->B.Y.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*EB->B.Y.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				//z-component of magnetic field
				name = "z";
				#ifdef HDF5_DOUBLE
				vec_values = CS->bField*EB->B.Z.subvec(iIndex,fIndex);
				saveToHDF5(group_ionSpecies, name, &vec_values);
				#elif defined HDF5_FLOAT
				fvec_values = conv_to<fvec>::from( CS->bField*EB->B.Z.subvec(iIndex,fIndex) );
				saveToHDF5(group_field, name, &fvec_values);
				#endif
				name.clear();

				delete group_field;
			}
		}

		delete group_fields;//Electromagnetic fields
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


template <class IT, class FT> void HDF<IT,FT>::saveFieldsVariables(const simulationParameters * params, twoDimensional::fields * EB, const characteristicScales * CS, const Group * group_iteration)
{}


template <class IT, class FT> void HDF<IT,FT>::saveIonsEnergy(const simulationParameters * params, const vector<IT> * IONS, const characteristicScales * CS, const Group * group_iteration){
	ENERGY_DIAGNOSTIC<IT,FT> energyOutputs(params);

	energyOutputs.computeKineticEnergyDensity(params, IONS);

	arma::vec kineticEnergyDensity = energyOutputs.getKineticEnergyDensity();

	try{
		string name;

		double units;
		CPP_TYPE cpp_type_value;
		arma::vec vec_values;
		arma::fvec fvec_values;


		name = "energy";
		Group * group_energy = new Group( group_iteration->createGroup( name ) );
		name.clear();

		// Ions energy
		name = "ions";
		Group * group_ions = new Group( group_energy->createGroup( name ) );
		name.clear();

		for(int ii=0; ii<IONS->size(); ii++){//Iterations over the ion species.
			stringstream ionSpec;
			ionSpec << (ii+1);

			name = "spp_" + ionSpec.str();
			Group * group_ionSpecies = new Group( group_ions->createGroup( name ) );
			name.clear();

			name = "kineticEnergyDensity";
			units = (params->dimensionality == 1) ? ( CS->mass*pow(CS->velocity,2)/CS->length ) : ( CS->mass*pow(CS->velocity,2)/pow(CS->length,2) );
			cpp_type_value = units*kineticEnergyDensity(ii);
			saveToHDF5(group_ionSpecies, name, &cpp_type_value);
			name.clear();

			delete group_ionSpecies;
		}//Iterations over the ion species.

		delete group_ions;
		// Ions energy

		delete group_energy;
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


template <class IT, class FT> void HDF<IT,FT>::saveFieldsEnergy(const simulationParameters * params, FT * EB, const characteristicScales * CS, const Group * group_iteration){
	ENERGY_DIAGNOSTIC<IT,FT> energyOutputs(params);

	energyOutputs.computeElectromagneticEnergyDensity(params, EB);

	arma::vec magneticEnergyDensity = energyOutputs.getMagneticEnergyDensity();

	arma::vec electricEnergyDensity = energyOutputs.getElectricEnergyDensity();

	try{
		string name;

		double units;
		CPP_TYPE cpp_type_value;
		arma::vec vec_values;
		arma::fvec fvec_values;


		name = "energy";
		Group * group_energy = new Group( group_iteration->createGroup( name ) );
		name.clear();


		// Fields energy
		name = "fields";
		Group * group_fields = new Group( group_energy->createGroup( name ) );
		name.clear();

		name = "E";
		Group * group_efield = new Group( group_fields->createGroup( name ) );
		name.clear();

		name = "X";
		units = CS->vacuumPermittivity*pow(CS->eField,2);
		cpp_type_value = units*electricEnergyDensity(0);
		saveToHDF5(group_efield, name, &cpp_type_value);
		name.clear();

		name = "Y";
		units = CS->vacuumPermittivity*pow(CS->eField,2);
		cpp_type_value = units*electricEnergyDensity(1);
		saveToHDF5(group_efield, name, &cpp_type_value);
		name.clear();

		name = "Z";
		units = CS->vacuumPermittivity*pow(CS->eField,2);
		cpp_type_value = units*electricEnergyDensity(2);
		saveToHDF5(group_efield, name, &cpp_type_value);
		name.clear();

		delete group_efield;

		name = "B";
		Group * group_bfield = new Group( group_fields->createGroup( name ) );
		name.clear();

		name = "X";
		units = pow(CS->bField,2)/CS->vacuumPermeability;
		cpp_type_value = units*magneticEnergyDensity(0);
		saveToHDF5(group_bfield, name, &cpp_type_value);
		name.clear();

		name = "Y";
		units = pow(CS->bField,2)/CS->vacuumPermeability;
		cpp_type_value = units*magneticEnergyDensity(1);
		saveToHDF5(group_bfield, name, &cpp_type_value);
		name.clear();

		name = "Z";
		units = pow(CS->bField,2)/CS->vacuumPermeability;
		cpp_type_value = units*magneticEnergyDensity(2);
		saveToHDF5(group_bfield, name, &cpp_type_value);
		name.clear();

		delete group_bfield;

		name.clear();

		delete group_fields;
		// Fields energy

		delete group_energy;
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


template <class IT, class FT> void HDF<IT,FT>::saveOutputs(const simulationParameters * params, const vector<IT> * IONS, FT * EB, const characteristicScales * CS, const int it, double totalTime){

	try{
		stringstream iteration;
		stringstream dn;

		string name, path;
		path = params->PATH + "/HDF5/";

		dn << params->mpi.COMM_RANK;

		H5std_string FILE_NAME;

		// Save particles data
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			name = path + "PARTICLES_FILE_" + dn.str() + ".h5";
		}else if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
			name = path + "FIELDS_FILE_" + dn.str() + ".h5";
		}

		FILE_NAME = name;

		name.clear();

		H5File * outputFile;

		if(it == 0){
			outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );// Create a new file using the default property lists.
		}else{
			outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );// Create a new file using the default property lists.
		}

		iteration << it;

		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group * group_iteration = new Group( outputFile->createGroup( group_iteration_name ) );

		CPP_TYPE cpp_type_value;

		name = "time";
		cpp_type_value = (CPP_TYPE)totalTime;
		saveToHDF5(group_iteration, name, &cpp_type_value);
		name.clear();

		// Save particles data
		if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR){
			saveIonsVariables(params, IONS, CS, group_iteration);

			saveIonsEnergy(params, IONS, CS, group_iteration);
		}else if (params->mpi.COMM_COLOR == FIELDS_MPI_COLOR){
			saveFieldsVariables(params, EB, CS, group_iteration);

			saveFieldsEnergy(params, EB, CS, group_iteration);
		}

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


template class HDF<oneDimensional::ionSpecies, oneDimensional::fields>;
template class HDF<twoDimensional::ionSpecies, twoDimensional::fields>;
