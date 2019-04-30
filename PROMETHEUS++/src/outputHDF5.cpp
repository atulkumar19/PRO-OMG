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



//Constructor of HDF5Obj class
HDF::HDF(inputParameters *params,meshGeometry *mesh,vector<ionSpecies> *IONS){

	try{
		stringstream dn;
		dn << params->mpi.rank_cart;

		string name, path;
		path = params->PATH + "/HDF5/";
		name = path + "main_D"  + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();

		Exception::dontPrint();

		// Create a new file using the default property lists.
		H5File *outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );

		H5std_string numOfDomains( "numOfDomains" );
		int nod[1] = {params->mpi.NUMBER_MPI_DOMAINS};
	   	hsize_t dims_nod[1] = {1};
		DataSpace *dataspace_nod = new DataSpace(1, dims_nod);
		DataSet *dataset_nod = new DataSet(outputFile->createDataSet( numOfDomains, PredType::NATIVE_INT, *dataspace_nod ));
		dataset_nod->write( nod, PredType::NATIVE_INT);
		delete dataspace_nod;
		delete dataset_nod;

		H5std_string numOutputFiles( "numOutputFiles" );
		int nof[1] = {(int)floor(params->timeIterations/params->saveVariablesEach)};
	   	hsize_t dims_nof[1] = {1};
		DataSpace *dataspace_nof = new DataSpace(1, dims_nof);
		DataSet *dataset_nof = new DataSet(outputFile->createDataSet( numOutputFiles, PredType::NATIVE_INT, *dataspace_nof ));
		dataset_nof->write( nof, PredType::NATIVE_INT);
		delete dataspace_nof;
		delete dataset_nof;

		//Geometry of the mesh
		Group *group_geo = new Group( outputFile->createGroup( "/geometry" ) );

		H5std_string cellDim( "finiteDiferences" );
		H5std_string numCell( "numberOfCells" );

#ifdef HDF5_DOUBLE
		CPP_TYPE cd[3] = {mesh->DX,mesh->DY,mesh->DZ};
#elif defined HDF5_FLOAT
		CPP_TYPE cd[3] = {(float)mesh->DX,(float)mesh->DY,(float)mesh->DZ};
#endif
	   	hsize_t dims_cd[1] = {3};
		DataSpace *dataspace_cd = new DataSpace(1, dims_cd);
		DataSet *dataset_cd = new DataSet(group_geo->createDataSet( cellDim, HDF_TYPE, *dataspace_cd ));
		dataset_cd->write( cd, HDF_TYPE );
		delete dataspace_cd;
		delete dataset_cd;

		unsigned int nc[3] = {mesh->dim(0),mesh->dim(1),mesh->dim(2)};
	   	hsize_t dims_nc[1] = {3};
		DataSpace *dataspace_nc = new DataSpace(1, dims_nc);
		DataSet *dataset_nc = new DataSet(group_geo->createDataSet( numCell, PredType::NATIVE_INT, *dataspace_nc ));
		dataset_nc->write( nc, PredType::NATIVE_INT );
		delete dataspace_nc;
		delete dataset_nc;

		//Saving the x-axis coordinates
		H5std_string xAxis( "xAxis");
		CPP_TYPE  xaxis[mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS];
		for(int ii=0;ii<mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS;ii++){
#ifdef HDF5_DOUBLE
				xaxis[ii] = mesh->nodes.X(ii);
#elif defined HDF5_FLOAT
				xaxis[ii] = (float)mesh->nodes.X(ii);
#endif
		}
		hsize_t dims_xaxis[1] = {mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS};
		DataSpace *dataspace_xaxis = new DataSpace(1, dims_xaxis);
		DataSet *dataset_xaxis = new DataSet(group_geo->createDataSet( xAxis, HDF_TYPE, *dataspace_xaxis ));
		dataset_xaxis->write( xaxis, HDF_TYPE );
		delete dataspace_xaxis;
		delete dataset_xaxis;

		delete group_geo;
		//Geometry of the mesh

		//Energy of ions and electromagnetic fields
//		Group *group_energy = new Group( outputFile->createGroup( "/energy" ) );
//		delete group_energy;

		//Electron temperature
		H5std_string electronTemperature( "Te" );
#ifdef HDF5_DOUBLE
		CPP_TYPE Te[1] = {params->BGP.Te};
#elif defined HDF5_FLOAT
		CPP_TYPE Te[1] = {(float)params->BGP.Te};
#endif
	   	hsize_t dims_Te[1] = {1};
		DataSpace *dataspace_Te = new DataSpace(1, dims_Te);
		DataSet *dataset_Te = new DataSet(outputFile->createDataSet( electronTemperature, HDF_TYPE, *dataspace_Te ));
		dataset_Te->write( Te, HDF_TYPE );
		delete dataspace_Te;
		delete dataset_Te;

		//Ions
		Group *group_ions = new Group( outputFile->createGroup( "/ions" ) );
		H5std_string numberOfIonSpecies( "numberOfIonSpecies" );
		H5std_string numberDensity( "numberDensity" );

		int nis[1] = {params->numberOfIonSpecies};
	   	hsize_t dims_nis[1] = {1};
		DataSpace *dataspace_nis = new DataSpace(1, dims_nis);
		DataSet *dataset_nis = new DataSet(group_ions->createDataSet( numberOfIonSpecies, PredType::NATIVE_INT, *dataspace_nis ));
		dataset_nis->write( nis, PredType::NATIVE_INT );
		delete dataspace_nis;
		delete dataset_nis;

#ifdef HDF5_DOUBLE
		CPP_TYPE nd[1] = {params->ne};
#elif defined HDF5_FLOAT
		CPP_TYPE nd[1] = {(float)params->ne};
#endif
	   	hsize_t dims_nd[1] = {1};
		DataSpace *dataspace_nd = new DataSpace(1, dims_nd);
		DataSet *dataset_nd = new DataSet(group_ions->createDataSet( numberDensity, HDF_TYPE, *dataspace_nd ));
		dataset_nd->write( nd, HDF_TYPE );
		delete dataspace_nd;
		delete dataset_nd;


		for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "/ions/species_" + ionSpec.str();
			Group *group_ionSpecies = new Group( outputFile->createGroup( name ) );
			name.clear();

			H5std_string fracNumDen( "Dn" );
#ifdef HDF5_DOUBLE
			CPP_TYPE dn[1] = {IONS->at(ii).BGP.Dn};
#elif defined HDF5_FLOAT
			CPP_TYPE dn[1] = {(float)IONS->at(ii).BGP.Dn};
#endif
		   	hsize_t dims_dn[1] = {1};
			DataSpace *dataspace_dn = new DataSpace(1, dims_dn);
			DataSet *dataset_dn = new DataSet(group_ionSpecies->createDataSet( fracNumDen, HDF_TYPE, *dataspace_dn ));
			dataset_dn->write( dn, HDF_TYPE );
			delete dataspace_dn;
			delete dataset_dn;

			H5std_string supPartProp( "superParticlesProperties" );
			CPP_TYPE spp[3] = {(CPP_TYPE)IONS->at(ii).NCP, (CPP_TYPE)IONS->at(ii).NSP, (CPP_TYPE)IONS->at(ii).nSupPartOutput};
		   	hsize_t dims_spp[1] = {3};
			DataSpace *dataspace_spp = new DataSpace(1, dims_spp);
			DataSet *dataset_spp = new DataSet(group_ionSpecies->createDataSet( supPartProp, HDF_TYPE, *dataspace_spp ));
			dataset_spp->write( spp, HDF_TYPE );
			delete dataspace_spp;
			delete dataset_spp;

			H5std_string temperature( "T" );
#ifdef HDF5_DOUBLE
			CPP_TYPE T[2] = {IONS->at(ii).BGP.Tpar, IONS->at(ii).BGP.Tper};
#elif defined HDF5_FLOAT
			CPP_TYPE T[2] = {(float)IONS->at(ii).BGP.Tpar, (float)IONS->at(ii).BGP.Tper};
#endif
		   	hsize_t dims_T[1] = {2};
			DataSpace *dataspace_T = new DataSpace(1, dims_T);
			DataSet *dataset_T = new DataSet(group_ionSpecies->createDataSet( temperature, HDF_TYPE, *dataspace_T ));
			dataset_T->write( T, HDF_TYPE );
			delete dataspace_T;
			delete dataset_T;

			H5std_string ionProp( "ionProperties" );
#ifdef HDF5_DOUBLE
			CPP_TYPE ip[4] = {IONS->at(ii).M, IONS->at(ii).Q, IONS->at(ii).Z, IONS->at(ii).colFreq};
#elif defined HDF5_FLOAT
			CPP_TYPE ip[4] = {(float)IONS->at(ii).M, (float)IONS->at(ii).Q, (float)IONS->at(ii).Z, 0.0f};
#endif
		   	hsize_t dims_ip[1] = {4};
			DataSpace *dataspace_ip = new DataSpace(1, dims_ip);
			DataSet *dataset_ip = new DataSet(group_ionSpecies->createDataSet( ionProp, HDF_TYPE, *dataspace_ip ));
			dataset_ip->write( ip, HDF_TYPE );
			delete dataspace_ip;
			delete dataset_ip;

			delete group_ionSpecies;
		}

		delete group_ions;
		//Ions

		//Electromagnetic fields
		H5std_string backgroundMagneticField( "Bo" );

#ifdef HDF5_DOUBLE
		CPP_TYPE Bo[3] = {params->BGP.Bx, params->BGP.By, params->BGP.Bz};
#elif defined HDF5_FLOAT
		CPP_TYPE Bo[3] = {(float)params->BGP.Bx, (float)params->BGP.By, (float)params->BGP.Bz};
#endif
	   	hsize_t dims_Bo[1] = {3};
		DataSpace *dataspace_Bo = new DataSpace(1, dims_Bo);
		DataSet *dataset_Bo = new DataSet(outputFile->createDataSet( backgroundMagneticField, HDF_TYPE, *dataspace_Bo ));
		dataset_Bo->write( Bo, HDF_TYPE );
		delete dataspace_Bo;
		delete dataset_Bo;

		//Electromagnetic fields

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
void HDF::siv_1D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	try{
		string name, path;
		stringstream iteration, dn;

		iteration << IT;
		dn << params->mpi.rank_cart;

		path = params->PATH + "/HDF5/";

#ifdef HDF5_SINGLE_FILE
		name = path + "file_D" + dn.str() + ".h5";
#elif defined HDF5_MULTIPLE_FILES
		name = path + "file_D" + dn.str() + "_" + iteration.str() + ".h5";
#endif

		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File *outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

#ifdef HDF5_SINGLE_FILE
		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group *group_iteration = new Group (outputFile->openGroup( group_iteration_name ));

		//Ions
		Group *group_ions = new Group( group_iteration->createGroup( "ions" ) );
#elif defined HDF5_MULTIPLE_FILES
		//Ions
		Group *group_ions = new Group( outputFile->createGroup( "ions" ) );
#endif

		for(int ii=0;ii<IONS->size();ii++){//Iterations over the ion species.

			stringstream ionSpec;
			ionSpec << (ii+1);
			name = "species_" + ionSpec.str();
			Group *group_ionSpecies = new Group( group_ions->createGroup( name ) );
			name.clear();

			Group *group_position = new Group( group_ionSpecies->createGroup( "X" ) );

			vec VAUX;
#ifdef HDF5_FLOAT
			fvec FVAUX;
#endif

			H5std_string xposition( "x" );

			CPP_TYPE *xpos;
			VAUX = CS->length*( IONS->at(ii).X(span(0,IONS->at(ii).nSupPartOutput - 1),0) );
#ifdef HDF5_DOUBLE
			xpos = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			xpos = FVAUX.memptr();
#endif
		   	hsize_t dims_xpos[1] = {(unsigned int)IONS->at(ii).nSupPartOutput};
			DataSpace *dataspace_xpos = new DataSpace(1, dims_xpos);
			DataSet *dataset_xpos = new DataSet(group_position->createDataSet( xposition, HDF_TYPE, *dataspace_xpos ));
			dataset_xpos->write( xpos, HDF_TYPE );
			delete dataspace_xpos;
			delete dataset_xpos;

			delete group_position;

			Group *group_velocity = new Group( group_ionSpecies->createGroup( "velocity" ) );

			H5std_string xvelocity( "vx" );

			CPP_TYPE *xvel;
			VAUX = CS->velocity*(tmpIONS->at(ii).velocity(span(0,IONS->at(ii).nSupPartOutput - 1),0));
#ifdef HDF5_DOUBLE
			xvel = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			xvel = FVAUX.memptr();
#endif
		   	hsize_t dims_xvel[1] = {(unsigned int)IONS->at(ii).nSupPartOutput};
			DataSpace *dataspace_xvel = new DataSpace(1, dims_xvel);
			DataSet *dataset_xvel = new DataSet(group_velocity->createDataSet( xvelocity, HDF_TYPE, *dataspace_xvel ));
			dataset_xvel->write( xvel, HDF_TYPE );
			delete dataspace_xvel;
			delete dataset_xvel;

			H5std_string yvelocity( "vy" );

			CPP_TYPE *yvel;
			VAUX = CS->velocity*(tmpIONS->at(ii).velocity(span(0,IONS->at(ii).nSupPartOutput - 1),1));
#ifdef HDF5_DOUBLE
			yvel = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			yvel = FVAUX.memptr();
#endif
		   	hsize_t dims_yvel[1] = {(unsigned int)IONS->at(ii).nSupPartOutput};
			DataSpace *dataspace_yvel = new DataSpace(1, dims_yvel);
			DataSet *dataset_yvel = new DataSet(group_velocity->createDataSet( yvelocity, HDF_TYPE, *dataspace_yvel ));
			dataset_yvel->write( yvel, HDF_TYPE );
			delete dataspace_yvel;
			delete dataset_yvel;

			H5std_string zvelocity( "vz" );

			CPP_TYPE *zvel;
			VAUX = CS->velocity*(tmpIONS->at(ii).velocity(span(0,IONS->at(ii).nSupPartOutput - 1),2));
#ifdef HDF5_DOUBLE
			zvel = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			zvel = FVAUX.memptr();
#endif
		   	hsize_t dims_zvel[1] = {(unsigned int)IONS->at(ii).nSupPartOutput};
			DataSpace *dataspace_zvel = new DataSpace(1, dims_zvel);
			DataSet *dataset_zvel = new DataSet(group_velocity->createDataSet( zvelocity, HDF_TYPE, *dataspace_zvel ));
			dataset_zvel->write( zvel, HDF_TYPE );
			delete dataspace_zvel;
			delete dataset_zvel;

			delete group_velocity;

			VAUX.reset();

			H5std_string numberDensity( "numberDensity" );

			CPP_TYPE *n;
			VAUX = IONS->at(ii).n.subvec(iIndex,fIndex)/CS->length;
#ifdef HDF5_DOUBLE
			n = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			n = FVAUX.memptr();
#endif
		   	hsize_t dims_n[1] = {(unsigned int)(params->meshDim(0))};
			DataSpace *dataspace_n = new DataSpace(1, dims_n);
			DataSet *dataset_n = new DataSet(group_ionSpecies->createDataSet( numberDensity, HDF_TYPE, *dataspace_n ));
			dataset_n->write( n, HDF_TYPE );
			delete dataspace_n;
			delete dataset_n;

			Group *group_bulkVelocity = new Group( group_ionSpecies->createGroup( "bulkVelocity" ) );


			H5std_string xBulkVelocity( "Ux" );

			CPP_TYPE *Ux;
			VAUX = CS->velocity*tmpIONS->at(ii).nv.X.subvec(iIndex,fIndex)/CS->length;
#ifdef HDF5_DOUBLE
			Ux = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			Ux = FVAUX.memptr();
#endif
		   	hsize_t dims_Ux[1] = {(unsigned int)params->meshDim(0)};
			DataSpace *dataspace_Ux = new DataSpace(1, dims_Ux);
			DataSet *dataset_Ux = new DataSet(group_bulkVelocity->createDataSet( xBulkVelocity, HDF_TYPE, *dataspace_Ux ));
			dataset_Ux->write( Ux, HDF_TYPE );
			delete dataspace_Ux;
			delete dataset_Ux;

			H5std_string yBulkVelocity( "Uy" );

			CPP_TYPE *Uy;
			VAUX = CS->velocity*tmpIONS->at(ii).nv.Y.subvec(iIndex,fIndex)/CS->length;
#ifdef HDF5_DOUBLE
			Uy = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			Uy = FVAUX.memptr();
#endif
		   	hsize_t dims_Uy[1] = {(unsigned int)params->meshDim(0)};
			DataSpace *dataspace_Uy = new DataSpace(1, dims_Uy);
			DataSet *dataset_Uy = new DataSet(group_bulkVelocity->createDataSet( yBulkVelocity, HDF_TYPE, *dataspace_Uy ));
			dataset_Uy->write( Uy, HDF_TYPE );
			delete dataspace_Uy;
			delete dataset_Uy;

			H5std_string zBulkVelocity( "Uz" );

			CPP_TYPE *Uz;
			VAUX = CS->velocity*tmpIONS->at(ii).nv.Z.subvec(iIndex,fIndex)/CS->length;
#ifdef HDF5_DOUBLE
			Uz = VAUX.memptr();
#elif defined HDF5_FLOAT
			armaCastDoubleToFloat(&VAUX, &FVAUX);
			Uz = FVAUX.memptr();
#endif
		   	hsize_t dims_Uz[1] = {(unsigned int)params->meshDim(0)};
			DataSpace *dataspace_Uz = new DataSpace(1, dims_Uz);
			DataSet *dataset_Uz = new DataSet(group_bulkVelocity->createDataSet( zBulkVelocity, HDF_TYPE, *dataspace_Uz ));
			dataset_Uz->write( Uz, HDF_TYPE );
			delete dataspace_Uz;
			delete dataset_Uz;

			delete group_bulkVelocity;

			delete group_ionSpecies;
		}//Iterations over the ion species.

		delete group_ions;
		//Ions*/

#ifdef HDF5_SINGLE_FILE
		delete group_iteration;
#endif

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
void HDF::siv_2D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT){

}
#endif

#ifdef THREED
void HDF::siv_3D(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT){

}
#endif


void HDF::saveIonsVariables(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, const characteristicScales * CS, const int IT){

	#ifdef ONED
		siv_1D(params, tmpIONS, IONS, CS, IT);
	#endif

	#ifdef TWOD
		siv_2D(params, tmpIONS, IONS, CS, IT);
	#endif

	#ifdef THREED
		siv_3D(params, tmpIONS, IONS, CS, IT);
	#endif

}


void HDF::saveFieldsVariables(const inputParameters * params, oneDimensional::electromagneticFields * EB, const characteristicScales * CS, const int IT){

	unsigned int iIndex(params->meshDim(0)*params->mpi.rank_cart+1);
	unsigned int fIndex(params->meshDim(0)*(params->mpi.rank_cart+1));

	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);


	try{
		vec VAUX;
#ifdef HDF5_FLOAT
		fvec FVAUX;
#endif

		string name, path;
		stringstream iteration, dn;

		iteration << IT;
		dn << params->mpi.rank_cart;

		path = params->PATH + "/HDF5/";

#ifdef HDF5_SINGLE_FILE
		name = path + "file_D" + dn.str() + ".h5";
#elif defined HDF5_MULTIPLE_FILES
		name = path + "file_D" + dn.str() + "_" + iteration.str() + ".h5";
#endif

		const H5std_string	FILE_NAME( name );
		name.clear();

		H5File *outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );//Open an existing file.

#ifdef HDF5_SINGLE_FILE
		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group *group_iteration = new Group (outputFile->openGroup( group_iteration_name ));

		Group *group_emf = new Group( group_iteration->createGroup( "emf" ) );//Electromagnetic fields
#elif defined HDF5_MULTIPLE_FILES
		Group *group_emf = new Group( outputFile->createGroup( "emf" ) );//Electromagnetic fields
#endif

		Group *group_emf_E = new Group( group_emf->createGroup( "E" ) );//Electric fields

		H5std_string xElectricField( "Ex" );

		CPP_TYPE *Ex;
		VAUX = 0.5*CS->eField*( EB->E.X.subvec(iIndex,fIndex) + EB->E.X.subvec(iIndex-1,fIndex-1) );
#ifdef HDF5_DOUBLE
		Ex = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		Ex = FVAUX.memptr();
#endif
	   	hsize_t dims_Ex[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_Ex = new DataSpace(1, dims_Ex);
		DataSet *dataset_Ex = new DataSet(group_emf_E->createDataSet( xElectricField, HDF_TYPE, *dataspace_Ex ));
		dataset_Ex->write( Ex, HDF_TYPE );
		delete dataspace_Ex;
		delete dataset_Ex;

		H5std_string yElectricField( "Ey" );

		CPP_TYPE *Ey;
		VAUX = CS->eField*EB->E.Y.subvec(iIndex,fIndex);
#ifdef HDF5_DOUBLE
		Ey = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		Ey = FVAUX.memptr();
#endif
	   	hsize_t dims_Ey[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_Ey = new DataSpace(1, dims_Ey);
		DataSet *dataset_Ey = new DataSet(group_emf_E->createDataSet( yElectricField, HDF_TYPE, *dataspace_Ey ));
		dataset_Ey->write( Ey, HDF_TYPE );
		delete dataspace_Ey;
		delete dataset_Ey;

		H5std_string zElectricField( "Ez" );

		CPP_TYPE *Ez;
		VAUX = CS->eField*EB->E.Z.subvec(iIndex,fIndex);
#ifdef HDF5_DOUBLE
		Ez = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		Ez = FVAUX.memptr();
#endif
	   	hsize_t dims_Ez[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_Ez = new DataSpace(1, dims_Ez);
		DataSet *dataset_Ez = new DataSet(group_emf_E->createDataSet( zElectricField, HDF_TYPE, *dataspace_Ez ));
		dataset_Ez->write( Ez, HDF_TYPE );
		delete dataspace_Ez;
		delete dataset_Ez;

		delete group_emf_E;//Electric field


		Group *group_emf_B = new Group( group_emf->createGroup( "B" ) );//Magnetic fields

		H5std_string xMagneticField( "Bx" );

		CPP_TYPE *Bx;
		VAUX = CS->bField*( 0.5*( EB->B.X.subvec(iIndex,fIndex) + EB->B.X.subvec(iIndex-1,fIndex-1) ) );
#ifdef HDF5_DOUBLE
		Bx = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		Bx = FVAUX.memptr();
#endif

	   	hsize_t dims_Bx[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_Bx = new DataSpace(1, dims_Bx);
		DataSet *dataset_Bx = new DataSet(group_emf_B->createDataSet( xMagneticField, HDF_TYPE, *dataspace_Bx ));
		dataset_Bx->write( Bx, PredType::NATIVE_DOUBLE );
		delete dataspace_Bx;
		delete dataset_Bx;

		H5std_string yMagneticField( "By" );

		CPP_TYPE *By;
		VAUX = CS->bField*( 0.5*( EB->B.Y.subvec(iIndex,fIndex) + EB->B.Y.subvec(iIndex-1,fIndex-1) ) );
#ifdef HDF5_DOUBLE
		By = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		By = FVAUX.memptr();
#endif
	   	hsize_t dims_By[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_By = new DataSpace(1, dims_By);
		DataSet *dataset_By = new DataSet(group_emf_B->createDataSet( yMagneticField, HDF_TYPE, *dataspace_By ));
		dataset_By->write( By, HDF_TYPE );
		delete dataspace_By;
		delete dataset_By;

		H5std_string zMagneticField( "Bz" );

		CPP_TYPE *Bz;
		VAUX = CS->bField*( 0.5*( EB->B.Z.subvec(iIndex,fIndex) + EB->B.Z.subvec(iIndex-1,fIndex-1) ) );
#ifdef HDF5_DOUBLE
		Bz = VAUX.memptr();
#elif defined HDF5_FLOAT
		armaCastDoubleToFloat(&VAUX, &FVAUX);
		Bz = FVAUX.memptr();
#endif
	   	hsize_t dims_Bz[1] = {(unsigned int)params->meshDim(0)};
		DataSpace *dataspace_Bz = new DataSpace(1, dims_Bz);
		DataSet *dataset_Bz = new DataSet(group_emf_B->createDataSet( zMagneticField, HDF_TYPE, *dataspace_Bz ));
		dataset_Bz->write( Bz, HDF_TYPE );
		delete dataspace_Bz;
		delete dataset_Bz;

		delete group_emf_B;//Electric field


		delete group_emf;//Electromagnetic fields

#ifdef HDF5_SINGLE_FILE
		delete group_iteration;
#endif

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

void HDF::saveOutputs(const inputParameters * params, const vector<ionSpecies> * tmpIONS, const vector<ionSpecies> * IONS, emf * EB, const characteristicScales * CS, const int IT, double totalTime){


	try{

		stringstream dn;
		dn << params->mpi.rank_cart;

		stringstream iteration;
		iteration << IT;

		string name, path;
		path = params->PATH + "/HDF5/";

#ifdef HDF5_SINGLE_FILE
		name = path + "file_D" + dn.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();
#elif defined HDF5_MULTIPLE_FILES
		name = path + "file_D" + dn.str() + "_" + iteration.str() + ".h5";
		const H5std_string	FILE_NAME( name );
		name.clear();
#endif


#ifdef HDF5_SINGLE_FILE
		H5File *outputFile;

		if(IT == 0){
			outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );// Create a new file using the default property lists.
		}else{
			outputFile = new H5File( FILE_NAME, H5F_ACC_RDWR );// Create a new file using the default property lists.
		}

		string group_iteration_name;
		group_iteration_name = "/" + iteration.str();
		Group *group_iteration = new Group( outputFile->createGroup( group_iteration_name ) );
#elif defined HDF5_MULTIPLE_FILES
		H5File *outputFile = new H5File( FILE_NAME, H5F_ACC_TRUNC );// Create a new file using the default property lists.
#endif

		H5std_string time( "time" );
		CPP_TYPE tm[1] = {(CPP_TYPE)totalTime};
	   	hsize_t dims_tm[1] = {1};
		DataSpace *dataspace_tm = new DataSpace(1, dims_tm);
#ifdef HDF5_SINGLE_FILE
		DataSet *dataset_tm = new DataSet(group_iteration->createDataSet( time, HDF_TYPE, *dataspace_tm ));
#elif defined HDF5_MULTIPLE_FILES
		DataSet *dataset_tm = new DataSet(outputFile->createDataSet( time, HDF_TYPE, *dataspace_tm ));
#endif
		dataset_tm->write( tm, HDF_TYPE );

		delete dataspace_tm;
		delete dataset_tm;

#ifdef HDF5_SINGLE_FILE
		delete group_iteration;
#endif

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

	saveIonsVariables(params, tmpIONS, IONS, CS, IT);

	saveFieldsVariables(params, EB, CS, IT);



}
