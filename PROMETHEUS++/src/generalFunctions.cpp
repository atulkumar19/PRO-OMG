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

#include "generalFunctions.h"

void GENERAL_FUNCTIONS::bCastTimestep(simulationParameters * params, int logicVariable){
	int changeTimestep;
	double * timeSteps;
	int * logic;
	double ststep;//smallest timestep

	logic = (int*)malloc(params->mpi.NUMBER_MPI_DOMAINS*sizeof(int));

	MPI_Allgather(&logicVariable, 1, MPI_INT, logic, 1, MPI_INT, params->mpi.MPI_TOPO);

	MPI_Barrier(params->mpi.MPI_TOPO);

	for(int ii=0; ii<params->mpi.NUMBER_MPI_DOMAINS; ii++){
		if(*(logic + ii) == 1)
			changeTimestep = 1;
	}

	if(changeTimestep == 1){
		timeSteps = (double*)malloc(params->mpi.NUMBER_MPI_DOMAINS*sizeof(double));
		MPI_Allgather(&params->DT, 1, MPI_DOUBLE, timeSteps, 1, MPI_DOUBLE, params->mpi.MPI_TOPO);

		ststep = *timeSteps;
		for(int ii=1; ii<params->mpi.NUMBER_MPI_DOMAINS; ii++){//Notice 'ii' starts at 1 instead of 0
			if( *(timeSteps + ii) < ststep )
				ststep = *(timeSteps + ii);
		}

		params->DT = ststep;
		free(timeSteps);
	}

	free(logic);

}

void GENERAL_FUNCTIONS::checkStability(simulationParameters * params, const meshParams *mesh, const characteristicScales * CS, const vector<ionSpecies> * IONS){

/*	#ifdef ONED
	double Cmax(1);
	#endif

	#ifdef TWOD
	double Cmax(1/sqrt(2));
	#endif

	#ifdef THREED
	double Cmax(1/sqrt(3));
	#endif
*/

	double CourantNumber(0);
	double vx(0),vy(0),vz(0);
	double aux(0);

	#ifdef ONED
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		if(IONS->at(ii).SPECIES != 0){
			double vxTmp(0);
			vec tmp = IONS->at(ii).V.col(0);
			vxTmp = abs(tmp.max());
			vx = (vxTmp > vx) ? vxTmp : vx;
		}
	}//The density still have units here.

	aux = vx/mesh->DX;
	#endif

	#ifdef TWOD
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		if(IONS->at(ii).SPECIES != 0){
			double vxTmp(0), vyTmp(0);
			vec tmp = IONS->at(ii).V.col(0);
			vxTmp = abs(tmp.max());
			vx = (vxTmp > vx) ? vxTmp : vx;
			tmp.reset();
			tmp = IONS->at(ii).V.col(1);
			vyTmp = abs(tmp.max());
			vy = (vyTmp > vy) ? vyTmp : vy;
		}
	}//The density still have units here.

	aux = vx/mesh->DX + vy/mesh->DY;
	#endif

	#ifdef THREED
	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){
		if(IONS->at(ii).SPECIES != 0){
			double vxTmp(0), vyTmp(0), vzTmp(0);
			vec tmp = IONS->at(ii).V.col(0);
			vxTmp = abs(tmp.max());
			vx = (vxTmp > vx) ? vxTmp : vx;
			tmp.reset();
			tmp = IONS->at(ii).V.col(1);
			vyTmp = abs(tmp.max());
			vy = (vyTmp > vy) ? vyTmp : vy;
			tmp.reset();
			tmp = IONS->at(ii).V.col(2);
			vzTmp = abs(tmp.max());
			vz = (vzTmp > vz) ? vzTmp : vz;
		}
	}//The density still have units here.

	aux = vx/mesh->DX + vy/mesh->DY + vz/mesh->DZ;
	#endif

	CourantNumber = params->DT*aux;

	//Reduce and broadcast timestep if the Courant number exceeds the number Cmax.
	double tmpDT(Cmax/aux);
	if(params->DT > tmpDT){
		params->DT =  0.5*tmpDT;
		logicVariable = 1;
	}

	bCastTimestep(params, logicVariable);

}

void GENERAL_FUNCTIONS::checkEnergy(simulationParameters * params, meshParams *mesh, characteristicScales * CS, vector<ionSpecies> * IONS, fields * EB, int IT){

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++){//Iteration over the ions' species
		int NSP(IONS->at(ii).NSP);
		int jj;
		#pragma omp parallel shared(NSP, IONS, ii) private(jj)
		{
			double tmpEnergy(0);
			#pragma omp for
			for(jj=0;jj<NSP;jj++){
				tmpEnergy += IONS->at(ii).V(jj,0)*IONS->at(ii).V(jj,0) //
							+ IONS->at(ii).V(jj,1)*IONS->at(ii).V(jj,1) //
							+ IONS->at(ii).V(jj,2)*IONS->at(ii).V(jj,2);
			}
			tmpEnergy *= 0.5*IONS->at(ii).M*IONS->at(ii).NCP;
			tmpEnergy *= CS->mass*CS->velocity*CS->velocity;//Convert to SI units.

			#pragma omp critical (kinetic_energy)
			{
			params->em->ionsEnergy(IT,ii) += tmpEnergy;
			}
		}//end of the parallel section
	}//Iteration over the ions' species

	forwardPBC_1D(&EB->E.X);
	forwardPBC_1D(&EB->E.Y);
	forwardPBC_1D(&EB->E.Z);

	forwardPBC_1D(&EB->B.X);
	forwardPBC_1D(&EB->B.Y);
	forwardPBC_1D(&EB->B.Z);

	int NX(EB->E.X.n_elem);
	vec tmpVector;

	#pragma omp parallel sections shared(EB, NX) private(tmpVector)
	{

	#pragma omp section
	{
	tmpVector = 0.5*( EB->E.X.subvec(1,NX-2) + EB->E.X.subvec(0,NX-3) );
	tmpVector = tmpVector % tmpVector;
	params->em->E_fieldEnergy(IT,0) = sum(tmpVector);
	params->em->E_fieldEnergy(IT,0) *= 0.5*F_EPSILON*(CS->length*mesh->DX)*(CS->eField*CS->eField);//SI units.
	}

	#pragma omp section
	{
	tmpVector = EB->E.Y.subvec(1,NX-2) % EB->E.Y.subvec(1,NX-2);
	params->em->E_fieldEnergy(IT,1) = sum(tmpVector);
	params->em->E_fieldEnergy(IT,1) *= 0.5*F_EPSILON*(CS->length*mesh->DX)*(CS->eField*CS->eField);//SI units.
	}

	#pragma omp section
	{
	tmpVector = EB->E.Z.subvec(1,NX-2) % EB->E.Z.subvec(1,NX-2);
	params->em->E_fieldEnergy(IT,2) = sum(tmpVector);
	params->em->E_fieldEnergy(IT,2) *= 0.5*F_EPSILON*(CS->length*mesh->DX)*(CS->eField*CS->eField);//SI units.
	}

	#pragma omp section
	{
	tmpVector = 0.5*( EB->B.Y.subvec(1,NX-2) + EB->B.Y.subvec(0,NX-3) ) - params->BGP.By;
	tmpVector = tmpVector % tmpVector;
	params->em->B_fieldEnergy(IT,1) = sum(tmpVector);
	params->em->B_fieldEnergy(IT,1) *= (CS->length*mesh->DX)*(CS->bField*CS->bField)/(2*F_MU);//SI units
	}

	#pragma omp section
	{
	tmpVector = 0.5*( EB->B.Z.subvec(1,NX-2) + EB->B.Z.subvec(0,NX-3) ) - params->BGP.Bz;
	tmpVector = tmpVector % tmpVector;
	params->em->B_fieldEnergy(IT,2) = sum(tmpVector);
	params->em->B_fieldEnergy(IT,2) *= (CS->length*mesh->DX)*(CS->bField*CS->bField)/(2*F_MU);//SI units
	}

	}//end of the parallel region

	for(int ii=0;ii<params->numberOfParticleSpecies;ii++)//Iteration over the ions' species
		params->em->totalEnergy(IT,0) += params->em->ionsEnergy(IT,ii);

	params->em->totalEnergy(IT,0) += params->em->E_fieldEnergy(IT,0) + params->em->E_fieldEnergy(IT,1) //
									+ params->em->E_fieldEnergy(IT,2);

	params->em->totalEnergy(IT,0) += params->em->B_fieldEnergy(IT,1) + params->em->B_fieldEnergy(IT,2);


	restoreVector(&EB->E.X);
	restoreVector(&EB->E.Y);
	restoreVector(&EB->E.Z);

	restoreVector(&EB->B.X);
	restoreVector(&EB->B.Y);
	restoreVector(&EB->B.Z);

	if( (IT+1) == params->transient ){//IT == params->transient
		//params->em->totalEnergy.col(1).fill( mean(params->em->totalEnergy.col(0).subvec(0,params->transient-1)) );
//		params->em->totalEnergy.col(1).fill( mean(params->em->totalEnergy.col(0).subvec(IT-3000,IT+3000)) );
		params->em->totalEnergy.col(1).fill( params->em->totalEnergy(IT) );
	}
}


void GENERAL_FUNCTIONS::saveDiagnosticsVariables(simulationParameters * params){

	string name, path;

	path = params->PATH + "/diagnostics/";

	int N(appliedFilters.size());
	vec AUX = zeros(N);
	for(int ii=0;ii<N;ii++){
		AUX(ii) = appliedFilters[ii];
	}

	name = path + "filters.dat";
	AUX.save(name,raw_ascii);

	N = smoothingParameter.size();
	AUX = zeros(N);
	for(int ii=0;ii<N;ii++){
		AUX(ii) = smoothingParameter[ii];
	}

	name = path + "smoothingParameter.dat";
	AUX.save(name,raw_ascii);

	name = path + "ionsEnergy.dat";
	params->em->ionsEnergy.save(name,raw_ascii);

	name = path + "electricFieldEnergy.dat";
	params->em->E_fieldEnergy.save(name,raw_ascii);

	name = path + "magneticFieldEnergy.dat";
	params->em->B_fieldEnergy.save(name,raw_ascii);

	name = path + "totalEnergy.dat";
	params->em->totalEnergy.save(name,raw_ascii);

}
