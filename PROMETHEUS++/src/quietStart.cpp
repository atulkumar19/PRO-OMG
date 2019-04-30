#include "quietStart.h"

QUIETSTART::QUIETSTART(const inputParameters * params, ionSpecies * ions){

    recalculateNumberSuperParticles(params,ions);

	unsigned int NTSP = (unsigned int)((int)(ions->NSP)*params->mpi.NUMBER_MPI_DOMAINS);

    QUIETSTART::dec = zeros<uvec>(NTSP);
    for(unsigned int ii=1; ii<NTSP; ii++){
        QUIETSTART::dec(ii) = ii;
    }
}

double QUIETSTART::recalculateNumberSuperParticles(const inputParameters * params, ionSpecies *ions){
	//Definition of the initial number of superparticles for each species
    double exponent;
    exponent = ceil(log(ions->NPC*params->meshDim(0)*params->mpi.NUMBER_MPI_DOMAINS)/log(2.0));
    ions->NSP = ceil( pow(2.0,exponent)/(double)params->mpi.NUMBER_MPI_DOMAINS );

	ions->nSupPartOutput = floor( (ions->pctSupPartOutput/100.0)*ions->NSP );
}

vector<int> QUIETSTART::dec2bin(int dec){
    vector<int> bin;
    while(dec != 0){
        bin.push_back(dec%2);
        dec = (int)floor((double)dec/2.0);
    }
    return(bin);
}

vector<int> QUIETSTART::dec2b3(int dec){
    vector<int> b3;
    while(dec != 0){
        b3.push_back(dec%3);
        dec = (int)floor((double)dec/3.0);
    }
    return(b3);
}

void QUIETSTART::bit_reversedFractions_base2(const inputParameters * params, ionSpecies * ions, vec * b2fr){
    const unsigned int sf(50);
    vec fracs = zeros(sf);
    for(unsigned int ii=0;ii<sf;ii++)
        fracs(ii) = 1.0/pow(2.0,(double)(ii+1));

	if(params->mpi.rank_cart != 0){
		unsigned int iInd = ions->NSP*params->mpi.rank_cart;
		for(unsigned int ii=0;ii<ions->NSP;ii++){
			vector<int> bin = dec2bin(QUIETSTART::dec(ii + iInd));
	    	for(unsigned int jj=0;jj<bin.size();jj++){
	        	(*b2fr)(ii) += ((double)bin.at(jj))*fracs(jj);
	    	}
		}
	}else{
		for(unsigned int ii=1;ii<ions->NSP;ii++){
			vector<int> bin = dec2bin(QUIETSTART::dec(ii));
	    	for(unsigned int jj=0;jj<bin.size();jj++){
	        	(*b2fr)(ii) += ((double)bin.at(jj))*fracs(jj);
	    	}
		}
	}
}

void QUIETSTART::bit_reversedFractions_base3(const inputParameters * params, ionSpecies * ions, vec * b3fr){
    const unsigned int sf(50);
    vec fracs = zeros(sf);
    for(unsigned int ii=0;ii<sf;ii++)
        fracs(ii) = 1.0/pow(3.0,(double)(ii+1));

	if(params->mpi.rank_cart != 0){
		unsigned int iInd = ions->NSP*params->mpi.rank_cart;
		for(unsigned int ii=0;ii<ions->NSP;ii++){
		    vector<int> b3 = dec2b3(QUIETSTART::dec(ii + iInd));
		    for(unsigned int jj=0;jj<b3.size();jj++){
		        (*b3fr)(ii) += ((double)b3.at(jj))*fracs(jj);
		    }
		}
	}else{
		for(unsigned int ii=1;ii<ions->NSP;ii++){
		    vector<int> b3 = dec2b3(QUIETSTART::dec(ii));
		    for(unsigned int jj=0;jj<b3.size();jj++){
		        (*b3fr)(ii) += ((double)b3.at(jj))*fracs(jj);
		    }
		}
	}
}

//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
void QUIETSTART::maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
				const string parDirection){
    ions->X = zeros<mat>(ions->NSP,3);
    ions->V = zeros<mat>(ions->NSP,3);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

    // Initialising positions
    vec b2fr = zeros(ions->NSP);
    bit_reversedFractions_base2(params, ions, &b2fr);

    ions->X.col(0) = b2fr;

    // Initialising gyro-angle

    vec b3fr = zeros(ions->NSP);
    bit_reversedFractions_base3(params, ions, &b3fr);

    b3fr *= 2.0*M_PI;

    vec R = zeros(ions->NSP);
    for(int ii=0;ii<ions->NSP;ii++)
        R(ii) = ((double)QUIETSTART::dec(ii) + 0.5 )/ions->NSP;

	if(parDirection.compare("x") == 0){

		ions->V.col(1) = ions->BGP.VTper*sqrt( -log( R ) ) % cos(b3fr);
		ions->V.col(2) = ions->BGP.VTper*sqrt( -log( R ) ) % sin(b3fr);

        srand(time(NULL));
        double randPhase( (int)(rand()%100)/100.0);
        randPhase *= 2.0*M_PI;

//		ions->V.col(0) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(b3fr + randPhase);
		ions->V.col(0) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(b3fr);

	}else if(parDirection.compare("z") == 0){

		ions->V.col(0) = ions->BGP.VTper*sqrt( -log( R ) ) % cos(b3fr);
		ions->V.col(1) = ions->BGP.VTper*sqrt( -log( R ) ) % sin(b3fr);

        srand(time(NULL));
        double randPhase( (int)(rand()%100)/100.0);
        randPhase *= 2.0*M_PI;

//		ions->V.col(2) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(b3fr + randPhase);
		ions->V.col(2) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(b3fr);

	}else if(parDirection.compare("xz") == 0){//To be modified.


	}else{
		exit(0);
	}
}

//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
void QUIETSTART::ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
					const string parDirection){
    ions->X = zeros<mat>(ions->NSP,3);
    ions->V = zeros<mat>(ions->NSP,3);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

    // Initialising positions
    vec b2fr = zeros(ions->NSP);
    bit_reversedFractions_base2(params, ions, &b2fr);

    ions->X.col(0) = b2fr;

    // Initialising gyro-angle

    vec b3fr = zeros(ions->NSP);
    bit_reversedFractions_base3(params, ions, &b3fr);

    b3fr *= 2.0*M_PI;

    vec R = zeros(ions->NSP);
    for(int ii=0;ii<ions->NSP;ii++)
        R(ii) = ((double)QUIETSTART::dec(ii) + 0.5 )/ions->NSP;

	vec tmp = randu<vec>(ions->NSP*params->mpi.NUMBER_MPI_DOMAINS);
	unsigned int iInd = ions->NSP*params->mpi.rank_cart;
	unsigned int fInd = iInd + ions->NSP - 1;

	vec phi = 2.0*M_PI*tmp.subvec(iInd,fInd);

	if(parDirection.compare("x") == 0){

		ions->V.col(1) = ions->BGP.VTper*cos(phi);
		ions->V.col(2) = ions->BGP.VTper*sin(phi);

        srand(time(NULL));
        double randPhase( (int)(rand()%100)/100.0);
        randPhase *= 2.0*M_PI;

//		ions->V.col(0) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(b3fr + randPhase);
		ions->V.col(0) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(phi);

	}else if(parDirection.compare("z") == 0){

		ions->V.col(0) = ions->BGP.VTper*cos(phi);
		ions->V.col(1) = ions->BGP.VTper*sin(phi);

		arma_rng::set_seed_random();
		tmp = randu<vec>(ions->NSP*params->mpi.NUMBER_MPI_DOMAINS);
		phi = 2*M_PI*tmp.subvec(iInd,fInd);

		ions->V.col(2) = ions->BGP.VTpar*sqrt( -log( R ) ) % sin(phi);

	}else if(parDirection.compare("xz") == 0){//To be modified.


	}else{
		exit(0);
	}
}
