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

#include "quietStart.h"

template <class IT> QUIETSTART<IT>::QUIETSTART(const simulationParameters * params, IT * ions){

    // Unitary vector along B field
	b1 = {params->BGP.Bx, params->BGP.By, params->BGP.Bz};
	b1 = arma::normalise(b1);

	if (arma::dot(b1,y) < PRO_ZERO){
		b2 = arma::cross(b1,y);
	}else{
		b2 = arma::cross(b1,z);
	}

	// Unitary vector perpendicular to b1 and b2
	b3 = arma::cross(b1,b2);


    recalculateNumberSuperParticles(params, ions);

	// Number of super-particles in entire simulation
	unsigned int NTSP = (unsigned int)((int)(2*ions->NSP)*params->mpi.MPIS_PARTICLES);

    QUIETSTART::dec = zeros<uvec>(NTSP);
    for(unsigned int ii=1; ii<NTSP; ii++){
        QUIETSTART::dec(ii) = ii;
    }
}


template <class IT> void QUIETSTART<IT>::recalculateNumberSuperParticles(const simulationParameters * params, IT * ions){
	//Definition of the initial number of superparticles for each species
    double exponent = ceil(log(ions->NPC*params->mesh.NUM_NODES_IN_SIM)/log(2.0));

	ions->NSP = ceil( pow(2.0,exponent)/(double)params->mpi.MPIS_PARTICLES );

	ions->nSupPartOutput = floor( (ions->pctSupPartOutput/100.0)*ions->NSP );
}


template <class IT> vector<int> QUIETSTART<IT>::dec2bin(int dec){
    vector<int> bin;
    while(dec != 0){
        bin.push_back(dec%2);
        dec = (int)floor((double)dec/2.0);
    }
    return(bin);
}


template <class IT> vector<int> QUIETSTART<IT>::dec2b3(int dec){
    vector<int> b3;
    while(dec != 0){
        b3.push_back(dec%3);
        dec = (int)floor((double)dec/3.0);
    }
    return(b3);
}


template <class IT> void QUIETSTART<IT>::bit_reversedFractions_base2(const simulationParameters * params, unsigned int NSP, vec * b2fr){
    const unsigned int sf(50);
    vec fracs = zeros(sf);

    for(unsigned int ii=0;ii<sf;ii++)
        fracs(ii) = 1.0/pow(2.0,(double)(ii+1));

	unsigned int iInd = NSP*((unsigned int)params->mpi.COMM_RANK);
	for(unsigned int ii=0;ii<NSP;ii++){
		vector<int> bin = dec2bin(QUIETSTART::dec(ii + iInd));
		for(unsigned int jj=0;jj<bin.size();jj++){
			(*b2fr)(ii) += ((double)bin.at(jj))*fracs(jj);
		}
	}

}


template <class IT> void QUIETSTART<IT>::bit_reversedFractions_base3(const simulationParameters * params, unsigned int NSP, vec * b3fr){
    const unsigned int sf(50);
    vec fracs = zeros(sf);

    for(unsigned int ii=0;ii<sf;ii++)
        fracs(ii) = 1.0/pow(3.0,(double)(ii+1));


	unsigned int iInd = NSP*((unsigned int)params->mpi.COMM_RANK);
	for(unsigned int ii=0;ii<NSP;ii++){
		vector<int> b3 = dec2b3(QUIETSTART::dec(ii + iInd));
		for(unsigned int jj=0;jj<b3.size();jj++){
			(*b3fr)(ii) += ((double)b3.at(jj))*fracs(jj);
		}
	}
}

//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution in 2D3V.
template <class IT> void QUIETSTART<IT>::maxwellianVelocityDistribution(const simulationParameters * params, IT * ions){
    ions->X = zeros<mat>(ions->NSP,3);
    ions->V = zeros<mat>(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
    ions->g = zeros(ions->NSP);
    ions->mu = zeros(ions->NSP);

    // Initialising positions
	vec b2fr = zeros(ions->NSP);
    bit_reversedFractions_base2(params, ions->NSP, &b2fr);

	vec b3fr = zeros(ions->NSP);
    bit_reversedFractions_base3(params, ions->NSP, &b3fr);

	ions->X.col(0) = b2fr;
	ions->X.col(1) = b3fr;

	// We scale the positions
	ions->X.col(0) *= params->mesh.LX;
	ions->X.col(1) *= params->mesh.LY;

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->VTper*sqrt( -log(1.0 - R) ) % cos(phi);
	arma::vec V3 = ions->VTper*sqrt( -log(1.0 - R) ) % sin(phi);

	arma_rng::set_seed_random();
	R = randu<vec>(ions->NSP);
	arma_rng::set_seed_random();
	phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V1 = ions->VTpar*sqrt( -log(1.0 - R) ) % sin(phi);


	for(int pp=0;pp<ions->NSP;pp++){
		ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
		ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
		ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

		ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
		ions->mu(pp) = 0.5*ions->g(pp)*ions->g(pp)*ions->M*( V2(pp)*V2(pp) + V3(pp)*V3(pp) )/params->BGP.Bo;
		ions->Ppar(pp) = ions->g(pp)*ions->M*V1(pp);
	}

    ions->avg_mu = mean(ions->mu);
}


//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution in 2D3V
template <class IT> void QUIETSTART<IT>::ringLikeVelocityDistribution(const simulationParameters * params, IT * ions){
    ions->X = zeros<mat>(ions->NSP,3);
    ions->V = zeros<mat>(ions->NSP,3);
	ions->g = zeros(ions->NSP);
    ions->Ppar = zeros(ions->NSP);
    ions->mu = zeros(ions->NSP);

    // Initialising positions
	vec b2fr = zeros(ions->NSP);
    bit_reversedFractions_base2(params, ions->NSP, &b2fr);

	vec b3fr = zeros(ions->NSP);
    bit_reversedFractions_base3(params, ions->NSP, &b3fr);

	ions->X.col(0) = b2fr;
	ions->X.col(1) = b3fr;

	// We scale the positions
	ions->X.col(0) *= params->mesh.LX;
	ions->X.col(1) *= params->mesh.LY;

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->VTper*cos(phi);
	arma::vec V3 = ions->VTper*sin(phi);

	arma_rng::set_seed_random();
	phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V1 = ions->VTpar*sqrt( -log(1.0 - R) ) % sin(phi);


	for(int pp=0;pp<ions->NSP;pp++){
		ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
		ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
		ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

		ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
        ions->mu(pp) = 0.5*ions->g(pp)*ions->g(pp)*ions->M*( V2(pp)*V2(pp) + V3(pp)*V3(pp) )/params->BGP.Bo;
		ions->Ppar(pp) = ions->g(pp)*ions->M*V1(pp);
	}

    ions->avg_mu = mean(ions->mu);
}


template class QUIETSTART<oneDimensional::ionSpecies>;
template class QUIETSTART<twoDimensional::ionSpecies>;
