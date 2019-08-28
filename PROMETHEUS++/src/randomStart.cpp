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

#include "randomStart.h"

RANDOMSTART::RANDOMSTART(const inputParameters * params){
	// Unitary vector along B field
	b1 = {sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0), \
	      sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0),\
		  cos(params->BGP.theta*M_PI/180.0)};

	// Unitary vector perpendicular to b1
	b2 = arma::cross(b1,y);

	// Unitary vector perpendicular to b1 and b2
	b3 = arma::cross(b1,b2);
}


void RANDOMSTART::ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions){

	ions->X = randu<mat>(ions->NSP,3);
	ions->V = zeros(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
	ions->g = zeros(ions->NSP);
	ions->mu = zeros(ions->NSP);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->BGP.VTper*cos(phi);
	arma::vec V3 = ions->BGP.VTper*sin(phi);

	arma_rng::set_seed_random();
	phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V1 = ions->BGP.VTpar*sqrt( -log(1.0 - R) ) % sin(phi);

	for(int pp=0;pp<ions->NSP;pp++){
		ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
		ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
		ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

		ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
		ions->mu(pp) = 0.5*ions->g(pp)*ions->g(pp)*ions->M*( V2(pp)*V2(pp) + V3(pp)*V3(pp) )/params->BGP.Bo;
		ions->Ppar(pp) = ions->g(pp)*ions->M*V1(pp);
	}

	ions->BGP.mu = mean(ions->mu);
}


//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
void RANDOMSTART::maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions){

	ions->X = randu<mat>(ions->NSP,3);
	ions->V = zeros(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
	ions->g = zeros(ions->NSP);
	ions->mu = zeros(ions->NSP);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->BGP.VTper*sqrt( -log(1.0 - R) ) % cos(phi);
	arma::vec V3 = ions->BGP.VTper*sqrt( -log(1.0 - R) ) % sin(phi);

	arma_rng::set_seed_random();
	R = randu<vec>(ions->NSP);
	arma_rng::set_seed_random();
	phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V1 = ions->BGP.VTpar*sqrt( -log(1.0 - R) ) % sin(phi);

	for(int pp=0;pp<ions->NSP;pp++){
		ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
		ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
		ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

		ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
		ions->mu(pp) = 0.5*ions->g(pp)*ions->g(pp)*ions->M*( V2(pp)*V2(pp) + V3(pp)*V3(pp) )/params->BGP.Bo;
		ions->Ppar(pp) = ions->g(pp)*ions->M*V1(pp);
	}

	ions->BGP.mu = mean(ions->mu);
}
