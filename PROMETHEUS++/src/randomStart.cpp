#include "randomStart.h"

//Function to set up the velocity distribution for the LAPD experiment.
void RANDOMSTART::beamVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
					const string parDirection){
	if(parDirection.compare("x") == 0){

		double Vb(8.3217E5);
		ions->V.col(0) +=  Vb;


	}else if(parDirection.compare("z") == 0){
		//Perpendicular components:
		//		ions->V.col(0)
		//		ions->V.col(1)
		//Parallel component
		//		ions->V.col(2)

		double Vperp(-4.5E5);
		double Vpar(-7E5);

//		double Vperp(-5.3491E5);
//		double Vpar(-6.3748E5);

		ions->V.col(0) +=  Vperp;
		ions->V.col(2) += Vpar;

	}else if(parDirection.compare("xz") == 0){//To be modified.

	}else{
		exit(0);
	}
}


void RANDOMSTART::ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
					const string parDirection){

	ions->X = randu<mat>(ions->NSP,3);

	ions->V = zeros(ions->NSP,3);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	if(parDirection.compare("x") == 0){
		ions->V.col(1) = ions->BGP.VTper*cos(phi);
		ions->V.col(2) = ions->BGP.VTper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->V.col(0) = ions->BGP.VTpar*sqrt( -log(1-R) ) % sin(phi);
	}else if(parDirection.compare("z") == 0){
		ions->V.col(0) = ions->BGP.VTper*cos(phi);
		ions->V.col(1) = ions->BGP.VTper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->V.col(2) = ions->BGP.VTpar*sqrt( -log(1-R) ) % sin(phi);
	}else if(parDirection.compare("xz") == 0){ //To be modified.
		double THETA(params->BGP.theta*M_PI/180.0);

		vec vx = ions->BGP.VTper*cos(phi);
		vec vy = ions->BGP.VTper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		vec vz = ions->BGP.VTpar*sqrt( -log(1-R) ) % sin(phi);

		ions->V.col(0) = vx*cos(THETA) + vz*sin(THETA);
		ions->V.col(2) = -vx*sin(THETA) + vz*cos(THETA);
	}else{
		exit(0);
	}
}


//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
void RANDOMSTART::maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions){

	ions->X = randu<mat>(ions->NSP,3);
	ions->V = zeros(ions->NSP,3);
	ions->g = zeros(ions->NSP);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	// Cartesian unitary vectors
	arma::vec x = {1.0, 0.0, 0.0};
	arma::vec y = {0.0, 1.0, 0.0};
	arma::vec z = {0.0, 0.0, 1.0};

	// Unitary vector along B field
	arma::vec b1 = {sin(params->BGP.theta*M_PI/180.0)*cos(params->BGP.phi*M_PI/180.0), \
				    sin(params->BGP.theta*M_PI/180.0)*sin(params->BGP.phi*M_PI/180.0),\
				    cos(params->BGP.theta*M_PI/180.0)};

	// Unitary vector perpendicular to b1
	arma::vec b2 = arma::cross(b1,z);

	// Unitary vector perpendicular to b1 and b2
	arma::vec b3 = arma::cross(b1,b2);

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->BGP.VTper*sqrt( -log(1-R) ) % cos(phi);
	arma::vec V3 = ions->BGP.VTper*sqrt( -log(1-R) ) % sin(phi);

	arma_rng::set_seed_random();
	R = randu<vec>(ions->NSP);
	arma_rng::set_seed_random();
	phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V1 = ions->BGP.VTpar*sqrt( -log(1-R) ) % sin(phi);

	for(int pp=0;pp<ions->NSP;pp++){
		ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
		ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
		ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

		ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
	}
}


void RANDOMSTART::shellVelocityDistribution(const inputParameters * params, ionSpecies * ions){
	ions->X = randu<mat>(ions->NSP,3);

	ions->V = zeros(ions->NSP,3);

	ions->BGP.VTper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.VTpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	vec theta = 2.0*M_PI*randu<vec>(ions->NSP);
	arma_rng::set_seed_random();
	vec phi = acos( 2.0*randu<vec>(ions->NSP) - 1.0 );

	ions->V.col(0) = ions->BGP.VTper*cos(theta) % sin(phi);
	ions->V.col(1) = ions->BGP.VTper*sin(theta) % sin(phi);
	ions->V.col(2) = ions->BGP.VTpar*cos(phi);
}
