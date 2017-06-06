#include "randomStart.h"

//Function to set up the velocity distribution for the LAPD experiment.
void RANDOMSTART::beamVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
					const string parDirection){
	if(parDirection.compare("x") == 0){

		double Vb(8.3217E5);
		ions->velocity.col(0) +=  Vb;
		

	}else if(parDirection.compare("z") == 0){
		//Perpendicular components:
		//		ions->velocity.col(0)
		//		ions->velocity.col(1)
		//Parallel component
		//		ions->velocity.col(2)

		double Vperp(-4.5E5);
		double Vpar(-7E5);

//		double Vperp(-5.3491E5);
//		double Vpar(-6.3748E5);
		
		ions->velocity.col(0) +=  Vperp;
		ions->velocity.col(2) += Vpar;

	}else if(parDirection.compare("xz") == 0){//To be modified.

	}else{
		exit(0);
	}
}

void RANDOMSTART::ringLikeVelocityDistribution(const inputParameters * params, ionSpecies * ions,\
					const string parDirection){

	ions->position = randu<mat>(ions->NSP,3);

	ions->velocity = zeros(ions->NSP,3);

	ions->BGP.V_Tper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.V_Tpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	if(parDirection.compare("x") == 0){
		ions->velocity.col(1) = ions->BGP.V_Tper*cos(phi);
		ions->velocity.col(2) = ions->BGP.V_Tper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->velocity.col(0) = ions->BGP.V_Tpar*sqrt( -log(1-R) ) % sin(phi);
	}else if(parDirection.compare("z") == 0){
		ions->velocity.col(0) = ions->BGP.V_Tper*cos(phi);
		ions->velocity.col(1) = ions->BGP.V_Tper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->velocity.col(2) = ions->BGP.V_Tpar*sqrt( -log(1-R) ) % sin(phi);
	}else if(parDirection.compare("xz") == 0){ //To be modified.
		double THETA(params->BGP.theta*M_PI/180.0);

		vec vx = ions->BGP.V_Tper*cos(phi);
		vec vy = ions->BGP.V_Tper*sin(phi);

		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		vec vz = ions->BGP.V_Tpar*sqrt( -log(1-R) ) % sin(phi);

		ions->velocity.col(0) = vx*cos(THETA) + vz*sin(THETA);
		ions->velocity.col(2) = -vx*sin(THETA) + vz*cos(THETA);
	}else{
		exit(0);
	}
}

//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
void RANDOMSTART::maxwellianVelocityDistribution(const inputParameters * params, ionSpecies * ions, \
					const string parDirection){

	ions->position = randu<mat>(ions->NSP,3);

	ions->velocity = zeros(ions->NSP,3);

	ions->BGP.V_Tper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.V_Tpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	vec phi = 2*M_PI*randu<vec>(ions->NSP);

	if(parDirection.compare("x") == 0){
		ions->velocity.col(1) = ions->BGP.V_Tper*sqrt( -log(1-R) ) % cos(phi);
		ions->velocity.col(2) = ions->BGP.V_Tper*sqrt( -log(1-R) ) % sin(phi);

		arma_rng::set_seed_random();
		R = randu<vec>(ions->NSP);
		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->velocity.col(0) = ions->BGP.V_Tpar*sqrt( -log(1-R) ) % sin(phi);
	}else if(parDirection.compare("z") == 0){
		ions->velocity.col(0) = ions->BGP.V_Tper*sqrt( -log(1-R) ) % cos(phi);
		ions->velocity.col(1) = ions->BGP.V_Tper*sqrt( -log(1-R) ) % sin(phi);

		arma_rng::set_seed_random();
		R = randu<vec>(ions->NSP);
		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		ions->velocity.col(2) = ions->BGP.V_Tpar*sqrt( -log(1.0-R) ) % sin(phi);
	}else if(parDirection.compare("xz") == 0){
		double THETA(params->BGP.theta*M_PI/180.0);

		vec vx = ions->BGP.V_Tper*sqrt( -log(1.0-R) ) % cos(phi);
		vec vy = ions->BGP.V_Tper*sqrt( -log(1.0-R) ) % sin(phi);

		ions->velocity.col(1) = vy;

		arma_rng::set_seed_random();
		R = randu<vec>(ions->NSP);
		arma_rng::set_seed_random();
		phi = 2.0*M_PI*randu<vec>(ions->NSP);

		vec vz = ions->BGP.V_Tpar*sqrt( -log(1-R) ) % sin(phi);

		ions->velocity.col(0) = vx*cos(THETA) + vz*sin(THETA);
		ions->velocity.col(2) = -vx*sin(THETA) + vz*cos(THETA);
	}else{
		exit(0);
	}
}

void RANDOMSTART::shellVelocityDistribution(const inputParameters * params, ionSpecies * ions){
	ions->position = randu<mat>(ions->NSP,3);

	ions->velocity = zeros(ions->NSP,3);

	ions->BGP.V_Tper = sqrt(2.0*F_KB*ions->BGP.Tper/ions->M);
	ions->BGP.V_Tpar = sqrt(2.0*F_KB*ions->BGP.Tpar/ions->M);

	vec theta = 2.0*M_PI*randu<vec>(ions->NSP);
	arma_rng::set_seed_random();
	vec phi = acos( 2.0*randu<vec>(ions->NSP) - 1.0 );

	ions->velocity.col(0) = ions->BGP.V_Tper*cos(theta) % sin(phi);
	ions->velocity.col(1) = ions->BGP.V_Tper*sin(theta) % sin(phi);
	ions->velocity.col(2) = ions->BGP.V_Tpar*cos(phi);
}
