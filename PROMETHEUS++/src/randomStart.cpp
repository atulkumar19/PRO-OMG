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

template <class IT> RANDOMSTART<IT>::RANDOMSTART(const simulationParameters * params){
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
}


template <class IT> void RANDOMSTART<IT>::ringLikeVelocityDistribution(const simulationParameters * params, IT * ions){
	arma_rng::set_seed_random();
	ions->X = randu<mat>(ions->NSP,3);

	ions->V = zeros(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
	ions->g = zeros(ions->NSP);
	ions->mu = zeros(ions->NSP);

	// We scale the positions
	ions->X.col(0) *= params->mesh.LX;
	ions->X.col(1) *= params->mesh.LY;

	arma::vec R = randu(ions->NSP);
	arma_rng::set_seed_random();
	arma::vec phi = 2.0*M_PI*randu<vec>(ions->NSP);

	arma::vec V2 = ions->VTper*sqrt( -log(1.0 - R) ) % cos(phi);
	arma::vec V3 = ions->VTper*sqrt( -log(1.0 - R) ) % sin(phi);

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


//This function creates a Maxwellian velocity distribution for ions with a homogeneous spatial distribution.
template <class IT> void RANDOMSTART<IT>::maxwellianVelocityDistribution(const simulationParameters * params, IT * ions){
	arma_rng::set_seed_random();
	ions->X = randu<mat>(ions->NSP,3);

	ions->V = zeros(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
	ions->g = zeros(ions->NSP);
	ions->mu = zeros(ions->NSP);

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

//*** @non-uniform
template <class IT> double RANDOMSTART<IT>::target(const simulationParameters * params, IT * ions, double X, double V3, double V2, double V1){


              arma::vec S = linspace(0,params->mesh.LX,200);
              arma::vec xx(1,1);
              xx(0,0)= X;

                double T3=0.0;
                arma::vec TT3(1,1);
                interp1(S,params->PP.Tper,xx,TT3);
                T3 = (ions->Tper)*TT3(0,0);   //Temperature profile in x

                double T2=0.0;
                arma::vec TT2(1,1);
                interp1(S,params->PP.Tper,xx,TT2);
                T2 =  (ions->Tper)*TT2(0,0);   //Temperature profile in y

                double T1=0.0;
                arma::vec TT1(1,1);
                interp1(S,params->PP.Tpar,xx,TT1);
                T1 =  (ions->Tpar)*TT1(0,0);   //Temperature profile in z

        double k3=sqrt((ions->M)/(2.0*M_PI*F_KB*T3));
	double k2=sqrt((ions->M)/(2.0*M_PI*F_KB*T2));
	double k1=sqrt((ions->M)/(2.0*M_PI*F_KB*T1));
	double s3=sqrt((ions->M)/(2.0*F_KB*T3));
	double s2=sqrt((ions->M)/(2.0*F_KB*T2));
	double s1=sqrt((ions->M)/(2.0*F_KB*T1));

	double h= (k1*k2*k3)*exp(-sqrt(2)*(((V1*V1)*(s1*s1))+((V2*V2)*(s2*s2))+((V3*V3)*(s3*s3)))); // Pdf for 3-Velocities with temperature profile


              double g=0.0;
              arma::vec gg(1,1);
              interp1(S,params->PP.ne,xx,gg);
              g = gg(0,0); //density profile



	return(g*h); //target 4-D Pdf
}


//*** @non-uniform
template <class IT> void RANDOMSTART<IT>::maxwellianVelocityDistribution_nonhomogeneous(const simulationParameters * params, IT * ions){
// Initialize to zero all ion variables
              ions->X = zeros(ions->NSP,3);
	ions->V = zeros(ions->NSP,3);
	ions->Ppar = zeros(ions->NSP);
	ions->g = zeros(ions->NSP);
	ions->mu = zeros(ions->NSP);

	// We use the Metropolis-Hastings algorithm for sampling the 4-D target PDF.

	arma::vec X(ions->NSP);  // Particles position along x-axis that will be generated using M-H.
	arma::vec V3(ions->NSP);  //Velocity profile in X
	arma::vec V2(ions->NSP);  //Velocity profile in Y
	arma::vec V1(ions->NSP); //Velocity profile in Z

              double X_test = 0.0;
              double V3_test= -1.0;
              double V2_test= -1.0;
              double V1_test= -1.0;




              std::default_random_engine generator(params->mpi.MPI_DOMAIN_NUMBER+1);

              std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0); // Generator of random, uniformly distributed numbers in [0,1)

              //std::cout << "From Rank: " << params->mpi.MPI_DOMAIN_NUMBER << std::endl;
              //std::cout << uniform_distribution(generator) << " " << std::endl;

	unsigned int iterator = 1;

              double ratio = 0.0;
              X(0)=0.0;
              V3(0)=0.0;
              V2(0)=0.0;
              V1(0)=0.0;

                while(iterator < ions->NSP){
                        X_test = uniform_distribution(generator)*params->mesh.LX;
                        V3_test= 10*(ions->VTper)*((uniform_distribution(generator))-0.5);
                        V2_test= 10*(ions->VTper)*((uniform_distribution(generator))-0.5);
                        V1_test= 10*(ions->VTpar)*((uniform_distribution(generator))-0.5);

                ratio = target(params, ions, X_test,V3_test, V2_test, V1_test)/target(params, ions,  X(iterator - 1),V3(iterator - 1),V2(iterator - 1),V1(iterator - 1));


                if (ratio >= 1.0){
                            X(iterator) = X_test;
                            V3(iterator) = V3_test;
                            V2(iterator) = V2_test;
                            V1(iterator) = V1_test;
                             iterator += 1;
                }else{
                              if (uniform_distribution(generator) < ratio){
                                  X(iterator) = X_test;
                                  V3(iterator) = V3_test;
                                  V2(iterator) = V2_test;
                                  V1(iterator) = V1_test;
                                  iterator += 1;

                              }
                     }
              }

                 for(int pp=0;pp<ions->NSP;pp++){

                         ions->V(pp,0) = V1(pp)*dot(b1,x) + V2(pp)*dot(b2,x) + V3(pp)*dot(b3,x);
                         ions->V(pp,1) = V1(pp)*dot(b1,y) + V2(pp)*dot(b2,y) + V3(pp)*dot(b3,y);
                         ions->V(pp,2) = V1(pp)*dot(b1,z) + V2(pp)*dot(b2,z) + V3(pp)*dot(b3,z);

                         ions->g(pp) = 1.0/sqrt( 1.0 - dot(ions->V.row(pp),ions->V.row(pp))/(F_C*F_C) );
                         ions->mu(pp) = 0.5*ions->g(pp)*ions->g(pp)*ions->M*( V2(pp)*V2(pp) + V3(pp)*V3(pp) )/params->BGP.Bo;
                         ions->Ppar(pp) = ions->g(pp)*ions->M*V1(pp);
        }
                ions->X.col(0) = X;
               // ions->X.col(0) *= params->mesh.LX;
               // ions->X.col(1) *= params->mesh.LY;

                ions->avg_mu = mean(ions->mu);
}


template class RANDOMSTART<oneDimensional::ionSpecies>;
template class RANDOMSTART<twoDimensional::ionSpecies>;
