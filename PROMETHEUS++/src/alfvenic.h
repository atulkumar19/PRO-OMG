#ifndef H_ALFVENIC
#define H_ALFVENIC

#include <cmath>
#include <iostream>
#include <vector>

#include <armadillo>
#include "structures.h"
#include "boundaryConditions.h"
#include "types.h"

#include "mpi_main.h"

using namespace arma;
using namespace std;

class ALFVENIC{
	#define A_kB 1.3807E-16
	#define A_C 2.9979E10
	#define A_Zp 1.0l
	#define A_Za 2.0l
	#define A_q 4.8032E-10
	#define A_qa A_q*A_Za
	#define A_qp A_q*A_Zp
	#define A_mp 1.6726E-24
	#define A_ma 6.6436605E-24
	#define A_me 9.1094E-28

	struct{
		double L;
		vec amp;
		vec wavenumber;
		vec angularFreq;
		vec phase;
		vector<arma::mat> Uo;
		vfield_vec dB;

		unsigned int numberOfTestModes;
		vec kappa;
		vec omega;
	}Aw;

	struct plasmaParams{//in cgs units

		double np, na, ne;
		double B;
		double wca, wcp, wce;
		double wpa, wpp, wpe;
		double VA;
		double dp;
		plasmaParams(const inputParameters * params,vector<ionSpecies> * IONS){
			np = (1/1E6)*IONS->at(0).BGP.Dn*params->ne;//1/cm^3
			na = (1/1E6)*IONS->at(1).BGP.Dn*params->ne;//1/cm^3
			ne = np + A_Za*na;

			B = (1E4)*params->BGP.Bo;//in gauss

			wca = A_qa*B/(A_ma*A_C);
			wcp = A_qp*B/(A_mp*A_C);
			wce = A_q*B/(A_me*A_C);
			wpe = sqrt( 4*M_PI*ne*A_q*A_q/A_me );
			wpp = sqrt( 4*M_PI*np*A_qp*A_qp/A_mp );
			wpa = sqrt( 4*M_PI*na*A_qa*A_qa/A_ma );

			VA = B/sqrt(4*M_PI*np*A_mp);
			dp = VA/wcp;
		}

	};

	double function(const plasmaParams *PP,double w,double k);

	void dispertionRelation(const plasmaParams *PP);

	double dispertionRelation(double w,plasmaParams * PP);

	double brentRoots(const plasmaParams *PP,double x1,double x2,double k,int ITMAX);

	void addMagneticPerturbations(fields * EB);

	void addVelocityPerturbations(const inputParameters * params,vector<ionSpecies> * IONS);

	void generateModes(const inputParameters * params,const meshGeometry * mesh,fields * EB,vector<ionSpecies> * IONS);

	void loadModes(const inputParameters * params,const meshGeometry * mesh,fields * EB,vector<ionSpecies> * IONS);


public:

	ALFVENIC(const inputParameters * params,const meshGeometry * mesh,fields * EB,vector<ionSpecies> * IONS);

	void normalize(const characteristicScales * CS);

	void addPerturbations(const inputParameters * params,vector<ionSpecies> * IONS,fields * EB);

};

#endif
