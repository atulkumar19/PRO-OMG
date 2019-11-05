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

#ifndef H_TYPES
#define H_TYPES

#include <vector>
#include <armadillo>

// * * * * * * * * NAMESPACES  * * * * * * * * //
namespace oneDimensional{
	class electromagneticFields;
	class ionSpeciesParams;
	class GCSpeciesParams;
}


namespace twoDimensional{
	class electromagneticFields;
}


namespace threeDimensional{
	class electromagneticFields;
}
// * * * * * * * * NAMESPACES  * * * * * * * * //


// * * * * * * * * VECTOR FIELD TYPES  * * * * * * * * //
class vfield_vec{

public:
	arma::vec X;
	arma::vec Y;
	arma::vec Z;

	vfield_vec(){};
	vfield_vec(unsigned int N) : X(N), Y(N), Z(N) {}

	vfield_vec operator + (vfield_vec R);
	vfield_vec operator += (vfield_vec R);
	vfield_vec operator - (vfield_vec R);
	vfield_vec operator -= (vfield_vec R);
	vfield_vec operator * (double s);
	vfield_vec operator *= (double s);
    friend vfield_vec operator * (double s, vfield_vec R );
	vfield_vec operator / (double s);
	vfield_vec operator / (vfield_vec R);
	vfield_vec operator /= (double s);
	vfield_vec operator /= (vfield_vec R);


	void fill(double value);
	void ones(unsigned int N);
	void zeros();
	void zeros(unsigned int N);
};


class vfield_mat{

public:
	arma::mat X;
	arma::mat Y;
	arma::mat Z;

	vfield_mat(){};
	vfield_mat(unsigned int N, unsigned int M) : X(N,M), Y(N,M), Z(N,M) {}

	vfield_mat operator + (vfield_mat R);
	vfield_mat operator += (vfield_mat R);
	vfield_mat operator - (vfield_mat R);
	vfield_mat operator -= (vfield_mat R);
	vfield_mat operator * (double s);
	vfield_mat operator *= (double s);
    friend vfield_mat operator * (double s, vfield_mat R );
	vfield_mat operator / (double s);
	vfield_mat operator / (vfield_mat R);
	vfield_mat operator /= (double s);
	vfield_mat operator /= (vfield_mat R);

	void fill(double value);
	void ones(unsigned int N, unsigned int M);
	void zeros();
	void zeros(unsigned int N, unsigned int M);
};


class vfield_cube{

public:
	arma::cube X;
	arma::cube Y;
	arma::cube Z;

	vfield_cube(){};
	vfield_cube(unsigned int N, unsigned int M, unsigned int P) : X(N,M,P), Y(N,M,P), Z(N,M,P) {}

	vfield_cube operator + (vfield_cube R);
	vfield_cube operator += (vfield_cube R);
	vfield_cube operator - (vfield_cube R);
	vfield_cube operator -= (vfield_cube R);
	vfield_cube operator * (double s);
	vfield_cube operator *= (double s);
    friend vfield_cube operator * (double s, vfield_cube R );
	vfield_cube operator / (double s);
	vfield_cube operator / (vfield_cube R);
	vfield_cube operator /= (double s);
	vfield_cube operator /= (vfield_cube R);

	void fill(double value);
	void ones(unsigned int N, unsigned int M, unsigned int P);
	void zeros();
	void zeros(unsigned int N, unsigned int M, unsigned int P);
};
// * * * * * * * * VECTOR FIELD TYPES  * * * * * * * * //


// * * * * * * * * ION VARIABLES AND PARAMETERS DERIVED TYPES  * * * * * * * * //
struct ionsInitialMeanParameters{
	double Dn;

	double Tpar;		// Parallel temperature.
	double Tper;		// Perpendicular temperature.
	double LarmorRadius;// Larmor radius.
	double VTper;		// Thermal velocity.
	double VTpar;		// Thermal velocity.
	double Wc;			// Average cyclotron frequency.
	double Wpi;			// Ion plasma frequency.
	double mu; 			// Average magnetic moment
};


class oneDimensional::ionSpeciesParams : public vfield_vec{

public:
	int SPECIES;
	int IC; // Initial condition IC=1 (Maxwellian), IC=2 (ring-like)
	double NSP; // Initial number of superparticles for the given ion species.
	double NCP; // Number of charged particles per superparticle.
	double NPC; // Number of superparticles per cell. When its value is zero, the particles are loaded from external files.
	double Q; 	// Charge.
	double Z; 	// Atomic number.
	double M; 	// Mass

	// variables for controlling super-particles' outputs
	double pctSupPartOutput;
	unsigned int nSupPartOutput;

	ionsInitialMeanParameters BGP;

	arma::mat X; 		// Ions position, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat V; 		// Ions' velocity, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat P; 		// Ions' momentum, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::vec g; 		// Ions' relativistic gamma factor.
	arma::ivec meshNode; // Position of each particle in the discrete mesh.

	// Guiding-center variables
	arma::vec mu; 	// Ions' magnetic moment.
	arma::vec Ppar; // Parallel momentum used in guiding-center orbits

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl, wxc, wxr;	// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxll, wxrr;		// Particles' weights w.r.t. the vertices of the grid cells. Third-order particle interpolation

	arma::vec wyl, wyc, wyr;
	arma::vec wzl, wzc, wzr;

	arma::vec n; 		// Ion density at time level "l + 1"
	arma::vec n_; 		// Ion density at time level "l - 1;
	arma::vec n__; 		// Ion density at time level "l - 2;
	arma::vec n___; 		// Ion density at time level "l - 3;
	vfield_vec nv; 		// Ion bulk velocity at time level "l + 1/2"
	vfield_vec nv_; 	// Ion bulk velocity at time level "l - 1/2"
	vfield_vec nv__; 	// Ion bulk velocity at time level "l - 3/2"
};
// * * * * * * * * ION VARIABLES AND PARAMETERS DERIVED TYPES  * * * * * * * * //


class oneDimensional::GCSpeciesParams : public vfield_vec{

public:
	int SPECIES;
	int IC; // Initial condition IC=1 (Maxwellian), IC=2 (ring-like)
	double NSP; // Initial number of superparticles for the given ion species.
	double NCP; // Number of charged particles per superparticle.
	double NPC; // Number of superparticles per cell. When its value is zero, the particles are loaded from external files.
	double Q; 	// Charge.
	double Z; 	// Atomic number.
	double M; 	// Mass

	// variables for controlling super-particles' outputs
	double pctSupPartOutput;
	unsigned int nSupPartOutput;

	ionsInitialMeanParameters BGP;

	arma::vec X; 		// Ions position, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat V; 		// Ions' velocity, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::vec g; 		// Ions' relativistic gamma factor.
	arma::vec Ppar; // Parallel momentum used in guiding-center orbits
	arma::vec mu; 	// Ions' magnetic moment.

	arma::ivec meshNode; // Position of each particle in the discrete mesh.

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl, wxc, wxr;	// Particles' weights w.r.t. the vertices of the grid cells
};
// * * * * * * * * GC VARIABLES AND PARAMETERS DERIVED TYPES  * * * * * * * * //


// * * * * * * * * ELECTROMAGNETIC FIELDS DERIVED TYPES  * * * * * * * * //
class oneDimensional::electromagneticFields : public vfield_vec{


public:
	vfield_vec E;
	vfield_vec B;
	vfield_vec _B;
	vfield_vec b;
	vfield_vec b_;

	electromagneticFields(){};
	electromagneticFields(unsigned int N) : E(N), B(N), b(N), b_(N), _B(N){};
	void zeros(unsigned int N);
	void fill(double A);
};


class twoDimensional::electromagneticFields : public vfield_mat{

public:
	vfield_mat E;
	vfield_mat B;
	vfield_mat b;
	vfield_mat b_;
	arma::mat _B;

	electromagneticFields(){};
	electromagneticFields(unsigned int N, unsigned int M) : E(N,M), B(N,M), b(N,M), b_(N,M), _B(N,M){};
	void zeros(unsigned int N, unsigned int M);
};


class threeDimensional::electromagneticFields : public vfield_cube{

public:
	vfield_cube E;
	vfield_cube B;
	vfield_cube b;
	vfield_cube b_;
	arma::cube _B;

	electromagneticFields(){};
	electromagneticFields(unsigned int N, unsigned int M, unsigned int P) : E(N,M,P), B(N,M,P), b(N,M,P), b_(N,M,P), _B(N,M,P){};
	void zeros(unsigned int N, unsigned int M, unsigned int P);
};
// * * * * * * * * ELECTROMAGNETIC FIELDS DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //

#endif
