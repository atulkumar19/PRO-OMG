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

#include <typeinfo>
#include <vector>
#include <armadillo>

// * * * * * * * * NAMESPACES  * * * * * * * * //
namespace oneDimensional{
	class fields;
	class ionSpecies;
	class GCSpecies;
}


namespace twoDimensional{
	class fields;
	class ionSpecies;
}


namespace threeDimensional{
	class fields;
}
// * * * * * * * * NAMESPACES  * * * * * * * * //


// * * * * * * * * TYPES IDENTIFIERS  * * * * * * * * //
class types_info{

public:
	types_info(){};
	~types_info(){};

	const std::type_info * ionSpecies_1D_type;
	const std::type_info * ionSpecies_2D_type;

	const std::type_info * GCSpecies_1D_type;

	const std::type_info * fields_1D_type;
	const std::type_info * fields_2D_type;
};
// * * * * * * * * TYPES IDENTIFIERS  * * * * * * * * //


// * * * * * * * * VECTOR FIELD TYPES  * * * * * * * * //
class vfield_vec{

public:
	arma::vec X;
	arma::vec Y;
	arma::vec Z;

	vfield_vec(){};
	vfield_vec(unsigned int N) : X(N), Y(N), Z(N) {}

	~vfield_vec(){};

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

	~vfield_mat(){};

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

	~vfield_cube(){};

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

class oneDimensional::ionSpecies : public vfield_vec{

public:
	int SPECIES;
	int IC; 					// Initial condition IC=1 (Maxwellian), IC=2 (ring-like)
	double NSP; 				// Initial number of superparticles for the given ion species.
	double NCP; 				// Number of charged particles per superparticle.
	double NPC; 				// Number of superparticles per cell. When its value is zero, the particles are loaded from external files.
	double Q; 					// Charge.
	double Z; 					// Atomic number.
	double M; 					// Mass

	// variables for controlling super-particles' outputs
	double pctSupPartOutput; 	//
	unsigned int nSupPartOutput;//

	double Dn;					//

	double go;					// Initial relativistic gamma
	double Tpar;				// Parallel temperature.
	double Tper;				// Perpendicular temperature.
	double LarmorRadius;		// Larmor radius.
	double VTper;				// Thermal velocity.
	double VTpar;				// Thermal velocity.
	double Wc;					// Average cyclotron frequency.
	double Wp;					// Plasma frequency.
	double avg_mu; 				// Average magnetic moment

	arma::mat X; 				// Ions position, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat V; 				// Ions' velocity, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat P; 				// Ions' momentum, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::vec g; 				// Ions' relativistic gamma factor.
	arma::ivec meshNode; 		// Ions' position in terms of the index of mesh node

	// Guiding-center variables
	arma::vec mu; 				// Ions' magnetic moment.
	arma::vec Ppar; 			// Parallel momentum used in guiding-center orbits

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxc;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxr;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxll;				// Particles' weights w.r.t. the vertices of the grid cells. Third-order particle interpolation
	arma::vec wxrr;				// Particles' weights w.r.t. the vertices of the grid cells. Third-order particle interpolation

	arma::vec n; 				// Ion density at time level "l + 1"
	arma::vec n_; 				// Ion density at time level "l - 1;
	arma::vec n__; 				// Ion density at time level "l - 2;
	arma::vec n___; 			// Ion density at time level "l - 3;
	vfield_vec nv; 				// Ion bulk velocity at time level "l + 1/2"
	vfield_vec nv_; 			// Ion bulk velocity at time level "l - 1/2"
	vfield_vec nv__; 			// Ion bulk velocity at time level "l - 3/2"

	ionSpecies(){};
	~ionSpecies(){};
};


class twoDimensional::ionSpecies : public vfield_mat{

public:
	int SPECIES;
	int IC; 					// Initial condition IC=1 (Maxwellian), IC=2 (ring-like)
	double NSP; 				// Initial number of superparticles for the given ion species.
	double NCP; 				// Number of charged particles per superparticle.
	double NPC; 				// Number of superparticles per cell. When its value is zero, the particles are loaded from external files.
	double Q; 					// Charge.
	double Z; 					// Atomic number.
	double M; 					// Mass

	// variables for controlling super-particles' outputs
	double pctSupPartOutput; 	//
	unsigned int nSupPartOutput;//

	double Dn;					//

	double go;					// Initial relativistic gamma
	double Tpar;				// Parallel temperature.
	double Tper;				// Perpendicular temperature.
	double LarmorRadius;		// Larmor radius.
	double VTper;				// Thermal velocity.
	double VTpar;				// Thermal velocity.
	double Wc;					// Average cyclotron frequency.
	double Wp;					// Plasma frequency.
	double avg_mu; 				// Average magnetic moment

	arma::mat X; 				// Ions position, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat V; 				// Ions' velocity, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat P; 				// Ions' momentum, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::vec g; 				// Ions' relativistic gamma factor.
	arma::imat meshNode; 		// Ions' position in terms of the index of mesh node

	// Guiding-center variables
	arma::vec mu; 				// Ions' magnetic moment.
	arma::vec Ppar; 			// Parallel momentum used in guiding-center orbits

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxc;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxr;				// Particles' weights w.r.t. the vertices of the grid cells

	arma::vec wyl;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wyc;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wyr;				// Particles' weights w.r.t. the vertices of the grid cells

	arma::mat n; 		// Ion density at time level "l + 1"
	arma::mat n_; 		// Ion density at time level "l - 1;
	arma::mat n__; 		// Ion density at time level "l - 2;
	arma::mat n___; 		// Ion density at time level "l - 3;
	vfield_mat nv; 		// Ion bulk velocity at time level "l + 1/2"
	vfield_mat nv_; 	// Ion bulk velocity at time level "l - 1/2"
	vfield_mat nv__; 	// Ion bulk velocity at time level "l - 3/2"

	ionSpecies(){};
	~ionSpecies(){};
};
// * * * * * * * * ION VARIABLES AND PARAMETERS DERIVED TYPES  * * * * * * * * //

class oneDimensional::GCSpecies : public vfield_vec{

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

	double Dn;

	double go;			// Initial relativistic gamma
	double Tpar;		// Parallel temperature.
	double Tper;		// Perpendicular temperature.
	double LarmorRadius;// Larmor radius.
	double VTper;		// Thermal velocity.
	double VTpar;		// Thermal velocity.
	double Wc;			// Average cyclotron frequency.
	double Wp;			// Plasma frequency.
	double avg_mu; 		// Average magnetic moment

	arma::vec X; 		// Ions position, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::mat V; 		// Ions' velocity, the dimension should be (NSP,3), where NP is the number of particles of the ion species.
	arma::vec g; 		// Ions' relativistic gamma factor.
	arma::vec Ppar; // Parallel momentum used in guiding-center orbits
	arma::vec mu; 	// Ions' magnetic moment.

	arma::ivec meshNode; // Position of each particle in the discrete mesh.

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl, wxc, wxr;	// Particles' weights w.r.t. the vertices of the grid cells

	GCSpecies(){};
	~GCSpecies(){};
};
// * * * * * * * * GC VARIABLES AND PARAMETERS DERIVED TYPES  * * * * * * * * //


// * * * * * * * * ELECTROMAGNETIC FIELDS DERIVED TYPES  * * * * * * * * //
class oneDimensional::fields : public vfield_vec{


public:
	vfield_vec E;
	vfield_vec B;
	vfield_vec _B;
	vfield_vec b;
	vfield_vec b_;

	fields(){};
	fields(unsigned int N) : E(N), B(N), b(N), b_(N), _B(N){};

	~fields(){};

	void zeros(unsigned int N);
	void fill(double A);
};


class twoDimensional::fields : public vfield_mat{

public:
	vfield_mat E;
	vfield_mat B;
	vfield_mat _B;
	vfield_mat b;
	vfield_mat b_;

	fields(){};
	fields(unsigned int NX, unsigned int NY) : E(NX,NY), B(NX,NY), b(NX,NY), b_(NX,NY), _B(NX,NY){};

	~fields(){};

	void zeros(unsigned int NX, unsigned int NY);
};


class threeDimensional::fields : public vfield_cube{

public:
	vfield_cube E;
	vfield_cube B;
	vfield_cube b;
	vfield_cube b_;
	arma::cube _B;

	fields(){};
	fields(unsigned int N, unsigned int M, unsigned int P) : E(N,M,P), B(N,M,P), b(N,M,P), b_(N,M,P), _B(N,M,P){};

	~fields(){};

	void zeros(unsigned int N, unsigned int M, unsigned int P);
};
// * * * * * * * * ELECTROMAGNETIC FIELDS DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //

#endif
