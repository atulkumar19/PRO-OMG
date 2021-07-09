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

// Declare namespaces:
// =============================================================================
namespace oneDimensional
{
	class fields;
	class ionSpecies;
}


namespace twoDimensional
{
	class fields;
	class ionSpecies;
}


// Declare type identifiers:
// =============================================================================
class types_info
{

public:
	types_info(){};
	~types_info(){};

	const std::type_info * ionSpecies_1D_type;
	const std::type_info * ionSpecies_2D_type;

	const std::type_info * fields_1D_type;
	const std::type_info * fields_2D_type;
};

// Declare vector field types:
// =============================================================================
class vfield_vec
{

public:
	arma::vec X;
	arma::vec Y;
	arma::vec Z;

	vfield_vec(){};
	vfield_vec(unsigned int N);

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


class vfield_mat
{

public:
	arma::mat X;
	arma::mat Y;
	arma::mat Z;

	vfield_mat(){};
	vfield_mat(unsigned int N, unsigned int M);

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

//  Structure to store each ion species initial condition parameters:
// =============================================================================
struct particle_IC
{
	int IC_type;             	   		// 1: Uniform profiles, 2: profiles from external files

	// Reference values for profiles:
	// ==============================
	double Tper;
	double Tpar;
	double densityFraction;

	// Name of external files:
	// =======================
	std::string Tper_fileName; 				// File containing normalized spatial profile of Tper
	std::string Tpar_fileName; 				// File containing normalized spatial profile of Tpar
	std::string densityFraction_fileName;	// File containing normalized spatial profile

	// Number of elements of profiles:
	// ===============================
	int Tper_NX;
	int Tpar_NX;
	int densityFraction_NX;

	// Variables to store profiles from external files:
	// ================================================
	arma::vec Tper_profile;
	arma::vec Tpar_profile;
	arma::vec densityFraction_profile;
};

//  Structure to store each ion species particle boundary condition parameters:
// =============================================================================
struct particle_BC
{
	int BC_type;					// 1: Warm plasma source, 2: NBI, 3: periodic, 4: reflecting

	// Particle source temperature:
	// ===========================
	double T;

	// Particle source beam energy:
	// ============================
	double E;

	// Particle source beam pitch angle:
	// ================================
	double eta;

	// Particle source rate:
	// =====================
	double G;						// Fueling rate of source in particles/second
	double sigma_x;					// Spatial spread of beam
	double mean_x;					// Spatial location of beam injection

	// Name of external file:
	// ======================
	std::string G_fileName;

	// Number of elements in external file:
	// ====================================
	int G_NS;

	// Variable to store profile from external file:
	// =============================================
	arma::vec G_profile;

	// Computational particle accumulators:
	// =============================================
	double S1;
	double S2;
	double GSUM;

	// Variable to store new particle weight based on fueling:
	// ======================================================
	double a_new;

	// Constructor:
	// ===========
	particle_BC()
	{
		BC_type   = 0;
		E 	  	  = 0;
		T     	  = 0;
		eta   	  = 0;
		G         = 0;
		sigma_x   = 0;
		mean_x    = 0;
		G_NS      = 0;
		GSUM      = 0;
		S1        = 0;
		S2        = 0;
		a_new 	  = 0;
	};

};

// Structure to hold RF operator terms:
// ====================================
struct particle_RFterms
{
	arma::vec phase;
	arma::vec udE3;
};

//  Define ION VARIABLES AND PARAMETERS DERIVED TYPES:
// =============================================================================
class oneDimensional::ionSpecies : public vfield_vec
{

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

	double densityFraction;		//

	double go;					// Initial relativistic gamma
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
 	arma::ivec pCount;
  	arma::vec eCount;
	arma::ivec mn; 				// Ions' position in terms of the index of mesh node
	arma::mat E;				// Electric field seen by particles when advancing particles velocity
	arma::mat B;				// Magnetic field seen by particles when advancing particles velocity

	// Guiding-center variables
	arma::vec mu; 				// Ions' magnetic moment.
	arma::vec Ppar; 			// Parallel momentum used in guiding-center orbits

	//These weights are used in the charge extrapolation and the force interpolation
	arma::vec wxl;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxc;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxr;				// Particles' weights w.r.t. the vertices of the grid cells

	arma::vec wxl_;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxc_;				// Particles' weights w.r.t. the vertices of the grid cells
	arma::vec wxr_;				// Particles' weights w.r.t. the vertices of the grid cells

    // Mesh-defined ion moments:
	arma::vec n; 				// Ion density at time level "l + 1"
	arma::vec n_; 				// Ion density at time level "l"
	arma::vec n__; 				// Ion density at time level "l - 1"
	arma::vec n___; 			// Ion density at time level "l - 2"
	vfield_vec nv; 				// Ion bulk velocity at time level "l + 1/2"
	vfield_vec nv_; 			// Ion bulk velocity at time level "l - 1/2"
	vfield_vec nv__; 			// Ion bulk velocity at time level "l - 3/2"
    arma::vec P11;				// Ion pressure tensor, component 1,1
    arma::vec P22;				// Ion pressure tensor, component 2,2
    arma::vec Tpar_m;			// Ion parallel temperature
    arma::vec Tper_m;			// Ion perpendicular temperature

	// Particle-defined ion moments:
	arma::vec n_p;
	arma::vec nv_p;
	arma::vec Tpar_p;
	arma::vec Tper_p;

	// Particle defined flags:
	arma::ivec f1;             	// Flag for left boundary
	arma::ivec f2;              // Flag for Right boundary
	arma::ivec f3;              // Flag for RF operator

	// Particle kinetic energy at boundaries:
	arma::ivec dE1;              // left boundary
	arma::ivec dE2;              // Right boundary
	arma::ivec dE3;              // RF operator

	// Particle weight:
	arma::vec a;                // Computational particle weigth

	// Initial condition parameters:
	particle_IC p_IC;

	// Boundary conditions:
	particle_BC p_BC;

	// Particle Rf terms:
	particle_RFterms p_RF;

	// Constructor:
	ionSpecies(){};

	// Destructor:
	~ionSpecies(){};
};


class twoDimensional::ionSpecies : public vfield_mat
{
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

	double densityFraction;					//

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
  	arma::ivec pCount;
  	arma::vec eCount;
	arma::imat mn; 				// Ions' position in terms of the index of mesh node
	arma::mat E;				// Electric field seen by particles when advancing particles velocity
	arma::mat B;				// Magnetic field seen by particles when advancing particles velocity

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

    // Mesh-defined ion moments:
	arma::mat n; 		// Ion density at time level "l + 1"
	arma::mat n_; 		// Ion density at time level "l"
	arma::mat n__; 		// Ion density at time level "l - 1"
	arma::mat n___; 	// Ion density at time level "l - 2"
	vfield_mat nv; 		// Ion bulk velocity at time level "l + 1/2"
	vfield_mat nv_; 	// Ion bulk velocity at time level "l - 1/2"
	vfield_mat nv__; 	// Ion bulk velocity at time level "l - 3/2"
	arma::vec P11;				// Ion pressure tensor, component 1,1
	arma::vec P22;				// Ion pressure tensor, component 2,2
	arma::vec Tpar_m;			// Ion parallel temperature
	arma::vec Tper_m;			// Ion perpendicular temperature

	// Particle-defined ion moments:
	arma::vec n_p;
	arma::vec nv_p;
	arma::vec Tpar_p;
	arma::vec Tper_p;

	// Particle defined flags:
	arma::ivec f1;             	// Flag for left boundary
	arma::ivec f2;              // Flag for Right boundary
	arma::ivec f3;              // Flag for RF operator

	// Particle kinetic energy at boundaries:
	arma::ivec dE1;              // left boundary
	arma::ivec dE2;              // Right boundary
	arma::ivec dE3;              // RF operator

	// Particle weight:
	arma::vec a;                // Computational particle weigth

	// Initial condition parameters:
	particle_IC p_IC;

	// Boundary conditions:
	particle_BC p_BC;

    ionSpecies(){};
	~ionSpecies(){};
};


//  Define ELECTROMAGNETIC FIELDS DERIVED TYPES:
// =============================================================================
class oneDimensional::fields : public vfield_vec
{

public:
	vfield_vec E;
	vfield_vec B;

	fields(){};
	fields(unsigned int N) : E(N), B(N){};

	~fields(){};

	void zeros(unsigned int N);
	void fill(double A);
};


class twoDimensional::fields : public vfield_mat
{

public:
	vfield_mat E;
	vfield_mat B;

	fields(){};
	fields(unsigned int NX, unsigned int NY) : E(NX,NY), B(NX,NY){};

	~fields(){};

	void zeros(unsigned int NX, unsigned int NY);
};

// * * * * * * * * ELECTROMAGNETIC FIELDS DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //


// * * * * * * * * SIMULATION CONTROL DERIVED TYPES  * * * * * * * * //

#endif
