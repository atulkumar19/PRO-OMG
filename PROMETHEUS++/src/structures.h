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

#ifndef H_STRUCTURES
#define H_STRUCTURES

#include <vector>
#include <string>
#include <map>
#include "armadillo"
#include "types.h"

#include<omp.h>
#include "mpi.h"

#define float_zero 1E-7
#define double_zero 1E-15


// Physical constants
#define F_E 1.602176E-19	// Electron charge in C (absolute value)
#define F_ME 9.109382E-31	// Electron mass in kg
#define F_MP 1.672621E-27	// Proton mass in kg
#define F_U 1.660538E-27	// Atomic mass unit in kg
#define F_KB 1.380650E-23	// Boltzmann constant in Joules/Kelvin
#define F_EPSILON 8.854E-12 // Vacuum permittivity in C^2/(N*m^2)
#define F_C 299792458.0 	// Light speed in m/s
#define F_MU (4*M_PI)*1E-7 	// Vacuum permeability in N/A^2
extern double F_EPSILON_DS; // Dimensionless vacuum permittivity
extern double F_E_DS; 		// Dimensionless vacuum permittivity
extern double F_ME_DS; 		// Dimensionless vacuum permittivity
extern double F_MU_DS; 		// Dimensionless vacuum permittivity
extern double F_C_DS; 		// Dimensionless vacuum permittivity


struct mpiParams{
	int NUMBER_MPI_DOMAINS;
	int MPI_DOMAIN_NUMBER;

	MPI_Comm MPI_TOPO;

	int MPI_DOMAINS_ALONG_X_AXIS;
	int MPI_DOMAINS_ALONG_Y_AXIS;
	int MPI_DOMAINS_ALONG_Z_AXIS;

	int MPI_CART_COORDS_1D[1];
	int MPI_CART_COORDS_2D[2];
	std::vector<int *> MPI_CART_COORDS;

	int MPI_DOMAIN_NUMBER_CART;
	int LEFT_MPI_DOMAIN_NUMBER_CART;
	int RIGHT_MPI_DOMAIN_NUMBER_CART;
	int UP_MPI_DOMAIN_NUMBER_CART;
	int DOWN_MPI_DOMAIN_NUMBER_CART;
};


struct energyMonitor{
	int it;
	double refEnergy;
	arma::mat ionsEnergy;//(time,species)
	arma::mat E_fieldEnergy;//(time,components)
	arma::mat B_fieldEnergy;//(time,components)
	arma::mat totalEnergy;
	energyMonitor(){};
	energyMonitor(int numberOfParticleSpecies,int timeIterations){
		it = 0;
		refEnergy = 0.0;
		ionsEnergy = arma::zeros(timeIterations,numberOfParticleSpecies);
		E_fieldEnergy = arma::zeros(timeIterations,3);
		B_fieldEnergy  = arma::zeros(timeIterations,3);
		totalEnergy = arma::zeros(timeIterations,2);
	}

};


struct meshParams{
	vfield_vec nodes;

	int NX_PER_MPI; // Number of mesh nodes along x-axis in subdomain (no ghost nodes considered)
	int NY_PER_MPI; // Number of mesh nodes along y-axis in subdomain (no ghost nodes considered)
	int NZ_PER_MPI; // Number of mesh nodes along z-axis in subdomain (no ghost nodes considered)

	int NX_IN_SIM; // Number of mesh nodes along x-axis in entire simulation domain (no ghost nodes considered)
	int NY_IN_SIM; // Number of mesh nodes along x-axis in entire simulation domain (no ghost nodes considered)
	int NZ_IN_SIM; // Number of mesh nodes along x-axis in entire simulation domain (no ghost nodes considered)

	double DX;
	double DY;
	double DZ;

	double LX;		// Size of simulation domain along x-axis
	double LY;		// Size of simulation domain along x-axis
	double LZ;		// Size of simulation domain along x-axis

	int SPLIT_DIRECTION;
};


struct backgroundPlasmaParameters{
	double ne;
	double Te;
	double Bo;
	double Bx;
	double By;
	double Bz;

	double theta; // Spherical polar angle (as measured from z-axis)
	double phi; // Spherical azimuthal angle (as measured from x-axis)

	double propVectorAngle; // Angle between the z-axis and the propagation vector.
};


struct simulationParameters{
	// List of variables in the outputs
	std::vector<std::string> outputs_variables;

	//Control parameters for the simulation
	std::string PATH;//Path to save the outputs. It must point to the directory where the folder outputFiles is.

	int argc;
	char **argv;

	int dimensionality; // Dimensionality of the simulation domain 1-D = 1; 2-D = 2
	int particleIntegrator; // particleIntegrator=1 (Boris'), particleIntegrator=2 (Vay's), particleIntegrator=3 (Relativistic GC).
	bool includeElectronInertia;
	bool quietStart; // Flag for using a quiet start

	bool restart;
	int weightingScheme; // TOS = 1; TSC (DEFAULT) = 1; NNS = 2; TOS = 3; TSC = 4;
	int BC; // BC = 1 full periodic, BC = 2
	int numberOfRKIterations;
	double smoothingParameter;
	int timeIterations;
	double simulationTime; // In units of the shorter ion gyro-period in the simulation
	int transient;//Transient time (in number of iterations).
	double DT;//Time step
	double DTc;//Ciclotron period fraction.
	int loadFields;
	int loadGrid;
	int usingHDF5;
	double outputCadence;//Save variables each "outputCadence" times the background ion cycloperiod.
	int outputCadenceIterations;
	arma::file_type outputFormat;//Outputs format (raw_ascii,raw_binary).

	// Parameters of mesh used in simulation
	meshParams mesh;

	//ions properties
	int numberOfParticleSpecies; // This species are evolved self-consistently with the fields
	int numberOfTracerSpecies; // This species are not self-consistently evolved with the fields

	backgroundPlasmaParameters BGP;

	int filtersPerIterationFields;
	int filtersPerIterationIons;
	int checkSmoothParameter;

	double ionLarmorRadius;
	double ionSkinDepth;
	double ionGyroPeriod;

	double DrL;
	double dp;

	int checkStability;
	int rateOfChecking;//Each 'rateOfChecking' iterations we use the CLF criteria for the particles to stabilize the simulation

	energyMonitor * em; // Structure to monitor energy conservation

	// Parameters of Alfven waves' spectrum
	unsigned int loadModes;
	unsigned int numberOfAlfvenicModes;//Number of Alfvenic waves for the initial condition
	unsigned int numberOfTestModes;
	double maxAngle;
	double fracMagEnerInj;//Fraction of background magnetic energy injected
	unsigned int shuffleModes;

	// MPI parameters
	mpiParams mpi;

	// Error codes
	std::map<int, std::string> errorCodes;

	types_info typesInfo;

	simulationParameters();
};


struct characteristicScales{
	double time;
	double velocity;
	double momentum;
	double length;
	double mass;
	double charge;
	double density;
	double eField;
	double bField;
	double pressure;
	double temperature;
	double magneticMoment;
	double resistivity;

	characteristicScales(){
		time = 0.0;
		velocity = 0.0;
		length = 0.0;
		mass = 0.0;
		charge = 0.0;
		density = 0.0;
		eField = 0.0;
		bField = 0.0;
		pressure = 0.0;
		magneticMoment = 0.0;
	}
};

struct fundamentalScales{
	double electronSkinDepth;
	double electronGyroPeriod;
	double electronGyroRadius;
	double * ionSkinDepth;
	double * ionGyroPeriod;
	double * ionGyroRadius;

	fundamentalScales(simulationParameters * params){
		electronSkinDepth = 0.0;
		electronGyroPeriod = 0.0;
		electronGyroRadius = 0.0;
		ionSkinDepth = new double[params->numberOfParticleSpecies];
		ionGyroPeriod = new double[params->numberOfParticleSpecies];
		ionGyroRadius = new double[params->numberOfParticleSpecies];
	}
};

#endif
