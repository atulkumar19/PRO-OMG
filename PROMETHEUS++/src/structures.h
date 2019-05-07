#ifndef H_STRUCTURES
#define H_STRUCTURES

#include <vector>
#include <string>
#include "armadillo"
#include "types.h"

#include<omp.h>
#include "mpi.h"

using namespace oneDimensional;//This namespace is used in the overloaded functions for the different structures in types.h

// Physical constants
#define F_E 1.602176E-19//Electron charge in C (absolute value)
#define F_ME 9.109382E-31//Electron mass in kg
#define F_MP 1.672621E-27//Proton mass in kg
#define F_U 1.660538E-27//Atomic mass unit in kg
#define F_KB 1.380650E-23//Boltzmann constant in Joules/Kelvin
#define F_EPSILON 8.854E-12 //Vacuum permittivity in C^2/(N*m^2)
#define F_C 299792458.0 //Light speed in m/s
#define F_MU (4*M_PI)*1E-7 //Vacuum permeability in N/A^2
extern double F_EPSILON_DS; // Dimensionless vacuum permittivity
extern double F_E_DS; // Dimensionless vacuum permittivity
extern double F_MU_DS; // Dimensionless vacuum permittivity
extern double F_C_DS; // Dimensionless vacuum permittivity


struct mpiStruct{
	int NUMBER_MPI_DOMAINS;
	int MPI_DOMAIN_NUMBER;
	MPI_Comm mpi_topo;
	int rank_cart, lRank, rRank;
};


struct energyMonitor{
	int it;
	double refEnergy;
	arma::mat ionsEnergy;//(time,species)
	arma::mat E_fieldEnergy;//(time,components)
	arma::mat B_fieldEnergy;//(time,components)
	arma::mat totalEnergy;
	energyMonitor(){};
	energyMonitor(int numberOfIonSpecies,int timeIterations){
		it = 0;
		refEnergy = 0.0;
		ionsEnergy = arma::zeros(timeIterations,numberOfIonSpecies);
		E_fieldEnergy = arma::zeros(timeIterations,3);
		B_fieldEnergy  = arma::zeros(timeIterations,3);
		totalEnergy = arma::zeros(timeIterations,2);
	}

};


typedef ionSpeciesParams ionSpecies;


typedef electromagneticFields emf;


struct meshGeometry{
	vfield_vec nodes;

	arma::uvec dim;

	double DX;
	double DY;
	double DZ;
};


struct backgroundParameters{
	double Te;
	double Bo;
	double Bx;
	double By;
	double Bz;

	double theta; // Spherical polar angle (as measured from z-axis)
	double phi; // Spherical azimuthal angle (as measured from x-axis)

	double propVectorAngle; // Angle between the z-axis and the propagation vector.
};


struct inputParameters{
	//Control parameters for the simulation
	std::string PATH;//Path to save the outputs. It must point to the directory where the folder outputFiles is.

	int argc;
	char **argv;

	int particleIntegrator; // particleIntegrator=1 (Boris'), particleIntegrator=2 (Vay's), particleIntegrator=3 (Relativistic GC).
	int quietStart; // Flag for using a quiet start

	int restart;
	int weightingScheme; //TSP = 1, CIC(volume weighting) = 2
	int BC;//BC = 1 full periodic, BC = 2
	int numberOfRKIterations;
	double smoothingParameter;
	int timeIterations;
	double simulationTime; // In units of the shorter ion gyro-period in the simulation
	int transient;//Transient time (in number of iterations).
	double DT;//Time step
	double DTc;//Ciclotron period fraction.
	double shorterIonGyroperiod;//Shorter ion cycloperiod.
	int loadFields;
	int loadGrid;
	int usingHDF5;
	double outputCadence;//Save variables each "outputCadence" times the background ion cycloperiod.
	int outputCadenceIterations;
	arma::file_type outputFormat;//Outputs format (raw_ascii,raw_binary).

	//Mesh geometry
	arma::uvec meshDim; //number of nodes in each direction. meshDim(0) = # of nodes along the x axis.

	//ions properties
	int numberOfIonSpecies;
	int numberOfTracerSpecies;

	double ne;

	backgroundParameters BGP;

	int filtersPerIterationFields;
	int filtersPerIterationIons;
	int checkSmoothParameter;

	double DrL;
	double dp;

	int checkStability;
	int rateOfChecking;//Each 'rateOfChecking' iterations we use the CLF criteria for the particles to stabilize the simulation

	energyMonitor * em;//Energy monitor

	unsigned int loadModes;
	unsigned int numberOfAlfvenicModes;//Number of Alfvenic waves for the initial condition
	unsigned int numberOfTestModes;
	double maxAngle;
	double fracMagEnerInj;//Fraction of background magnetic energy injected
	unsigned int shuffleModes;

	mpiStruct mpi;
};


struct characteristicScales{
	double time;
	double velocity;
	double length;
	double mass;
	double charge;
	double density;
	double eField;
	double bField;
	double pressure;
	double temperature;

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
	}
};

#endif
