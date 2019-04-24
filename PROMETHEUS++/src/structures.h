#ifndef H_STRUCTURES
#define H_STRUCTURES

#include <vector>
#include <string>
#include<omp.h>

#include "armadillo"
#include "types.h"

#include "mpi.h"

using namespace oneDimensional;//This namespace is used in the overloaded functions for the different structures in types.h

#define F_E 1.602176E-19//Electron charge in C (absolute value)
#define F_ME 9.109382E-31//Electron mass in kg
#define F_MP 1.672621E-27//Proton mass in kg
#define F_U 1.660538E-27//Atomic mass unit in kg
#define F_KB 1.380650E-23//Boltzmann constant in Joules/Kelvin
#define F_EPSILON 8.854E-12 //Vacuum permittivity in C^2/(N*m^2)
#define F_C 299792458.0 //Light speed in m/s
#define F_MU (4*M_PI)*1E-7 //Vacuum permeability in N/A^2


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

	double backgroundTemperature;
	double backgroundBField;
	double Bx;
	double By;
	double Bz;

	//Angle between the z-axis and the background magnetic field.
	double theta;
	//Angle between the z-axis and the propagation vector.
	double phi;
};


struct inputParameters{
	//Control parameters for the simulation
	std::string PATH;//Path to save the outputs. It must point to the directory where the folder outputFiles is.

	int argc;
	char **argv;

	int quietStart; // Flag for using a quiet start

	int restart;
	int weightingScheme; //TSP = 1, CIC(volume weighting) = 2
	int BC;//BC = 1 full periodic, BC = 2
	int numberOfRKIterations;
	double smoothingParameter;
	int timeIterations;
	int transient;//Transient time (in number of iterations).
	double DT;//Time step
	double DTc;//Ciclotron period fraction.
	double backgroundTc;//Background ion cycloperiod.
	int loadFields;
	int loadGrid;
	int usingHDF5;
	double saveVariablesEach;//Save variables each "saveVariablesEach" times the background ion cycloperiod.
	arma::file_type outputFormat;//Outputs format (raw_ascii,raw_binary).

	//Mesh geometry
	arma::uvec meshDim; //number of nodes in each direction. meshDim(0) = # of nodes along the x axis.

	//ions properties
	int numberOfIonSpecies;
	int numberOfTracerSpecies;

	double totalDensity;

	backgroundParameters BGP;

	int filtersPerIteration;
	int filtersPerIterationIonsVariables;
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
