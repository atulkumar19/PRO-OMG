#include "energyDiagnostic.h"

// Constructors

template <class IT, class FT> ENERGY_DIAGNOSTIC<IT, FT>::ENERGY_DIAGNOSTIC(const simulationParameters * params, const FT * EB, const vector<IT> * IONS){
    kineticEnergyDensity = zeros(IONS->size());

    magneticEnergyDensity = zeros(3);

    electricEnergyDensity = zeros(3);

    computeKineticEnergyDensity(params, IONS);

    computeElectromagneticEnergyDensity(params, EB);
}



template <class IT, class FT> void ENERGY_DIAGNOSTIC<IT, FT>::computeKineticEnergyDensity(const simulationParameters * params, const vector<IT> * IONS){
    for(int ss=0; ss<IONS->size(); ss++){
        kineticEnergyDensity(ss) = arma::sum(IONS->at(ss).g - 1.0)*IONS->at(ss).NCP*IONS->at(ss).M*F_C_DS*F_C_DS;

        kineticEnergyDensity(ss) = (params->dimensionality == 1) ? (kineticEnergyDensity(ss)/params->mesh.DX) : (kineticEnergyDensity(ss)/(params->mesh.DX*params->mesh.DY));
    }
}


template <class IT, class FT> void ENERGY_DIAGNOSTIC<IT, FT>::computeElectromagneticEnergyDensity(const simulationParameters * params, const oneDimensional::fields * EB){
    // Indices of subdomain
    unsigned int iIndex = params->mpi.iIndex;
	unsigned int fIndex = params->mpi.fIndex;

    arma::vec E_X = arma::pow(EB->B.X.subvec(iIndex,fIndex) - params->BGP.Bx, 2);
    arma::vec E_Y = arma::pow(EB->B.Y.subvec(iIndex,fIndex) - params->BGP.By, 2);
    arma::vec E_Z = arma::pow(EB->B.Z.subvec(iIndex,fIndex) - params->BGP.Bz, 2);

    magneticEnergyDensity(0) = 0.5*arma::sum( E_X )/F_MU_DS;
    magneticEnergyDensity(1) = 0.5*arma::sum( E_Y )/F_MU_DS;
    magneticEnergyDensity(2) = 0.5*arma::sum( E_Z )/F_MU_DS;

    E_X = arma::pow(EB->E.X.subvec(iIndex,fIndex), 2);
    E_Y = arma::pow(EB->E.Y.subvec(iIndex,fIndex), 2);
    E_Z = arma::pow(EB->E.Z.subvec(iIndex,fIndex), 2);

    electricEnergyDensity(0) = 0.5*F_EPSILON_DS*arma::sum( E_X );
    electricEnergyDensity(1) = 0.5*F_EPSILON_DS*arma::sum( E_Y );
    electricEnergyDensity(2) = 0.5*F_EPSILON_DS*arma::sum( E_Z );
}


template <class IT, class FT> void ENERGY_DIAGNOSTIC<IT, FT>::computeElectromagneticEnergyDensity(const simulationParameters * params, const twoDimensional::fields * EB){
    // Indices of subdomain
	unsigned int irow = params->mpi.irow;
	unsigned int frow = params->mpi.frow;
	unsigned int icol = params->mpi.icol;
	unsigned int fcol = params->mpi.fcol;

    arma::mat E_X = arma::pow(EB->B.X.submat(irow,icol,frow,fcol) - params->BGP.Bx, 2);
    arma::mat E_Y = arma::pow(EB->B.Y.submat(irow,icol,frow,fcol) - params->BGP.By, 2);
    arma::mat E_Z = arma::pow(EB->B.Z.submat(irow,icol,frow,fcol) - params->BGP.Bz, 2);

    magneticEnergyDensity(0) = 0.5*arma::sum( arma::sum( E_X ) )/F_MU_DS;
    magneticEnergyDensity(1) = 0.5*arma::sum( arma::sum( E_Y ) )/F_MU_DS;
    magneticEnergyDensity(2) = 0.5*arma::sum( arma::sum( E_Z ) )/F_MU_DS;

    E_X = arma::pow(EB->E.X.submat(irow,icol,frow,fcol), 2);
    E_Y = arma::pow(EB->E.Y.submat(irow,icol,frow,fcol), 2);
    E_Z = arma::pow(EB->E.Z.submat(irow,icol,frow,fcol), 2);

    electricEnergyDensity(0) = 0.5*F_EPSILON_DS*arma::sum( arma::sum( E_X ) );
    electricEnergyDensity(1) = 0.5*F_EPSILON_DS*arma::sum( arma::sum( E_Y ) );
    electricEnergyDensity(2) = 0.5*F_EPSILON_DS*arma::sum( arma::sum( E_Z ) );
}


template <class IT, class FT> arma::vec ENERGY_DIAGNOSTIC<IT, FT>::getKineticEnergyDensity(){
    return(kineticEnergyDensity);
}

template <class IT, class FT> arma::vec ENERGY_DIAGNOSTIC<IT, FT>::getMagneticEnergyDensity(){
    return(magneticEnergyDensity);
}

template <class IT, class FT> arma::vec ENERGY_DIAGNOSTIC<IT, FT>::getElectricEnergyDensity(){
    return(electricEnergyDensity);
}


template class ENERGY_DIAGNOSTIC<oneDimensional::ionSpecies, oneDimensional::fields>;
template class ENERGY_DIAGNOSTIC<twoDimensional::ionSpecies, twoDimensional::fields>;
