#include <math.h>
#include "rfOperator.h"

using namespace std;

rfOperator::rfOperator()
{}

void rfOperator::MPI_AllreduceDouble(const simulationParameters * params, double * v)
{
    double recvbuf = 0;

    MPI_Allreduce(v, &recvbuf, 1, MPI_DOUBLE, MPI_SUM, params->mpi.COMM);

    *v = recvbuf;
}

void rfOperator::checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        // RF boundary locations:
        // ======================
        double x1 = params->RF.x1;
        double x2 = params->RF.x2;

        // Loop over all ion species:
        // ==========================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles is "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(x1, x2, params, IONS, aa, CS, std::cout) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                double X = IONS->at(aa).X(ii,0);

                if ( (X > x1) && (X < x2) )
                {
                    IONS->at(aa).f3(ii) = 1;
                }

            } // Particle loop

        } // Species loop

    } // Particle MPI guard

}

void rfOperator::checkResonanceAndFlag(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::calculateErf(simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        // RF electric field to apply:
        // ======================
        double E_RF = 10/CS->eField;

        // Total power absorbed by unit electric field E_RF:
        // ============================================
        double uE3_total = 0;

        // RF parameters:
        // ==============
        double kper = params->RF.kper;
        double kpar = params->RF.kpar;
        double w    = params->RF.freq*2*M_PI;

        // Current time:
        // ============
        double t = params->currentTime;

        // Time step:
        // ==========
        double DT = params->DT;

        // Calculate udE3(ii) by applying unit electric field:
        // ==================================================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles in "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Mass of "aa" species:
            // =====================
            double Ma = IONS->at(aa).M;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(Ma, params, IONS, EB, aa, CS, t, kper, kpar, w, E_RF, DT, std::cout) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                int tid = omp_get_thread_num();
                if ( IONS->at(aa).f3(ii) == 1 )
                {
                    // Interpolate magnetic field at particle position:
                    // ================================================
                    // BX:
                    double BXp;
                    interpolateScalarField(ii,params,&IONS->at(aa),&EB->B.X,&BXp);
                    IONS->at(aa).B(ii,0) = BXp;

                    // BY:
                    double BYp;
                    interpolateScalarField(ii,params,&IONS->at(aa),&EB->B.Y,&BYp);
                    IONS->at(aa).B(ii,1) = BYp;

                    // Record current velocity vector:
                    // ==============================
                    arma::rowvec V = IONS->at(aa).V.row(ii);

                    // Record current position:
                    // =======================
                    double x = IONS->at(aa).X(ii,0);

                    // Advance ion velocity due to RF electric field:
                    // ==============================================
                    arma::rowvec Vold;
                    double dKE = 0;
                    double dKE1  = 0;
                    int numit = params->RF.numit;
                    double rho0    = 0;
                    double cosPhi0 = 0;
                    double sinPhi0 = 0;
                    double phase0  = 0;

                    double w_ci  = (fabs(IONS->at(aa).Q)*BXp/Ma);
                    double r0    = params->geometry.r2;
                    double BX0   = params->em_IC.BX;

                    double KE0 = 0.5*Ma*dot(V,V);

                    int ii_star;

                    //double gyroperiod = 2*M_PI/w_ci;
                    //numit = round(gyroperiod/DT)


                    for(int jj=0; jj<numit; jj++)
                    {
                        // Calculate RF term:
                        // =================
                        double vperp  = sqrt( pow(V(1),2) + pow(V(2),2) );
                        double vy     = arma::as_scalar(V(1));
                        double vz     = arma::as_scalar(V(2));
                        double cosPhi = -(vz/vperp);
                        double sinPhi = +(vy/vperp);

                        double rho   = fabs(vperp/w_ci);
                        double ry    = r0*sqrt(BX0/BXp) + rho*cosPhi;
                        double phase = kper*ry + kpar*x - w*t;

                        // Store RF terms for next calculation cycle:
                        IONS->at(aa).p_RF.rho(ii)    = rho;
                        IONS->at(aa).p_RF.cosPhi(ii) = cosPhi;
                        IONS->at(aa).p_RF.sinPhi(ii) = sinPhi;
                        IONS->at(aa).p_RF.phase(ii)  = phase;

                        if ( jj == 0)
                        {
                            rho0    = rho;
                            cosPhi0 = cosPhi;
                            sinPhi0 = sinPhi;
                            phase0  = phase;

                            ii_star = ii;
                        }

                        Vold = V;
                        advanceVelocity(ii,params,&IONS->at(aa),&V,E_RF,DT);

                        /*
                        if (params->mpi.IS_PARTICLES_ROOT)
                        {
                            if (ii == ii_star)
                            {
                                if (tid == 1)
                                {
                                    cout << sqrt(dot(V,V))*CS->velocity << endl;
                                }
                            }
                        }
                        */

                        dKE = 0.5*Ma*(dot(V,V) - dot(Vold,Vold));
                        dKE1 += dKE;
                    } // jj loop

                    // Calculate final kinetic energy:
                    double KE1 = 0.5*Ma*dot(V,V);

                    // Average change in kinetic energy
                    double dKE2 = (KE1 - KE0)/numit;
                    dKE1 = dKE1/numit;

                    if (params->mpi.IS_PARTICLES_ROOT)
                    {
                        if (ii == ii_star)
                        {
                                //cout << "dKE1 = " << dKE1*CS->energy/F_E << endl;
                                //cout << "dKE2 = " << dKE2*CS->energy/F_E << endl;

                        }
                    }



                    // Restore RF terms in IONS:
                    IONS->at(aa).p_RF.rho(ii)    = rho0;
                    IONS->at(aa).p_RF.cosPhi(ii) = cosPhi0;
                    IONS->at(aa).p_RF.sinPhi(ii) = sinPhi0;
                    IONS->at(aa).p_RF.phase(ii)  = phase0;

                    // Calculate udE3:
                    // ===============
                    IONS->at(aa).p_RF.udE3(ii) = dKE1;

                } // f3 guard

            } // ii particle loop

        } // Species Loop

        // Accumulate udE3 into uE3_total:
        // ================================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of computational particles per process:
            int NSP(IONS->at(aa).NSP);

            // Super particle conversion factor:
            double alpha = IONS->at(aa).NCP;

            // Variable to accumulate udE3:
            double S1 = 0;

            #pragma omp parallel default(none) shared(S1, params, IONS, aa, std::cout) firstprivate(NSP)
            {
                // Computational particle accumulators:
                double S1_private = 0;

                #pragma omp for
                for(int ii=0; ii<NSP; ii++)
                {
                    if ( IONS->at(aa).f3(ii) == 1 )
                    {
                        double a = IONS->at(aa).a(ii);
                        S1_private += a*IONS->at(aa).p_RF.udE3(ii);
                    }
                }

                #pragma omp critical
                S1 += S1_private;

            } // pragma omp parallel

            // AllReduce S1 and S2 over all particle MPIs:
            MPI_AllreduceDouble(params,&S1);

            // Accumulate computational particles:
            uE3_total += alpha*S1/DT;

        } // Species loop

        // Calculate Erf:
        // ==============
        params->RF.Erf = E_RF*sqrt(params->RF.Prf/uE3_total);

        if (params->mpi.IS_PARTICLES_ROOT)
        {
            cout << uE3_total << endl;
        }


    } // Particle MPI guard

}

void rfOperator::calculateErf(simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::fill4Ghosts(arma::vec * v)
{
	int N = v->n_elem;

    v->subvec(N-2,N-1) = v->subvec(N-4,N-3);
    v->subvec(0,1)     = v->subvec(2,3);
}

void rfOperator::interpolateScalarField(int ii,const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, double * F_p)
{
	// Total number of mesh grids + 4 ghost cells:
	int NX =  params->mesh.NX_IN_SIM + 4;

	// Create temporary storage for mesh-defined quantity:
	arma::vec F = zeros(NX);

	// Populate F:
	F.subvec(1,NX-2) = *F_m;
	fill4Ghosts(&F);

	// Nearest grid point with ghost cells:
	int ix = IONS->mn(ii) + 2;

	// Interpolate mesh-defined quantity to iith particle:
	*(F_p) = 0;
	*(F_p) += IONS->wxl(ii)*F(ix-1);
	*(F_p) += IONS->wxc(ii)*F(ix);
	*(F_p) += IONS->wxr(ii)*F(ix+1);
}

void rfOperator::interpolateScalarField(int ii,const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::vec * F_m, double * F_p)
{

}

void rfOperator::advanceVelocity(int ii, const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::rowvec * V, double E_RF, double DT)
{
    // Place holders for EM fields:
    arma::mat Ep = zeros(1, 3);
    arma::mat Bp = zeros(1, 3);

    // Assign value to magnetic fields:
    int ip = 0;
    Bp(ip,0) = IONS->B(ii,0); // BX
    Bp(ip,1) = IONS->B(ii,1); // BY
    Bp(ip,2) = IONS->B(ii,2); // BZ

    // Extract larmour radius:
    double rL_i = IONS->p_RF.rho(ii);

    // Extract gyro-phase terms:
    double cosPhi = IONS->p_RF.cosPhi(ii);
    double sinPhi = IONS->p_RF.sinPhi(ii);
    double phase = IONS->p_RF.phase(ii);

    // Assign value to electric fields:
    double EX = 0;
    double EY = 0;
    double EZ = 0;
    switch (params->RF.handedness)
    {
    case -1: // E_minus (Left handed polarization)
            EX = 0;
            EY = +0.5*E_RF*cos(phase);
            EZ = -0.5*E_RF*sin(phase);
        case 0: // E (Linear polarization)
            EX = 0;
            EY = +E_RF*cos(phase);
            EZ = 0;
        case 1: // E_plus (Right handed polarization)
            EX = 0;
            EY = +0.5*E_RF*cos(phase);
            EZ = +0.5*E_RF*sin(phase);
    }

    Ep(ip,0) = EX;
    Ep(ip,1) = EY;
    Ep(ip,2) = EZ;

    // Magnetic field gradient:
    double dBx = Bp(ip,1)/(-0.5*(params->geometry.r2));

    // Particle defined magnetic field:
    Bp(ip,1) = -0.5*(params->geometry.r2*sqrt(params->em_IC.BX/Bp(ip,0)) + rL_i*cosPhi)*dBx;
    Bp(ip,2) = -0.5*(rL_i*sinPhi)*dBx;

    // Create placeholders for V calculation:
    arma::rowvec U = arma::zeros<rowvec>(3);
    arma::rowvec VxB = arma::zeros<rowvec>(3);
    arma::rowvec tau = arma::zeros<rowvec>(3);
    arma::rowvec up = arma::zeros<rowvec>(3);
    arma::rowvec t = arma::zeros<rowvec>(3);
    arma::rowvec upxt = arma::zeros<rowvec>(3);

    // VxB term:
    VxB = arma::cross(V->row(ip), Bp.row(ip));

    // Calculate new V:
    double A = IONS->Q*DT/IONS->M;
    double g = 1.0/sqrt( 1.0 -  dot(V->row(ip),V->row(ip))/(F_C_DS*F_C_DS) );
    U = g*V->row(ip);
    U += 0.5*A*(Ep.row(ip) + VxB); // U_hs = U_L + 0.5*a*(E + cross(V, B)); % Half step for velocity
    tau = 0.5*A*Bp.row(ip); // tau = 0.5*q*dt*B/m;
    up = U + 0.5*A*Ep.row(ip); // up = U_hs + 0.5*a*E;
    double gp = sqrt( 1.0 + dot(up, up)/(F_C_DS*F_C_DS) ); // gammap = sqrt(1 + up*up');
    double sigma = gp*gp - dot(tau, tau); // sigma = gammap^2 - tau*tau';
    double us = dot(up, tau)/F_C_DS; // us = up*tau'; % variable 'u^*' in paper
    g = sqrt(0.5)*sqrt( sigma + sqrt( sigma*sigma + 4.0*( dot(tau, tau) + us*us ) ) );// gamma = sqrt(0.5)*sqrt( sigma + sqrt(sigma^2 + 4*(tau*tau' + us^2)) );
    t = tau/g; 			// t = tau/gamma;
    double s = 1.0/( 1.0 + dot(t, t) ); // s = 1/(1 + t*t'); % variable 's' in paper

    upxt = arma::cross(up, t);

    U = s*( up + dot(up, t)*t+ upxt ); 	// U_L = s*(up + (up*t')*t + cross(up, t));
    V->row(ip) = U/g;	// V = U_L/gamma;
}

void rfOperator::advanceVelocity(int ii, const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::rowvec * V, double E_RF, double DT)
{
}

void rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, oneDimensional::fields * EB, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
        // =====================
        int numIonSpecies = IONS->size();

        // RF electric field to apply:
        // ======================
        double E_RF = params->RF.Erf;

        // RF parameters:
        // ==============
        double kper = params->RF.kper;
        double kpar = params->RF.kpar;
        double w    = params->RF.freq*2*M_PI;

        // Current time:
        // ============
        double t = params->currentTime;

        // Time step:
        // ==========
        double DT = params->DT;

        // Applying electric field:
        // ========================
        for (int aa=0; aa<numIonSpecies; aa++)
        {
            // Number of particles in "aa" species:
            // ===================================
            int NSP = IONS->at(aa).NSP;

            // Mass of "aa" species:
            // =====================
            double Ma = IONS->at(aa).M;

            // Particle loop:
            // ==================================
            #pragma omp parallel for default(none) shared(Ma, params, IONS, EB, aa, CS, t, kper, kpar, w, E_RF, DT) firstprivate(NSP)
            for(int ii=0; ii<NSP; ii++)
            {
                if ( IONS->at(aa).f3(ii) == 1 )
                {
                    arma::rowvec V = IONS->at(aa).V.row(ii);
                    //advanceVelocity(ii,params,&IONS->at(aa),&V,params->RF.Erf,params->DT);
                    IONS->at(aa).V.row(ii) = V;
                    IONS->at(aa).dE3(ii)   = 0.5*Ma*dot(V,V);  // ***** this variable needs to be cleared onced accumulated ***
                    IONS->at(aa).f3(ii)    = 0;
                } // f3 guard

            } // Particle Loop

        } // Species loop

    }// MPI guard

}

void rfOperator::applyRfOperator(const simulationParameters * params, const characteristicScales * CS, twoDimensional::fields * EB, vector<twoDimensional::ionSpecies> * IONS)
{

}

void rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    // Particle MPI sentinel
    //      params->RF.E3 = 0;
    //      double DT = params->DT;
    //      loop over all Species
    //      {
    //          double alpha = IONS->at(aa).NCP;
    //          IONS->at(aa).RF.E3 = 0;
    //          loop over all particles
    //          if (f3 == 1)
    //          {
    //              double a   = IONS->at(aa).a(ii);
    //              double dE3 = IONS->at(aa).RF.dE3(ii); [J]
    //              IONS->at(aa).RF.E3 += (alpha/DT)*a*dE3; [W]
    //          }
    //          MPI_AllreduceDouble on IONS->at(aa).RF.E3
    //        }
    //        params->RF.E3 += IONS->at(aa).RF.E3; [W]
}

void rfOperator::calculateAppliedPrf(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{

}
