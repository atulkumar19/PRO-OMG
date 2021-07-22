#include <math.h>
#include "collisionOperator.h"

using namespace std;

collisionOperator::collisionOperator()
{}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<oneDimensional::ionSpecies> * IONS)
{
    if (params->mpi.COMM_COLOR == PARTICLES_MPI_COLOR)
    {
        // Number of ION species:
    	// =====================
    	int numIonSpecies = IONS->size();

        // Time step:
        // =========
        double dt = params->DT*CS->time;

    	for (int aa=0; aa<numIonSpecies; aa++)
    	{
            // Number of particles is "aa" species:
        	// ===================================
        	int NSP_a = IONS->at(aa).NSP;

        	// Species "aa" parameters:
        	// =======================
        	double Ma = IONS->at(aa).M*CS->mass;
        	double Za = IONS->at(aa).Z;

            // Initialize Species "bb" parameters:
            // =======================
            double Mb(0.0);
            double Zb(0.0);
            arma::vec nb  =  zeros(NSP_a,1);
            arma::vec Tb  =  zeros(NSP_a,1);
            arma::vec uxb =  zeros(NSP_a,1);

            // Initialize total ion density and flux density:
        	// ===========================================
        	arma::vec nUx_i = zeros(NSP_a,1);
        	arma::vec n_i   = zeros(NSP_a,1);

        	for (int bb=0; bb<(numIonSpecies+1); bb++)
            {
                // Background species "bb" conditions:
				// ==================================
				if (bb < numIonSpecies) // Ions:
				{
					// Background parameters:
					Mb = IONS->at(bb).M*CS->mass;
					Zb = IONS->at(bb).Z;

					// Interpolate moments:
					interpolateIonMoments(params,IONS,aa,bb);
					arma::vec n_p    = IONS->at(aa).n_p/CS->length;
					arma::vec nv_p   = IONS->at(aa).nv_p*CS->velocity/CS->length;
					arma::vec Tpar_p = IONS->at(aa).Tpar_p*CS->temperature*F_KB/F_E;
					arma::vec Tper_p = IONS->at(aa).Tper_p*CS->temperature*F_KB/F_E;

					// Background conditions:
					nb = n_p;
					Tb = 0.5*(Tpar_p + Tper_p);
					uxb = nv_p/n_p;

					// Accumulate total ion density and ion flux density:
					n_i   = n_i + n_p*Zb;
					nUx_i = nUx_i + nv_p*Zb;
				}
				else // Electrons:
				{
					// Background parameters:
					Mb = F_ME;
					Zb = -1;

					// Background conditions:
					nb  = n_i;
					Tb  = ones(NSP_a,1)*params->f_IC.Te*CS->temperature*F_KB/F_E;
					uxb = nUx_i/n_i;
				}

                // Apply collisions to all particles:
				// ==================================
                #pragma omp parallel for default(none) shared(params, IONS, aa, CS, Ma, Za, Mb, Zb, nb, Tb, uxb, dt) firstprivate(NSP_a)
				for(int ii=0; ii<NSP_a; ii++)
				{
					// Species "aa":
					// =============================================================================
					// Velocities:
					double vxa = IONS->at(aa).V(ii,0)*CS->velocity;
					double vya = IONS->at(aa).V(ii,1)*CS->velocity;
					double vza = IONS->at(aa).V(ii,2)*CS->velocity;

					// Convert to ion species "bb" frame:
					// =============================================================================
					double wxa = vxa - uxb(ii);
					double wya = vya;
					double wza = vza;

                    // Convert velocity from cartesian to spherical coordinate system:
					// =============================================================================
					double w(0.0);
					double xi(0.0);
					double phi(0.0);
					cartesian2Spherical(&wxa, &wya, &wza, &w, &xi, &phi);

                    // Apply Monte-Carlo collision operator:
					// =============================================================================
					double w0   = w;
					double xi0  = xi;
					double phi0 = phi;

					double wTb = sqrt(2*F_E*Tb(ii)/Mb);
					double xab = w0/wTb;

					// Velocity operator:
					u_CollisionOperator(&w0,xab,wTb,nb(ii),Tb(ii),Mb,Zb,Za,Ma,dt);

					// Pitch angle operator:
					xi_CollisionOperator(&xi0,xab,wTb,nb(ii),Tb(ii),Mb,Zb,Za,Ma,dt);

					// Gyro-phase operator:
					phi_CollisionOperator(&phi0,xi0,xab,wTb,nb(ii),Tb(ii),Mb,Zb,Za,Ma,dt);

					// Final Velocity:
					// =============================================================================
					w = w0;

					// Final pitch angle:
					// =============================================================================
					xi = xi0;
					// Reflective boundary condition:
					if (xi > 1)
					{
						xi = +1 - fmod(xi,+1);
					}
					else if (xi < -1)
					{
						xi = -1 - fmod(xi,-1);
					}

					// Final gyro-phase angle:
					// =============================================================================
					phi = phi0;
					// Periodic boundary condition:
					if (phi > M_PI)
					{
						phi = phi - 2*M_PI;
					}
					else if (phi < -M_PI)
					{
						phi = phi + 2*M_PI;
					}

					// Convert velocity from spherical to cartesian coordinate sytem:
					// =====================================================================
					Spherical2Cartesian(&w,&xi,&phi,&wxa,&wya,&wza);

					// Back to lab frame and normalize:
					// =====================================================================
					IONS->at(aa).V(ii,0) = (wxa + uxb(ii))/CS->velocity;
					IONS->at(aa).V(ii,1) = wya/CS->velocity;
					IONS->at(aa).V(ii,2) = wza/CS->velocity;
                } // "ii" particle loop

            } // "bb" species loop

        } // "aa" species loop

    } // MPI if statement

}

void collisionOperator::ApplyCollisionOperator(const simulationParameters * params, const characteristicScales * CS, vector<twoDimensional::ionSpecies> * IONS)
{}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, vector<oneDimensional::ionSpecies> * IONS, int a, int b)
{
    //  Number of computational particles:
    int NSP(IONS->at(a).NSP);

    // Create particle-defined quantities:
    arma::vec n_p    = zeros(NSP,1);
    arma::vec nv_p   = zeros(NSP,1);
    arma::vec Tpar_p = zeros(NSP,1);
    arma::vec Tper_p = zeros(NSP,1);

    // Create mesh defined quantities:
    arma::vec n_m    = IONS->at(b).n;
    arma::vec nv_m   = IONS->at(b).nv.X;
    arma::vec Tpar_m = IONS->at(b).Tpar_m;
    arma::vec Tper_m = IONS->at(b).Tper_m;

    // Interpolate:
    interpolateScalarField(params, &IONS->at(a), &n_m   , &n_p   );
    interpolateScalarField(params, &IONS->at(a), &nv_m  , &nv_p  );
    interpolateScalarField(params, &IONS->at(a), &Tpar_m, &Tpar_p);
    interpolateScalarField(params, &IONS->at(a), &Tper_m, &Tper_p);

    // Assign values:
    IONS->at(a).n_p    = n_p;
    IONS->at(a).nv_p   = nv_p;
    IONS->at(a).Tpar_p = Tpar_p;
    IONS->at(a).Tper_p = Tper_p;
}

void collisionOperator::interpolateIonMoments(const simulationParameters * params, vector<twoDimensional::ionSpecies> * IONS, int a, int b)
{}

void collisionOperator::interpolateScalarField(const simulationParameters * params, oneDimensional::ionSpecies * IONS, arma::vec * F_m, arma::vec * F_p)
{
	// Triangular Shape Cloud (TSC) scheme. See Sec. 5-3-2 of R. Hockney and J. Eastwood, Computer Simulation Using Particles.
	//		wxl		   wxc		wxr
	// --------*------------*--------X---*--------
	//				    0       x

	//wxc = 0.75 - (x/H)^2
	//wxr = 0.5*(1.5 - abs(x)/H)^2
	//wxl = 0.5*(1.5 - abs(x)/H)^2

	int NX =  params->mesh.NX_IN_SIM + 4; //Mesh size along the X axis (considering the gosht cell)
	int NSP(IONS->NSP);

	arma::vec F = zeros(NX);

	F.subvec(1,NX-2) = *F_m;

	fill4Ghosts(&F);

	//Contrary to what may be thought,F_p is declared as shared because the private index ii ensures
	//that each position is accessed (read/written) by one thread at the time.
	#pragma omp parallel for default(none) shared(params, IONS, F_p, F) firstprivate(NSP)
	for(int ii=0; ii<NSP; ii++)
	{
		int ix = IONS->mn(ii) + 2;

		(*F_p)(ii) += IONS->wxl(ii)*F(ix-1);
		(*F_p)(ii) += IONS->wxc(ii)*F(ix);
		(*F_p)(ii) += IONS->wxr(ii)*F(ix+1);

	}//End of the parallel region
}

void collisionOperator::interpolateScalarField(const simulationParameters * params, twoDimensional::ionSpecies * IONS, arma::mat * F_m, arma::mat * F_p)
{}

void collisionOperator::fill4Ghosts(arma::vec * v)
{
	int N = v->n_elem;

	//v->subvec(N-2,N-1) = v->subvec(2,3);
	//v->subvec(0,1) = v->subvec(N-4,N-3);
    
    v->subvec(N-2,N-1) = v->subvec(N-4,N-3);
    v->subvec(0,1)     = v->subvec(2,3);
}

void collisionOperator::cartesian2Spherical(double * wx, double * wy, double * wz, double * w, double * xi, double * phi)
{
    *w = sqrt( pow(*wx,2.0) + pow(*wy,2.0) + pow(*wz,2.0) );
    *xi = (*wx)/(*w);
    *phi = atan2(-*wy,*wz);
}

void collisionOperator::Spherical2Cartesian(double * w, double * xi, double * phi, double * wx, double * wy, double * wz)
{
    double wper = (*w)*sqrt( 1.0 - pow((*xi),2.0) );
    *wx   = (*w)*(*xi);
    *wy   = -wper*sin(*phi);
    *wz   = +wper*cos(*phi);
}

void collisionOperator::u_CollisionOperator(double * w, double xab,double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt)
{
    double BoozerFactor = (double)0.5;
    double nu_E_dt(0.0);
    int energyOperatorModel = 2;

    // Normalized collision rate:
    nu_E_dt = BoozerFactor*nu_E(xab,nb,Tb,Mb,Zb,Za,Ma,energyOperatorModel)*dt;

    // Calculate substeps:
    int Nstep   = round(nu_E_dt/0.4) + 1;
    double dt_s = (double) dt/Nstep;

    // Limit substepping:
    if (Nstep > 20)
    {
        cout << "Nstep for 'w' operator = " << Nstep << endl;
        Nstep = 20;
    }

    // Apply operator:
    nu_E_dt  = nu_E_dt/Nstep;

    for (int kk = 0; kk<Nstep; kk++)
    {
        double E0 = (0.5*Ma*pow((*w),2.0))/F_E;
        double A = -2.0*nu_E_dt*E0;
        double B = 2.0*nu_E_dt*( 1.5 + E_nuE_d_nu_E_dE(xab))*Tb;

        // Random number between 0 and 1:
        double randomNumber = (double) rand()/RAND_MAX;
        double Rm;

        // -1 or +1 function:
        if (randomNumber - 0.5 < 0)
        {
            Rm = -1.0;
        }
        else
        {
            Rm = +1.0;
        }

        double C = 2.0*Rm*sqrt(Tb*E0*nu_E_dt);
        E0 = E0 + A + B + C;
        *(w) = sqrt(2.0*F_E*E0/Ma);
    }

}

void collisionOperator:: xi_CollisionOperator(double * xi, double xab, double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt)
{
    // Normalized collisional rate:
    // ===========================
    double nu_D_dt(0.0);
    nu_D_dt = nu_D(xab,nb,Tb,Mb,Zb,Za,Ma)*dt;

    // Calculate substeps:
    // ===========================
    int Nstep   = round(nu_D_dt/0.4) + 1;
    double dt_s = (double) dt/Nstep;

    // Recalculate normalized rate:
    // ============================
    nu_D_dt  = nu_D_dt/Nstep;

    // Limit substepping:
    // ===========================
    if (Nstep > 20)
    {
        cout << "Nstep for 'xi' operator = " << Nstep << endl;
        Nstep = 20;
    }

    // Apply operator:
    // ===========================
    for (int kk = 0; kk<Nstep; kk++)
    {
        // Deterministic part:
        // ==================
        double A = -(*xi)*nu_D_dt;
        double B = 0.0;

        // Stochastic part:
        // ===============
        // Random number between 0 and 1:
        double randomNumber = (double) rand()/RAND_MAX;
        double Rm;

        // -1 or +1 function:
        if (randomNumber - 0.5 < 0)
        {
            Rm = -1.0;
        }
        else
        {
            Rm = +1.0;
        }

        double C = Rm*sqrt( (1.0 - pow(*(xi),2.0))*nu_D_dt );

        // Monte-Carlo change:
        // ==================
       *(xi) = *(xi) + A + B + C;
    }

}

void collisionOperator::phi_CollisionOperator(double * phi, double xi, double xab, double wTb, double nb, double Tb, double Mb, double Zb, double Za, double Ma, double dt)
{
    // Normalized collisional rate:
    // ===========================
    double nu_D_dt(0.0);
    nu_D_dt = nu_D(xab,nb,Tb,Mb,Zb,Za,Ma)*dt;

    // Calculate substeps:
    // ===========================
    int Nstep   = round(nu_D_dt/0.4) + 1;
    double dt_s = (double) dt/Nstep;

    // Recalculate normalized rate:
    // ============================
    nu_D_dt  = nu_D_dt/Nstep;

    // Limit substepping:
    // ===========================
    if (Nstep > 20)
    {
        cout << "Nstep for 'phi' operator = " << Nstep << endl;
        Nstep = 20;
    }

    // Apply operator:
    // ===========================
    for (int kk = 0; kk<Nstep; kk++)
    {
        // Deterministic part:
        // ==================
        double A = 0.0;
        double B = 0.0;

        // Stochastic part:
        // ===============
        // Random number between 0 and 1:
        double randomNumber = (double) rand()/RAND_MAX;
        double Rm;

        // -1 or +1 function:
        if (randomNumber - 0.5 < 0)
        {
            Rm = -1.0;
        }
        else
        {
            Rm = +1.0;
        }

        if (xi==1)
        {
            xi = 0.999;
        }

        double C = Rm*sqrt( nu_D_dt/(1.0 - pow(xi,2.0)) );

        // Monte-Carlo change:
        // ==================
       *(phi) = *(phi) + A + B + C;
    }

}

double collisionOperator::nu_E(double xab, double nb, double Tb, double Mb, double Zb, double Za, double Ma, int energyOperatorModel)
{
    double y;

    if (energyOperatorModel == 1)
    {
        // From Hinton 1983 EQ 92 and L. Chen 1988 EQ 50
        y = nu_ab0(nb,Tb,Mb,Zb,Za,Ma)*( (2.0*(Ma/Mb)*Gb(xab)/xab) - (erfp(xab)/(pow(xab,2.0)) ) );
    }
    else if  (energyOperatorModel == 2)
    {
        // From L. Chen 1983 EQ 57 commonly used for NBI
        y = nu_ab0(nb,Tb,Mb,Zb,Za,Ma)*(2.0*(Ma/Mb))*(Gb(xab)/xab);
    }

    return y;
}

double collisionOperator::nu_D(double xab, double nb, double Tb, double Mb, double Zb, double Za, double Ma)
{
    double y = nu_ab0(nb,Tb,Mb,Zb,Za,Ma)*(erf(xab) - Gb(xab))/pow(xab,3.0);
    return y;
}

double collisionOperator::nu_ab0(double nb, double Tb, double Mb, double Zb, double Za, double Ma)
{
    double wTb = sqrt(2.0*F_E*Tb/Mb);
    double y  = nb*pow(F_E,4.0)*pow(Za*Zb,2.0)*logA(nb,Tb)/(2.0*M_PI*Ma*Ma*F_EPSILON*F_EPSILON*pow(wTb,3.0));

    return y;
}

double collisionOperator::logA(double nb, double Tb)
{
    double y = 30.0 - log( sqrt( nb*pow(Tb,-3.0/2) ) );
    return y;
}

double collisionOperator::Gb(double xab)
{
    double y;

    if (xab < 0.01)
    {
        y = (2.0/sqrt(M_PI))*xab/3;
    }
    else
    {
        y = (erf(xab) - xab*erfp(xab))/(2.0*pow(xab,2.0));
    }

    return y;
}

double collisionOperator::erfp(double xab)
{
    double y = (2.0/sqrt(M_PI))*exp(-pow(xab,2.0));
    return y;
}

double collisionOperator::erfpp(double xab)
{
    double y = -(4.0*xab/sqrt(M_PI))*exp(-pow(xab,2.0));
    return y;
}

double collisionOperator::E_nuE_d_nu_E_dE(double xab)
{
    double y = 0.5*(  ( 3.0*xab*erfp(xab) - 3.0*erf(xab) - (pow(xab,2.0))*erfpp(xab) )/( erf(xab) - (xab*erfp(xab)) )  );
    return y;
}

/*
if (true)
{
    cout << "xab: " << xab << endl;
    cout << "nb: " << nb << endl;
    cout << "Tb: " << Tb << endl;
    cout << "Mb: " << Mb << endl;
    cout << "Zb: " << Zb << endl;
    cout << "Ma: " << Ma << endl;
    cout << "Za: " << Za << endl;
    cout << "nu_E: " << nu_E_dt/dt << endl;

    cout << "nu_ab0: " << nu_ab0(nb,Tb,Mb,Zb,Za,Ma) << endl;
    cout << "Gb: " << Gb(xab) << endl;
    cout << "logA: " << logA(nb,Tb) << endl;
    cout << "pow(Tb,-3/2): " << pow(Tb,-3.0/2) << endl;
}
*/
