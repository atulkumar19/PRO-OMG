#include "alfvenic.h"

ALFVENIC::ALFVENIC(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS){

	if(params->numberOfAlfvenicModes > 0){
		if(params->loadModes == 0){
			generateModes(params,mesh,EB,IONS);
		}else{
			loadModes(params,mesh,EB,IONS);
		}
	}
}

void ALFVENIC::generateModes(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS){
	plasmaParams PP(params,IONS);

	Aw.amp.set_size(params->numberOfAlfvenicModes);
	Aw.wavenumber.set_size(params->numberOfAlfvenicModes);
	Aw.angularFreq.set_size(params->numberOfAlfvenicModes);
	Aw.phase.set_size(params->numberOfAlfvenicModes);
	Aw.dB.zeros(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);

	Aw.numberOfTestModes = params->numberOfTestModes;
	Aw.kappa.set_size(Aw.numberOfTestModes);
	Aw.omega.set_size(Aw.numberOfTestModes);

	mat afBrackets(Aw.numberOfTestModes,2);//angular frequency brackets (limits) for the root finding

	Aw.L = mesh->nodes.X(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS-1) + mesh->DX;
	vec x = (mesh->nodes.X + mesh->DX/2);

	for(int ii=0;ii<Aw.numberOfTestModes;ii++){
		Aw.kappa(ii) = 2*M_PI*(double)(ii+1)/(Aw.L/3);// In meters! in the dispertion relation it must be cm!
	}

	vec w = linspace<vec>(0,0.999*PP.wca,100);

	for(int jj=0;jj<Aw.numberOfTestModes;jj++){
		double w1,w2;
		for(int ii=0;ii<w.size()-1;ii++){
			double ratio;
			ratio = function(&PP,w(ii),Aw.kappa(jj)/100)/function(&PP,w(ii+1),Aw.kappa(jj)/100);//Factor 1/100 to transform to cm^-1
			if( ratio < 0 ){
				w1 = w(ii);
				w2 = w(ii+1);
				break;
			}
		}
		Aw.omega(jj) = brentRoots(&PP,w1,w2,Aw.kappa(jj)/100,100);//Factor 1/100 to transform to cm^-1
	}

	double wmin(0.048*PP.wcp),wmax(0.427*PP.wcp);
	uvec I = find(Aw.omega > wmin,params->numberOfAlfvenicModes,"first");

	Aw.wavenumber = Aw.kappa(I);
	Aw.angularFreq = Aw.omega(I);
	Aw.amp = 1/Aw.angularFreq;
	Aw.amp = params->BGP.Bo*sqrt( params->fracMagEnerInj*Aw.amp/sum(Aw.amp) );


	if(params->mpi.rank_cart == 0)
		Aw.phase = (2*M_PI*params->maxAngle/360)*randu<vec>(params->numberOfAlfvenicModes);
	MPI_ARMA_VEC mpi_phase(params->numberOfAlfvenicModes);
	MPI_Bcast(Aw.phase.memptr(),1,mpi_phase.type,0,params->mpi.mpi_topo);

	double PHI(params->BGP.propVectorAngle*M_PI/180);

	for(int ii=0;ii<params->numberOfAlfvenicModes;ii++){//Here the staggered grid is not taken into account.
		double Bx(Aw.amp(ii)*cos(PHI)),By(Aw.amp(ii)),Bz(Aw.amp(ii)*sin(PHI));

		Aw.dB.X += -Bx*sin( Aw.wavenumber(ii)*x + Aw.phase(ii) );
		Aw.dB.Y += By*cos( Aw.wavenumber(ii)*x + Aw.phase(ii) );
		Aw.dB.Z += Bz*sin( Aw.wavenumber(ii)*x + Aw.phase(ii) );
	}

	for(int ii=0;ii<params->numberOfIonSpecies;ii++){
		mat u_amp(params->numberOfAlfvenicModes,3);
		if(ii==0){//Protons
			for(int jj=0;jj<params->numberOfAlfvenicModes;jj++){
				double phVel = Aw.angularFreq(jj)/Aw.wavenumber(jj);
				u_amp(jj,0) = 0;
				u_amp(jj,1) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wcp);
				u_amp(jj,2) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wcp);
			}
		}else if(ii==1){//Alpha-particles
			for(int jj=0;jj<params->numberOfAlfvenicModes;jj++){
				double phVel = Aw.angularFreq(jj)/Aw.wavenumber(jj);
				u_amp(jj,0) = 0;
				u_amp(jj,1) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wca);
				u_amp(jj,2) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wca);
			}
		}
		Aw.Uo.push_back(u_amp);
	}

}

void ALFVENIC::loadModes(const inputParameters * params,const meshGeometry * mesh,emf * EB,vector<ionSpecies> * IONS){
	plasmaParams PP(params,IONS);

	mat spectra;
	bool status;

	if(params->argc == 3){
		string tmp(params->argv[2]);
		string tmpString = "inputFiles/spectra_" + tmp + ".alfven";
		status = spectra.load(tmpString,raw_ascii);
	}else{
		status = spectra.load("inputFiles/spectra.alfven",raw_ascii);
	}

	if(status && (params->numberOfAlfvenicModes == spectra.n_rows)){

		Aw.L = mesh->nodes.X(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS-1) + mesh->DX;
		vec x = (mesh->nodes.X + mesh->DX/2);

		Aw.amp.set_size(spectra.n_rows);
		Aw.wavenumber.set_size(spectra.n_rows);
		Aw.angularFreq.set_size(spectra.n_rows);
		Aw.phase.set_size(spectra.n_rows);
		Aw.dB.zeros(mesh->dim(0)*params->mpi.NUMBER_MPI_DOMAINS);

		for(int ii=0;ii<spectra.n_rows;ii++){
			Aw.angularFreq(ii) = spectra(ii,0);
			Aw.amp(ii) = sqrt( spectra(ii,1) );
			Aw.wavenumber(ii) = 100*dispertionRelation(Aw.angularFreq(ii),&PP);// in m^{-1}.
			Aw.phase(ii) = spectra(ii,2);
		}

		if(params->shuffleModes == 1)
			Aw.phase = shuffle(Aw.phase);

		double PHI(params->BGP.propVectorAngle*M_PI/180);

//		cout << sum(Aw.amp%Aw.amp) << '\t' << spectra.n_rows << '\n';

		for(int ii=0;ii<spectra.n_rows;ii++){//Here the staggered grid is not taken into account.
			double Bx(Aw.amp(ii)*cos(PHI)),By(Aw.amp(ii)),Bz(Aw.amp(ii)*sin(PHI));
			double Cx,Cy;
			Cx = (2/Aw.L)*sin( 0.5*Aw.wavenumber(ii)*Aw.L )*cos( 0.5*(Aw.wavenumber(ii)*Aw.L + 2*Aw.phase(ii)) );
			Cy = (2/Aw.L)*sin( 0.5*Aw.wavenumber(ii)*Aw.L )*sin( 0.5*(Aw.wavenumber(ii)*Aw.L + 2*Aw.phase(ii)) );

			Aw.dB.X += -Bx*( sin( Aw.wavenumber(ii)*x + Aw.phase(ii) ) - Cx*x );
			Aw.dB.Y += By*( cos( Aw.wavenumber(ii)*x + Aw.phase(ii) ) + Cy*x );
			Aw.dB.Z += Bz*( sin( Aw.wavenumber(ii)*x + Aw.phase(ii) ) - Cx*x );
		}

		double sc = sqrt( mean( (Aw.dB.X%Aw.dB.X) + (Aw.dB.Y%Aw.dB.Y) + (Aw.dB.Z%Aw.dB.Z) ) );

		Aw.dB.zeros(Aw.dB.X.n_elem);

		for(int ii=0;ii<spectra.n_rows;ii++){//Here the staggered grid is not taken into account.
			Aw.amp(ii) *= sqrt(params->fracMagEnerInj)*params->BGP.Bo/sc;

			double Bx(Aw.amp(ii)*cos(PHI)),By(Aw.amp(ii)),Bz(Aw.amp(ii)*sin(PHI));
			double Cx, Cy;

			Cx = (2/Aw.L)*sin( 0.5*Aw.wavenumber(ii)*Aw.L )*cos( 0.5*(Aw.wavenumber(ii)*Aw.L + 2*Aw.phase(ii)) );
			Cy = (2/Aw.L)*sin( 0.5*Aw.wavenumber(ii)*Aw.L )*sin( 0.5*(Aw.wavenumber(ii)*Aw.L + 2*Aw.phase(ii)) );

			Aw.dB.X += -Bx*( sin( Aw.wavenumber(ii)*x + Aw.phase(ii) ) - Cx*x );
			Aw.dB.Y += By*( cos( Aw.wavenumber(ii)*x + Aw.phase(ii) ) + Cy*x );
			Aw.dB.Z += Bz*( sin( Aw.wavenumber(ii)*x + Aw.phase(ii) ) - Cx*x );
		}

		for(int ii=0;ii<params->numberOfIonSpecies;ii++){
			mat u_amp(spectra.n_rows,3);
			if(ii==0){//Protons
				for(int jj=0;jj<spectra.n_rows;jj++){
					double phVel = Aw.angularFreq(jj)/Aw.wavenumber(jj);
					u_amp(jj,0) = 0;
					u_amp(jj,1) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wcp);
					u_amp(jj,2) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wcp);
				}
				//				u_amp.save("U_protons.dat",raw_ascii);
			}else if(ii==1){//Alpha-particles
				for(int jj=0;jj<spectra.n_rows;jj++){
					double phVel = Aw.angularFreq(jj)/Aw.wavenumber(jj);
					u_amp(jj,0) = 0;
					u_amp(jj,1) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wca);
					u_amp(jj,2) = -(Aw.amp(jj)/params->BGP.Bo)*phVel/(1-Aw.angularFreq(jj)/PP.wca);
				}
				//				u_amp.save("U_alphas.dat",raw_ascii);
			}
			Aw.Uo.push_back(u_amp);
		}

		//		Aw.wavenumber.save("k.dat",raw_ascii);
		//		Aw.angularFreq.save("w.dat",raw_ascii);
//		exit(0);
	}else{
		exit(1);
	}
}


void ALFVENIC::dispertionRelation(const plasmaParams *PP){
	vec w = linspace<vec>(0,0.999*PP->wca,1000);
	vec k;

	k = (w/A_C) % sqrt( 1 - PP->wpe*PP->wpe/(w%(w+PP->wce)) - PP->wpp*PP->wpp/(w%(w-PP->wcp)) - PP->wpa*PP->wpa/(w%(w-PP->wca)) );

	w.save("freq.dat",raw_ascii);
	k.save("wnumb.dat",raw_ascii);
}

double ALFVENIC::dispertionRelation(double w,plasmaParams * PP){
	double k;
	k = (w/A_C)*sqrt( 1 - PP->wpe*PP->wpe/(w*(w+PP->wce)) - PP->wpp*PP->wpp/(w*(w-PP->wcp)) - PP->wpa*PP->wpa/(w*(w-PP->wca)) );
	return(k);
}

/* From the dispersion relation we have the following equation to
   calculate the ratio w/k used in the perturbation amplitudes */
double ALFVENIC::function(const plasmaParams *PP,double w,double k){
	double fw(0);
	fw += w*w*(w + PP->wce)*(w - PP->wcp)*(w - PP->wca) - w*(w - PP->wcp)*(w - PP->wca)*PP->wpe*PP->wpe;
	fw += - w*(w + PP->wce)*(w - PP->wca)*PP->wpp*PP->wpp - w*(w + PP->wce)*(w - PP->wcp)*PP->wpa*PP->wpa;
	fw += - (w + PP->wce)*(w - PP->wcp)*(w - PP->wca)*k*k*A_C*A_C;
	return(fw);
}

double ALFVENIC::brentRoots(const plasmaParams *PP,double x1,double x2,double k,int ITMAX){

	double nearZero(1E-20);
	double tolerance (1E-10);
	double FPP(1E-11);

	double a(x1), b(x2), c(x2);
	double d, e, tol, xm, fa(function(PP,x1,k)), fb(function(PP,x2,k)), fc(function(PP,x2,k));
	double P, Q, R, S, T;

	if( (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) ){
		cout << "Root not bracketed.\n";
		exit(1);
	}

	for(int it=0;it<ITMAX;it++){
		if( (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) ){
			c = a;
			fc = fa;
			e = b - a;
			d = e;
		}
		if( abs(fc) < abs(fb) ){
			a = b;
			b = c;
			c = a;
			fa = fb;
			fb = fc;
			fc = fa;
		}

		tol = 2.0*FPP*abs(b) + 0.5*tolerance;
		xm = 0.5*(c - b);
//		cout << "abs(xm) = "  << abs(xm) << " tol = " << tol << '\n';
//		cout << "f(b) = " << fb << " b = "<< b << '\n';
		if(abs(xm) < tol || abs(fb) < nearZero)
			return(b);

		if( abs(e) > tol  && abs(fa) > abs(fb) ){
			S = fb/fa;
			if(abs(a - c) < nearZero){
				P = 2.0*xm*S;
				Q = 1.0 - S;
			}else{
				T = fa/fc;
				R = fb/fc;
				P = S*(2.0*xm*T*(T-R) - (b-a)*(R-1.0));
				Q = (T-1.0)*(R-1.0)*(S-1.0);
			}

			if(P > 0.0)
				Q = -Q;

			P = abs(P);

			double min1(3.0*xm*Q - abs(tol*Q)),min2(abs(e*Q));
			if(2.0*P < min1 < min2 ? min1 : min2){
				e = d;
				d = P/Q;
			}else{
				d = xm;
				e = d;
			}
		}else{
			d = xm;
			e = d;
		}

		a = b;
		fa = fb;
		if(abs(d) > tol){
			b += d;
		}else{
			if (xm > 0)
				b += fabs(tol);
		    else
				b -= fabs(tol);
		}
		fb = function(PP,b,k);

//		cout << "f(b) = " << fb << " b = "<< b << '\n';
//		cout << "Iteration " << it << '\n';
	}
//	cout << "Number of iterations exceeded\n";
	exit(1);

}

void ALFVENIC::normalize(const characteristicScales * CS){
	Aw.L /= CS->length;
	if(Aw.Uo.size() > 0){
		Aw.dB /= CS->bField;
		for(int ii=0;ii<Aw.Uo.size();ii++){
			Aw.Uo[ii] /= CS->velocity;
		}
		Aw.wavenumber *= CS->length;
		Aw.angularFreq *= CS->time;
	}
}

void ALFVENIC::addMagneticPerturbations(emf * EB){
	unsigned int NX(EB->B.X.n_elem);

	EB->B.X.subvec(1,NX-2) += Aw.dB.X;
	EB->B.Y.subvec(1,NX-2) += Aw.dB.Y;
	EB->B.Z.subvec(1,NX-2) += Aw.dB.Z;
}

void ALFVENIC::addVelocityPerturbations(const inputParameters * params,vector<ionSpecies> * IONS){

	double PHI(params->BGP.propVectorAngle*M_PI/180);

	for(unsigned int ii=0;ii<params->numberOfIonSpecies;ii++){
		for(int jj=0;jj<params->numberOfAlfvenicModes;jj++){
			double vx(Aw.Uo[ii](jj,2)*cos(PHI)), vy(Aw.Uo[ii](jj,1)), vz(Aw.Uo[ii](jj,2)*sin(PHI));
			double Cx, Cy;
			Cx = (2/Aw.L)*sin( 0.5*Aw.wavenumber(jj)*Aw.L )*cos( 0.5*(Aw.wavenumber(jj)*Aw.L + 2*Aw.phase(jj)) );
			Cy = (2/Aw.L)*sin( 0.5*Aw.wavenumber(jj)*Aw.L )*sin( 0.5*(Aw.wavenumber(jj)*Aw.L + 2*Aw.phase(jj)) );

			IONS->at(ii).V.col(0) += -vx*( sin( Aw.wavenumber(jj)*IONS->at(ii).X.col(0) + Aw.phase(jj)) - Cx*IONS->at(ii).X.col(0));
			IONS->at(ii).V.col(1) += vy*( cos( Aw.wavenumber(jj)*IONS->at(ii).X.col(0) + Aw.phase(jj)) + Cy*IONS->at(ii).X.col(0) );
			IONS->at(ii).V.col(2) += vz*( sin( Aw.wavenumber(jj)*IONS->at(ii).X.col(0) + Aw.phase(jj)) - Cx*IONS->at(ii).X.col(0) );


		}
	}

}

void ALFVENIC::addPerturbations(const inputParameters * params,vector<ionSpecies> * IONS,emf * EB){

	if(params->numberOfAlfvenicModes > 0){
		addMagneticPerturbations(EB);
		addVelocityPerturbations(params,IONS);
	}

}
