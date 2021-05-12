% Objective:
% Read and analyze the output data produced by Prometheus++
% Data has been produced by a initial implementation of a 4D Metropolis
% algorithm in Prometheus++:

clear all;
close all;

myDir = 'm2021_01_13/HDF5';
%  myDir = 'm2020_10_20e/HDF5'; % bad agreement
% myDir = 'm2020_10_20d/HDF5'; % good agreement
%  myDir = 'm2020_10_16a/HDF5'; % good agreement
% myDir = 'C:\Users\nfc\OneDrive - Oak Ridge National Laboratory\Projects\LDRD_Project\Prometheus\mirror-simulations\m2020_10_16a\HDF5';
cd(myDir)
% Read HDF5 data:
% =========================================================================

fileName = 'main.h5';
m = HDF2Struct_v2(fileName);
ss  = m.geometry.xAxis;

ranksFields = 2;
for ii = 1:ranksFields
        fileName = ['FIELDS_FILE_',num2str(ii-1),'.h5'];
        eb{ii} = HDF2Struct_v2(fileName);
        
        if ii == 1
            tii = fieldnames(eb{ii});
            numOut = length(tii);
        end

        for jj=1:numOut  
            dum = tii{jj};
            kk = str2double(dum(2:end)) + 1;
            % Magnetic field:
            bbx{ii}{kk} = eb{ii}.(tii{jj}).fields.B.x; % x
            bby{ii}{kk} = eb{ii}.(tii{jj}).fields.B.y; % y
            bbz{ii}{kk} = eb{ii}.(tii{jj}).fields.B.z; % z

            % Electric field:
            eex{ii}{kk} = eb{ii}.(tii{jj}).fields.E.x; % x
            eey{ii}{kk} = eb{ii}.(tii{jj}).fields.E.y; % x
            eez{ii}{kk} = eb{ii}.(tii{jj}).fields.E.z; % x
        end
end

ranksParticles = 38;
for ii = 1:ranksParticles
    fileName = ['PARTICLES_FILE_',num2str(ii-1),'.h5'];
    d{ii} = HDF2Struct_v2(fileName);

    for jj=1:numOut
        dum = tii{jj};
        kk = str2double(dum(2:end)) + 1;    
        xx{ii}{kk}   = d{ii}.(tii{jj}).ions.spp_1.X;
        pc{ii}(kk)   = d{ii}.(tii{jj}).ions.spp_1.pCount;
        ec{ii}(kk)   = d{ii}.(tii{jj}).ions.spp_1.eCount;
        vv1{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,1); % x
        vv2{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,2); % y
        vv3{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,3); % z
        nn{kk}       = d{1}.(tii{jj}).ions.spp_1.n;
        uux{kk}      = d{1}.(tii{jj}).ions.spp_1.U.x; % x
        uuy{kk}      = d{1}.(tii{jj}).ions.spp_1.U.y; % x
        uuz{kk}      = d{1}.(tii{jj}).ions.spp_1.U.z; % x
               
        ttName{kk}       = tii{jj};
    end

end

cd ..
%%
% Concatenate data:
% =========================================================================
s  = ss;

% Fields:
for jj = 1:numOut
    Bx(:,jj) = [bbx{1}{jj};bbx{2}{jj}];
    By(:,jj) = [bby{1}{jj};bby{2}{jj}];
    Bz(:,jj) = [bbz{1}{jj};bbz{2}{jj}];
    
    Ex(:,jj) = [eex{1}{jj};eex{2}{jj}];
    Ey(:,jj) = [eey{1}{jj};eey{2}{jj}];
    Ez(:,jj) = [eez{1}{jj};eez{2}{jj}];    
end

% Particles:
for jj = 1:numOut
    Uxd = uux{jj};
    Uyd = uuy{jj};
    Uzd = uuz{jj};
    Ned = nn{jj};
    Xd  = xx{1}{jj};
    pcd = pc{1}(jj);
    ecd = ec{1}(jj);
    V1d = vv1{1}{jj};
    V2d = vv2{1}{jj};
    V3d = vv3{1}{jj};
    for ii  = 2:ranksParticles
        Xd = [Xd;xx{ii}{jj}];
        pcd = pcd+ pc{ii}(jj);
        ecd = ecd+ ec{ii}(jj);
        V1d = [V1d;vv1{ii}{jj}];
        V2d = [V2d;vv2{ii}{jj}];
        V3d = [V3d;vv3{ii}{jj}];
    end
    % Gather in time dependent variable:
    Ne(:,jj) = Ned;
    Ux(:,jj) = Uxd;
    Uy(:,jj) = Uyd;
    Uz(:,jj) = Uzd;
    X(:,jj)  = Xd;
    V1(:,jj) = V1d;
    V2(:,jj) = V2d;
    V3(:,jj) = V3d;
    pCount(:,jj)=(pcd);
    eCount(:,jj)=(ecd);
end

% Physical constants:
% =========================================================================
e_c = 1.6020e-19;
k_B = 1.3806e-23;
m_p = 1.6726e-27;
m_e = 9.1094e-31;
mu0 = 4*pi*1e-7;
c = 3e8;
E_0=m_p*c^2;

% Derived quantities:
% =========================================================================
% Temperatures and thermal velocities:
M = m.ions.spp_1.M;
Tpar = m.ions.spp_1.Tpar*k_B/e_c;
Tper = m.ions.spp_1.Tper*k_B/e_c;
vTper = sqrt(1*e_c*Tper/M);
vTpar = sqrt(1*e_c*Tpar/M);
% Real particles for each computational particle:
 NCP = m.ions.spp_1.NCP;

% Total number of computational particles:
NS = m.ions.spp_1.NSP*ranksParticles;

% Scaling factor for PDF:
N = NS*NCP;

%% Calculate statistics:
close all
% Statistics:
% =========================================================================
for jj = 1
    % Scale for X:
    NN = 500;
    xbin = linspace(0,max(X(:,jj)),NN);
    % Scale for vpar and vper:
    vxbin = linspace(-8*vTpar,+8*vTpar,NN);
    vybin = linspace(-8*vTper,+8*vTper,NN);
    vzbin = vybin;
    KEbin=linspace(0,5*max(Tper),NN);
end

ux = vxbin(1:end-1);
uy = vybin(1:end-1);
uz = vybin(1:end-1);

%% 1D distributions:
% Important note: We need to remember that the X vector below represents
% the position of each super-particle which whose charge density is charge
% per unit area. Thus we need to add compression to correctly extract the
% fluid variables.

try 
    clear n Vx Vy Vz Px Py Pz Tx Ty Tz
end

% x vector:
xVec = linspace(ss(1),ss(end),NN-1);
% x increment:
dx = xVec(2) - xVec(1);

% Magnetic field interpolated at positions of the histograms:
BBx = interp1(ss,Bx,xVec); 


% Select model:
modelType = 2;
switch modelType
    case 1
        compFactor=1;
    case 2
        % Calculate the compression factor to convert quantites to physical space:
        B0 = BBx(round(length(xVec)/2));
        compFactor = BBx/B0;
end 

for jj = 1:numOut
    fxvx{jj} = histcounts2(X(:,jj),V1(:,jj),xbin,vxbin,'Normalization','pdf');
    fxvy{jj} = histcounts2(X(:,jj),V2(:,jj),xbin,vybin,'Normalization','pdf');
    fxvz{jj} = histcounts2(X(:,jj),V3(:,jj),xbin,vzbin,'Normalization','pdf');
    fsqvzvy{jj} = histcounts2(X(:,jj),sqrt(V3(:,jj).^2+V2(:,jj).^2),xbin,vzbin,'Normalization','pdf');
    
    rng=find(X(:,jj)>2 & X(:,jj)<3);
    fvxvy{jj} = histcounts2(V1(rng,jj),V2(rng,jj),vxbin,vybin,'Normalization','pdf');
    
    % Removed Temporarely:
    %fkex{jj}=histcounts2(X(:,jj),P_KE(:,jj),xbin,KEbin,'Normalization','pdf');

    % Zeroth moment:
    n(:,jj)  =   N*compFactor(:,1).*trapz(ux,fxvx{jj}.*ux.^0,2);
    % First moment:
    Vx(:,jj) =   N*compFactor(:,1).*trapz(ux,fxvx{jj}.*ux.^1,2)./n(:,jj);
    Vy(:,jj) =   N*compFactor(:,1).*trapz(uy,fxvy{jj}.*uy.^1,2)./n(:,jj);
    Vz(:,jj) =   N*compFactor(:,1).*trapz(uz,fxvz{jj}.*uz.^1,2)./n(:,jj);
    % Second moment:
    Px(:,jj) = M*N*compFactor(:,1).*trapz(ux,fxvx{jj}.*(ux - Vx(:,jj)).^2,2);
    Py(:,jj) = M*N*compFactor(:,1).*trapz(uy,fxvy{jj}.*(uy - Vy(:,jj)).^2,2);
    Pz(:,jj) = M*N*compFactor(:,1).*trapz(uz,fxvz{jj}.*(uz - Vz(:,jj)).^2,2);
    
    % Temperature:
    Tx(:,jj) = Px(:,jj)./(n(:,jj)*e_c);
    Ty(:,jj) = Py(:,jj)./(n(:,jj)*e_c);
    Tz(:,jj) = Pz(:,jj)./(n(:,jj)*e_c);
end

%% 3D plot:
% jj = 11;
% rng=find(X(:,jj)>0 & X(:,jj)<2);
% figure; plot3(X(rng,jj),V2(rng,jj),V3(rng,jj),'k.','markersize',1)

%% Plot data:
% =========================================================================
 close all
jj = 25;

% Plot distribution function:
% =========================================================================
figure;
surf(xbin(1:end-1),vxbin(1:end-1)./vTpar,fxvx{jj}','LineStyle','none');
title('x-Vx');
xlabel('x')
ylabel('Vx')

figure;
surf(xbin(1:end-1),vybin(1:end-1)./vTper,fxvy{jj}','LineStyle','none');
title('x-Vy')
xlabel('x')
ylabel('Vy')

figure;
surf(xbin(1:end-1),vzbin(1:end-1)./vTper,fxvz{jj}','LineStyle','none');
title('x-Vz');
xlabel('x')
ylabel('Vz')

figure;
surf(vxbin(1:end-1)./vTpar,vybin(1:end-1)./vTper,fvxvy{jj}','LineStyle','none');
title('Vx-Vy')
xlabel('Vx')
ylabel('Vy')

% figure;
% surf(xbin(1:end-1),KEbin(1:end-1),fkex{jj}','LineStyle','none');
% title('x-KE')
% xlabel('x')
% ylabel('KE')

% Plasma pressure:
% =========================================================================
figure 
hold on
hP(1) = plot(xbin(1:end-1),Px(:,jj),'k')
hP(2) = plot(xbin(1:end-1),Py(:,jj),'r')
hP(3) = plot(xbin(1:end-1),Pz(:,jj),'g')
hP(4) = plot(xbin(1:end-1),Px(:,8),'k--')
hP(5) = plot(xbin(1:end-1),Py(:,8),'r--')
hP(6) = plot(xbin(1:end-1),Pz(:,8),'g--')

grid on
legend(hP(1:3),'Px','Py','Pz')
title('Ion Pressure')
ylim([0,max(Px(:,jj))*1.2])

% Ion Temperatures:
% =========================================================================
figure 
hold on
hT(1) = plot(xbin(1:end-1),Tx(:,jj),'k')
hT(2) = plot(xbin(1:end-1),Ty(:,jj),'r')
hT(3) = plot(xbin(1:end-1),Tz(:,jj),'g')
legend(hT,'Tx','Ty','Tz')
title('Ion Temperature')
ylim([0,1.2e3])
xlabel('x(in meters)');
ylabel('Temperature in KeV')
grid on

% Plasma density:
% =========================================================================
figure 
hold on
plot(xbin(1:end-1),n(:,jj));hold on;
 plot(ss(1:length(Ne)),Ne(:,jj));
title('Plasma density')
xlabel('x(in meters)');
ylabel('density')
grid on

% Ion drift velocities:
% =========================================================================
figure 
hold on
frame=30;
plot(ss(1:length(Ux(:,jj))),movmean(Ux(:,jj)/(vTpar),frame),'k')
plot(ss(1:length(Uy(:,jj))),movmean(Uy(:,jj)/(vTpar),frame),'b')
plot(ss(1:length(Uz(:,jj))),movmean(Uz(:,jj)/(vTpar),frame),'r')
title('Drift velocities')
ylim([-1,1]*max(Ux(:,jj)/vTpar)*1.2)
grid on

% Ion Hall electric fields:
% =========================================================================
frame=30;
Ex_Hall =movmean( -( Uy(:,jj).*Bz(:,jj) - Uz(:,jj).*By(:,jj)),frame);
Ey_Hall =movmean( -(-Ux(:,jj).*Bz(:,jj) + Uz(:,jj).*Bx(:,jj)),frame);
Ez_Hall =movmean( -( Ux(:,jj).*By(:,jj) - Uy(:,jj).*Bx(:,jj)),frame);

figure 
hold on
plot(ss(1:length(Uy(:,jj))),Ex_Hall,'b')
plot(ss(1:length(Ux(:,jj))),Ez_Hall,'r')
plot(ss(1:length(Uz(:,jj))),Ey_Hall,'k')
title('Hall terms')
grid on

% Profiles:
% =========================================================================
figure 
% Bx:
subplot(3,1,1)
plot(ss(1:length(Bx(:,jj))),Bx(:,jj))
title('Bx')
xlabel('x(in meters)');
ylabel('B-field in Tesla')
grid on

% Br and Bz:
subplot(3,1,2)
hold on
hS(1)=plot(ss(1:length(By(:,jj))),By(:,jj),'k')
hS(2)=plot(ss(1:length(By(:,1))),By(:,1),'m.')
hS(3)=plot(ss(1:length(Bz(:,jj))),Bz(:,jj),'b')
hS(4)=plot(ss(1:length(Bz(:,1))),Bz(:,1),'g')
legend(hS,'By','By at t=0','Bz','Bz at t=0')
title('Bx,By')
xlabel('x(in meters)');
ylabel('B-field in Tesla')
% ylim([0,1.5]);
grid on

% Electric field components:
subplot(3,1,3)
frame=30;
plot(ss(1:length(Ex(:,jj))),movmean(Ex(:,jj),frame),'b');hold on;
plot(ss(1:length(Ey(:,jj))),movmean(Ey(:,jj),frame),'k');
plot(ss(1:length(Ez(:,jj))),movmean(Ez(:,jj),frame),'r');
% title('$E_\parallel$','interpreter','latex','fontSize',fontSize.title);
xlabel('x(in meters)');
ylabel('E-field in V/m')
% ylim([0,1.5]);
grid on

%% Comparing the fluid's kinetic energy with plasma pressure:
close all

% x increment:
dx = xVec(2) - xVec(1);

% Magnetic field interpolated at positions of the histograms:
BBx = interp1(ss,Bx,xVec); 

% Calculate the compression factor to convert quantites to physical space:
B0 = BBx(round(length(xVec)/2));
compFactor = BBx/B0;

%Parallel drift interpolated on querry points
UUUx = interp1(ss,Ux,xVec);

% Pressures:
% =========================================================================
% Calculating the plasma's kinetic energy density:
% P_KE   = M*n.*UUUx.^2;
P_KE   = (M*n.*Vx.^2);

% Total parallel pressure due to electrons and ions:
P_PAR = (e_c*n*m.Te*(k_B/e_c) + Px);

% Total perpendicular pressure due to electrons and ions:
P_PER = (e_c*n*m.Te*(k_B/e_c) + Py);

% Add both pressures:
P_SUM = P_KE + P_PAR;

% Force densities:
% =========================================================================
% Kinetic energy force density:
frame = 50;
bx = BBx(1:end-1,:);  
switch modelType
    case 1 % No magnetic compression
        F_KE = movmean(diff(P_KE),frame)/dx;
    case 2 % With magnetic compression
        F_KE = movmean(bx.*diff(P_KE./BBx),frame)/dx;
end

% Parallel pressure gradient force:
F_PAR =  movmean(diff(P_PAR),frame)/dx;

% Mirror force:
pper  = P_PER(1:end-1,:);
ppar  = P_PAR(1:end-1,:);
F_MIR = movmean(((pper- ppar)./bx).*diff(BBx)/dx,frame);

% Plot Force densities:
% =========================================================================
figure('color','w')
hold on
rng = 25;
fF(1) = plot(xVec(1:end-1),F_KE(:,rng),'k','LineWidth',2);
fF(2) = plot(xVec(1:end-1),F_PAR(:,rng),'r','LineWidth',2);
fF(3) = plot(xVec(1:end-1),F_MIR(:,rng),'g','LineWidth',2);
plot(xVec(1:end-1),F_KE(:,rng)+F_PAR(:,rng)+F_MIR(:,rng),'c','LineWidth',2);
title('Force density terms','interpreter','latex','fontSize',13)

switch modelType
    case 1
        legendText{1} = '$\partial(P_{KE})/\partial x$';
    case 2
        legendText{1} = '$B_x\frac{\partial}{\partial{x}}(\frac{P_{KE}}{B_x})$';
end
legendText{2} = '$\frac{\partial}{\partial{x}}P_{\parallel}$';
legendText{3} = '$(\frac{P_{\perp} - P_{\parallel}}{B_x})\frac{\partial}{\partial{x}}(B_x)$';

hLeg = legend(fF,legendText);
set(hLeg,'interpreter','Latex','fontSize',12,'Location','best')
box on
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Nm$^{-3}$]','interpreter','latex','fontSize',13)
set(gca,'fontName','Times','FontSize',12)

% Plot Force densities other presentation:
% =========================================================================
figure('color','w')
hold on
rng = 25;
fFb(1) = plot(xVec(1:end-1),+F_KE(:,rng)+F_PAR(:,rng),'k','LineWidth',2);
fFb(2) = plot(xVec(1:end-1),-F_MIR(:,rng),'g','LineWidth',2);
title('Force density terms','interpreter','latex','fontSize',13)

switch modelType
    case 1
        legendText{1} = '$\partial(P_{KE})/\partial x + \frac{\partial}{\partial{x}}P_{\parallel}$';
    case 2
        legendText{1} = '$B_x\frac{\partial}{\partial{x}}(\frac{P_{KE}}{B_x}) + \frac{\partial}{\partial{x}}P_{\parallel}$';
end
legendText{2} = '$-(\frac{P_{\perp} - P_{\parallel}}{B_x})\frac{\partial}{\partial{x}}(B_x)$';

hLegb = legend(fFb,legendText);
set(hLegb,'interpreter','Latex','fontSize',12,'Location','best')
box on
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Nm$^{-3}$]','interpreter','latex','fontSize',13)
set(gca,'fontName','Times','FontSize',12)

% Plot energy densities:
% =========================================================================
figure('color','w')
hold on
rng = 25;
fP(1) = plot(xVec,P_KE(:,rng),'k','LineWidth',2);
fP(2) = plot(xVec,P_PAR(:,rng),'r','LineWidth',2);
fP(3) = plot(xVec,P_PER(:,rng),'g','LineWidth',2);
plot(xVec,P_PER(:,rng)-P_PAR(:,rng),'c','LineWidth',2);
title('Energy density terms','interpreter','latex','fontSize',13)
legendText{1} = '$P_{KE}$';
legendText{2} = '$P_{\parallel}$';
legendText{3} = '$P_{\perp}$';
hLeg = legend(fP,legendText);
set(hLeg,'interpreter','Latex','fontSize',12,'Location','best')
box on
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Jm$^{-3}$]','interpreter','latex','fontSize',13)
set(gca,'fontName','Times','FontSize',12)

return

% Plot Energy densities:
% -------------------------------------------------------------------------
figure('color','w')
hold on
rng = 25;
fP(1) = plot(xVec,P_KE(:,rng),'k','LineWidth',2);
fP(2) = plot(xVec,P_PAR(:,rng),'r','LineWidth',2);
fP(3) = plot(xVec,P_SUM(:,rng),'g','LineWidth',2);
fP(4) = plot(xVec,P_PER(:,rng),'m','LineWidth',2);
plot(xVec,P_PER(:,rng)-P_PAR(:,rng),'c','LineWidth',2)
title('Pressure terms','interpreter','latex','fontSize',13)
legendText{1} = '$Mnu_{\parallel}^2$';
legendText{2} = '$P_{\parallel}$';
legendText{3} = '$Mnu_{\parallel}^2+P_{\parallel}$';
legendText{4} = '$P_{\perp}$';
hLeg = legend(fP,legendText);
set(hLeg,'interpreter','Latex','fontSize',12,'Location','best')
box on
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Jm$^{-3}$]','interpreter','latex','fontSize',13)
set(gca,'fontName','Times','FontSize',12)

return

% Plot mirror force:
figure('color','w')
hold on
frame = 71;
hm(1)=plot(ss(1:end-1),F_Mirror_a(:,rng));
hm(2)=plot(ss(1:end-1),movmean(F_Mirror_b(:,rng),frame),'r');
title('Mirror force','interpreter','latex','fontSize',13);
legendText{1} = '$a$';
legendText{2} = '$b$';

hLeg = legend(hm,legendText(1:2));
set(hLeg,'interpreter','Latex','fontSize',12,'Location','best')
box on
box on

% Plot kinetic force vs pressure forces:
% -------------------------------------------------------------------------
try 
    clear legendText
end
figure('color','w')
hold on
hFs(1) = plot(ss(1:end-1),movmean(F_KE(:,rng),frame),'r','LineWidth',2);
hFs(2) = plot(ss(1:end-1),movmean(-F_PPar(:,rng),frame) + F_Mirror_a(:,rng),'g','LineWidth',2);
title('Force balance','interpreter','latex','fontSize',13)
legendText{1} = '$\frac{\partial(Mnu_{\parallel}^2)}{\partial{s}}$';
legendText{2} = '$-\frac{\partial{P_{\parallel}}}{\partial{s}} + (\frac{P_{\parallel} - P_{\perp}}{B})\frac{\partial B}{\partial{s}}$';
hLeg = legend(hFs,legendText);
set(hLeg,'interpreter','Latex','fontSize',16,'Location','best')
box on
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Nm$^{-3}$]','interpreter','latex','fontSize',13)
set(gca,'fontName','Times','FontSize',12)


% Plot individual forces:
% -------------------------------------------------------------------------
figure('color','w')
hold on
hFa(1) = plot(ss(1:end-1),movmean(F_KE(:,rng),frame),'k','LineWidth',2);
hFa(2) = plot(ss(1:end-1),movmean(F_PPar(:,rng),frame) ,'r','LineWidth',2);
hFa(3) = plot(ss(1:end-1),-F_Mirror_a(:,rng),'g','LineWidth',2);

legendText{1} = '$\frac{\partial(Mnu_{\parallel}^2)}{\partial{s}}$';
legendText{2} = '$-\frac{\partial{P_{\parallel}}}{\partial{s}}$';
legendText{3} = '$(\frac{P_{\parallel} - P_{\perp}}{B})\frac{B}{\partial{s}}$';

legF = legend(hFa,legendText);
set(legF,'interpreter','Latex','fontSize',16,'Location','best')
box on
set(gca,'fontName','Times','FontSize',12)
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Nm$^{-3}$]','interpreter','latex','fontSize',13)


% ELectric field
% -------------------------------------------------------------------------
figure('color','w')
plot(ss,movmean(Ex(:,rng),31),'k','LineWidth',2)
box on
set(gca,'fontName','Times','FontSize',12)
xlim([0,max(xVec)])
xlabel('x [m]','interpreter','latex','fontSize',13)
ylabel('[Vm$^{-1}$]','interpreter','latex','fontSize',13)
title('$E_{\parallel}$','interpreter','latex','fontSize',16)

% Plot ionization source at steady-state time:
G = diff(n.*Vx)/dx;
figure('color','w')

frame = 25;
plot(xVec(1:end-1),movmean(G(:,rng),frame))
box on

%% Functions:

function data=HDF2Struct_v2(f,verbose)
%HDF2STRUCT - Reads HDF5 files into structure
%
% Syntax:  data = HDF2Struct(file_name, verbose)
%
% Inputs:
%    file_name - String. Path to the hdf5 file
%    verbose   - Boolean. Whether or not to print warnings when renaming
%    variables with invalid matlab names
%
% Outputs:
%    data - Matlab structure containing the read data
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
% Author: Luca Amerio
% email: lucamerio89@gmail.com
% March 2019; Last revision: 22-March-2019
%------------- BEGIN CODE --------------
if nargin<2
    verbose=false;
end
data=struct;
loadcontent('/');
    function loadcontent(pathStr)
        %Gets info of current group (or root)
        info=h5info(f,pathStr);
        
        %Loading variables
        for vari=1:length(info.Datasets)
            var=info.Datasets(vari).Name;
            fields  = strsplit(pathStr,'/');   % Ignore initial blank field later
            fields(cellfun(@isempty,fields))=[];
            
            % Validate variable name:
            varName=validateFieldName(var);
            
            % Validate field name:
            fieldsName=validateFieldName(fields);
            
            % Assign data to "data" structure:
            data = setfield(data,fieldsName{:},varName{:},h5read(f,[pathStr,'/',var]));
        end
        
        %Loading attributes
        for atti=1:length(info.Attributes)
            att=info.Attributes(atti).Name;
            fields  = strsplit(pathStr,'/');   % Ignore initial blank field later
            fields(cellfun(@isempty,fields))=[];
            attName=validateFieldName(att);
            fieldsName=validateFieldName(fields);
            data = setfield(data,fieldsName{:},attName{:},h5readatt(f,pathStr,att));
        end
        
        %Loading groups (calls loadcontent recursively for the selected
        %group)
        for gi=1:length(info.Groups)
            loadcontent(info.Groups(gi).Name);
        end
        
        % HDF naming convention allows names unsupported by matlab. This 
        % funtion tryies to clean them when possible.
        function name=validateFieldName(name)
            if ischar(name)
                name={name};
            elseif iscellstr(name)
            else
                error('Input must be either a string or a cell array of strings')
            end
            
            check=~cellfun(@isvarname,name);
            
            if any(check)
                if verbose
                    warning('"%s" is not a valid field name\n',name{check})
                end
                for i=find(check)
                    if any(name{i}==' ')
                        name_new=strrep(name{i},' ','');
                        if verbose
                            warning('"%s" is not a valid field name\nchanging "%s" to "%s"\n',name{i},name{i},name_new)
                        end
                        name{i}=name_new;
                    elseif isnumeric(str2num(name{i}))
                        name{i} = ['t',name{i}];
                    else
                        error('"%s" is not a valid field name\n',name{i})
                    end
                end
            end
        end
    end
end