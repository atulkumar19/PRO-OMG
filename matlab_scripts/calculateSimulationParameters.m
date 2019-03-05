function ST = calculateSimulationParameters(Bo,betae,betai,Zi,mi_amu,ne)
% Script to calculate simulation parameters of PRO++
% NOTE: All quantities are in SI units.
%
% Bo: Magnetic field (Teslas)
% betae: Electron plasma beta
% betai: Ion plasma beta
% Zi: Ion atomic number
% mi_amu: Ion mass in atomic unit masses
% ne: Electron density (m^-3)
% Example: P = calculateSimulationParameters(1.37E-7,1.0,1.0E-2,1.0,1.007,1E16)

% Physical constants
kB = 1.38E-23; % Boltzmann constant
mu0 = (4E-7)*pi; % Magnetic permeability of vacuum
ep0 = 8.854E-12; % Electric permittivity of vacuum
c=2.9979E8; % Speed of light
amu = 1.660539E-27; % Atomic mass unit in kg

qe = 1.602176E-19; % Electron charge
me = 9.109383E-31; % Electron mass

% Ions and plasma parameters
mi = mi_amu*amu; % Ion mass
qi = Zi*qe; % Ion charge
ni = ne/Zi; % Ion density (quasi-neutrality condition)

Te = betae*Bo^2/(2.0*mu0*ne); % Electron temperature (Joules)
Ti = betai*Bo^2/(2.0*mu0*ne); % Ion temperature (Joules)

% Fundamental frequencies
wci = qi*Bo/mi; % Ion cyclotron frequency
wce = qe*Bo/me; % Electron cyclotron frequency
wpe = sqrt( (ne*qe^2)/(ep0*me) ); % Electron plasma frequency
wpi = sqrt( (ni*qi^2)/(ep0*mi) ); % Ion plasma frequency
wlh = sqrt( (wce*wci*(wci^2 + wpi^2))/(wpi^2 + wci*wce) ); % Lower hibrid frequency

VA = Bo/sqrt(mu0*ni*mi); % Alfven speed

lD = sqrt( ep0*kB*Te/(ni*qi^2) ); % Debye length

VTe = sqrt(2.0*Te/me); % Electron thermal speed
VTi = sqrt(2.0*Ti/mi); % Ion thermal speed

% Lenght scales
rLe = VTe/wce; % Ion's Larmor radius
dpe = c/wpe; % Ions skin depth

rLi = VTi/wci; % Ion's Larmor radius
dpi = c/wpi; % Ions skin depth

% Assigning values to resulting structure
ST.qe = qe;
ST.mu0 = mu0;
ST.ep0 = ep0;
ST.me = me;
ST.mi = mi;
ST.ne = ne;
ST.ni = ni;
ST.VTe = VTe;
ST.VTi = VTi;
ST.wci = wci;
ST.wce = wce;
ST.Te = Te/qe;
ST.Ti = Ti/qe;
ST.dpi = dpi;
ST.rLi = rLi;

disp(['Electron temperature:' num2str(ST.Te) ' eV'])
disp(['Ion temperature:' num2str(ST.Ti) ' eV'])
disp(['Ion skin depth:' num2str(ST.dpi) ' m'])
disp(['Ions Larmor radius:' num2str(ST.rLi) ' m'])
end