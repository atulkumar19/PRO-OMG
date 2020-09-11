function ST = postprocessSimulation(path)
% Example: ST = postprocessSimulation('../PROMETHEUS++/outputFiles/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/test2/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/dispersion_relation/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/warm_plasma/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/GC/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/Tests/warm_plasma/HDF5/')

close all

% Physical constants
ST.kB = 1.38E-23; % Boltzmann constant
ST.mu0 = (4E-7)*pi; % Magnetic permeability of vacuum
ST.ep0 = 8.854E-12; % Electric permittivity of vacuum
ST.c = 2.9979E8; % Speed of light
ST.amu = 1.660539E-27; % Atomic mass unit in kg

ST.qe = 1.602176E-19; % Electron charge
ST.me = 9.109383E-31; % Electron mass

ST.path = path;

ST.params = loadSimulationParameters(ST);

ST = loadData(ST);

ST.time = loadTimeVector(ST);

EnergyConservation(ST);

if (ST.params.dimensionality == 1)
    FourierAnalysis1D(ST,'B','z');
else
    FourierAnalysis2D(ST,'B','z');
end
end


function params = loadSimulationParameters(ST)
params = struct;
info = h5info([ST.path 'main.h5']);

params.info = info;

if ~isempty(info.Datasets)
    for ii=1:numel(info.Datasets)
        params.(info.Datasets(ii).Name) = h5read(info.Filename, ['/' info.Datasets(ii).Name]);
    end
end

for ii=1:numel(info.Groups)
    groupName = strsplit(info.Groups(ii).Name,'/');
    
    for jj=1:numel(info.Groups(ii).Datasets)
        datasetName = info.Groups(ii).Datasets(jj).Name;
        params.(groupName{end}).(datasetName) = h5read(info.Filename, ['/' groupName{end} '/' datasetName]);
    end
    
    if ~isempty(info.Groups(ii).Groups)
        for jj=1:numel(info.Groups(ii).Groups)
            subGroupName = strsplit(info.Groups(ii).Groups(jj).Name,'/');
            for kk=1:numel(info.Groups(ii).Groups(jj).Datasets)
                datasetName = info.Groups(ii).Groups(jj).Datasets(kk).Name;
                params.(groupName{end}).(subGroupName{end}).(datasetName) = ...
                    h5read(info.Filename, [info.Groups(ii).Groups(jj).Name '/' datasetName]);
            end
        end
    end
end
end

function ST = loadData(ST)
ST.fields_data = struct;
ST.ions_data = struct;


% First, we load ions data
numberOfOutputs_ions = [];

for ff=1:ST.params.numMPIsParticles
    info = h5info([ST.path ['PARTICLES_FILE_' num2str(ff-1) '.h5']]);
    
    numberOfOutputs_ions(ff)= numel(info.Groups);
%     numberOfOutputs_ions(ff)= 5000; 
    
    for ii=1:numel(info.Groups)
        groupName = strsplit(info.Groups(ii).Name,'/');
        
        for jj=1:numel(info.Groups(ii).Datasets)
            datasetName = info.Groups(ii).Datasets(jj).Name;
            ST.ions_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(datasetName) = ...
                h5read(info.Filename, ['/' groupName{end} '/' datasetName]);
        end
        
        if ~isempty(info.Groups(ii).Groups)
            for jj=1:numel(info.Groups(ii).Groups)
                subGroupName = strsplit(info.Groups(ii).Groups(jj).Name,'/');
                
                for kk=1:numel(info.Groups(ii).Groups(jj).Datasets)
                    datasetName = info.Groups(ii).Groups(jj).Datasets(kk).Name;
                    ST.ions_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(datasetName) = ...
                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Name '/' datasetName]);
                end
                
                if ~isempty(info.Groups(ii).Groups(jj).Groups)
                    for kk=1:numel(info.Groups(ii).Groups(jj).Groups)
                        subSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Name,'/');
                        
                        for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Datasets)
                            datasetName = info.Groups(ii).Groups(jj).Groups(kk).Datasets(ll).Name;
                            
                            ST.ions_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(datasetName) = ...
                                h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Name '/' datasetName]);
                        end
                        
                        if ~isempty(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                            for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                                subSubSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name,'/');
                                
                                for mm=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets)
                                    datasetName = info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets(mm).Name;
                                    
                                    ST.ions_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(subSubSubGroupName{end}).(datasetName) = ...
                                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name '/' datasetName]);
                                end
                            end
                        end
                    end
                end
                
            end
        end
        
    end
end

% Then, we load fields data
numberOfOutputs_fields = [];

for ff=1:ST.params.numMPIsFields
    info = h5info([ST.path ['FIELDS_FILE_' num2str(ff-1) '.h5']]);
    
    numberOfOutputs_fields(ff)= numel(info.Groups);
%     numberOfOutputs_fields(ff)= 5000;%
    
    for ii=1:numel(info.Groups)
        groupName = strsplit(info.Groups(ii).Name,'/');
        
        for jj=1:numel(info.Groups(ii).Datasets)
            datasetName = info.Groups(ii).Datasets(jj).Name;
            ST.fields_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(datasetName) = ...
                h5read(info.Filename, ['/' groupName{end} '/' datasetName]);
        end
        
        if ~isempty(info.Groups(ii).Groups)
            for jj=1:numel(info.Groups(ii).Groups)
                subGroupName = strsplit(info.Groups(ii).Groups(jj).Name,'/');
                
                for kk=1:numel(info.Groups(ii).Groups(jj).Datasets)
                    datasetName = info.Groups(ii).Groups(jj).Datasets(kk).Name;
                    ST.fields_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(datasetName) = ...
                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Name '/' datasetName]);
                end
                
                if ~isempty(info.Groups(ii).Groups(jj).Groups)
                    for kk=1:numel(info.Groups(ii).Groups(jj).Groups)
                        subSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Name,'/');
                        
                        for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Datasets)
                            datasetName = info.Groups(ii).Groups(jj).Groups(kk).Datasets(ll).Name;
                            
                            ST.fields_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(datasetName) = ...
                                h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Name '/' datasetName]);
                        end
                        
                        if ~isempty(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                            for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                                subSubSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name,'/');
                                
                                for mm=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets)
                                    datasetName = info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets(mm).Name;
                                    
                                    ST.fields_data.(['MPI' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(subSubSubGroupName{end}).(datasetName) = ...
                                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name '/' datasetName]);
                                end
                            end
                        end
                    end
                end
                
            end
        end
        
    end
end

ST.numberOfOutputs = min([numberOfOutputs_ions numberOfOutputs_fields]);
end

function time = loadTimeVector(ST)
time = zeros(1,ST.numberOfOutputs);

for ii=1:ST.numberOfOutputs
    time(ii) = ST.ions_data.(['MPI0_O' num2str(ii-1)]).time;
end

disp(["Simulation time analysed: " + num2str(time(end)/ST.params.scales.ionGyroPeriod(1)) ])
end


function FourierAnalysis1D(ST, field, component)
if strcmp(component,'x')
    component_num = 1;
elseif strcmp(component,'y')
    component_num = 2;
elseif strcmp(component,'z')
    component_num = 3;
end


NMPI_P = ST.params.numMPIsParticles;
NMPI_F = ST.params.numMPIsFields;
NX_PER_MPI = ST.params.geometry.NX; % Number of cells per domain
NX_IN_SIM = ST.params.geometry.NX_IN_SIM; % Number of cells in the whole domain

% Plasma parameters
qi = ST.params.ions.spp_1.Q;
mi = ST.params.ions.spp_1.M;
Bo = sqrt(dot(ST.params.Bo,ST.params.Bo));

ne = ST.params.ions.ne;
ni = ST.params.ions.ne;
wpi = sqrt(ni*((qi)^2)/(mi*ST.ep0));

wci = qi*Bo/mi; % Ion cyclotron frequency
wce = ST.qe*Bo/ST.me; % Electron cyclotron frequency

wci = double(wci);
wce = double(wce);
wpi = double(wpi);

tauci = 2*pi/wci;

% Alfven speed
VA = Bo/sqrt( ST.mu0*ne*mi );
VA = double(VA);

% Lower hybrid frequency
wlh = sqrt( wpi^2*wci*wce/( wci*wce + wpi^2 ) );
wlh = wlh/wci;

disp(['Lower hybrid frequency: ' num2str(wlh)]);

% % % % % % Fourier variables % % % % % % 
time = ST.time;
DT = mean(diff(time));

% Dimensionless time in units of ion cyclotron period
time = time/tauci;

% *** @tomodify
NT = int32(ST.numberOfOutputs); % Number of snapshots
% NT = find((time-2.5)>0, 1);

Df = 1.0/(DT*double(NT));
fmax = 1.0/(2.0*double(DT)); % Nyquist theorem
fAxis = 0:Df:fmax-Df;

wAxis = 2*pi*fAxis/wci;


DX = ST.params.geometry.DX;
Dk = 1.0/(DX*double(NX_IN_SIM));
kmax = 1.0/(2.0*DX);
kAxis = 0:Dk:kmax-Dk;
kAxis = 2.0*pi*kAxis;
xAxis = ST.c*kAxis/wpi;
% % % % % % Fourier variables % % % % % % 

% Whistlers dispersion relation
k_wh = sqrt( (1 + (VA/ST.c)^2)*( wAxis.^2./(1 + wAxis) ) );

% Ion cyclotron wave dispersion relation

k_ic = @(w) sqrt( (wci/wpi)^2*(w.^2.*(1.0 + (wpi/wci)^2 - w)./(1 - w)) );

F = zeros(NT,NX_IN_SIM);

for ii=1:NT
    for dd=1:NMPI_F
        ix = (dd-1)*NX_PER_MPI + 1;
        fx = dd*NX_PER_MPI;
            
        if strcmp(field,'B')
            F(ii,ix:fx) = ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component)  - ST.params.Bo(component_num);
        else
            F(ii,ix:fx) = ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component);
        end
    end
end


% NK = floor( numel(xAxis)/2 );

% one-to-one comparative with 2-D simulations
NK = numel(xAxis);
NW = numel(wAxis);

% x-axis
kSpacex = zeros(NT, NK);
for ii=1:NT
    A = fft( squeeze(F(ii,:)) );
    kSpacex(ii,:) = A(1:NK);
end


fourierSpacex = zeros(NW, NK);
for ii=1:NK
    A = fft(hanning(double(NT)).* kSpacex(:,ii) );
    fourierSpacex(:,ii) = A(1:NW);
end

FX = squeeze( fourierSpacex.*conj(fourierSpacex) );
Spectrum_x = sum(FX,2)/NK;

Spectrum_x = Spectrum_x/double(NX_IN_SIM);


wk_fig = figure;


% Propagation along x-axis
A = FX;
% Magnetoacoustic wave
z = linspace(0,max([max(xAxis), max(wAxis)]),10);

figure(wk_fig)
subplot(1,4,1)
ax = plot(log10(Spectrum_x),wAxis,'k');
%     set(ax.Parent(1),'XDir','reverse')
box on; grid on; grid minor
xlabel('Intensity ($log_{10}$)', 'Interpreter', 'latex')
ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')

figure(wk_fig)
subplot(1,4,[2 4])
imagesc(xAxis,wAxis,log10(A(1:numel(wAxis),1:numel(xAxis))));
hold on;plot(xAxis, wlh*ones(size(xAxis)),'k--',z,z,'k--');hold off;
hold on;plot(k_wh, wAxis,'k--');hold off;
w_ic = linspace(0,1,100);
hold on;plot(k_ic(w_ic), w_ic,'k--', xAxis, ones(size(xAxis)), 'k--');hold off;
axis xy; colormap(jet); colorbar
try
    axis([0 max(xAxis) 0 max(wAxis)])
end
xlabel('$ck_x/\omega_p$', 'Interpreter', 'latex')
ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')
title(['$' field '_' component '(x)$'],'interpreter','latex')
end


function FourierAnalysis2D(ST, field, component)
if strcmp(component,'x')
    component_num = 1;
elseif strcmp(component,'y')
    component_num = 2;
elseif strcmp(component,'z')
    component_num = 3;
end


NMPI_P = ST.params.numMPIsParticles;
NMPI_F = ST.params.numMPIsFields;
NX_PER_MPI = ST.params.geometry.NX; % Number of cells per domain
NX_IN_SIM = ST.params.geometry.NX_IN_SIM; % Number of cells in the whole domain
NY_PER_MPI = ST.params.geometry.NY; % Number of cells per domain
NY_IN_SIM = ST.params.geometry.NY_IN_SIM; % Number of cells in the whole domain
% SPLIT_DIRECTION = ST.params.geometry.SPLIT_DIRECTION;

% Plasma parameters
qi = ST.params.ions.spp_1.Q;
mi = ST.params.ions.spp_1.M;
Bo = sqrt(dot(ST.params.Bo,ST.params.Bo));

ne = ST.params.ions.ne;
ni = ST.params.ions.ne;
wpi = sqrt(ni*((qi)^2)/(mi*ST.ep0));

wci = qi*Bo/mi; % Ion cyclotron frequency
wce = ST.qe*Bo/ST.me; % Electron cyclotron frequency

wci = double(wci);
wce = double(wce);
wpi = double(wpi);

tauci = 2*pi/wci;

% Alfven speed
VA = Bo/sqrt( ST.mu0*ne*mi );
VA = double(VA);

% Lower hybrid frequency
wlh = sqrt( wpi^2*wci*wce/( wci*wce + wpi^2 ) );
wlh = wlh/wci;

disp(['Lower hybrid frequency: ' num2str(wlh)]);

% % % % % % Fourier variables % % % % % % 
time = ST.time;
DT = mean(diff(time));

time = time/tauci;

% *** @tomodify
NT = int32(ST.numberOfOutputs); % Number of snapshots
% NT = find((time-2.5)>0, 1);

Df = 1.0/(DT*double(NT));
fmax = 1.0/(2.0*double(DT)); % Nyquist theorem
fAxis = 0:Df:fmax-Df;

wAxis = 2*pi*fAxis/wci;


DX = ST.params.geometry.DX;
Dk = 1.0/(DX*double(NX_IN_SIM));
kmax = 1.0/(2.0*DX);
kAxis = 0:Dk:kmax-Dk;
kAxis = 2.0*pi*kAxis;
xAxis = ST.c*kAxis/wpi;

DY = ST.params.geometry.DY;
Dk = 1.0/(DY*double(NY_IN_SIM));
kmax = 1.0/(2.0*DY);
kAxis = 0:Dk:kmax-Dk;
kAxis = 2.0*pi*kAxis;
yAxis = ST.c*kAxis/wpi;
% % % % % % Fourier variables % % % % % % 

% Whistlers dispersion relation
k_wh = sqrt( (1 + (VA/ST.c)^2)*( wAxis.^2./(1 + wAxis) ) );

% Ion cyclotron wave dispersion relation

k_ic = @(w) sqrt( (wci/wpi)^2*(w.^2.*(1.0 + (wpi/wci)^2 - w)./(1 - w)) );

F = zeros(NT,NX_IN_SIM,NY_IN_SIM);

for ii=1:NT
    for dd=1:NMPI_F
        if (ST.params.geometry.SPLIT_DIRECTION == 0)
            ix = (dd-1)*NX_PER_MPI + 1;
            fx = dd*NX_PER_MPI;
            iy = 1;
            fy = NY_PER_MPI;
        else
            ix = 1;
            fx = NX_PER_MPI;
            iy = (dd-1)*NY_PER_MPI + 1;
            fy = dd*NY_PER_MPI;
        end
        
        if strcmp(field,'B')
            F(ii,ix:fx,iy:fy) = ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component)  - ST.params.Bo(component_num);
        else
            F(ii,ix:fx,iy:fy) = ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component);
        end
    end
end


% 2D FFT
if floor(NX_IN_SIM/2) < numel(xAxis)
    NKx = floor(NX_IN_SIM/2);
else
    NKx = numel(xAxis);
end

if floor(NY_IN_SIM/2) < numel(yAxis)
    NKy = floor(NY_IN_SIM/2);
else
    NKy = numel(yAxis);
end

if floor(NT/2) < numel(wAxis)
    NW = floor(NT/2);
else
    NW = numel(wAxis);
end

kSpace = zeros(NT,NKx,NKy);
for ii=1:NT
    A = fft2(squeeze(F(ii,:,:)));
    kSpace(ii,:,:) = A(1:NKx,1:NKy);
end


A = fft(kSpace,NT,1);
fourierSpace = zeros(NW,NKx,NKy);
fourierSpace = A(1:NW,:,:);

% NKxi = floor(NKx/2);
% NKyi = floor(NKy/2);

% one-to-one comparison with 1-D simulations
NKxi = NKx;
NKyi = NKy;

fig = figure;


% Magnetoacoustic wave
z = linspace(0,max([max(xAxis), max(wAxis)]),10);

% w-kx space (ky=0)
A = squeeze(fourierSpace(:,:,1));
A = log10(A.*conj(A));

figure(fig)
subplot(3,6,[2 3])
imagesc(xAxis(1:NKx), wAxis(1:NW), A)
hold on;plot(xAxis(1:NKx), wlh*ones(size(xAxis(1:NKx))),'k--',z,z,'k--');hold off;
hold on;plot(k_wh, wAxis,'k--');hold off;
w_ic = linspace(0,1,100);
hold on;plot(k_ic(w_ic), w_ic,'k--', xAxis(1:NKx), ones(size(xAxis(1:NKx))), 'k--');hold off;
xlabel('$K_x$', 'Interpreter', 'latex')
ylabel('$\omega$', 'Interpreter', 'latex')
axis xy; box on; grid on;
colormap(jet)

A = squeeze(fourierSpace(:,1:NKxi,1));
A = A.*conj(A);
A = log10(sum(A,2)/NKxi);

figure(fig)
subplot(3,6,1)
plot(A, wAxis(1:NW))
ylabel('$\omega$', 'Interpreter', 'latex')
box on; grid on;

% w-kx space (ky integrated)
A = fourierSpace(:,:,1:1:NKyi);
A = log10( sum(A.*conj(A),3)/NKyi );

figure(fig)
subplot(3,6,[5 6])
imagesc(xAxis(1:NKx), wAxis(1:NW), A)
hold on;plot(xAxis(1:NKx), wlh*ones(size(xAxis(1:NKx))),'k--',z,z,'k--');hold off;
hold on;plot(k_wh, wAxis,'k--');hold off;
w_ic = linspace(0,1,100);
hold on;plot(k_ic(w_ic), w_ic,'k--', xAxis(1:NKx), ones(size(xAxis(1:NKx))), 'k--');hold off;
xlabel('$K_x$', 'Interpreter', 'latex')
ylabel('$\omega$', 'Interpreter', 'latex')
axis xy; box on; grid on;
colormap(jet)

A = fourierSpace(:,1:NKxi,1:1:NKyi);
A = A.*conj(A);
A = sum(A,3)/NKyi;
A = sum(A,2)/NKxi;
A = log10(A);

figure(fig)
subplot(3,6,4)
plot(A, wAxis(1:NW))
ylabel('$\omega$', 'Interpreter', 'latex')
box on; grid on;


% Magnetoacoustic wave
z = linspace(0,max([max(yAxis), max(yAxis)]),10);

% w-ky space (kx=0)
A = squeeze(fourierSpace(:,1,:));
A = log10(A.*conj(A));

figure(fig)
subplot(3,6,[8 9])
imagesc(yAxis(1:NKy), wAxis(1:NW), A)
hold on;plot(yAxis(1:NKy), wlh*ones(size(yAxis(1:NKy))),'k--',z,z,'k--');hold off;
hold on;plot(k_wh, wAxis,'k--');hold off;
w_ic = linspace(0,1,100);
hold on;plot(k_ic(w_ic), w_ic,'k--', yAxis(1:NKy), ones(size(yAxis(1:NKy))), 'k--');hold off;
xlabel('$K_y$', 'Interpreter', 'latex')
ylabel('$\omega$', 'Interpreter', 'latex')
axis xy; box on; grid on;
colormap(jet)


A = squeeze(fourierSpace(:,1,1:1:NKyi));
A = A.*conj(A);
A = log10(sum(A,2)/NKxi);

figure(fig)
subplot(3,6,7)
plot(A, wAxis(1:NW))
ylabel('$\omega$', 'Interpreter', 'latex')
box on; grid on;

% w-ky space (kx integrated)
A = fourierSpace(:,1:1:NKxi,:);
A = log10( squeeze(sum(A.*conj(A),2))/NKxi );

figure(fig)
subplot(3,6,[11 12])
imagesc(yAxis(1:NKy), wAxis(1:NW), A)
hold on;plot(yAxis(1:NKy), wlh*ones(size(yAxis(1:NKy))),'k--',z,z,'k--');hold off;
hold on;plot(k_wh, wAxis,'k--');hold off;
w_ic = linspace(0,1,100);
hold on;plot(k_ic(w_ic), w_ic,'k--', yAxis(1:NKy), ones(size(yAxis(1:NKy))), 'k--');hold off;
xlabel('$K_y$', 'Interpreter', 'latex')
ylabel('$\omega$', 'Interpreter', 'latex')
axis xy; box on; grid on;
colormap(jet)


A = fourierSpace(:,1:1:NKxi,1:NKyi);
A = A.*conj(A);
A = sum(A,3)/NKyi;
A = sum(A,2)/NKxi;
A = log10(A);

figure(fig)
subplot(3,6,10)
plot(A, wAxis(1:NW))
ylabel('$\omega$', 'Interpreter', 'latex')
box on; grid on;

% Frequency spectrum (kx and ky integrated)
A = fourierSpace(:,1:1:NKxi,1:1:NKyi);
A = sum(sum(A.*conj(A),3),2)/(NKxi*NKyi);
A = A/double(NX_IN_SIM*NY_IN_SIM);
A = log10(A);

figure(fig)
subplot(3,6,[14 17])
plot(wAxis(1:NW), A)
ylabel('$\omega$', 'Interpreter', 'latex')
axis xy; box on; grid on;
colormap(jet)




end


function EnergyConservation(ST)
% Diagnostic to monitor energy transfer/conservation
NT = ST.numberOfOutputs;
NSPP = ST.params.ions.numberOfParticleSpecies;
NMPI_P = ST.params.numMPIsParticles;
NMPI_F = ST.params.numMPIsFields;
DX = ST.params.geometry.DX;
DY = ST.params.geometry.DY;
ionGyroPeriod = ST.params.scales.ionGyroPeriod(1);

kineticEnergyDensity = zeros(NSPP,NT);
electricEnergyDensity = zeros(4,NT);
magneticEnergyDensity = zeros(4,NT);

% First we calculate the kinetic energy of the simulated ions
ilabels = {};

for ss=1:NSPP
    mi = ST.params.ions.(['spp_' num2str(ss)]).M;
    NCP = ST.params.ions.(['spp_' num2str(ss)]).NCP;
    
    for ii=1:NT
        for dd=1:NMPI_P           
            kineticEnergyDensity(ss,ii) = kineticEnergyDensity(ss,ii) + ST.ions_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.ions.(['spp_' num2str(ss)]).kineticEnergyDensity;
        end
        
    ilabels{ss} = ['Species ' num2str(ss)];
    end
end
ilabels{NSPP + 1} = 'Total';


% Energy of electromagnetic fields
for ii=1:NT
    for dd=1:NMPI_F
        electricEnergyDensity(1,ii) = electricEnergyDensity(1,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.E.X;
        electricEnergyDensity(2,ii) = electricEnergyDensity(2,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.E.Y;
        electricEnergyDensity(3,ii) = electricEnergyDensity(3,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.E.Z;
        
        magneticEnergyDensity(1,ii) = magneticEnergyDensity(1,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.B.X;
        magneticEnergyDensity(2,ii) = magneticEnergyDensity(2,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.B.Y;
        magneticEnergyDensity(3,ii) = magneticEnergyDensity(3,ii) + ST.fields_data.(['MPI' num2str(dd-1) '_O' num2str(ii-1)]).energy.fields.B.Z;
    end
end

electricEnergyDensity(4,:) = sum(electricEnergyDensity(1:3,:),1);
magneticEnergyDensity(4,:) = sum(magneticEnergyDensity(1:3,:),1);

ET = sum(kineticEnergyDensity,1) + electricEnergyDensity(4,:) + magneticEnergyDensity(4,:);

% Change in total energy as a percentage of the initial ion energy
ET_Ei = zeros(NSPP,NT);
for ss=1:NSPP
    ET_Ei(ss,:) = 100.0*(ET - ET(1))/kineticEnergyDensity(ss,1);
end

% Relative change in total energy
ET = 100.0*(ET - ET(1))/ET(1);

% Change in kinetic energy w.r.t. initial condition
kineticEnergyDensity = kineticEnergyDensity - kineticEnergyDensity(:,1);

% Change in magnetic energy w.r.t. initial condition
magneticEnergyDensity = magneticEnergyDensity - magneticEnergyDensity(:,1);

% Change in electric energy w.r.t. initial condition
electricEnergyDensity = electricEnergyDensity - electricEnergyDensity(:,1);

time = ST.time/ionGyroPeriod;

% Figures to show energy conservation
fig = figure('name','Energy conservation');
for ss=1:NSPP
    figure(fig)
    subplot(5,1,1)
    hold on;
    plot(time, kineticEnergyDensity(ss,:), '--')
    hold off
    box on; grid on;
    xlim([min(time) max(time)])
    xlabel('Time (s)','interpreter','latex')
    ylabel('$\Delta \mathcal{E}_K$ (J/m$^3$)','interpreter','latex')
end

figure(fig)
subplot(5,1,1)
hold on;
plot(time, sum(kineticEnergyDensity,1))
hold off
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_K$ (J/m$^3$)','interpreter','latex')
legend(ilabels,'interpreter','latex')

figure(fig);
subplot(5,1,2)
plot(time, magneticEnergyDensity)
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_B$ (J/m$^3$)','interpreter','latex')
legend({'$B_x$', '$B_y$', '$B_z$', '$B$'},'interpreter','latex')

figure(fig);
subplot(5,1,3)
plot(time, electricEnergyDensity)
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_E$ (J/m$^3$)','interpreter','latex')
legend({'$E_x$', '$E_y$', '$E_z$', '$E$'},'interpreter','latex')

figure(fig);
subplot(5,1,4)
plot(time, sum(kineticEnergyDensity,1),'r', time, magneticEnergyDensity(4,:), 'b-', time, electricEnergyDensity(4,:), 'k')
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}$ (J/m$^3$)','interpreter','latex')
legend({'$K_i$', '$B$', '$E$'},'interpreter','latex')

figure(fig);
subplot(5,1,5)
for ss=1:NSPP
    hold on; plot(time, ET_Ei(ss,:),'--'); hold off
end
hold on;plot(time, ET);hold off
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_T$ (\%)','interpreter','latex')
legend(ilabels,'interpreter','latex')

end


