function ST = postprocessSimulation(path)
% Example: ST = postprocessSimulation('../PROMETHEUS++/outputFiles/warm_plasma/HDF5/')
% Physical constants

close all

ST.kB = 1.38E-23; % Boltzmann constant
ST.mu0 = (4E-7)*pi; % Magnetic permeability of vacuum
ST.ep0 = 8.854E-12; % Electric permittivity of vacuum
ST.c=2.9979E8; % Speed of light
ST.amu = 1.660539E-27; % Atomic mass unit in kg

ST.qe = 1.602176E-19; % Electron charge
ST.me = 9.109383E-31; % Electron mass

ST.path = path;

ST.params = loadSimulationParameters(ST);

ST.data = loadData(ST);

FourierAnalysis(ST,'E');

% EnergyDiagnostic(ST);
end

function params = loadSimulationParameters(ST)
params = struct;
info = h5info([ST.path 'main_D0.h5']);

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

function data = loadData(ST)
data = struct;

for ff=1:ST.params.numOfDomains
    info = h5info([ST.path ['file_D' num2str(ff-1) '.h5']]);
    
    for ii=1:numel(info.Groups)       
        groupName = strsplit(info.Groups(ii).Name,'/');
        
        for jj=1:numel(info.Groups(ii).Datasets)
            datasetName = info.Groups(ii).Datasets(jj).Name;
            data.(['D' num2str(ff-1) '_O' groupName{end}]).(datasetName) = ...
                h5read(info.Filename, ['/' groupName{end} '/' datasetName]);
        end
        
        if ~isempty(info.Groups(ii).Groups)
            for jj=1:numel(info.Groups(ii).Groups)
                subGroupName = strsplit(info.Groups(ii).Groups(jj).Name,'/');
                
                for kk=1:numel(info.Groups(ii).Groups(jj).Datasets)
                    datasetName = info.Groups(ii).Groups(jj).Datasets(kk).Name;
                    data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(datasetName) = ...
                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Name '/' datasetName]);
                end
                
                if ~isempty(info.Groups(ii).Groups(jj).Groups)
                    for kk=1:numel(info.Groups(ii).Groups(jj).Groups)
                        subSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Name,'/');
                        
                        for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Datasets)
                            datasetName = info.Groups(ii).Groups(jj).Groups(kk).Datasets(ll).Name;
                            
                            data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(datasetName) = ...
                                h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Name '/' datasetName]);
                        end
                        
                        if ~isempty(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                            for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                                subSubSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name,'/');
                                
                                for mm=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets)
                                    datasetName = info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets(mm).Name;
                                    
                                    data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(subSubSubGroupName{end}).(datasetName) = ...
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

end

function FourierAnalysis(ST,EMF)
% Plasma parameters
qi = ST.params.ions.species_1.ionProperties(2);
mi = ST.params.ions.species_1.ionProperties(1);
Bo = sqrt(dot(ST.params.Bo,ST.params.Bo));

wci = qi*Bo/mi; % Ion cyclotron frequency
wce = ST.qe*Bo/ST.me; % Electron cyclotron frequency

ni = ST.params.ions.numberDensity;
wpi = sqrt(ni*((qi)^2)/(mi*ST.ep0));

% Lower hybrid frequency
wlh = sqrt( wpi^2*wci*wce/( wci*wce + wpi^2 ) );
wlh = wlh/wci;

disp(['Lower hybrid frequency: ' num2str(wlh)]);

NT = ST.params.numOutputFiles; % Number of snapshots
ND = ST.params.numOfDomains; % Number of domains
NXPD = ST.params.geometry.numberOfCells(1); % Number of cells per domain
NXTD = ND*NXPD; % Number of cells in the whole domain

time = zeros(1,NT);

F = zeros(NT,NXTD);

for ii=1:NT
    time(ii) = ST.data.(['D0_O' num2str(ii-1)]).time;
    for dd=1:ND
        F(ii,(dd-1)*NXPD + 1:dd*NXPD) = ...
            ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).emf.(EMF).([EMF 'y']);
    end
end

DT = mean(diff(time));
Df = 1.0/(DT*double(NT)); 
fmax = 1.0/(2.0*double(DT)); % Nyquist theorem
fAxis = 0:Df:fmax-Df;

wAxis = 2*pi*fAxis/wci;


DX = mean(diff(ST.params.geometry.xAxis));
Dk = 1.0/(DX*double(NXTD));
kmax = 1.0/(2.0*DX);
kAxis = 0:Dk:kmax-Dk;
kAxis = 2.0*pi*kAxis;

xAxis = ST.c*kAxis/wpi;

kSpace = zeros(NT,NXTD);
for ii=1:NT
    kSpace(ii,:) = fft(F(ii,:));
end

fourierSpace = zeros(NT,NXTD);
for ii=1:NXTD
    fourierSpace(:,ii) = fft(hanning(double(NT)).*kSpace(:,ii));
end

A = fourierSpace.*conj(fourierSpace);
z = linspace(0,max([max(xAxis), max(wAxis)]),10);

figure
imagesc(xAxis,wAxis,log10(A(1:NT/2,1:NXTD/2)));
hold on;plot(xAxis, wlh*ones(size(xAxis)),'k--',z,z,'k--');hold off;
axis xy; colormap(jet); colorbar
axis([0 max(xAxis) 0 max(wAxis)])
xlabel('$ck/\omega_p$', 'Interpreter', 'latex')
ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')
end

function EnergyDiagnostic(ST)
% Diagnostic to monitor energy transfer/conservation
NT = ST.params.numOutputFiles;
NSPP = ST.params.ions.numberOfIonSpecies;
ND = ST.params.numOfDomains;

% First we calculate the kinetic energy of the simulated ions
Ei = zeros(NSPP,NT);
EB = zeros(3,NT);
EB = zeros(3,NT);

for ss=1:NSPP
    mi = ST.params.ions.(['species_' num2str(ss)]).ionProperties(1);
    NSP = ST.params.ions.(['species_' num2str(ss)]).superParticlesProperties(3);
    
    for ii=1:NT
        aux_x = 0;
        aux_y = 0;
        aux_z = 0;
        
        for dd=1:ND
            aux_x = aux_x + sum(ST.data.(['D' num2str(dd-1) '_O' num2str(ii)]).ions.(['species_' num2str(ss)]).velocity.vx.^2);
            aux_y = aux_y + sum(ST.data.(['D' num2str(dd-1) '_O' num2str(ii)]).ions.(['species_' num2str(ss)]).velocity.vy.^2);
            aux_z = aux_z + sum(ST.data.(['D' num2str(dd-1) '_O' num2str(ii)]).ions.(['species_' num2str(ss)]).velocity.vz.^2);
        end
        
        aux_x = aux_x/NSP;
        aux_y = aux_y/NSP;
        aux_z = aux_z/NSP;
        
        
        Ei(ss,ii) = 0.5*mi*(aux_x + aux_y + aux_z);
    end
end

end