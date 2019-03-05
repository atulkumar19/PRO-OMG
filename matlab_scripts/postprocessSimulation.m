function ST = postprocessSimulation(path)
% Example: ST = postprocessSimulation('../PROMETHEUS++/outputFiles/warm_plasma/HDF5/')
% Physical constants

close all

kB = 1.38E-23; % Boltzmann constant
mu0 = (4E-7)*pi; % Magnetic permeability of vacuum
ep0 = 8.854E-12; % Electric permittivity of vacuum
c=2.9979E8; % Speed of light
amu = 1.660539E-27; % Atomic mass unit in kg

qe = 1.602176E-19; % Electron charge
me = 9.109383E-31; % Electron mass

ST.path = path;

ST.params = loadSimulationParameters(ST);

ST.data = loadData(ST);

FourierAnalysis(ST,'E');
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

DX = mean(diff(ST.params.geometry.xAxis));
Dk = 1.0/(DX*double(NXTD));
kmax = 1.0/(2.0*DX);
kAxis = 0:Dk:kmax-Dk;

kSpace = zeros(NT,NXTD);
for ii=1:NT
    kSpace(ii,:) = fft(F(ii,:));
end

fourierSpace = zeros(NT,NXTD);
for ii=1:NXTD
    fourierSpace(:,ii) = fft(hanning(double(NT)).*kSpace(:,ii));
end

A = fourierSpace.*conj(fourierSpace);

figure
imagesc(kAxis,fAxis,log10(A(1:NT/2,1:NXTD/2)));
axis xy; colormap(jet); colorbar
xlabel('$k$ ($m^-1$)', 'Interpreter', 'latex')
ylabel('$f$ (Hz)', 'Interpreter', 'latex')

end