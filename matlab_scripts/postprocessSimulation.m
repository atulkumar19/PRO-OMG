function ST = postprocessSimulation(path)
% Example: ST = postprocessSimulation('../PROMETHEUS++/outputFiles/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/test2/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/dispersion_relation/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/warm_plasma/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/outputFiles/GC/HDF5/')
% ST = postprocessSimulation('../PROMETHEUS++/Tests/warm_plasma/HDF5/')


% Physical constants

close all

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

% FO_test_1D(ST);

% FO_test_2D(ST);

% FourierAnalysis(ST,'B','x');
% FourierAnalysis(ST,'B','y');
FourierAnalysis(ST,'B','z');

% FourierAnalysis(ST,'E','x');
% FourierAnalysis(ST,'E','y');
% FourierAnalysis(ST,'E','z');

EnergyDiagnostic(ST);

% testFieldInterpolation(ST);
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

function ST = loadData(ST)
ST.data = struct;

numberOfOutputs = [];

for ff=1:ST.params.numOfDomains
    info = h5info([ST.path ['file_D' num2str(ff-1) '.h5']]);
    
    numberOfOutputs(ff)= numel(info.Groups);
    
    for ii=1:numel(info.Groups)
        groupName = strsplit(info.Groups(ii).Name,'/');
        
        for jj=1:numel(info.Groups(ii).Datasets)
            datasetName = info.Groups(ii).Datasets(jj).Name;
            ST.data.(['D' num2str(ff-1) '_O' groupName{end}]).(datasetName) = ...
                h5read(info.Filename, ['/' groupName{end} '/' datasetName]);
        end
        
        if ~isempty(info.Groups(ii).Groups)
            for jj=1:numel(info.Groups(ii).Groups)
                subGroupName = strsplit(info.Groups(ii).Groups(jj).Name,'/');
                
                for kk=1:numel(info.Groups(ii).Groups(jj).Datasets)
                    datasetName = info.Groups(ii).Groups(jj).Datasets(kk).Name;
                    ST.data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(datasetName) = ...
                        h5read(info.Filename, [info.Groups(ii).Groups(jj).Name '/' datasetName]);
                end
                
                if ~isempty(info.Groups(ii).Groups(jj).Groups)
                    for kk=1:numel(info.Groups(ii).Groups(jj).Groups)
                        subSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Name,'/');
                        
                        for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Datasets)
                            datasetName = info.Groups(ii).Groups(jj).Groups(kk).Datasets(ll).Name;
                            
                            ST.data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(datasetName) = ...
                                h5read(info.Filename, [info.Groups(ii).Groups(jj).Groups(kk).Name '/' datasetName]);
                        end
                        
                        if ~isempty(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                            for ll=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups)
                                subSubSubGroupName = strsplit(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Name,'/');
                                
                                for mm=1:numel(info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets)
                                    datasetName = info.Groups(ii).Groups(jj).Groups(kk).Groups(ll).Datasets(mm).Name;
                                    
                                    ST.data.(['D' num2str(ff-1) '_O' groupName{end}]).(subGroupName{end}).(subSubGroupName{end}).(subSubSubGroupName{end}).(datasetName) = ...
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

ST.numberOfOutputs = min(numberOfOutputs);
end

function time = loadTimeVector(ST)
time = zeros(1,ST.numberOfOutputs);

for ii=1:ST.numberOfOutputs
    time(ii) = ST.data.(['D0_O' num2str(ii-1)]).time;
end

end


function FO_test_1D(ST)
% Function for testing ExB drift of a GC particle in constant perpendicular
% electric and magnetic fields.
NT = int64(ST.numberOfOutputs);
NSPP = int64(ST.params.ions.numberOfParticleSpecies);
ND = int64(ST.params.numOfDomains);
DX = int64(ST.params.geometry.DX);


ilabels = {};

for ss=1:NSPP
    mi = ST.params.ions.(['spp_' num2str(ss)]).M;
    qi = ST.params.ions.(['spp_' num2str(ss)]).Q;
    NCP = int64(ST.params.ions.(['spp_' num2str(ss)]).NCP);
    NSP = int64(ST.params.ions.(['spp_' num2str(ss)]).NSP_OUT);
    NPARTICLES = ND*NSP;
    
    X = zeros(NPARTICLES,NT);
    V = zeros(NPARTICLES,3,NT);
    
    for ii=1:NT
        for dd=1:ND
            iIndex = NSP*(dd - 1) + 1;
            fIndex = iIndex + NSP - 1;
            
            X(iIndex:fIndex,ii) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).X;
            V(iIndex:fIndex,:,ii) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).V;
        end
    end
    
    ilabels{ss} = ['Species ' num2str(ss)];
    
    % Time
    t = ST.time;
    
    % Plot test particle position and velocity
    ii = randi(NPARTICLES);
    
    x = squeeze(X(ii,:));
    v = squeeze(V(ii,:,:));
    
    fig = figure;
    subplot(4,1,1)
    plot(t, x, 'b.')
    xlabel('Time [s]','interpreter','latex')
    ylabel('$X(t)$ [m]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,2)
    plot(t, v(1,:), 'b.')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_x$ [m/s]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,3)
    plot(t, v(2,:), 'b.')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_y$ [m/s]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,4)
    plot(t, v(3,:), 'b.')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_y$ [m/s]','interpreter','latex')
    
end


end

function FO_test_2D(ST)
% Function for testing ExB drift of a GC particle in constant perpendicular
% electric and magnetic fields.
NT = int64(ST.numberOfOutputs);
NSPP = int64(ST.params.ions.numberOfParticleSpecies);
ND = int64(ST.params.numOfDomains);
DX = int64(ST.params.geometry.DX);


ilabels = {};

for ss=1:NSPP
    mi = ST.params.ions.(['spp_' num2str(ss)]).M;
    qi = ST.params.ions.(['spp_' num2str(ss)]).Q;
    NCP = int64(ST.params.ions.(['spp_' num2str(ss)]).NCP);
    NSP = int64(ST.params.ions.(['spp_' num2str(ss)]).NSP_OUT);
    NPARTICLES = ND*NSP;
    Wc = qi*ST.params.Bo(3)/mi;
    Eo = 1.0E5;
    Bo = ST.params.Bo(3);
    
    X = zeros(NPARTICLES,2,NT);
    V = zeros(NPARTICLES,3,NT);
    
    for ii=1:NT
        for dd=1:ND
            iIndex = NSP*(dd - 1) + 1;
            fIndex = iIndex + NSP - 1;
            
            X(iIndex:fIndex,:,ii) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).X;
            V(iIndex:fIndex,:,ii) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).V;
        end
    end
    
    ilabels{ss} = ['Species ' num2str(ss)];
    
    % Time
    t = ST.time;
    
    % Plot test particle position and velocity
    ii = randi(NPARTICLES);
    
    x = squeeze(X(ii,:,:));
    v = squeeze(V(ii,:,:));
    
    xs = v(1,1)*sin(Wc*t)/Wc - (v(2,1) + Eo/Bo)*cos(Wc*t)/Wc + x(1,1) + (v(2,1) + Eo/Bo)/Wc;
    xs = mod(xs,ST.params.geometry.LX);
    xs(xs<0) = xs(xs<0) + ST.params.geometry.LX;
    
    ys = v(1,1)*cos(Wc*t)/Wc + (v(2,1) + Eo/Bo)*sin(Wc*t)/Wc - Eo*t/Bo + x(2,1) - v(1,1)/Wc;
    ys = mod(ys,ST.params.geometry.LY);
    ys(ys<0) = ys(ys<0) + ST.params.geometry.LY;
    
    vxs = v(1,1)*cos(Wc*t) + (v(2,1) + Eo/Bo)*sin(Wc*t);
    vys = -v(1,1)*sin(Wc*t) + (v(2,1) + Eo/Bo)*cos(Wc*t) - Eo/Bo;
    
    fig = figure;
    subplot(4,1,1)
    plot(x(1,:), x(2,:), 'b.', xs,ys,'r.', x(1,1), x(2,1),'sm', xs(1),ys(1),'go')
    xlabel('$X$ [m]','interpreter','latex')
    ylabel('$Y$ [m]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,2)
    plot(t, v(1,:), 'b.',t,vxs,'r')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_x$ [m/s]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,3)
    plot(t, v(2,:), 'b.',t,vys,'r')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_y$ [m/s]','interpreter','latex')
    
    figure(fig)
    subplot(4,1,4)
    plot(t, v(3,:), 'b.')
    xlabel('Time (s)','interpreter','latex')
    ylabel('$V_y$ [m/s]','interpreter','latex')
    
end


end


function FourierAnalysis(ST,field,component)
if strcmp(component,'x')
    component_num = 1;
elseif strcmp(component,'y')
    component_num = 2;
elseif strcmp(component,'z')
    component_num = 3;
end


ND = ST.params.numOfDomains; % Number of domains
NX_PER_MPI = ST.params.geometry.NX; % Number of cells per domain
NX_IN_SIM = ST.params.geometry.NX_IN_SIM; % Number of cells in the whole domain
NY_PER_MPI = ST.params.geometry.NY; % Number of cells per domain
NY_IN_SIM = ST.params.geometry.NY_IN_SIM; % Number of cells in the whole domain
SPLIT_DIRECTION = ST.params.geometry.SPLIT_DIRECTION;

% Plasma parameters
qi = ST.params.ions.spp_1.Q;
mi = ST.params.ions.spp_1.M;
Bo = sqrt(dot(ST.params.Bo,ST.params.Bo));

wci = qi*Bo/mi; % Ion cyclotron frequency
wce = ST.qe*Bo/ST.me; % Electron cyclotron frequency

ni = ST.params.ions.ne;
wpi = sqrt(ni*((qi)^2)/(mi*ST.ep0));

wci = double(wci);
wce = double(wce);
wpi = double(wpi);


% Lower hybrid frequency
wlh = sqrt( wpi^2*wci*wce/( wci*wce + wpi^2 ) );
wlh = wlh/wci;

disp(['Lower hybrid frequency: ' num2str(wlh)]);

% % % % % % Fourier variables % % % % % % 
time = ST.time;
DT = mean(diff(time));

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


F = zeros(NT,NX_IN_SIM,NY_IN_SIM);

for ii=1:NT
    for dd=1:ND   
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
            F(ii,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component)  - ST.params.Bo(component_num);
        else
            F(ii,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.(field).(component);
        end
    end
end

% x-axis
kSpacex = zeros(NT,NX_IN_SIM);
for ii=1:NT
    kSpacex(ii,:) = fft( squeeze(F(ii,:,1)) );
end

fourierSpacex = zeros(NT,NX_IN_SIM);
for ii=1:NX_IN_SIM
    fourierSpacex(:,ii) = fft(hanning(double(NT)).*kSpacex(:,ii));
end

if (ST.params.dimensionality == 2)
    % y-axis
    kSpacey = zeros(NT,NY_IN_SIM);
    for ii=1:NT
        kSpacey(ii,:) = fft( squeeze(F(ii,1,:)) );
    end
    
    fourierSpacey = zeros(NT,NY_IN_SIM);
    for ii=1:NY_IN_SIM
        fourierSpacey(:,ii) = fft( hanning(double(NT)).*kSpacey(:,ii) );
    end
end

wk_fig = figure;

if (ST.params.dimensionality == 1)
    % Propagation along x-axis
    A = fourierSpacex.*conj(fourierSpacex);
    % Magnetoacoustic wave
    z = linspace(0,max([max(xAxis), max(wAxis)]),10);
    
    figure(wk_fig)
    imagesc(xAxis,wAxis,log10(A(1:NT/2,1:NX_IN_SIM/2)));
    hold on;plot(xAxis, wlh*ones(size(xAxis)),'k--',z,z,'k--');hold off;
    axis xy; colormap(jet); colorbar
    axis([0 max(xAxis) 0 max(wAxis)])
    xlabel('$ck_x/\omega_p$', 'Interpreter', 'latex')
    ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')
    title(['$' field '_' component '(x)$'],'interpreter','latex')
else
    % Propagation along x-axis
    A = fourierSpacex.*conj(fourierSpacex);
    % Magnetoacoustic wave
    z = linspace(0,max([max(xAxis), max(wAxis)]),10);
    
    figure(wk_fig)
    subplot(2,1,1)
    imagesc(xAxis,wAxis,log10(A(1:NT/2,1:NX_IN_SIM/2)));
    hold on;plot(xAxis, wlh*ones(size(xAxis)),'k--',z,z,'k--');hold off;
    axis xy; colormap(jet); colorbar
    try
        axis([0 max(xAxis) 0 max(wAxis)])
    end
    xlabel('$ck_x/\omega_p$', 'Interpreter', 'latex')
    ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')
    title(['$' field '_' component '(x)$'],'interpreter','latex')
    
    % Propagation along x-axis
    A = fourierSpacey.*conj(fourierSpacey);
    % Magnetoacoustic wave
    z = linspace(0,max([max(yAxis), max(wAxis)]),10);
    
    figure(wk_fig)
    subplot(2,1,2)
    imagesc(yAxis,wAxis,log10(A(1:NT/2,1:NY_IN_SIM/2)));
    hold on;plot(yAxis, wlh*ones(size(yAxis)),'k--',z,z,'k--');hold off;
    axis xy; colormap(jet); colorbar
    try
        axis([0 max(xAxis) 0 max(wAxis)])
    end
    xlabel('$ck_y/\omega_p$', 'Interpreter', 'latex')
    ylabel('$\omega/\Omega_i$', 'Interpreter', 'latex')
    title(['$' field '_' component '(y)$'],'interpreter','latex')
end
end

function EnergyDiagnostic(ST)
% Diagnostic to monitor energy transfer/conservation
NT = ST.numberOfOutputs;
NSPP = ST.params.ions.numberOfParticleSpecies;
ND = ST.params.numOfDomains;
DX = ST.params.geometry.DX;
DY = ST.params.geometry.DY;
ionGyroPeriod = ST.params.scales.ionGyroPeriod(1);

% First we calculate the kinetic energy of the simulated ions
Ei = zeros(NSPP,NT);
ilabels = {};

for ss=1:NSPP
    mi = ST.params.ions.(['spp_' num2str(ss)]).M;
    NCP = ST.params.ions.(['spp_' num2str(ss)]).NCP;
    
    for ii=1:NT
        for dd=1:ND
            g = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).g;
            
            Ei(ss,ii) = Ei(ss,ii) + sum(g - 1.0)*mi*ST.c^2;
        end
        
%         Ei(ss,ii) = NCP*Ei(ss,ii);
        
        if(ST.params.dimensionality == 1)
            Ei(ss,ii) = NCP*Ei(ss,ii)/DX;
        else
            Ei(ss,ii) = NCP*Ei(ss,ii)/(DX*DY);
        end
    end
    
    ilabels{ss} = ['Species ' num2str(ss)];
end
ilabels{NSPP + 1} = 'Total';


% Energy of electromagnetic fields
EBx = zeros(1,NT);
EBy = zeros(1,NT);
EBz = zeros(1,NT);

EEx = zeros(1,NT);
EEy = zeros(1,NT);
EEz = zeros(1,NT);

for ii=1:NT
    for dd=1:ND
        Bx = ST.params.Bo(1) - ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.x;
        By = ST.params.Bo(2) - ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.y;
        Bz = ST.params.Bo(3) - ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.z;
        
        EBx(ii) = EBx(ii) + sum(sum(Bx.^2));
        EBy(ii) = EBy(ii) + sum(sum(By.^2));
        EBz(ii) = EBz(ii) + sum(sum(Bz.^2));
        
        Ex = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.x;
        Ey = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.y;
        Ez = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.z;
        
        EEx(ii) = EEx(ii) + sum(sum(Ex.^2));
        EEy(ii) = EEy(ii) + sum(sum(Ey.^2));
        EEz(ii) = EEz(ii) + sum(sum(Ez.^2));
    end
end

EBx = 0.5*EBx/ST.mu0;
EBy = 0.5*EBy/ST.mu0;
EBz = 0.5*EBz/ST.mu0;
EB = EBx + EBy + EBz;

EEx = 0.5*ST.ep0*EEx;
EEy = 0.5*ST.ep0*EEy;
EEz = 0.5*ST.ep0*EEz;
EE = EEx + EEy + EEz;

% if(ST.params.dimensionality == 1)
%     EBx = EBx*DX;
%     EBy = EBy*DX;
%     EBz = EBz*DX;
%     EB = EB*DX;
%     
%     EEx = EEx*DX;
%     EEy = EEy*DX;
%     EEz = EEz*DX;
%     EE = EE*DX;
% else
%     EBx = EBx*DX*DY;
%     EBy = EBy*DX*DY;
%     EBz = EBz*DX*DY;
%     EB = EB*DX*DY;
%     
%     EEx = EEx*DX*DY;
%     EEy = EEy*DX*DY;
%     EEz = EEz*DX*DY;
%     EE = EE*DX*DY;
% end

ET = sum(Ei,1) + EE + EB;

% Change in total energy as a percentage of the initial ion energy
ET_Ei = zeros(NSPP,NT);
for ss=1:NSPP
    ET_Ei(ss,:) = 100.0*(ET - ET(1))/Ei(ss,1);
end

% Relative change in total energy
ET = 100.0*(ET - ET(1))/ET(1);

% Change in kinetic energy w.r.t. initial condition
Ei = Ei - Ei(:,1);

% Change in magnetic energy w.r.t. initial condition
EBx = EBx - EBx(1);
EBy = EBy - EBy(1);
EBz = EBz - EBz(1);
EB = EB - EB(1);

% Change in electric energy w.r.t. initial condition
EEx = EEx - EEx(1);
EEy = EEy - EEy(1);
EEz = EEz - EEz(1);
EE = EE - EE(1);

time = ST.time/ionGyroPeriod;

% Figures to show energy conservation
fig = figure('name','Energy conservation');
for ss=1:NSPP
    figure(fig)
    subplot(5,1,1)
    hold on;
    plot(time, Ei(ss,:), '--')
    hold off
    box on; grid on;
    xlim([min(time) max(time)])
    xlabel('Time (s)','interpreter','latex')
    ylabel('$\Delta \mathcal{E}_K$ (J/m$^3$)','interpreter','latex')
end

figure(fig)
subplot(5,1,1)
hold on;
plot(time, sum(Ei,1))
hold off
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_K$ (J/m$^3$)','interpreter','latex')
legend(ilabels,'interpreter','latex')

figure(fig);
subplot(5,1,2)
plot(time, EBx, 'r--', time, EBy, 'b-.', time, EBz, 'c-.', time, EB, 'k-')
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_B$ (J/m$^3$)','interpreter','latex')
legend({'$B_x$', '$B_y$', '$B_z$', '$B$'},'interpreter','latex')

figure(fig);
subplot(5,1,3)
plot(time, EEx, 'r--', time, EEy, 'b--', time, EEz, 'c--', time, EE, 'k-')
box on; grid on;
xlim([min(time) max(time)])
xlabel('Time (s)','interpreter','latex')
ylabel('$\Delta \mathcal{E}_E$ (J/m$^3$)','interpreter','latex')
legend({'$E_x$', '$E_y$', '$E_z$', '$E$'},'interpreter','latex')

figure(fig);
subplot(5,1,4)
plot(time, sum(Ei,1),'r', time, EB, 'b-', time, EE, 'k')
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


function testFieldInterpolation(ST)
% Diagnostic to monitor energy transfer/conservation
NT = ST.numberOfOutputs;
NS = ST.params.ions.numberOfParticleSpecies;
ND = ST.params.numOfDomains;
NX = double(ST.params.geometry.NX);
NY = double(ST.params.geometry.NY);
DX = ST.params.geometry.DX;
DY = ST.params.geometry.DY;
xNodes = ST.params.geometry.xAxis;
if (ST.params.dimensionality == 2)
    yNodes = ST.params.geometry.yAxis;
end
LX = ST.params.geometry.LX;
LY = ST.params.geometry.LY;

for ss=1:NS
    NSP = double(ST.params.ions.(['spp_' num2str(ss)]).NSP_OUT);
    
    X = zeros(NT,ND*NSP,ST.params.dimensionality);
    Ep = zeros(NT,ND*NSP,3);
    Bp = zeros(NT,ND*NSP,3);
    E = zeros(NT,3,NX,NY);
    B = zeros(NT,3,NX,NY);
    U = zeros(NT,3,NX,NY);
    n = zeros(NT,NX,NY);
    
    for ii=1:NT
        for dd=1:ND
            iIndex = (dd-1)*NSP + 1;
            fIndex = dd*NSP;
            X(ii,iIndex:fIndex,:) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).X;
            Ep(ii,iIndex:fIndex,:) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).E;
            Bp(ii,iIndex:fIndex,:) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).B;
            
            if (ST.params.geometry.SPLIT_DIRECTION == 0)
                ix = (dd-1)*NX + 1;
                fx = dd*NX;
                iy = 1;
                fy = NY;
            else
                ix = 1;
                fx = NX;
                iy = (dd-1)*NY + 1;
                fy = dd*NY;
            end
            
            E(ii,1,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.x;
            E(ii,2,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.y;
            E(ii,3,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.E.z;
            
            B(ii,1,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.x;
            B(ii,2,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.y;
            B(ii,3,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).fields.B.z;
            
            n(ii,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).n;
            U(ii,1,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).U.x;
            U(ii,2,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).U.y;
            U(ii,3,ix:fx,iy:fy) = ST.data.(['D' num2str(dd-1) '_O' num2str(ii-1)]).ions.(['spp_' num2str(ss)]).U.z;
        end
    end
    
    % Iterations to plot
    its = [1 randi(NT-1) NT];
    %     its = [1 1 1];
    
    fig_E = figure;
    fig_B = figure;
    
    if (ST.params.dimensionality == 1)
        % Ex
        xAxis = xNodes + 0.5*DX;
        for it=1:numel(its)
            F = squeeze( E(its(it),1,:,:) );
            Fp = squeeze( Ep(its(it),:,1) );
            
            figure(fig_E)
            subplot(3,3,it)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$E_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ey
        xAxis = xNodes;
        for it=1:numel(its)
            F = squeeze( E(its(it),2,:,:) );
            Fp = squeeze( Ep(its(it),:,2) );
            
            figure(fig_E)
            subplot(3,3,it+3)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$E_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ez
        xAxis = xNodes;
        for it=1:numel(its)
            F = squeeze( E(its(it),3,:,:) );
            Fp = squeeze( Ep(its(it),:,3) );
            
            figure(fig_E)
            subplot(3,3,it+6)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$E_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        
        % Bx
        xAxis = xNodes;
        for it=1:numel(its)
            F = squeeze( B(its(it),1,:,:) );
            Fp = squeeze( Bp(its(it),:,1) );
            
            figure(fig_B)
            subplot(3,3,it)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$B_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % By
        xAxis = xNodes + 0.5*DX;
        for it=1:numel(its)
            F = squeeze( B(its(it),2,:,:) );
            Fp = squeeze( Bp(its(it),:,2) );
            
            figure(fig_B)
            subplot(3,3,it+3)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$B_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Bz
        xAxis = xNodes + 0.5*DX;
        for it=1:numel(its)
            F = squeeze( B(its(it),3,:,:) );
            Fp = squeeze( Bp(its(it),:,3) );
            
            figure(fig_B)
            subplot(3,3,it+6)
            plot(xAxis,F,'bo-', X(its(it),:),Fp,'k.')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$B_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Number density diagnostic
        fig_n = figure;
        
        xAxis = xNodes;
        for it=1:numel(its)
            F = squeeze( n(its(it),:,:) )';
            
            figure(fig_n)
            subplot(4,3,it)
            plot(xAxis,F,'bo-')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$n$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ux
        for it=1:numel(its)
            F = squeeze( U(its(it),1,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+3)
            plot(xAxis,F,'bo-')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$U_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Uy
        for it=1:numel(its)
            F = squeeze( U(its(it),2,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+6)
            plot(xAxis,F,'bo-')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$U_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Uz
        for it=1:numel(its)
            F = squeeze( U(its(it),3,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+9)
            plot(xAxis,F,'bo-')
            box on; grid on;
            xlabel('$x$','Interpreter','latex')
            ylabel('$U_z$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
    else
        % Ex
        xAxis = xNodes + 0.5*DX;
        yAxis = yNodes;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( E(its(it),1,:,:) )';
            Fp = squeeze( Ep(its(it),:,1) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_E)
            subplot(3,3,it)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            hold on;scatter3(Xp(:,1)+LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1)-LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)+LY,Fp,'r.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)-LY,Fp,'r.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$E_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ey
        xAxis = xNodes;
        yAxis = yNodes + 0.5*DY;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( E(its(it),2,:,:) )';
            Fp = squeeze( Ep(its(it),:,2) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_E)
            subplot(3,3,it+3)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            hold on;scatter3(Xp(:,1)+LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1)-LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)+LY,Fp,'r.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)-LY,Fp,'r.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$E_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ez
        xAxis = xNodes;
        yAxis = yNodes;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( E(its(it),3,:,:) )';
            Fp = squeeze( Ep(its(it),:,3) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_E)
            subplot(3,3,it+6)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            hold on;scatter3(Xp(:,1)+LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1)-LX,Xp(:,2),Fp,'b.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)+LY,Fp,'r.');hold off
            hold on;scatter3(Xp(:,1),Xp(:,2)-LY,Fp,'r.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$E_z$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Bx
        xAxis = xNodes;
        yAxis = yNodes + 0.5*DY;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( B(its(it),1,:,:) )';
            Fp = squeeze( Bp(its(it),:,1) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_B)
            subplot(3,3,it)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$B_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % By
        xAxis = xNodes + 0.5*DX;
        yAxis = yNodes;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( B(its(it),2,:,:) )';
            Fp = squeeze( Bp(its(it),:,2) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_B)
            subplot(3,3,it+3)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$B_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Bz
        xAxis = xNodes;
        yAxis = yNodes;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( B(its(it),3,:,:) )';
            Fp = squeeze( Bp(its(it),:,3) );
            Xp = squeeze(X(its(it),:,:));
            
            figure(fig_B)
            subplot(3,3,it+6)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            hold on;scatter3(Xp(:,1),Xp(:,2),Fp,'k.');hold off
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$B_z$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        
        % Number density diagnostic
        fig_n = figure;
        
        xAxis = xNodes;
        yAxis = yNodes;
        [XGRID,YGRID] = meshgrid(xAxis,yAxis);
        
        for it=1:numel(its)
            F = squeeze( n(its(it),:,:) )';
            
            figure(fig_n)
            subplot(4,3,it)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$n$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Ux
        for it=1:numel(its)
            F = squeeze( U(its(it),1,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+3)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$U_x$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Uy
        for it=1:numel(its)
            F = squeeze( U(its(it),2,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+6)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$U_y$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
        
        % Uz
        for it=1:numel(its)
            F = squeeze( U(its(it),3,:,:) )';
            
            figure(fig_n)
            subplot(4,3,it+9)
            surf(XGRID,YGRID,F,'FaceAlpha',0.5)
            box on; grid on;colormap(jet)
            xlabel('$x$','Interpreter','latex')
            ylabel('$y$','Interpreter','latex')
            zlabel('$U_z$','Interpreter','latex')
            title(['$t$=' num2str(ST.time(its(it))) ' s'],'Interpreter','latex')
        end
    end
    
    
end
end
