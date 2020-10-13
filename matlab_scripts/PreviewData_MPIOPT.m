% Objective:
% Read and analyze the output data produced by Prometheus++
% Data has been produced by a initial implementation of a 4D Metropolis
% algorithm in Prometheus++:
% Repo: https://github.com/atulkumar19/ldrdPrometheous.git
% commit hash: e198525824da838db957751ebcf760944bc78d34

% note:
% This script is incomplete, we need to do the integration over all
% velocity space correctly

clear all;
close all;

myDir = '../PROMETHEUS++/outputFiles/HDF5/';
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
        vv1{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,1); % x
        vv2{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,2); % y
        vv3{ii}{kk}  = d{ii}.(tii{jj}).ions.spp_1.V(:,3); % z
        nn{kk}       = d{1}.(tii{jj}).ions.spp_1.n;
        uux{kk}      = d{1}.(tii{jj}).ions.spp_1.U.x; % x
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
    Ned = nn{jj};
    Xd  = xx{1}{jj};
    V1d = vv1{1}{jj};
    V2d = vv2{1}{jj};
    V3d = vv3{1}{jj};
    for ii  = 2:ranksParticles
        Xd = [Xd;xx{ii}{jj}];
        V1d = [V1d;vv1{ii}{jj}];
        V2d = [V2d;vv2{ii}{jj}];
        V3d = [V3d;vv3{ii}{jj}];
    end
    % Gather in time dependent variable:
    Ne(:,jj) = Ned;
    Ux(:,jj) = Uxd;
    X(:,jj)  = Xd;
    V1(:,jj) = V1d;
    V2(:,jj) = V2d;
    V3(:,jj) = V3d;
end

% Physical constants:
% =========================================================================
e_c = 1.6020e-19;
k_B = 1.3806e-23;
m_p = 1.6726e-27;
m_e = 9.1094e-31;
mu0 = 4*pi*1e-7;

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
% NCP = 1.24E20/NS;

% Scaling factor for PDF:
N = NS*NCP;


%% Calculate statistics:
close all
% Statitistics:
% =========================================================================
for jj = 1
    % Scale for X:
    xbin = linspace(0,max(X(:,jj)),100);
    % Scale for vpar and vper:
    vxbin = linspace(-5*vTpar,+5*vTpar,100);
    vybin = linspace(-5*vTper,+5*vTper,100);
    vzbin = vybin;
    KEbin=linspace(0,5*max(Tper),100);
end

ux = vxbin(1:end-1);
uy = vybin(1:end-1);
uz = vybin(1:end-1);
KE=0.5*M*(V1.^2 + V2.^2 + V3.^2)/e_c;

% 1D distributions:
%%
for jj = 1:numOut
    fxvx{jj} = histcounts2(X(:,jj),V1(:,jj),xbin,vxbin,'Normalization','pdf');
    fxvy{jj} = histcounts2(X(:,jj),V2(:,jj),xbin,vybin,'Normalization','pdf');
    fxvz{jj} = histcounts2(X(:,jj),V3(:,jj),xbin,vzbin,'Normalization','pdf');
    rng=find(X(:,jj)>0.8& X(:,jj)<1.5);
    fvxvy{jj} = histcounts2(V1(rng,jj),V3(rng,jj),vxbin,vybin,'Normalization','pdf');
    
    fkex{jj}=histcounts2(X(:,jj),KE(:,jj),xbin,KEbin,'Normalization','pdf');



    n(:,jj)  =   N*trapz(ux,fxvx{jj}.*ux.^0,2);
    Px(:,jj) = (2/3)*M*N*trapz(ux,fxvx{jj}.*ux.^2,2);
    Py(:,jj) = (2/3)*M*N*trapz(uy,fxvy{jj}.*uy.^2,2);
    Pz(:,jj) = (2/3)*M*N*trapz(uz,fxvz{jj}.*uz.^2,2);
end

%%
% Plot distributions:
% =========================================================================
close all
jj = 1;

figure;
surf(xbin(1:end-1),vxbin(1:end-1)./vTpar,fxvx{jj}','LineStyle','none');
figure;
surf(xbin(1:end-1),vybin(1:end-1)./vTper,fxvy{jj}','LineStyle','none');
title('x-Vy')
figure;
surf(vxbin(1:end-1)./vTpar,vybin(1:end-1)./vTper,fvxvy{jj}','LineStyle','none');
title('Vx-Vy')

figure;
surf(xbin(1:end-1),KEbin(1:end-1),fkex{jj}','LineStyle','none');
title('x-KE')






figure 
hold on
hP(1) = plot(xbin(1:end-1),Px(:,jj),'k')
hP(2) = plot(xbin(1:end-1),Py(:,jj),'r')
hP(3) = plot(xbin(1:end-1),Pz(:,jj),'g')
grid on

legend(hP,'Px','Py','Pz')
title('Ion Pressure')
ylim([0,max(Px(:,jj))*1.2])

Tx(:,jj) = Px(:,jj)./(n(:,jj)*e_c);
Ty(:,jj) = Py(:,jj)./(n(:,jj)*e_c);
Tz(:,jj) = Pz(:,jj)./(n(:,jj)*e_c);

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

figure 
hold on
plot(xbin(1:end-1),n(:,jj));hold on;
% plot(ss(1:length(Ne)),Ne(:,jj));

title('Plasma density')
ylim([0,max(n(:,jj))*1.2])
xlabel('x(in meters)');
ylabel('density')

grid on

figure 
hold on
plot(ss(1:length(Ux(:,jj))),Ux(:,jj)/(vTpar))
title('Ux')
ylim([-1,1]*max(Ux(:,jj)/vTpar)*1.2)
grid on

figure 
subplot(3,1,1)
plot(ss(1:length(Bx(:,jj))),Bx(:,jj))
title('Bx')
xlabel('x(in meters)');
ylabel('B-field in Tesla')

% ylim([0,1.5]);
grid on
subplot(3,1,2)
plot(ss(1:length(By(:,jj))),By(:,jj))
title('Br')
xlabel('x(in meters)');
ylabel('B-field in Tesla')
% ylim([0,1.5]);
grid on
subplot(3,1,3)
plot(ss(1:length(Ex(:,jj))),Ex(:,jj))
title('Ex')
xlabel('x(in meters)');
ylabel('E-field in V/m')
% ylim([0,1.5]);
grid on

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