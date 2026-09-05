function [q_mat_x,q_mat_y,I_mat,distrp]=Scatter2D(rg,Nois,V,Nu,DETpix,SD_dist,lambda,det_side,PSF0,dist_type, dist_param, Weight_power)
%% simulate a specific scattering case with noise added
% rg Nu V are the scattering parameters (Nu=phi'')
%Nois is the noise level (compared to max signal=1).
% NEGATIVE Nois value provides the number of photons per pixel at the
% center. Then we assume the noise level is the square root of the photon number in each pixel.
%Detpix shows the number of pixels in the detector side, default is 1000
% SD_dist: Distance sample to detector in cm (default 150 cm),
% detector half side is det_side/2 (default 3.5cm),
% wavelength lambda [nm] (default=0.15 nm)
% PSF0 option to incorporate initial measurement PSF (before noise). a matrix which is
% convolved with the simulated scattering. default is ones(1)
% Outside the guinier region the model is invalid, so intensity becomes NaN
%   dist_type  {'normal','lognormal','schulz','exponential','boltzmann',
%               'triangular','uniform'} - calls size_distribution_discrete
%               for the r_g distribution type (mean=rg, rel variance=V)
%  dist_param.N= number of r_g in the distribution
% default distribution- gaussian (N=11)
%
% Weight_power adds a weight to the scattering of each particle: in powers of r 
% default is 0- every r has the same weight (mass): eg IDP
% for solid spheres Weight_power=6: the intensity is proportional to m^2=r^6
% for spherical shells the power is 4: mass=r^2
%
% SPECIAL CASES- if Nu=-1/63 exactly, we use Pedersen 1997 exact formula
% (and not the second order in (q*r)^2 )
%  Nu=0.000666 - thin disks
%  Nu=11/225 - thin rods
%  Nu=-1/45 - spherical shell
%  Nu=1/18 - gaussian chain
%
%
% returns a meshgrid of q on the detector, x- and y-values in q_mat_x and
% q_mat_y (nm-1), respectively, and intensity matrix I_mat
% 
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(rg,Nois,V,Nu,DETpix,SD_dist,lambda,det_side,PSF0)
% usage examples:
%spherical particles (Nu=-1/63) with mean(r_g)= 4 nm; relative variance V, 
% diamond typical PSF and
%noise:
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(4,-1e3,V,-1/63,1000,360,0.1,7,bartlett2d(3,15)); 
% 
%Beck laboratory typical sizes, zero noise, gaussian chain model (Nu=+1/18)  
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(4,0,V,+1/18,1000,150,0.15,7,bartlett2d(25,25))
%
% note that usually the guinier region is small, and most pixels are
% useless (nan). multiply DETpix and det_side by a factor (0.5) to get the
% same results with less effort.

noisemodel=1; %default for constant noise level
try PSF0=PSF0/sum(PSF0(:));
catch PSF0=1;
end
PSF0=makeKernelOddCentered(PSF0); %make it odd so that the center remains at the same place
try rg=abs(rg);
catch
    rg=5;
end
try noisemodel=sign(Nois);
catch
    Nois=0;
    noisemodel=1;
end
try Nu=Nu;
catch
    Nu=-1/63; %spherical default
end
try V=abs(V);
catch
    V=0.1;
end
try r_g=rg;
catch
    r_g=5;
end
try isfinite(Weight_power);
catch
    Weight_power=0;
end

%pixel size = d/L*4pi/lambda=75um/150cm*4pi/lambda= 4e-3 nm^-1
% dqpix=4e-3;

try SD_dist=SD_dist;
catch
    SD_dist=150;  % Sample-detector dist (cm)
end
try lambda=lambda;
catch
    lambda=0.15; %wavelength nm
end
try det_hside=det_side/2;
catch
    det_hside=3.5; %half side (cm)
end

% Distance to detector 150, detector half side is det_hside, lambda=0.15 nm
maxq=4*pi/lambda*det_hside/SD_dist;
try Npix=round(DETpix/2);
catch
    Npix=501;
end
qv=linspace(-maxq,maxq,2*Npix+1);
[qvx,qvy]=meshgrid(qv);
qvr2=(qvx.^2+qvy.^2);
qvr4=qvr2.*qvr2;
qvr=sqrt(qvr2);
nu=[Nu];
ff=@(qr) exp(-qr.^2/3+0.5*nu*qr.^4); %monolithic formfactor (function of unitless q*r_g)
try n_radii=dist_param.N;
catch
    n_radii=11; % number of radii in guinier distribution
end
% try [r_vect,p] = size_distribution_discrete(n_radii, V, dist_type, [],0,25,Weight_power); % max(r)=4 is good for 99% of weighted CDF for boltzmann 
% try [r_vect,p] = size_distribution_discrete_no_max(n_radii, V, dist_type,0.995, 0, 0); % Thershold instead of max
switch Nu  % Overriding the Weighting power for certain cases
    case -1/63 %exact spherical (Pedersen1997)
        Weight_power=6;
    case +1/18 %exact Debye Gaussian chain (Debye 1949)
        Weight_power=0;
    case -1/45 %exact shell
        Weight_power=4;
    case 11/225 %exact thin rod phi''=11/25, phi'''=1236/99225
        Weight_power=2;
    case 0.000666 %exact thin disk  phi''=0, phi'''=1/270
        Weight_power=4;
end
try [r_vect,p] = size_distribution_discrete_no_max(n_radii, V, dist_type,0.995, 0, Weight_power); % Thershold instead of max
catch
% [r_vect, p] = gaussian_discrete(n_radii, V); r_vect=r_vect+1;
fprintf('error in dist\n')
[r_vect,p] = size_distribution_discrete(n_radii, V, [],'normal',0,4,Weight_power);
end
distrp.r=r_vect*rg;
distrp.p=p;
F=zeros(size(qvr2));
for ii=1:n_radii
    if 0,     F=F+p(ii)*(ff(qvr*r_g*(r_vect(ii)))); %scattering of a simple ensemble
    else
        s = rg * (r_vect(ii));     % scalar scale
        s2 = s * s;                    % s^2
        s4 = s2 * s2;                  % s^4

        % compute exponent using precomputed qvr2 / qvr4
        % exponent = - (s^2/3)*qvr2 + 0.5*nu*(s^4)*qvr4;
        % compute as two scaled arrays added together
        exponent = -(s2/3) .* qvr2;       % MxM
        exponent = exponent + (0.5 * nu * s4) .* qvr4;
        if abs((1/nu-round(1/nu))-0.000666)<1e-6  %add another order
            %example: nu=1/(18+0.000666+0.0123e-8) -> phi'''=0.0123
            ordersix=1e8*(abs(1/nu-round(1/nu))-0.000666); %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
        end
        if abs(nu-0.000666)<1e-6  %add another order for nu=0 
            %example: nu=(0.000666+0.0123e-8) -> phi'''=0.0123
            ordersix=1e8*(abs(nu-0.000666)); %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
        end

        exponent(abs(-(s2/3) .* qvr2)< 1 * abs((0.5 * nu * s4) .* qvr4))=nan; % eliminate anomalies outside the guinier region

        if Nu==-1/63 %exact spherical (Pedersen1997)
            GF=sqrt(5/3); %factor from sphere's radius to r_g
            expo=log((3*(sin(qvr*s*GF)-GF*s*qvr.*cos(s*qvr*GF))./qvr./qvr2/s/s2/GF^3).^2);
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=6; 
        end
        if Nu==+1/18 %exact Debye Gaussian chain (Debye 1949)
            GF=1; %factor from sphere's radius to r_g

            expo=log(2*(exp(-(qvr*s*GF).^2)+(qvr*s*GF).^2-1)./(qvr*s*GF).^4);
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=0;
        end

        if Nu==-1/45 %exact shell
            GF=1; %factor from sphere's radius to r_g

            expo=2*log((sin(qvr*s*GF))./(qvr*s*GF));
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=4;
        end

        if Nu==11/225 %exact thin rod phi''=11/25, phi'''=1236/99225
            GF=sqrt(12); %factor from rod's length to r_g
            u=qvr*s*GF+10*eps;
            %             expo=log(2*sinint(u)./u-4*sin(u/2).^2./u.^2);
            % High speed version (for x > 0)
            Si = imag(expint(1i * u)) + pi/2;
            expo=log(2*Si./u-4*sin(u/2).^2./u.^2);
            expo(abs(u)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=2;
        end

        if Nu==11/225+0.000666 % approx thin rod phi''=11/25, phi'''=412/33075
            s = rg * (r_vect(ii));     % scalar scale
            s2 = s * s;                    % s^2
            s4 = s2 * s2;                  % s^4

            nu=11/225;

            % compute exponent using precomputed qvr2 / qvr4
            % exponent = - (s^2/3)*qvr2 + 0.5*nu*(s^4)*qvr4;
            % compute as two scaled arrays added together
            exponent = -(s2/3) .* qvr2;       % MxM
            exponent = exponent + (0.5 * nu * s4) .* qvr4;
            ordersix=412/33075; %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
            exponent(abs(-(s2/3) .* qvr2)< 1 * abs((0.5 * nu * s4) .* qvr4))=nan; % eliminate anomalies outside the guinier region
        end

        if Nu==0.000666 %exact thin disk  phi''=0, phi'''=1/270
            GF=sqrt(2); %factor from disk's radius to r_g
            u=qvr*s*GF;
            expo=log(2./u.^2.*(1-besselj(1,2*u)./u));
            expo(abs(u)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=4;
        end
        
        
%         F = F + p(ii) * exp(exponent);
        F = F + r_vect(ii).^Weight_power.*p(ii) * exp(exponent);


    end
end

% Delt=sqrt(V);
% F=ff(qvr*r_g*(1+Delt))+ff(qvr*r_g*(1-Delt)); %scattering of a simple ensemble
% F=F/2; % max F is 1
F=filter2(PSF0,F,"same");
%         F=F+Nois*rand(size(F));  % add noise
if noisemodel<0
    F=F*abs(Nois); % Nois is the number of photons at the peak
    F=F+sqrt(F).*randn(size(F));
    F=F/abs(Nois);


else
    %             F=F+Nois*(1-2*rand(size(F)));  % add not just positive noise
    F=F+Nois*randn(size(F));  % add gaussian noise
end
q_mat_x=qvx;
q_mat_y=qvy;
F(F<0)=0; %remove scarce negatives
I_mat=F;
end
%%
function kernel_out = makeKernelOddCentered(kernel_in)
% makeKernelOddCentered  Expand even-sized kernel to odd while keeping center.
%   kernel_out = makeKernelOddCentered(kernel_in)
%   If rows are even, expands to rows+1 by splitting each original row's
%   weights half/half between adjacent rows. Same for columns.
%
% Examples:
%   makeKernelOddCentered([1 1])        -> [0.5 1 0.5]
%   makeKernelOddCentered([1;1])       -> [0.5; 1; 0.5]
%   makeKernelOddCentered([1 2;3 4])   -> 3x3 result (center preserved)

kernel_out = kernel_in;

% If rows even: create rows+1 and distribute each row half/half
[r, c] = size(kernel_out);
if mod(r,2) == 0
    newr = r + 1;
    tmp = zeros(newr, c);
    % for each original row i, split into tmp(i,:) and tmp(i+1,:)
    for i = 1:r
        tmp(i, :)   = tmp(i, :)   + 0.5 * kernel_out(i, :);
        tmp(i+1, :) = tmp(i+1, :) + 0.5 * kernel_out(i, :);
    end
    kernel_out = tmp;
end

% If cols even: create cols+1 and distribute each column half/half
[r, c] = size(kernel_out);
if mod(c,2) == 0
    newc = c + 1;
    tmp = zeros(r, newc);
    % for each original col j, split into tmp(:,j) and tmp(:,j+1)
    for j = 1:c
        tmp(:, j)   = tmp(:, j)   + 0.5 * kernel_out(:, j);
        tmp(:, j+1) = tmp(:, j+1) + 0.5 * kernel_out(:, j);
    end
    kernel_out = tmp;
end
end
%%
function [x, p] = gaussian_discrete(N, V, alpha)
%GAUSSIAN_DISCRETE  Discrete zero-mean Gaussian with length N and variance V.
%   [x,p] = gaussian_discrete(N, V, alpha)
%     N      : number of points (integer >= 2)
%     V      : target variance (>0)
%     alpha  : (optional) shape parameter for probabilities; default 0.5
%              p_i ∝ exp(-alpha * u_i^2), with symmetric grid u
%
%   Returns:
%     x : Nx1 locations (symmetric about 0), s.t. sum((x-μ)^2 .* p) = V exactly
%     p : Nx1 probabilities (sum(p)=1), mean zero by symmetry
%
%   Works for small N (2..11) and large N. No root finding.

if nargin < 3 || isempty(alpha), alpha = 0.5; end
if ~(isscalar(N) && N==round(N) && N>=2), error('N must be integer >= 2'); end
if ~(isscalar(V) && V>=0), error('V must be positive'); end

if V==0
    x=zeros(N,1);
    p=ones(N,1)/N;
    return
end
% --- symmetric base grid u (unitless) ---
if mod(N,2)==1
    m = (N-1)/2;
    u = (-m:m).';                % includes 0
else
    m = N/2;
    v = ((0:m-1) + 0.5).';
    u = [-flipud(v); v];         % ±0.5, ±1.5, ...
end

% --- probabilities from Gaussian shape on u (independent of scale) ---
w = exp(-alpha * (u.^2));        % unnormalized
% enforce exact symmetry of p for numerical robustness
if mod(N,2)==1
    mid = (N+1)/2;
    for k = 1:mid-1
        wk = 0.5*(w(mid-k) + w(mid+k));
        w(mid-k) = wk; w(mid+k) = wk;
    end
else
    for k = 1:N/2
        wk = 0.5*(w(k) + w(N+1-k));
        w(k) = wk; w(N+1-k) = wk;
    end
end
p = w / sum(w);

% --- choose scale a so that variance equals V exactly ---
% With x = a*u and mean zero by symmetry:
% Var = E[x^2] = a^2 * E[u^2]_p  =>  a = sqrt(V / E[u^2]_p).
Eu2 = sum((u.^2) .* p);
a = sqrt(V / Eu2);

x = a * u;
end
%%
function K = bartlett2d(n, m, mode)
%BARTLETT2D  Normalized 2D triangular (Bartlett-like) kernel with nonzero edges.
%   K = bartlett2d(n, m)          % raised-edge triangular (edges > 0)
%   K = bartlett2d(n, m, 'zero')  % classic Bartlett (edges = 0)
%
%   No toolboxes required.

    if nargin < 3, mode = 'raised'; end

    if nargin == 1, m=n; end

    wy = tri1d(n, mode);   % column
    wx = tri1d(m, mode).'; % row

    K = wy * wx;           % separable outer product
    s = sum(K(:));
    if s > 0, K = K / s; end
end
%%
function w = tri1d(N, mode)
% 1D triangular window; 'raised' keeps nonzero endpoints.

    if N <= 0
        w = zeros(0,1); return
    elseif N == 1
        w = 1; return
    end

    idx = (0:N-1).';
    c   = (N-1)/2;                 % geometric center

    switch lower(mode)
        case 'zero'   % classic Bartlett (endpoints = 0 for N>=3)
            if N == 2
                % With only two taps, use flat weights (normalized later)
                w = [1; 1];
            else
                w = 1 - abs(idx - c) / c;
            end
        otherwise     % 'raised' (nonzero endpoints)
            d = (N+1)/2;           % larger denominator -> nonzero edges
            w = 1 - abs(idx - c) / d;
            w(w < 0) = 0;          % numerical safety for very small N
    end
end
%%
function [x, p] = size_distribution_discrete_old(N, Vrel, dist_type, dist_param, xmin, xmax)
%SIZE_DISTRIBUTION_DISCRETE
% Discrete positive-support size distributions (Tomchuk et al. J. Appl. Cryst. (2023). 56, 1099–1107 types),
% enforcing exact mean and variance by x-renormalization.
%
% [x,p] = size_distribution_discrete(N, Vrel, Mean, dist_type, dist_param, xmin)
%
% Inputs:
%   N          integer >=2
%   Vrel       relative variance: Var/Mean^2  (p = sqrt(Vrel))
%   Mean       desired mean (>0)- always 1
%   dist_type  {'normal','lognormal','schulz','exponential','boltzmann',
%               'triangular','uniform'}
%   dist_param optional parameters depending on distribution
%   xmin       minimal allowed x-value (default 0)
%   dist_param.do_not_redistribute : flag that indicates to adhere to linearly spaced
%   radii between xmin-xmax
%
% Outputs:
%   x  Nx1 radii, x >= xmin
%   p  Nx1 probabilities, sum=1
%
% The function chooses a base grid u>0, constructs pdf(u), normalizes to p,
% then enforces mean and variance *exactly* by affine transform of x.
%
% Author: ChatGPT (2025)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<3 || isempty(dist_type), dist_type='normal'; end
if nargin<4, dist_param=[]; end
if nargin<5 || isempty(xmin), xmin=0; end
if nargin<6 || isempty(xmax), xmax=4; end
do_not_redistribute=1; %default value
if isfield(dist_param,'do_not_redistribute'),
    if dist_param.do_not_redistribute==~do_not_redistribute;
        do_not_redistribute=~do_not_redistribute; 
    end
end
Mean=1;
if Vrel<=0/N^2  % negligible variance has trivial distribution
    x=Mean*ones(N,1);
    p=x/sum(x(:));
    return
end

if ~(isscalar(N) && N>=2 && N==round(N))
    error('N must be integer >=2');
end
if ~(isscalar(Vrel) && Vrel>=0)
    error('Vrel must be >=0');
end
if ~(isscalar(Mean) && Mean>0)
    error('Mean must be >0');
end
if ~(isscalar(xmin) && xmin>=0)
    error('xmin must be >=0');
end

% absolute variance requested:
VarAbs = (sqrt(Vrel)*Mean)^2;

% ----------------------------------------------------------------------
% 1. base grid (positive)
% ----------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ADAPTIVE GRID u FOR LOW N
% - Build a fine temporary grid u_fine
% - Compute fine pdf w_fine (unnormalized)
% - Determine the region containing >= (1 - 1/N) of the mass
% - Select N points uniformly within that region
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Step 1: fine grid over a wide range ---
% Use 1000 points between very small and several times the Mean
u_fine = linspace(xmin, xmax, 200).';

% --- Step 2: compute the unnormalized PDF at the fine grid ---
switch lower(dist_type)
    case 'normal'
        sigma = sqrt(Vrel)*Mean;
        w_fine = exp(-(u_fine - Mean).^2/(2*sigma^2));

    case 'lognormal'
        if isempty(dist_param)
            s = sqrt(log(1+Vrel));
        else
            s = dist_param;
        end
        R0 = Mean / sqrt(1+exp(s^2)-1);
        w_fine = exp(-(log(u_fine./R0)).^2/(2*s^2)) ./ u_fine;

    case 'schulz'
        if isempty(dist_param)
            p = sqrt(Vrel); z = 1/p^2 - 1;
        else
            z = dist_param;
        end
        w_fine = u_fine.^z .* exp(-(z+1)*u_fine/Mean);

    case 'exponential'
        if isempty(dist_param), lambda = 1/Mean;
        else, lambda = dist_param; end
        w_fine = lambda*exp(-lambda*u_fine);

    case 'boltzmann'
        if isempty(dist_param), b = Mean*sqrt(Vrel/2);
        else, b = dist_param; end
        w_fine = exp(-sqrt(2)*abs(u_fine-Mean)/b);

    case 'triangular'
        if isempty(dist_param)
            sigma = sqrt(VarAbs);
        else
            sigma = dist_param;
        end
        L = sigma*sqrt(6);
        w_fine = max(0, 1 - abs(u_fine-Mean)/L);

    case 'uniform'
        if isempty(dist_param)
            sigma = sqrt(VarAbs);
        else
            sigma = dist_param;
        end
        L = sigma*sqrt(3);
        w_fine = (u_fine >= Mean-L & u_fine <= Mean+L);

    otherwise
        error('Unknown distribution type.');
end

% Avoid negativity
w_fine = max(w_fine,0);

% --- Step 3: find cumulative mass and the effective high-mass region ---
W = w_fine / sum(w_fine);     % normalized
C = cumsum(W);                % cumulative

% Cut off the lowest (1/N) mass from each tail
cut = 1/N;

% indices where C ∈ [cut, 1-cut]
i1 = find(C >= cut, 1, 'first');
i2 = find(C <= 1-cut, 1, 'last');

if isempty(i1), i1 = 1; end
if isempty(i2), i2 = length(u_fine); end

u_lo = u_fine(i1);
u_hi = u_fine(i2);

% --- Step 4: final u-grid (N points) evenly spaced over the truncated region ---
u = linspace(u_lo, u_hi, N+1).';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if do_not_redistribute
    u = linspace(xmin,xmax, N+1).';   % strictly positive; scale irrelevant
end
u(1)=[];
% ----------------------------------------------------------------------
% 2. unnormalized PDF on u
% ----------------------------------------------------------------------
switch lower(dist_type)

    case 'normal'
        % Eq. 17
        w_fine = exp(-(u_fine - Mean).^2 / (2 * sigma^2));
        
    case 'lognormal'
        % Eq. 10 & 11a
        s = sqrt(log(1 + Vrel));
        m = log(Mean) - 0.5 * s^2;
        w_fine = (1 ./ u_fine) .* exp(-(log(u_fine) - m).^2 / (2 * s^2));
        
    case 'schulz'
        % Eq. 15: z = 1/p^2 - 1
        z = 1/Vrel - 1;
        w_fine = u_fine.^z .* exp(-(z + 1) * u_fine / Mean);
        
    case 'exponential'
        % Eq. 16: p is fixed at 1
        lambda = 1 / Mean;
        w_fine = exp(-lambda * u_fine);
        
    case 'boltzmann'
        % Eq. 18: First-order exponential
        w_fine = exp(-sqrt(2) * abs(u_fine - Mean) / sigma);
        
    case 'triangular'
        % Eq. 19: Width is sigma * sqrt(6)
        L = sigma * sqrt(6);
        w_fine = max(0, 1 - abs(u_fine - Mean) / L);
        
    case 'uniform'
        % Eq. 20: Width is sigma * sqrt(3)
        L = sigma * sqrt(3);
        w_fine = (u_fine >= Mean - L & u_fine <= Mean + L);

    otherwise
        error('Unknown distribution type');
end

% ----------------------------------------------------------------------
% 3. normalize PDF → probabilities
% ----------------------------------------------------------------------
w = max(w,0);
p = w / sum(w);

% ----------------------------------------------------------------------
% 4. initial x = u (will be rescaled)
% ----------------------------------------------------------------------
x = u;

% ----------------------------------------------------------------------
% 5. enforce mean and variance by affine transform of x
% ----------------------------------------------------------------------
mu  = sum(p .* x);
var = sum(p .* (x - mu).^2);

if var == 0
    error('Computed variance is zero; distribution too narrow.');
end

% We want:
%   x_new = a*(x - mu) + Mean
%   a = sqrt(VarAbs / var)
if ~strcmp(dist_type,'exponential')
    a = sqrt(VarAbs / var);
    x = a*(x - mu) + Mean;
end
% ----------------------------------------------------------------------
% 6. enforce xmin
% ----------------------------------------------------------------------
if 1 || xmin > 0
    minx = min(x);
    if minx < xmin
        x = x + (xmin - minx);
    end
end

% ----------------------------------------------------------------------
% 7. verify & renormalize again if needed
% ----------------------------------------------------------------------
mu2  = sum(p .* x);
var2 = sum(p .* (x - mu2).^2);

% If drift >1%, reapply exact fix
if ~strcmp(dist_type,'exponential')
    if abs(mu2-Mean)/Mean > 0.01 || abs(var2-VarAbs)/VarAbs > 0.01
        % exact fix:
        a2 = sqrt(VarAbs / var2);
        x = a2*(x - mu2) + Mean;

        % enforce xmin again
        minx = min(x);
        if minx < xmin
            x = x + (xmin - minx);
        end
    end
end

% last check
mu2  = sum(p .* x);
var2 = sum(p .* (x - mu2).^2);
if ~strcmp(dist_type,'exponential')
    if abs(mu2-Mean)/Mean > 0.01 || abs(var2-VarAbs)/VarAbs > 0.01
        warning('Distribution mean or var was not reached')
    end
end
figure(432)
plot(x,p,'-','DisplayName',num2str(Vrel));
% disp(sum(p))
hold on
end

%%
function [x, p] = size_distribution_discrete(N, Vrel, dist_type, dist_param, xmin, xmax, weight_power)
% SIZE_DISTRIBUTION_DISCRETE 
% Discretizes size distributions using Importance Sampling based on 
% Tomchuk et al. (2023) definitions.
%
% [x, p] = size_distribution_discrete(N, Vrel, dist_type, dist_param, xmin, xmax, weight_power)

% 1. Handle Defaults
if nargin < 3 || isempty(dist_type), dist_type = 'normal'; end
if nargin < 4, dist_param = []; end
if nargin < 5 || isempty(xmin), xmin = 0; end
if nargin < 6 || isempty(xmax), xmax = 5; end 
if nargin < 7 || isempty(weight_power), weight_power = 0; end

Mean = 1; % Normalized mean per user request
% Per Tomchuk Eq. 2: p = sigma/Mean. Thus sigma = Mean * sqrt(Vrel)
sigma = sqrt(Vrel) * Mean; 

% 2. Create high-resolution fine grid
u_fine = linspace(xmin, xmax, 2000).';

% 3. Calculate PDF on fine grid (Tomchuk Table 1 & Section 2.2)
switch lower(dist_type)
    case 'normal'
        % Eq. 17
        w_fine = exp(-(u_fine - Mean).^2 / (2 * sigma^2));
        
    case 'lognormal'
        % Eq. 10 & 11a
        s = sqrt(log(1 + Vrel));
        m = log(Mean) - 0.5 * s^2;
        w_fine = (1 ./ u_fine) .* exp(-(log(u_fine) - m).^2 / (2 * s^2));
        
    case 'schulz'
        % Eq. 15: z = 1/p^2 - 1
        z = 1/Vrel - 1;
        w_fine = u_fine.^z .* exp(-(z + 1) * u_fine / Mean);
        
    case 'exponential'
        % Eq. 16: p is fixed at 1
        lambda = 1 / Mean;
        w_fine = exp(-lambda * u_fine);
        
    case 'boltzmann'
        % Eq. 18: First-order exponential
        w_fine = exp(-sqrt(2) * abs(u_fine - Mean) / sigma);
        
    case 'triangular'
        % Eq. 19: Width is sigma * sqrt(6)
        L = sigma * sqrt(6);
        w_fine = max(0, 1 - abs(u_fine - Mean) / L);
        
    case 'uniform'
        % Eq. 20: Width is sigma * sqrt(3)
        L = sigma * sqrt(3);
        w_fine = (u_fine >= Mean - L & u_fine <= Mean + L);
        
    otherwise
        error('Unknown distribution type: %s', dist_type);
end

% Numerical cleaning
w_fine(isnan(w_fine)) = 0;
w_fine = max(w_fine, 0);

% 4. Importance Weighting (Inverse Transform Sampling)
% weight_power: 0 (Number), 2 (Rods), 4 (Shells), 6 (Spheres)
w_weighted = w_fine .* (u_fine .^ weight_power);

% Normalize and compute Cumulative Distributions
% Tiny epsilon added to linspace to ensure strict monotonicity for interp1
C_weight = cumsum(w_weighted) / sum(w_weighted) + linspace(0, 1e-12, length(u_fine)).';
C_number = cumsum(w_fine) / sum(w_fine);

% 5. Partition weighted CDF into N equal segments
edges_weight = linspace(0, 1, N + 1);

% Map importance edges back to radii (r)
r_edges = interp1(C_weight, u_fine, edges_weight, 'linear', 'extrap');

% 6. Calculate discrete subpopulation properties
x = zeros(N, 1);
p = zeros(N, 1);

for i = 1:N
    % Subpopulation radius is the midpoint of the bin
    x(i) = (r_edges(i) + r_edges(i+1)) / 2;
    
    % Probability mass is the change in the Number-CDF across the bin edges
    p_high = interp1(u_fine, C_number, r_edges(i+1), 'linear', 'extrap');
    p_low  = interp1(u_fine, C_number, r_edges(i),   'linear', 'extrap');
    p(i) = p_high - p_low;
end

% Final renormalization to ensure sum(p) == 1
p = max(p, 0); 
p = p / sum(p);
fprintf('%s mean=%f ',dist_type, sum(p.*x))
fprintf('dV=%f\n',sum(p.*x.^2)./sum(p.*x).^2-1-Vrel)
% Optional: Plotting for verification
% figure (432); 
% subplot(211); plot(u_fine, w_fine/sum(w_fine),DisplayName=dist_type); hold on; plot(x, p, '+-','DisplayName',dist_type);
% ylabel('Probability'); legend (Location="bestoutside")
% subplot (212); plot(u_fine, C_weight, '-','DisplayName',dist_type,'LineWidth',3);legend (Location="bestoutside")
% ylabel('Weighted Cumulative'); hold on
end
%%
function [x, p] = size_distribution_discrete_no_max(N, Vrel, dist_type, threshold, xmin, weight_power)
% Revised for exact moment matching using Center-of-Mass binning
if nargin < 3 || isempty(dist_type), dist_type = 'normal'; end
if nargin < 4 || isempty(threshold), threshold = 0.9995; end
if nargin < 5 || isempty(xmin), xmin = 0; end
if nargin < 6 || isempty(weight_power), weight_power = 0; end

Mean = 1; 
sigma = sqrt(Vrel) * Mean; 

% 1. Adaptive xmax to capture the 'mass' (Tomchuk Eq 13 context)
if strcmpi(dist_type, 'lognormal') || strcmpi(dist_type, 'schulz')|| strcmpi(dist_type, 'boltzmann')
    % Log-normal distributions stretch exponentially under high-power weighting
    s_tmp = sqrt(log(1 + Vrel));
    m_weighted = log(Mean) - 0.5 * s_tmp^2 + weight_power * s_tmp^2;
    % Set ceiling to capture 5 standard deviations in the log-domain
    temp_xmax = exp(m_weighted + 5 * s_tmp);
else
    temp_xmax = Mean + 20 * sigma;
end 
u_fine = linspace(xmin, temp_xmax, 10000).'; % High-res grid for integration

% 2. PDF Calculation (Tomchuk et al. 2023 definitions)
switch lower(dist_type)
    case 'normal'
        w_fine = exp(-(u_fine - Mean).^2 / (2 * sigma^2)); % 
    case 'lognormal'
        s = sqrt(log(1 + Vrel)); % 
        m = log(Mean) - 0.5 * s^2;
        w_fine = (1 ./ u_fine) .* exp(-(log(u_fine) - m).^2 / (2 * s^2)); % 
    case 'schulz'
        % Eq. 15: z = 1/p^2 - 1
        z = 1/Vrel - 1;
        
        % Compute in the log-domain to stabilize large powers of z
        ln_w = z * log(u_fine) - (z + 1) * u_fine / Mean;
        
        % Protection for u_fine = 0 case (log(0) = -Inf, -Inf * z = -Inf)
        ln_w(isnan(ln_w)) = -inf;
        
        % Affine peak shift transformation: sets maximum value to 0 
        % This guarantees exp(0) = 1 at the peak, preventing underflow/overflow cascades
        ln_w = ln_w - max(ln_w);
        w_fine = exp(ln_w);
    case 'boltzmann'
        w_fine = exp(-sqrt(2) * abs(u_fine - Mean) / sigma); % 
    case 'triangular'
        L = sigma * sqrt(6); % 
        w_fine = max(0, 1 - abs(u_fine - Mean) / L); % 
    case 'uniform'
        L = sigma * sqrt(3); % 
        w_fine = (u_fine >= Mean - L & u_fine <= Mean + L); % 
end

w_fine(isnan(w_fine) | isinf(w_fine)) = 0;
w_fine = w_fine / sum(w_fine);

% 3. Calculate Cumulative Distributions
% Use 'weighted' CDF if weight_power > 0 (Importance Sampling)
w_weighted = w_fine .* (u_fine .^ weight_power);
C_weight = cumsum(w_weighted) / sum(w_weighted);
C_number = cumsum(w_fine) / sum(w_fine);

% Ensure strict monotonicity for interp1
C_weight = C_weight + linspace(0, 10e-12, length(C_weight)).';

% 4. Exact Partitioning
edges_prob = linspace(0, threshold, N + 1);
r_edges = interp1(C_weight, u_fine, edges_prob, 'linear', 'extrap');

x = zeros(N, 1);
p = zeros(N, 1);

% Cumulative product for center-of-mass calculation: Integral of (r * PDF)
C_moment1 = cumsum(w_fine .* u_fine);

for i = 1:N
    % Find indices in fine grid falling within this bin
    idx = u_fine >= r_edges(i) & u_fine < r_edges(i+1);
    
    % Probability of this bin (Exact Number Fraction)
    p_high = interp1(u_fine, C_number, r_edges(i+1), 'linear', 'extrap');
    p_low  = interp1(u_fine, C_number, r_edges(i),   'linear', 'extrap');
    p(i) = p_high - p_low;
    
    % Center of Mass x(i): Integral(r*P(r) dr) / Integral(P(r) dr)
    m1_high = interp1(u_fine, C_moment1, r_edges(i+1), 'linear', 'extrap');
    m1_low  = interp1(u_fine, C_moment1, r_edges(i),   'linear', 'extrap');
    x(i) = (m1_high - m1_low) / p(i);
end

% 5. THE AFFINE FIX: Force exactly Mean=1 and Var=Vrel
curr_mu  = sum(p .* x);
curr_var = sum(p .* (x - curr_mu).^2);

% Scale and shift radii
scale_factor = sqrt(Vrel / curr_var);
x = scale_factor * (x - curr_mu) + Mean;

% Ensure physical validity (no negative radii)
x = max(x, xmin);
% Final cleanup: ensure sum(p) = 1 
p = p / sum(p);
% fprintf('%s mean=%f, dV=%f, max(r)=%f\n',dist_type, sum(p.*x),sum(p.*x.^2)./sum(p.*x).^2-1-Vrel,max(x))
% fprintf('')
% Optional: Plotting for verification
% figure (432); 
% subplot(211); plot(u_fine, w_fine/sum(w_fine),DisplayName=dist_type); hold on; plot(x, p, '+-','DisplayName',dist_type);
% ylabel('Probability'); legend (Location="bestoutside")
% subplot (212); plot(u_fine, C_weight, '-','DisplayName',dist_type,'LineWidth',2);legend (Location="bestoutside")
% ylabel('Weighted Cumulative'); hold on
% plot(x, interp1(u_fine, C_weight,x), '+','HandleVisibility','off')

end