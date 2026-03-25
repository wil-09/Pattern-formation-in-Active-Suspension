% ===============================================================%
% System:
%   n_t + div(u n) = Δ n - μ0 γ0 γ ∂_y n - χ0 div(n ∇c)
%   c_t + div(u c) = α Δ c - β n c
%   ν u_xx + u_yy + (ν-1) u_xy - u + χ1 c_x - (Pe / (1+n)^2) n_x = 0
%   (ν-1) u_xy + ν v_yy + v_xx - v + χ1 c_y - (Pe / (1+n)^2) n_y - μ0 γ0 n = 0
%
% Numerical method: 2D pseudo-spectral (FFT), periodic BCs,
% IMEX (diffusion implicit; advection/chemotaxis/reaction explicit).
%% ===============================================================

clear; clc; close all;

%% -------------------- Parameters --------------------
% Domain & grid
Lx = 50;                 % domain length in x
Ly = 50;                 % domain length in y
Nx = 256*2;                % grid points in x (use power of 2 for fast FFT)
Ny = 256*2;                % grid points in y
dx = Lx / Nx;            % grid spacing (not used directly in spectral ops)
dy = Ly / Ny;

% Time stepping
dt      = 0.01;          % time step (reduce if simulation becomes unstable)
tFinal  = 200;            % final time
plotEvery = 1000;          % visualization frequency (in time steps)

% Physical / model coefficients (tune to obtain plume-like patterns)
alpha = 0.25;             % diffusion of c
beta  = 1;                % reaction rate in c_t (consumption by n)
chi0  = 10;                % chemotactic strength in n-equation
Pe    = 1.2;              % buoyancy-like coupling from n gradients to flow
mu0   = 0.05;              % coupling constant
nu    = 4;                % anisotropy parameter in elliptic system
gamma0 = 0.7;             % coupling constant. The first figure saved were produced for gamma = 0.7
gamma  = 0.8;             % vertical drift magnitude factor for n
chi1   = chi0*gamma;

% Numerical hygiene
clipNneg = true;         % clip n to be >= 0 each step (prevents undershoot)
dealise  = true;         % apply 2/3-rule dealiasing mask on nonlinear terms

rng(2);                  % random seed (for reproducible perturbations)

%% -------------------- Grids & Fourier operators --------------------
% Physical grids
x = linspace(0, Lx, Nx+1); x(end) = [];  % periodic
y = linspace(0, Ly, Ny+1); y(end) = [];
[X, Y] = meshgrid(x, y);

% Wavenumbers consistent with unshifted fft2 order
kx = (1*2*pi/Lx) * [0:(Nx/2-1), -Nx/2:-1];
ky = (1*2*pi/Ly) * [0:(Ny/2-1), -Ny/2:-1];
[KX, KY] = meshgrid(kx, ky);
K2 = KX.^2 + KY.^2;

% IMEX denominators for diffusion
denN = 1 + dt * K2;            % for n (diffusion coefficient = 1)
denC = 1 + dt * alpha * K2;    % for c

% Dealiasing (2/3 rule) mask
if dealise
    kx_cut = (2/3) * (max(abs(kx)));
    ky_cut = (2/3) * (max(abs(ky)));
    dealiasMask = (abs(KX) <= kx_cut) & (abs(KY) <= ky_cut);
else
    dealiasMask = true(size(KX));
end

%% -------------------- Initial conditions --------------------
% Start with a localized blob of n plus small random perturbations
n0 = 1.0;
sigma_n = 2.0;
x0 = Lx*0.5;
y0 = Ly*0.65;  % off-center to favor plumes in y
n = n0 * exp(-((X-x0).^2 + (Y-y0).^2)/(2*sigma_n^2));
% n = n .* (1 + 0.02*randn(size(n)));
n = n0 + 0.02*randn(size(n));
% n = n0 + 0.02.*cos(KX.*X + KY.*Y);
n = max(n, 0);

% Initial c field: uniform background with small perturbations
c0 = 1.0;
c  = c0 + 0.02*randn(size(n));
% c = c0 + 0.02.*cos(KX.*X + KY.*Y);

% Pre-allocate u, v
u = zeros(Ny, Nx);
v = zeros(Ny, Nx);

% In the Fourrier space, the velocity field u and v are calculated as
% u_f = c_1u.*n + c_2u.*c + c_3u.*fftn(ifftn(i*kxm.*n)./(1 + ifftn(n)).^2) + c_4u.*fftn(ifftn(i*kxm.*n)./(1 + ifftn(n)).^2);
% v_f = c_1v.*n + c_2v.*c + c_3v.*fftn(ifftn(i*kym.*n)./(1 + ifftn(n)).^2) + c_4v.*fftn(ifftn(i*kym.*n)./(1 + ifftn(n)).^2);

%% -------------------- Visualization setup --------------------
% Precompute quiver grid indices (subsample for performance/clarity)
quiverSub = max(1, round(Nx/30));
Xq = X(1:quiverSub:end, 1:quiverSub:end);
Yq = Y(1:quiverSub:end, 1:quiverSub:end);

fig = figure('Color','w');
layout = tiledlayout(fig,2,2,'Padding','compact','TileSpacing','compact');

ax1 = nexttile(layout, 1);
h_n = imagesc(x, y, n, 'Parent', ax1);
axis(ax1, 'image'); ax1.YDir = 'normal';
ax1.Colormap = turbo;
cb_n = colorbar(ax1);
try
    ax1.Title.String = 'n(x,y,t)';
catch
    % Some MATLAB versions may throw listener callback errors during figure setup.
end
hold(ax1, 'on');
q4 = quiver(ax1, Xq, Yq, zeros(size(Xq)), zeros(size(Yq)), 'k', 'AutoScale','on', 'AutoScaleFactor', 1.2);
hold(ax1, 'off');

ax2 = nexttile(layout, 2);
h_c = imagesc(x, y, c, 'Parent', ax2);
axis(ax2, 'image'); ax2.YDir = 'normal';
ax2.Colormap = turbo;
cb_c = colorbar(ax2);
try
    ax2.Title.String = 'c(x,y,t)';
catch
    % Some MATLAB versions may throw listener callback errors during figure setup.
end

ax3 = nexttile(layout, 3); % Vorticity highlights defects in the flow field. % Use a sub-sampled quiver overlay to show velocity vectors.
ax3.Colormap = turbo;
% Initial vorticity map
vort = zeros(size(n));   sn = zeros(size(n)); 
ss   = zeros(size(n));
h_vort = imagesc(x, y, vort, 'Parent', ax3);
axis(ax3, 'image'); ax3.YDir = 'normal';
cb_vort = colorbar(ax3);
try
    ax3.Title.String = 'W(x,y,t)';
catch
    % Some MATLAB versions may throw listener callback errors during figure setup.
end
hold(ax3, 'on');
q3 = quiver(ax3, Xq, Yq, zeros(size(Xq)), zeros(size(Yq)), 'k', 'AutoScale','on', 'AutoScaleFactor', 1.2);
hold(ax3, 'off');

ax4 = nexttile(layout, 4);
ax4.Colormap = hsv;
thetaField = zeros(size(n));
h_theta = imagesc(x, y, thetaField, 'Parent', ax4);
axis(ax4, 'image'); ax4.YDir = 'normal';
cb_theta = colorbar(ax4);
try
    ax4.Title.String = 'Director \theta(x,y,t) + defects';
catch
    % Some MATLAB versions may throw listener callback errors during figure setup.
end
hold(ax4, 'on');
q = quiver(ax4, Xq, Yq, zeros(size(Xq)), zeros(size(Yq)), 'k', 'AutoScale','on', 'AutoScaleFactor', 1.0);
hPlus = scatter(ax4, nan, nan, 36, 'r', 'filled', 'MarkerEdgeColor', 'k');
hMinus = scatter(ax4, nan, nan, 36, 'c', 'filled', 'MarkerEdgeColor', 'k');
trackLineHandles = [];
hold(ax4, 'off');
drawnow;

% Defect-core tracking state
tracks = struct('id', {}, 'sign', {}, 'x', {}, 'y', {}, 't', {}, 'lastFrame', {});
nextTrackId = 1;
trackMaxDist = 1.5;  % physical distance threshold for frame-to-frame linking
trackMaxGap = 2;     % maximum skipped plotted frames for keeping a track alive
frameCount = 0;

%% -------------------- Time integration loop --------------------
nSteps = ceil(tFinal / dt);
for step = 1:nSteps
    % ---- FFTs of current fields ----
    N_hat = fft2(n);
    C_hat = fft2(c);

    % ---- Spectral gradients ----
    nx = real(ifft2(1i*KX .* N_hat));     ny = real(ifft2(1i*KY .* N_hat));
    cx = real(ifft2(1i*KX .* C_hat));     cy = real(ifft2(1i*KY .* C_hat));

    % ---- Build sources for (u,v) elliptic system ----
    % Sx = Pe/(1+n)^2 * n_x ;
    S_hat = fft2(Pe./(1 + n));
    U_hat = 1i*KX.*(chi1*C_hat + S_hat)./(1 + nu*K2) + mu0*gamma0*KX.*KY*(nu - 1).*N_hat./((1 + K2).*(1 + nu*K2));
    V_hat = 1i*KY.*(chi1*C_hat + S_hat)./(1 + nu*K2) - mu0*gamma0*(1 + nu*KX.^2 + KY.^2).*N_hat./((1 + K2).*(1 + nu*K2));

    % Zero out any NaN/Inf (e.g., pathological division; should not happen)
    U_hat(~isfinite(U_hat)) = 0;
    V_hat(~isfinite(V_hat)) = 0;

    if dealise
        U_hat = U_hat .* dealiasMask;
        V_hat = V_hat .* dealiasMask;
    end

    % Back to physical space
    u = real(ifft2(U_hat));
    v = real(ifft2(V_hat));

    % ---- Build explicit RHS terms for n and c ----
    % Advection: div(u n) and div(u c)
    un = u .* n;
    vn = v .* n;
    uc = u .* c;
    vc = v .* c;

    div_un = real(ifft2(1i*KX .* fft2(un) + 1i*KY .* fft2(vn)));
    div_uc = real(ifft2(1i*KX .* fft2(uc) + 1i*KY .* fft2(vc)));
    nablan = real(ifft2(-K2.*N_hat)); nablac = real(ifft2(-K2.*C_hat)); 

    % Chemotaxis: div( n ∇c ) = ∂x(n cx) + ∂y(n cy)
    n_cx = n .* cx;
    n_cy = n .* cy;
    div_n_gradc = real(ifft2(1i*KX .* fft2(n_cx) + 1i*KY .* fft2(n_cy)));

    % Vertical drift term in n: - μ0 γ0 γ ∂_y n
    drift_n = - (mu0 * gamma0 * gamma) * ny;

    % RHS for n (explicit part)
    RHSn = nablan - div_un + drift_n - chi0 * div_n_gradc;

    % RHS for c (explicit part)
    RHSc = alpha * nablac - div_uc - beta * (n .* c) + 1 * beta * c;

    % Optional dealiasing on RHS
    if dealise
        RHSn_hat = fft2(RHSn) .* dealiasMask;
        RHSc_hat = fft2(RHSc) .* dealiasMask;
    else
        RHSn_hat = fft2(RHSn);
        RHSc_hat = fft2(RHSc);
    end

    % ---- IMEX update (diffusion implicit) ----
    N_hat_new = (N_hat + dt * RHSn_hat) ./ denN;
    C_hat_new = (C_hat + dt * RHSc_hat) ./ denC;

    if dealise
        N_hat_new = N_hat_new .* dealiasMask;
        C_hat_new = C_hat_new .* dealiasMask;
    end

    n = real(ifft2(N_hat_new));
    c = real(ifft2(C_hat_new));

    % Hygiene: prevent small negative values from Gibbs oscillations
    if clipNneg
        n = max(n, 0);
    end

    % ---- Visualization ----
    if mod(step, plotEvery) == 0 || step == 1 || step == nSteps
        frameCount = frameCount + 1;

        % Ensure the figure is active
        figure(fig);

        % Update n and c fields
        h_n.CData = n;
        try
            ax1.Title.String = sprintf('n(x,y,t),  t = %.2f', step*dt);
        catch
            % ignore graphics update errors
        end
        q4.UData = u(1:quiverSub:end, 1:quiverSub:end);
        q4.VData = v(1:quiverSub:end, 1:quiverSub:end);

        h_c.CData = c;
        try
            ax2.Title.String = sprintf('c(x,y,t),  t = %.2f', step*dt);
        catch
            % ignore graphics update errors
        end

        % Update vorticity (defects) and quiver overlay
        vx = real(ifft2(1i*KX .* fft2(v))); vy = real(ifft2(1i*KY .* fft2(v)));
        ux = real(ifft2(1i*KX .* fft2(u))); uy = real(ifft2(1i*KY .* fft2(u)));
        vort = vx - uy; sn = ux - vy; ss = vx + uy;

        % Director/orientation extracted from velocity field
        thetaField = mod(atan2(v, u), pi);

        % Winding-number defect detection on director field
        [xDef, yDef, qDef] = detectWindingDefects(thetaField, x, y);

        % Track defect cores over time
        [tracks, nextTrackId] = updateDefectTracks(tracks, nextTrackId, xDef, yDef, qDef, step*dt, frameCount, trackMaxDist, trackMaxGap);
        
        % Symmetric color scaling around zero to highlight vortices/defects
        vmax = max(abs(vort(:)));
        if vmax > 0
            caxis(ax3, [-vmax, vmax]);
        end
        
        h_vort.CData = vort;
        try
            ax3.Title.String = sprintf('W(x,y,t), t = %.2f', step*dt);
        catch
            % ignore graphics update errors
        end
        % Update quiver arrow vectors (sub-sampled for clarity)
        q3.UData = u(1:quiverSub:end, 1:quiverSub:end);
        q3.VData = v(1:quiverSub:end, 1:quiverSub:end);
        
        % Update director field, defect cores, and trajectories
        h_theta.CData = thetaField;
        caxis(ax4, [0, pi]);
        try
            ax4.Title.String = sprintf('Director field + defects,  t = %.2f', step*dt);
        catch
        end

        % Director glyphs (unit vectors from orientation)
        thetaSub = thetaField(1:quiverSub:end, 1:quiverSub:end);
        q.UData = cos(thetaSub);
        q.VData = sin(thetaSub);

        posMask = qDef > 0;
        negMask = qDef < 0;
        hPlus.XData = xDef(posMask);
        hPlus.YData = yDef(posMask);
        hMinus.XData = xDef(negMask);
        hMinus.YData = yDef(negMask);

        if ~isempty(trackLineHandles)
            validHandles = isgraphics(trackLineHandles);
            delete(trackLineHandles(validHandles));
        end
        trackLineHandles = [];
        hold(ax4, 'on');
        for k = 1:numel(tracks)
            if numel(tracks(k).x) > 1
                if tracks(k).sign > 0
                    lineColor = [1, 0, 0];
                else
                    lineColor = [0, 0.7, 0.9];
                end
                trackLineHandles(end+1) = plot(ax4, tracks(k).x, tracks(k).y, '-', 'Color', lineColor, 'LineWidth', 1.2);
            end
        end
        hold(ax4, 'off');

        drawnow;
    end
end

defectTracks = tracks;

% disp('Simulation complete.');

function [xDef, yDef, qDef] = detectWindingDefects(thetaField, x, y)
% Detect +/- 1/2 topological defects using winding numbers on 2*theta.

theta2 = 2 * thetaField;

% Plaquette corner values
t00 = theta2(1:end-1, 1:end-1);
t10 = theta2(1:end-1, 2:end);
t11 = theta2(2:end, 2:end);
t01 = theta2(2:end, 1:end-1);

% Wrapped phase increments around each plaquette
d1 = wrapPi(t10 - t00);
d2 = wrapPi(t11 - t10);
d3 = wrapPi(t01 - t11);
d4 = wrapPi(t00 - t01);

% Director charge q = sum(d(2*theta)) / (4*pi)
charge = (d1 + d2 + d3 + d4) / (4*pi);

% Keep robust detections near +/-1/2
mask = abs(charge) > 0.25;
qRaw = charge(mask);
qDef = 0.5 * sign(qRaw);

% Cell-center coordinates
xc = x(1:end-1) + 0.5 * (x(2) - x(1));
yc = y(1:end-1) + 0.5 * (y(2) - y(1));
[XC, YC] = meshgrid(xc, yc);
xDef = XC(mask);
yDef = YC(mask);
end

function [tracks, nextTrackId] = updateDefectTracks(tracks, nextTrackId, xDef, yDef, qDef, tNow, frameIdx, maxDist, maxGap)
% Nearest-neighbor tracking with sign consistency.

usedTrack = false(1, numel(tracks));

for i = 1:numel(xDef)
    thisSign = qDef(i);
    bestTrack = 0;
    bestDist = inf;

    for k = 1:numel(tracks)
        if usedTrack(k)
            continue;
        end
        if tracks(k).sign ~= thisSign
            continue;
        end
        if frameIdx - tracks(k).lastFrame > maxGap
            continue;
        end

        dxk = xDef(i) - tracks(k).x(end);
        dyk = yDef(i) - tracks(k).y(end);
        dist = hypot(dxk, dyk);

        if dist < bestDist
            bestDist = dist;
            bestTrack = k;
        end
    end

    if bestTrack > 0 && bestDist <= maxDist
        tracks(bestTrack).x(end+1) = xDef(i);
        tracks(bestTrack).y(end+1) = yDef(i);
        tracks(bestTrack).t(end+1) = tNow;
        tracks(bestTrack).lastFrame = frameIdx;
        usedTrack(bestTrack) = true;
    else
        newTrack.id = nextTrackId;
        newTrack.sign = thisSign;
        newTrack.x = xDef(i);
        newTrack.y = yDef(i);
        newTrack.t = tNow;
        newTrack.lastFrame = frameIdx;
        tracks(end+1) = newTrack;
        usedTrack(end+1) = true;
        nextTrackId = nextTrackId + 1;
    end
end
end

function a = wrapPi(a)
% Wrap angle to [-pi, pi].
a = atan2(sin(a), cos(a));
end
