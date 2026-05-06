"""
Plume-forming model with:
  - Polar (+/-1) defect detection from flow director
  - Kinetic energy diagnostics
  - Defect density and correlation analysis (scatter + lagged xcorr)
  
System:
  n_t + div(u n) = Δ n - μ0 γ0 γ ∂_y n - χ0 div(n ∇c)
  c_t + div(u c) = α Δ c - β n c
  ν u_xx + u_yy + (ν-1) u_xy - u + χ1 c_x - (Pe/(1+n)^2) n_x = 0
  (ν-1) u_xy + ν v_yy + v_xx - v + χ1 c_y
       - (Pe/(1+n)^2) n_y - μ0 γ0 n = 0

Numerical method: 2D pseudo-spectral (FFT), periodic BCs,
IMEX (diffusion implicit; advection/chemotaxis/reaction explicit).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import interpolate
import warnings

# ==================== Parameters ====================
Lx = 50
Ly = 50
Nx = 256 * 2
Ny = 256 * 2
dx = Lx / Nx
dy = Ly / Ny

dt = 0.01
tFinal = 200
plotEvery = 1000  # visualization/tracking interval
defectEvery = 1   # compute defects & rhoD every this many steps

alpha = 0.25
beta = 1.0
chi0 = 10.0
Pe = 1.2
mu0 = 5e-2
nu = 4.0
gamma0 = 7e-1
gamma = 8e-1
chi1 = chi0 * gamma

clipNneg = True
dealise = True

np.random.seed(2)

# ==================== Local functions (defined first) ====================

def detect_winding_defects_polar(thetaField, u, v, x, y):
    """
    Detect integer (+/-1) POLAR topological defects from flow-defined director.
    q = (1/2π) ∮ dθ  around each plaquette
    """
    speed = np.sqrt(u**2 + v**2)
    theta = thetaField.copy()
    theta[speed < 1e-4] = np.nan  # mask ill-defined director

    t00 = theta[:-1, :-1]
    t10 = theta[:-1, 1:]
    t11 = theta[1:, 1:]
    t01 = theta[1:, :-1]

    valid = np.isfinite(t00) & np.isfinite(t10) & np.isfinite(t11) & np.isfinite(t01)

    def wrap(a):
        return np.arctan2(np.sin(a), np.cos(a))

    d1 = wrap(t10 - t00)
    d2 = wrap(t11 - t10)
    d3 = wrap(t01 - t11)
    d4 = wrap(t00 - t01)
    winding = (d1 + d2 + d3 + d4) / (2 * np.pi)

    mask = valid & (np.abs(winding) > 0.75)
    qDef = np.sign(winding[mask]) * np.round(np.abs(winding[mask])).astype(int)

    xc = x[:-1] + 0.5 * (x[1] - x[0])
    yc = y[:-1] + 0.5 * (y[1] - y[0])
    XC, YC = np.meshgrid(xc, yc)

    xDef = XC[mask]
    yDef = YC[mask]

    return xDef, yDef, qDef


def update_defect_tracks(tracks, nextTrackId, xDef, yDef, qDef, tNow, frameIdx, maxDist, maxGap):
    """
    Nearest-neighbor tracking with sign consistency.
    Tracked only at plotted frames.
    """
    usedTrack = [False] * len(tracks)

    for i in range(len(xDef)):
        thisSign = qDef[i]
        bestTrack = -1
        bestDist = np.inf

        for k in range(len(tracks)):
            if usedTrack[k]:
                continue
            if tracks[k]['sign'] != thisSign:
                continue
            if frameIdx - tracks[k]['lastFrame'] > maxGap:
                continue

            dist = np.hypot(xDef[i] - tracks[k]['x'][-1], yDef[i] - tracks[k]['y'][-1])
            if dist < bestDist:
                bestDist = dist
                bestTrack = k

        if bestTrack >= 0 and bestDist <= maxDist:
            tracks[bestTrack]['x'].append(xDef[i])
            tracks[bestTrack]['y'].append(yDef[i])
            tracks[bestTrack]['t'].append(tNow)
            tracks[bestTrack]['lastFrame'] = frameIdx
            usedTrack[bestTrack] = True
        else:
            newTrack = {
                'id': nextTrackId,
                'sign': thisSign,
                'x': [xDef[i]],
                'y': [yDef[i]],
                't': [tNow],
                'lastFrame': frameIdx
            }
            tracks.append(newTrack)
            usedTrack.append(True)
            nextTrackId += 1

    return tracks, nextTrackId


def xcorr_coeff(a, b, maxLag):
    """
    Cross-correlation normalized to coefficient (no toolbox).
    Returns C(lag) for lags = -maxLag:maxLag.
    """
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()
    a = a - np.mean(a)
    b = b - np.mean(b)

    Na = len(a)
    lags = np.arange(-maxLag, maxLag + 1)
    C = np.zeros(len(lags))

    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if den == 0:
        return C, lags

    for ii, L in enumerate(lags):
        if L >= 0:
            aa = a[:Na - L]
            bb = b[L:Na]
        else:
            L = -L
            aa = a[L:Na]
            bb = b[:Na - L]
        C[ii] = np.sum(aa * bb) / den

    return C, lags

# ==================== Grids & Fourier operators ====================
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y)

kx = (2 * np.pi / Lx) * np.concatenate([np.arange(Nx // 2), np.arange(-Nx // 2, 0)])
ky = (2 * np.pi / Ly) * np.concatenate([np.arange(Ny // 2), np.arange(-Ny // 2, 0)])
KX, KY = np.meshgrid(kx, ky)
K2 = KX**2 + KY**2

denN = 1 + dt * K2
denC = 1 + dt * alpha * K2

if dealise:
    kx_cut = (2 / 3) * np.max(np.abs(kx))
    ky_cut = (2 / 3) * np.max(np.abs(ky))
    dealiasMask = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)
else:
    dealiasMask = np.ones_like(KX, dtype=bool)

# ==================== Initial conditions ====================
n0 = 1
n = n0 + 0.02 * np.cos(KX * X + KY * Y)
c0 = 1.0
c = c0 + 0.02 * np.cos(KX * X + KY * Y)

u = np.zeros((Ny, Nx))
v = np.zeros((Ny, Nx))

# ==================== Diagnostics storage ====================
nSteps = int(np.ceil(tFinal / dt))
time_vec = np.arange(1, nSteps + 1) * dt

Ek_time = np.zeros(nSteps)
rhoD_time = np.full(nSteps, np.nan)
Ndef_time = np.full(nSteps, np.nan)

# ==================== Visualization setup ====================
quiverSub = max(1, round(Nx / 30))
Xq = X[::quiverSub, ::quiverSub]
Yq = Y[::quiverSub, ::quiverSub]

fig = plt.figure(figsize=(14, 10), facecolor='w')
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])
h_n = ax1.imshow(n, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='turbo')
ax1.set_aspect('equal')
plt.colorbar(h_n, ax=ax1)
ax1.set_title('n(x,y,t)')
q4 = ax1.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), scale=25)

ax2 = fig.add_subplot(gs[0, 1])
h_c = ax2.imshow(c, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='turbo')
ax2.set_aspect('equal')
plt.colorbar(h_c, ax=ax2)
ax2.set_title('c(x,y,t)')

ax3 = fig.add_subplot(gs[1, 0])
Q = np.zeros_like(n)
h_Q = ax3.imshow(Q, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='turbo')
ax3.set_aspect('equal')
plt.colorbar(h_Q, ax=ax3)
ax3.set_title('Q(x,y,t)')
q3 = ax3.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), scale=25)

ax4 = fig.add_subplot(gs[1, 1])
thetaField = np.zeros_like(n)
h_theta = ax4.imshow(thetaField, extent=[x[0], x[-1], y[0], y[-1]], origin='lower', cmap='hsv', vmin=0, vmax=2*np.pi)
ax4.set_aspect('equal')
plt.colorbar(h_theta, ax=ax4)
ax4.set_title('Director θ + defects')
q = ax4.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), scale=20)
hPlus = ax4.scatter([], [], s=36, c='r', edgecolors='k', zorder=5)
hMinus = ax4.scatter([], [], s=36, c='c', edgecolors='k', zorder=5)
trackLineHandles = []

plt.ion()
plt.pause(0.001)

# Defect-core tracking state
tracks = []
nextTrackId = 1
trackMaxDist = 1.5
trackMaxGap = 2
frameCount = 0

# ==================== Time integration loop ====================
for step in range(nSteps):
    # ---- FFTs ----
    N_hat = np.fft.fft2(n)
    C_hat = np.fft.fft2(c)

    # ---- Gradients ----
    ny = np.real(np.fft.ifft2(1j * KY * N_hat))
    cx = np.real(np.fft.ifft2(1j * KX * C_hat))
    cy = np.real(np.fft.ifft2(1j * KY * C_hat))

    # ---- Flow solve in Fourier space ----
    S_hat = np.fft.fft2(Pe / (1 + n))
    U_hat = (1j * KX * (chi1 * C_hat + S_hat) / (1 + nu * K2) +
             mu0 * gamma0 * KX * KY * (nu - 1) * N_hat / ((1 + K2) * (1 + nu * K2)))
    V_hat = (1j * KY * (chi1 * C_hat + S_hat) / (1 + nu * K2) -
             mu0 * gamma0 * (1 + nu * KX**2 + KY**2) * N_hat / ((1 + K2) * (1 + nu * K2)))

    U_hat[~np.isfinite(U_hat)] = 0
    V_hat[~np.isfinite(V_hat)] = 0

    if dealise:
        U_hat = U_hat * dealiasMask
        V_hat = V_hat * dealiasMask

    u = np.real(np.fft.ifft2(U_hat))
    v = np.real(np.fft.ifft2(V_hat))

    # ---- Kinetic energy ----
    Ek_time[step] = dx * dy * np.sum(0.5 * (u**2 + v**2))

    # ---- RHS for n and c ----
    div_un = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u * n) + 1j * KY * np.fft.fft2(v * n)))
    div_uc = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u * c) + 1j * KY * np.fft.fft2(v * c)))
    nablan = np.real(np.fft.ifft2(-K2 * N_hat))
    nablac = np.real(np.fft.ifft2(-K2 * C_hat))

    div_n_gradc = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(n * cx) + 1j * KY * np.fft.fft2(n * cy)))
    drift_n = -(mu0 * gamma0 * gamma) * ny
    RHSn = nablan - div_un + drift_n - chi0 * div_n_gradc
    RHSc = alpha * nablac - div_uc - beta * (n * c) + beta * c

    # ---- Dealias nonlinear terms ----
    if dealise:
        RHSn_hat = np.fft.fft2(RHSn) * dealiasMask
        RHSc_hat = np.fft.fft2(RHSc) * dealiasMask
    else:
        RHSn_hat = np.fft.fft2(RHSn)
        RHSc_hat = np.fft.fft2(RHSc)

    # ---- IMEX update ----
    n = np.real(np.fft.ifft2((N_hat + dt * RHSn_hat) / denN))
    c = np.real(np.fft.ifft2((C_hat + dt * RHSc_hat) / denC))

    if clipNneg:
        n = np.maximum(n, 0)

    # ---- Defects & defect density ----
    if (step + 1) % defectEvery == 0:
        thetaField = np.arctan2(v, u)
        xDef_all, yDef_all, qDef_all = detect_winding_defects_polar(thetaField, u, v, x, y)

        Ndef_time[step] = len(qDef_all)
        rhoD_time[step] = Ndef_time[step] / (Lx * Ly)

    # ---- Visualization & tracking ----
    if (step + 1) % plotEvery == 0 or step == 0 or step == nSteps - 1:
        frameCount += 1
        tNow = (step + 1) * dt

        # Compute defects if not computed this step
        if (step + 1) % defectEvery != 0:
            thetaField = np.arctan2(v, u)
            xDef_all, yDef_all, qDef_all = detect_winding_defects_polar(thetaField, u, v, x, y)
            Ndef_time[step] = len(qDef_all)
            rhoD_time[step] = Ndef_time[step] / (Lx * Ly)

        # Track defects
        tracks, nextTrackId = update_defect_tracks(
            tracks, nextTrackId, xDef_all, yDef_all, qDef_all,
            tNow, frameCount, trackMaxDist, trackMaxGap)

        # Update plots
        h_n.set_data(n)
        ax1.set_title(f'n(x,y,t), t = {tNow:.2f}')
        q4.set_UVC(u[::quiverSub, ::quiverSub], v[::quiverSub, ::quiverSub])

        h_c.set_data(c)
        ax2.set_title(f'c(x,y,t), t = {tNow:.2f}')

        # Vorticity and Okubo-Weiss parameter
        ux = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u)))
        vx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
        uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
        vy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(v)))
        vort = vx - uy
        sn = ux - vy
        ss = vx + uy
        Q = sn**2 + ss**2 - vort**2
        h_Q.set_data(Q)
        vmax = np.max(np.abs(vort))
        if vmax > 0:
            h_Q.set_clim(-vmax, vmax)
        ax3.set_title(f'Q(x,y,t), t={tNow:.2f}, E_k={Ek_time[step]:.3e}')
        q3.set_UVC(u[::quiverSub, ::quiverSub], v[::quiverSub, ::quiverSub])

        # Director & defects
        h_theta.set_data(np.mod(thetaField, 2 * np.pi))
        ax4.set_title(f'θ + defects, t={tNow:.2f}, ρ_d={rhoD_time[step]:.3e}')

        thetaSub = thetaField[::quiverSub, ::quiverSub]
        q.set_UVC(np.cos(thetaSub), np.sin(thetaSub))

        posMask = qDef_all > 0
        negMask = qDef_all < 0
        hPlus.set_offsets(np.column_stack([xDef_all[posMask], yDef_all[posMask]]))
        hMinus.set_offsets(np.column_stack([xDef_all[negMask], yDef_all[negMask]]))

        # Remove old track lines
        for line in trackLineHandles:
            line.remove()
        trackLineHandles = []

        for track in tracks:
            if len(track['x']) > 1:
                lineColor = 'r' if track['sign'] > 0 else 'c'
                line, = ax4.plot(track['x'], track['y'], '-', color=lineColor, linewidth=1.1)
                trackLineHandles.append(line)

        plt.pause(0.001)

defectTracks = tracks

# ==================== Post-run correlation analysis ====================
# Clean NaNs if defectEvery > 1
rhoD = rhoD_time.copy()
Ndef = Ndef_time.copy()

if np.any(np.isnan(rhoD)):
    idx = np.where(~np.isnan(rhoD))[0]
    if len(idx) >= 2:
        f = interpolate.interp1d(time_vec[idx], rhoD[idx], kind='linear', fill_value='extrapolate')
        rhoD = f(time_vec)
    else:
        warnings.warn('Not enough defect samples to interpolate. Set defectEvery=1.')

# 1) Time series overlay
fig1, ax = plt.subplots(figsize=(10, 6), facecolor='w')
ax1_twin = ax
ax2_twin = ax.twinx()

line1 = ax1_twin.plot(time_vec, Ek_time, 'k', linewidth=1.4, label='$E_k$')
ax1_twin.set_ylabel('Kinetic energy $E_k$', color='k')
ax1_twin.tick_params(axis='y', labelcolor='k')

line2 = ax2_twin.plot(time_vec, rhoD, 'r', linewidth=1.4, label='$\\rho_d$')
ax2_twin.set_ylabel('Defect density $\\rho_d$', color='r')
ax2_twin.tick_params(axis='y', labelcolor='r')

ax.set_xlabel('Time')
ax.set_title('$E_k(t)$ and defect density $\\rho_d(t)$')
ax.grid(True, alpha=0.3)
fig1.tight_layout()

# 2) Scatter: E_k vs rho_d
fig2, ax = plt.subplots(figsize=(10, 6), facecolor='w')
scatter = ax.scatter(rhoD, Ek_time, s=18, c=time_vec, cmap='viridis')
ax.set_xlabel('Defect density $\\rho_d$')
ax.set_ylabel('Kinetic energy $E_k$')
ax.set_title('Scatter: $E_k$ vs $\\rho_d$ (colored by time)')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Time')
ax.grid(True, alpha=0.3)
fig2.tight_layout()

# 3) Pearson correlation coefficient
R = np.corrcoef(Ek_time, rhoD)
print(f'Pearson corr(E_k, rho_d) = {R[0, 1]:.3f}')

# 4) Lagged cross-correlation
maxLagTime = 5  # physical time units
maxLag = int(np.round(maxLagTime / dt))
C, lags = xcorr_coeff(Ek_time, rhoD, maxLag)

fig3, ax = plt.subplots(figsize=(10, 6), facecolor='w')
ax.plot(lags * dt, C, linewidth=1.5)
ax.set_xlabel('Time lag')
ax.set_ylabel('Cross-correlation (coeff)')
ax.set_title('Lagged cross-correlation: $E_k$ vs $\\rho_d$')
ax.grid(True, alpha=0.3)
fig3.tight_layout()

plt.show()
