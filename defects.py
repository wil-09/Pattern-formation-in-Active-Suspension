import numpy as np
import matplotlib.pyplot as plt


def wrap_pi(a):
    return np.arctan2(np.sin(a), np.cos(a))


def detect_winding_defects(theta_field, x, y):
    theta2 = 2.0 * theta_field

    t00 = theta2[:-1, :-1]
    t10 = theta2[:-1, 1:]
    t11 = theta2[1:, 1:]
    t01 = theta2[1:, :-1]

    d1 = wrap_pi(t10 - t00)
    d2 = wrap_pi(t11 - t10)
    d3 = wrap_pi(t01 - t11)
    d4 = wrap_pi(t00 - t01)

    charge = (d1 + d2 + d3 + d4) / (4.0 * np.pi)

    mask = np.abs(charge) > 0.25
    q_raw = charge[mask]
    q_def = 0.5 * np.sign(q_raw)

    xc = x[:-1] + 0.5 * (x[1] - x[0])
    yc = y[:-1] + 0.5 * (y[1] - y[0])
    xc_grid, yc_grid = np.meshgrid(xc, yc)

    x_def = xc_grid[mask]
    y_def = yc_grid[mask]
    return x_def, y_def, q_def


def update_defect_tracks(tracks, next_track_id, x_def, y_def, q_def, t_now, frame_idx, max_dist, max_gap):
    used_track = np.zeros(len(tracks), dtype=bool)

    for i in range(len(x_def)):
        this_sign = q_def[i]
        best_track = -1
        best_dist = np.inf

        for k in range(len(tracks)):
            if used_track[k]:
                continue
            if tracks[k]["sign"] != this_sign:
                continue
            if frame_idx - tracks[k]["lastFrame"] > max_gap:
                continue

            dxk = x_def[i] - tracks[k]["x"][-1]
            dyk = y_def[i] - tracks[k]["y"][-1]
            dist = np.hypot(dxk, dyk)

            if dist < best_dist:
                best_dist = dist
                best_track = k

        if best_track >= 0 and best_dist <= max_dist:
            tracks[best_track]["x"].append(float(x_def[i]))
            tracks[best_track]["y"].append(float(y_def[i]))
            tracks[best_track]["t"].append(float(t_now))
            tracks[best_track]["lastFrame"] = int(frame_idx)
            used_track[best_track] = True
        else:
            tracks.append(
                {
                    "id": int(next_track_id),
                    "sign": float(this_sign),
                    "x": [float(x_def[i])],
                    "y": [float(y_def[i])],
                    "t": [float(t_now)],
                    "lastFrame": int(frame_idx),
                }
            )
            used_track = np.append(used_track, True)
            next_track_id += 1

    return tracks, next_track_id


def run_simulation():
    Lx = 50
    Ly = 50
    Nx = 256 * 2
    Ny = 256 * 2

    dt = 0.01
    t_final = 200
    plot_every = 1000

    alpha = 0.25
    beta = 1
    chi0 = 10
    Pe = 1.2
    mu0 = 0.05
    nu = 4
    gamma0 = 0.7
    gamma = 0.8
    chi1 = chi0 * gamma

    clip_n_neg = True
    dealise = True

    rng = np.random.default_rng(2)

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, Nx // 2), np.arange(-Nx // 2, 0)))
    ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, Ny // 2), np.arange(-Ny // 2, 0)))
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX**2 + KY**2

    den_n = 1 + dt * K2
    den_c = 1 + dt * alpha * K2

    if dealise:
        kx_cut = (2 / 3) * np.max(np.abs(kx))
        ky_cut = (2 / 3) * np.max(np.abs(ky))
        dealias_mask = (np.abs(KX) <= kx_cut) & (np.abs(KY) <= ky_cut)
    else:
        dealias_mask = np.ones_like(KX, dtype=bool)

    n0 = 1.0
    n = n0 + 0.02 * rng.standard_normal((Ny, Nx))
    n = np.maximum(n, 0)

    c0 = 1.0
    c = c0 + 0.02 * rng.standard_normal((Ny, Nx))

    u = np.zeros((Ny, Nx))
    v = np.zeros((Ny, Nx))

    quiver_sub = max(1, int(round(Nx / 30)))
    Xq = X[::quiver_sub, ::quiver_sub]
    Yq = Y[::quiver_sub, ::quiver_sub]

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    h_n = ax1.imshow(n, origin="lower", extent=[0, Lx, 0, Ly], cmap="turbo", aspect="equal")
    fig.colorbar(h_n, ax=ax1)
    ax1.set_title("n(x,y,t)")
    q4 = ax1.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), color="k")

    h_c = ax2.imshow(c, origin="lower", extent=[0, Lx, 0, Ly], cmap="turbo", aspect="equal")
    fig.colorbar(h_c, ax=ax2)
    ax2.set_title("c(x,y,t)")

    vort = np.zeros_like(n)
    h_vort = ax3.imshow(vort, origin="lower", extent=[0, Lx, 0, Ly], cmap="turbo", aspect="equal")
    fig.colorbar(h_vort, ax=ax3)
    ax3.set_title("W(x,y,t)")
    q3 = ax3.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), color="k")

    theta_field = np.zeros_like(n)
    h_theta = ax4.imshow(theta_field, origin="lower", extent=[0, Lx, 0, Ly], cmap="hsv", aspect="equal", vmin=0, vmax=np.pi)
    fig.colorbar(h_theta, ax=ax4)
    ax4.set_title("Director θ(x,y,t) + defects")
    q = ax4.quiver(Xq, Yq, np.zeros_like(Xq), np.zeros_like(Yq), color="k")
    h_plus = ax4.scatter([], [], s=36, c="r", edgecolors="k")
    h_minus = ax4.scatter([], [], s=36, c="c", edgecolors="k")
    track_line_handles = []

    tracks = []
    next_track_id = 1
    track_max_dist = 1.5
    track_max_gap = 2
    frame_count = 0

    n_steps = int(np.ceil(t_final / dt))

    for step in range(1, n_steps + 1):
        N_hat = np.fft.fft2(n)
        C_hat = np.fft.fft2(c)

        nx = np.real(np.fft.ifft2(1j * KX * N_hat))
        ny = np.real(np.fft.ifft2(1j * KY * N_hat))
        cx = np.real(np.fft.ifft2(1j * KX * C_hat))
        cy = np.real(np.fft.ifft2(1j * KY * C_hat))

        S_hat = np.fft.fft2(Pe / (1 + n))
        U_hat = 1j * KX * (chi1 * C_hat + S_hat) / (1 + nu * K2) + mu0 * gamma0 * KX * KY * (nu - 1) * N_hat / ((1 + K2) * (1 + nu * K2))
        V_hat = 1j * KY * (chi1 * C_hat + S_hat) / (1 + nu * K2) - mu0 * gamma0 * (1 + nu * KX**2 + KY**2) * N_hat / ((1 + K2) * (1 + nu * K2))

        U_hat[~np.isfinite(U_hat)] = 0
        V_hat[~np.isfinite(V_hat)] = 0

        if dealise:
            U_hat *= dealias_mask
            V_hat *= dealias_mask

        u = np.real(np.fft.ifft2(U_hat))
        v = np.real(np.fft.ifft2(V_hat))

        un = u * n
        vn = v * n
        uc = u * c
        vc = v * c

        div_un = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(un) + 1j * KY * np.fft.fft2(vn)))
        div_uc = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(uc) + 1j * KY * np.fft.fft2(vc)))
        nabla_n = np.real(np.fft.ifft2(-K2 * N_hat))
        nabla_c = np.real(np.fft.ifft2(-K2 * C_hat))

        n_cx = n * cx
        n_cy = n * cy
        div_n_gradc = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(n_cx) + 1j * KY * np.fft.fft2(n_cy)))

        drift_n = -(mu0 * gamma0 * gamma) * ny

        rhs_n = nabla_n - div_un + drift_n - chi0 * div_n_gradc
        rhs_c = alpha * nabla_c - div_uc - beta * (n * c) + beta * c

        if dealise:
            rhs_n_hat = np.fft.fft2(rhs_n) * dealias_mask
            rhs_c_hat = np.fft.fft2(rhs_c) * dealias_mask
        else:
            rhs_n_hat = np.fft.fft2(rhs_n)
            rhs_c_hat = np.fft.fft2(rhs_c)

        N_hat_new = (N_hat + dt * rhs_n_hat) / den_n
        C_hat_new = (C_hat + dt * rhs_c_hat) / den_c

        if dealise:
            N_hat_new *= dealias_mask
            C_hat_new *= dealias_mask

        n = np.real(np.fft.ifft2(N_hat_new))
        c = np.real(np.fft.ifft2(C_hat_new))

        if clip_n_neg:
            n = np.maximum(n, 0)

        if step % plot_every == 0 or step == 1 or step == n_steps:
            frame_count += 1
            t_now = step * dt

            h_n.set_data(n)
            ax1.set_title(f"n(x,y,t),  t = {t_now:.2f}")
            q4.set_UVC(u[::quiver_sub, ::quiver_sub], v[::quiver_sub, ::quiver_sub])

            h_c.set_data(c)
            ax2.set_title(f"c(x,y,t),  t = {t_now:.2f}")

            vx = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(v)))
            uy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(u)))
            ux = np.real(np.fft.ifft2(1j * KX * np.fft.fft2(u)))
            vy = np.real(np.fft.ifft2(1j * KY * np.fft.fft2(v)))
            vort = vx - uy

            theta_field = np.mod(np.arctan2(v, u), np.pi)
            x_def, y_def, q_def = detect_winding_defects(theta_field, x, y)

            tracks, next_track_id = update_defect_tracks(
                tracks,
                next_track_id,
                x_def,
                y_def,
                q_def,
                t_now,
                frame_count,
                track_max_dist,
                track_max_gap,
            )

            vmax = np.max(np.abs(vort))
            if vmax > 0:
                h_vort.set_clim(-vmax, vmax)

            h_vort.set_data(vort)
            ax3.set_title(f"W(x,y,t), t = {t_now:.2f}")
            q3.set_UVC(u[::quiver_sub, ::quiver_sub], v[::quiver_sub, ::quiver_sub])

            h_theta.set_data(theta_field)
            ax4.set_title(f"Director field + defects,  t = {t_now:.2f}")
            theta_sub = theta_field[::quiver_sub, ::quiver_sub]
            q.set_UVC(np.cos(theta_sub), np.sin(theta_sub))

            pos_mask = q_def > 0
            neg_mask = q_def < 0
            h_plus.set_offsets(np.column_stack((x_def[pos_mask], y_def[pos_mask])) if np.any(pos_mask) else np.empty((0, 2)))
            h_minus.set_offsets(np.column_stack((x_def[neg_mask], y_def[neg_mask])) if np.any(neg_mask) else np.empty((0, 2)))

            for handle in track_line_handles:
                handle.remove()
            track_line_handles = []

            for tr in tracks:
                if len(tr["x"]) > 1:
                    line_color = (1, 0, 0) if tr["sign"] > 0 else (0, 0.7, 0.9)
                    (line,) = ax4.plot(tr["x"], tr["y"], "-", color=line_color, linewidth=1.2)
                    track_line_handles.append(line)

            plt.pause(0.001)

    return tracks


if __name__ == "__main__":
    defectTracks = run_simulation()
