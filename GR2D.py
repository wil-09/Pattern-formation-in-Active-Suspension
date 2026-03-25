import numpy as np
import matplotlib.pyplot as plt


def main():
    alpha = 0.25
    beta = 10
    gamma0 = 100
    ki0 = 10
    gamma = 1
    mu0 = 1
    Pe = 1.2
    ki1 = ki0 * gamma
    nu = 4
    n0 = 1
    c0 = 1

    kx0, kxf = -5, 5
    ky0, kyf = -5, 5
    m = 512 * 2

    dkx = (kxf - kx0) / m
    dky = (kyf - ky0) * dkx / (kxf - kx0)

    kx = np.arange(kx0, kyf + dkx, dkx)
    ky = np.arange(ky0, kyf + dky, dky)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)

    term1 = (
        -k**2 * (1 + alpha)
        - beta * n0
        - k**2 * (Pe * n0 / (1 + n0) ** 2 + ki1 * c0) / (1 + nu * k**2)
        + 1j * ky * mu0 * gamma0 * (2 * n0 - gamma + n0 / (1 + nu * k**2))
    )

    term2 = (
        -alpha * k**2
        - beta * n0
        - ki1 * c0 * k**2 / (1 + nu * k**2)
        + 1j * ky * mu0 * gamma0 * n0
    )

    term3 = (
        -k**2 * (1 + Pe * n0 * k**2 / ((1 + n0) ** 2 * (1 + nu * k**2)))
        + 1j * ky * mu0 * gamma0 * (n0 - gamma + n0 / (1 + nu * k**2))
    )

    term4 = n0 * c0 * k**2 * (ki0 + ki1 / (1 + nu * k**2)) * (
        beta + (Pe / (1 + n0) ** 2 + 1j * mu0 * gamma0 * ky) / (1 + nu * k**2)
    )

    disc = term1**2 - 4 * (term2 * term3 - term4)

    lambda2 = (-term1 - np.sqrt(disc)) / 2

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="3d")
    surf = ax1.plot_surface(kx, ky, np.real(lambda2), cmap="jet", linewidth=0, antialiased=False)
    ax1.set_xlabel("k_x")
    ax1.set_ylabel("k_y")
    ax1.set_zlabel("λ_r")
    fig1.colorbar(surf, ax=ax1, shrink=0.7)

    plt.figure()
    contour = plt.contour(kx, ky, np.real(lambda2))
    plt.xlabel("k_x")
    plt.ylabel("k_y")
    plt.colorbar(contour)

    plt.show()


if __name__ == "__main__":
    main()
