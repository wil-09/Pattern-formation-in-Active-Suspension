function GR2D
% this code plot the growth rate of the linear stability analysis in 1D
clear all; close all; clc
i = sqrt(-1);
set(0, 'defaultaxesfontsize', 20, 'defaultaxesfontWeight', 'bold', 'defaultaxesLineWidth', 1)
% fig1ad: alpha = 1; beta = 10; gamma0 = 1000; ki0 = 10; gamma = 1; mu0 = 1; Pe = 1.2; ki1 = ki0*gamma; nu = 4; n0 = 10; c0 = 01;
% fig1be: alpha = 1; beta = 10; gamma0 = 1000; ki0 = 10; gamma = 0.01; mu0 = 1; Pe = 1.2; ki1 = ki0/gamma; nu = 4; n0 = 1; c0 = 1;
% fig1cf: alpha = 0.1; beta = 10; gamma0 = 1000; ki0 = 10; gamma = 1; mu0 = 0.05; Pe = 1.2; ki1 = ki0/gamma; nu = 4; n0 = 1; c0 = 1;

% parameters of the model
% alpha = 0.01; beta = 10; gamma0 = 1000; ki0 = 10; gamma = 1; mu0 = 0.05;
% Pe = 1.20; ki1 = ki0*gamma; nu = 4; n0 = 0.01; c0 = 0.01;

alpha = 0.25; beta = 10; gamma0 = 100; ki0 = 10; gamma = 1; 
mu0 = 1; Pe = 1.2; ki1 = ki0*gamma; nu = 4; n0 = 1; c0 = 1;

% wave vector grid
kx0 = -5; kxf = 5; ky0 = -5; kyf = 5; m = 512.*2;
dkx = (kxf - kx0)/m; dky = (kyf - ky0)*dkx/(kxf - kx0);
kx = kx0:dkx:kyf; ky = ky0:dky:kyf;
[kx,ky] = meshgrid(kx,ky); k = sqrt(kx.^2 + ky.^2);

% b = -k.^2.*(1 + alpha) - beta.*n0 - k.^2.*(Pe.*n0./(1 + n0).^2 + ki1.*c0)./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*(2.*n0 - gamma + n0./(1 + nu.*k.^2));
% 
% c = (-alpha.*k.^2 - beta.*n0 - ki1.*c0.*k.^2./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*n0).*(-k.^2.*(1 + Pe.*n0.*k.^2./((1 + n0).^2.*(1 + nu.*k.^2))) + i.*ky.*mu0.*gamma0.*(n0 - gamma + n0./(1 + nu.*k.^2))) - n0.*c0.*k.^2.*(ki0 + ki1./(1 + nu.*k.^2)).*(beta + (Pe./(1 + n0).^2 + i.*mu0.*gamma0.*ky)./(1 + nu.*k.^2)); 
% lambda1 = (-(-k.^2.*(1 + alpha) - beta.*n0 - k.^2.*(Pe.*n0./(1 + n0).^2 + ki1.*c0)./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*(2.*n0 - gamma + n0./(1 + nu.*k.^2))) + sqrt((-k.^2.*(1 + alpha) - beta.*n0 - k.^2.*(Pe.*n0./(1 + n0).^2 + ki1.*c0)./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*(2.*n0 - gamma + n0./(1 + nu.*k.^2))).^2 - 4.*((-alpha.*k.^2 - beta.*n0 - ki1.*c0.*k.^2./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*n0).*(-k.^2.*(1 + Pe.*n0.*k.^2./((1 + n0).^2.*(1 + nu.*k.^2))) + i.*ky.*mu0.*gamma0.*(n0 - gamma + n0./(1 + nu.*k.^2))) - n0.*c0.*k.^2.*(ki0 + ki1./(1 + nu.*k.^2)).*(beta + (Pe./(1 + n0).^2 + i.*mu0.*gamma0.*ky)./(1 + nu.*k.^2)))));

lambda2 = (-(-k.^2.*(1 + alpha) - beta.*n0 - k.^2.*(Pe.*n0./(1 + n0).^2 + ki1.*c0)./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*(2.*n0 - gamma + n0./(1 + nu.*k.^2))) - sqrt((-k.^2.*(1 + alpha) - beta.*n0 - k.^2.*(Pe.*n0./(1 + n0).^2 + ki1.*c0)./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*(2.*n0 - gamma + n0./(1 + nu.*k.^2))).^2 - 4.*((-alpha.*k.^2 - beta.*n0 - ki1.*c0.*k.^2./(1 + nu.*k.^2) + i.*ky.*mu0.*gamma0.*n0).*(-k.^2.*(1 + Pe.*n0.*k.^2./((1 + n0).^2.*(1 + nu.*k.^2))) + i.*ky.*mu0.*gamma0.*(n0 - gamma + n0./(1 + nu.*k.^2))) - n0.*c0.*k.^2.*(ki0 + ki1./(1 + nu.*k.^2)).*(beta + (Pe./(1 + n0).^2 + i.*mu0.*gamma0.*ky)./(1 + nu.*k.^2)))));

% figure
% mesh(kx, ky, real(lambda1)); hold on
% xlabel 'k_x', ylabel 'k_y', zlabel '\lambda_r'

% figure
% mesh(kx, ky, real(lambda2)); grid off; colormap(jet)
% xlabel 'k_x', ylabel 'k_y', zlabel '\lambda_r'

figure
mesh(kx, ky, real(lambda2)); grid off; colormap(jet)
xlabel 'k_x', ylabel 'k_y', zlabel '\lambda_r'
colorbar 

figure
contour(kx, ky, real(lambda2)); grid off
xlabel 'k_x', ylabel 'k_y'
colorbar

% figure
% mesh(kx, ky, imag(lambda2)); colormap(jet)
% xlabel 'k_x', ylabel 'k_y', zlabel '\lambda_i'