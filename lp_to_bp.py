#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
nsmax = 100
nf = 5

# Wavenumbers
wn = np.linspace(0, nsmax, nsmax+1)

# Explicit low-pass filters
lpf = np.zeros((nsmax+1, nf-1))
for i in range(nf-1):
  L = 0.4*(i+1)*(nsmax+1)/nf
  for k in range(nsmax+1):
    lpf[k,i] = np.exp(-0.5*(k/L)**2)

# Equivalent band-pass filters
bpf = np.ones((nsmax+1, nf))
bpf[:,0] = lpf[:,0]
for j in range(1,nf):
  bpf[:,j] *= (1.0-lpf[:,0])
for i in range(1,nf-1):
  bpf[:,i] *= lpf[:,i]
  for j in range(i+1,nf):
    bpf[:,j] *= (1.0-lpf[:,i])
bpf_sum = np.sum(bpf, axis=1)

# Plot
fig,ax = plt.subplots(ncols=1, nrows=2, figsize=(8,8))
ax[0].plot(wn, lpf, linewidth=2.0)
ax[0].set_title("Explicit low-pass filters", fontsize=18)
ax[0].set_xlabel("Wavenumber", fontsize=14)
ax[0].set_xlim(0, nsmax)
ax[0].set_ylim(0, 1.1)
ax[1].plot(wn, bpf, linewidth=2.0)
ax[1].plot(wn, bpf_sum, '--k', linewidth=2.0)
ax[1].set_title("Equivalent band-pass filters", fontsize=18)
ax[1].set_xlabel("Wavenumber", fontsize=14)
ax[1].set_xlim(0, nsmax)
ax[1].set_ylim(0, 1.1)
fig.tight_layout(pad=2.5)
plt.savefig('lp_to_bp.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop lp_to_bp.pdf lp_to_bp.pdf')
