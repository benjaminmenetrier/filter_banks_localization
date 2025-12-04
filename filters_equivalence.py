#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
nsmax = 100
nf = 5

# Wavenumbers
wn = np.linspace(0, nsmax, nsmax+1)

# Explicit filters
lpf = np.zeros((nsmax+1, nf-1))
hpf = np.zeros((nsmax+1, nf-1))
for i in range(nf-1):
  L = 0.4*(i+1)*(nsmax+1)/nf
  for k in range(nsmax+1):
    lpf[k,i] = np.exp(-0.5*(k/L)**2)
    hpf[k,i] = 1.0-lpf[k,i]

# Equivalent band-pass filters
bpf_from_lpf = np.ones((nsmax+1, nf))
bpf_from_hpf = np.ones((nsmax+1, nf))
bpf_from_lpf[:,0] = lpf[:,0]
bpf_from_hpf[:,nf-1] = hpf[:,nf-2]
for j in range(1,nf):
  bpf_from_lpf[:,j] *= (1.0-lpf[:,0])
  bpf_from_hpf[:,nf-1-j] *= (1.0-hpf[:,nf-2])
for i in range(1,nf-1):
  bpf_from_lpf[:,i] *= lpf[:,i]
  bpf_from_hpf[:,nf-1-i] *= hpf[:,nf-2-i]
  for j in range(i+1,nf):
    bpf_from_lpf[:,j] *= (1.0-lpf[:,i])
    bpf_from_hpf[:,nf-1-j] *= (1.0-hpf[:,nf-2-i])
bpf_from_lpf_sum = np.sum(bpf_from_lpf, axis=1)
bpf_from_hpf_sum = np.sum(bpf_from_hpf, axis=1)

# Plot
fig,ax = plt.subplots(ncols=2, nrows=2, figsize=(16,8))
ax[0][0].plot(wn, lpf, linewidth=2.0)
ax[0][0].set_title("Explicit low-pass filters", fontsize=18)
ax[0][0].set_xlabel("Wavenumber", fontsize=14)
ax[0][0].set_xlim(0, nsmax)
ax[0][0].set_ylim(0, 1.1)

ax[0][1].plot(wn, hpf, linewidth=2.0)
ax[0][1].set_title("Complementary high-pass filters", fontsize=18)
ax[0][1].set_xlabel("Wavenumber", fontsize=14)
ax[0][1].set_xlim(0, nsmax)
ax[0][1].set_ylim(0, 1.1)

ax[1][0].plot(wn, bpf_from_lpf, linewidth=2.0)
ax[1][0].plot(wn, bpf_from_lpf_sum, '--k', linewidth=2.0)
ax[1][0].set_title("Equivalent band-pass filters, from low-pass filters", fontsize=18)
ax[1][0].set_xlabel("Wavenumber", fontsize=14)
ax[1][0].set_xlim(0, nsmax)
ax[1][0].set_ylim(0, 1.1)

ax[1][1].plot(wn, bpf_from_hpf, linewidth=2.0)
ax[1][1].plot(wn, bpf_from_hpf_sum, '--k', linewidth=2.0)
ax[1][1].set_title("Equivalent band-pass filters, from high-pass filters", fontsize=18)
ax[1][1].set_xlabel("Wavenumber", fontsize=14)
ax[1][1].set_xlim(0, nsmax)
ax[1][1].set_ylim(0, 1.1)

fig.tight_layout(pad=2.5)
plt.savefig('filters_equivalence.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop filters_equivalence.pdf filters_equivalence.pdf')
