#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
nsmax = 100
nf = 4

# Wavenumbers
wn = np.linspace(0, nsmax, nsmax+1)

# Explicit band-pass filters
bpf = np.zeros((nsmax+1, nf))
mode = 0.0
halfWidth = 0.05*(nsmax+1)
for i in range(0,nf-1):
  for k in range(nsmax+1):
    if k < mode-halfWidth:
      bpf[k,i] = 0.0
    elif k < mode:
      bpf[k,i] = 0.5*(np.cos(math.pi*(k-mode)/halfWidth)+1.0)
    elif k < mode+2.0*halfWidth:
      bpf[k,i] = 0.5*(np.cos(math.pi*(k-mode)/(2.0*halfWidth))+1.0)
    else:
      bpf[k,i] = 0.0
    bpf[k,nf-1] += bpf[k,i]
  halfWidth *= 2
  mode += halfWidth
for k in range(nsmax+1):
  bpf[k,nf-1] = 1.0-bpf[k,nf-1]

bpf_sum = np.sum(bpf, axis=1)

# Equivalent low-pass filters
lpf = np.ones((nsmax+1, nf-1))
lpf[:,0] = bpf[:,0]
for j in range(1,nf-1):
  for k in range(nsmax+1):
    if lpf[k,0] < 1.0:
      lpf[k,j] *= 1.0/(1.0-lpf[k,0])
for i in range(1,nf-1):
  beforeMode = True
  for k in range(nsmax+1):
    if beforeMode:
      if bpf[k,i] > 0:
        beforeMode = False
        lpf[k,i] *= bpf[k,i]
      else:
        lpf[k,i] = 1.0
    else:
      lpf[k,i] *= bpf[k,i]
  for j in range(i+1,nf-1):
    for k in range(nsmax+1):
      if lpf[k,i] < 1.0:
        lpf[k,j] *= 1.0/(1.0-lpf[k,i])

# Plot
fig,ax = plt.subplots(ncols=1, nrows=2, figsize=(8,8))
ax[0].plot(wn, bpf, linewidth=2.0)
ax[0].plot(wn, bpf_sum, '--k', linewidth=2.0)
ax[0].set_title("Explicit band-pass filters", fontsize=18)
ax[0].set_xlabel("Wavenumber", fontsize=14)
ax[0].set_xlim(0, nsmax)
ax[0].set_ylim(0, 1.1)
ax[1].plot(wn, lpf, linewidth=2.0)
ax[1].set_title("Equivalent low-pass filters", fontsize=18)
ax[1].set_xlabel("Wavenumber", fontsize=14)
ax[1].set_xlim(0, nsmax)
ax[1].set_ylim(0, 1.1)
fig.tight_layout(pad=2.5)
plt.savefig('bp_to_lp.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop bp_to_lp.pdf bp_to_lp.pdf')
