#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.fft import fft, ifft

# Parameters
nsmax = 100
nf = 4
colors = ["blue","orange","green","red"]
Lfactor = [0.1, 0.2, 0.4]
eLfactor = [0.1, 0.15, 0.2]
ngpmid = nsmax
ngphalf = 30

# Wavenumbers
wn = np.linspace(0, nsmax, nsmax+1)

# Grid-points
gp = np.linspace(1, 2*nsmax, 2*nsmax)-ngpmid-1

# Gaspari-Cohn function
def gc99(r):
   if r<0.5:
      value = 1.0-r
      value = 1.0+8.0/5.0*r*value
      value = 1.0-3.0/4.0*r*value
      value = 1.0-20.0/3.0*r**2*value
   else:
      if r<1.0:
         value = 1.0-r/3.0
         value = 1.0-8.0/5.0*r*value
         value = 1.0+3.0/4.0*r*value
         value = 1.0-2.0/3.0*r*value
         value = 1.0-5.0/2.0*r*value
         value = 1.0-12.0*r*value
         value = -value/(3.0*r)
      else:
        value = 0.0
   return value

# Filtering length-scales
L = np.zeros((nf-1))
eL = np.zeros((nf-1))
for i in range(nf-1):
  L[i] = Lfactor[i]*(i+1)*(nsmax+1)
  eL[i] = eLfactor[i]*(i+1)*(nsmax+1)

# Low-pass filters
lpf = np.zeros((nsmax+1, nf-1))
elpf = np.zeros((nsmax+1, nf-1))
Lflat = 0.0
for i in range(nf-1):
  for k in range(nsmax+1):
    lpf[k,i] = gc99(k/L[i])
  if i > 0:
    for k in range(nsmax+1):
      if k < Lflat:
        elpf[k,i] = 1.0
      else:
        elpf[k,i] = gc99((k-Lflat)/eL[i])
  else:
    for k in range(nsmax+1):
      elpf[k,i] = gc99(k/L[i])
  Lflat += eL[i]

# Low-pass convolution functions
lpfct = np.zeros((2*nsmax, nf-1))
elpfct = np.zeros((2*nsmax, nf-1))
for i in range(nf-1):
  x = np.zeros((2*nsmax))
  x[ngpmid] = 1.0
  y = fft(x)
  for k in range(nsmax+1):
    y[k] *= lpf[k,i]
    if k > 0:
      y[2*nsmax-k] *= lpf[k,i]   
  yinv = ifft(y)
  for j in range(2*nsmax):
    lpfct[j,i] = yinv[j].real

  x = np.zeros((2*nsmax))
  x[ngpmid] = 1.0
  y = fft(x)
  for k in range(nsmax+1):
    y[k] *= elpf[k,i]
    if k > 0:
      y[2*nsmax-k] *= elpf[k,i]   
  yinv = ifft(y)
  for j in range(2*nsmax):
    elpfct[j,i] = yinv[j].real

# Plot
fig,ax = plt.subplots(ncols=2, nrows=2, figsize=(12,6))
ax[0][0].set_prop_cycle(color=colors[0:nf-1])
ax[0][0].plot(wn, lpf, linewidth=2.0)
ax[0][0].set_title("Low-pass filters", fontsize=15)
ax[0][0].set_xlabel("Wavenumber", fontsize=14)
ax[0][0].set_xlim(0, nsmax)
ax[0][0].set_ylim(0, 1.1)

ax[0][1].set_prop_cycle(color=colors[0:nf-1])
ax[0][1].plot(wn, elpf, linewidth=2.0)
ax[0][1].set_title("Embedded low-pass filters", fontsize=15)
ax[0][1].set_xlabel("Wavenumber", fontsize=14)
ax[0][1].set_xlim(0, nsmax)
ax[0][1].set_ylim(0, 1.1)

ax[1][0].set_prop_cycle(color=colors)
ax[1][0].axhline(y=0.0, color='gray', linestyle=':')
ax[1][0].plot(gp[ngpmid-ngphalf:ngpmid+ngphalf], lpfct[ngpmid-ngphalf:ngpmid+ngphalf,:], linewidth=2.0)
ax[1][0].set_title("Corresponding convolution functions", fontsize=15)
ax[1][0].set_xlabel("Grid-point", fontsize=14)
ax[1][0].set_xlim(-ngphalf, ngphalf)
ax[1][0].set_ylim(-0.2*np.max(lpfct), 1.1*np.max(lpfct))

ax[1][1].set_prop_cycle(color=colors)
ax[1][1].axhline(y=0.0, color='gray', linestyle=':')
ax[1][1].plot(gp[ngpmid-ngphalf:ngpmid+ngphalf], elpfct[ngpmid-ngphalf:ngpmid+ngphalf,:], linewidth=2.0)
ax[1][1].set_title("Corresponding convolution functions", fontsize=15)
ax[1][1].set_xlabel("Grid-point", fontsize=14)
ax[1][1].set_xlim(-ngphalf, ngphalf)
ax[1][1].set_ylim(-0.2*np.max(elpfct), 1.1*np.max(elpfct))

fig.tight_layout(pad=1.0)
filename = 'band_pass_functions.pdf'
plt.savefig(filename, format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + filename + ' ' + filename)
