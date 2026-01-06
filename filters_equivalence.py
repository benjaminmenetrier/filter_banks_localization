#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
nsmax = 100
nf = 4
colors = ["blue","orange","green","red"]
#mode = "low"
#mode = "high"
#mode = "complementary"
mode = "embedded"
#mode = "pseudo-embedded"
if mode == "embedded":
  Lfactor = [0.1, 0.15, 0.2]
else:
  Lfactor = [0.1, 0.2, 0.4]

# Wavenumbers
wn = np.linspace(0, nsmax, nsmax+1)

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
for i in range(nf-1):
  L[i] = Lfactor[i]*(i+1)*(nsmax+1)

# Low-pass filters
lpf = np.zeros((nsmax+1, nf-1))
Lflat = 0.0
for i in range(nf-1):
  if (mode == "embedded") and i > 0:
    for k in range(nsmax+1):
      if k < Lflat:
        lpf[k,i] = 1.0
      else:
        lpf[k,i] = gc99((k-Lflat)/L[i])
  else:
    for k in range(nsmax+1):
      lpf[k,i] = gc99(k/L[i])
  Lflat += L[i]

# High-pass filters (sorted in opposite order)
hpf = np.zeros((nsmax+1, nf-1))
for i in range(nf-1):
  if mode == "high":
    for k in range(nsmax+1):
      hpf[k,i] = gc99((nsmax-k)/L[i])
  elif mode == "complementary":
    for k in range(nsmax+1):
      hpf[k,i] = 1.0-lpf[k,nf-2-i]

# Equivalent band-pass filters
if mode == "pseudo-embedded":
  bpf_from_lpf = np.zeros((nsmax+1, nf))
  bpf_from_hpf = np.zeros((nsmax+1, nf))
  bpf_from_lpf[:,0] = lpf[:,0]
  bpf_from_hpf[:,0] = hpf[:,0]
  for j in range(1,nf-1):
    bpf_from_lpf[:,j] = lpf[:,j]-lpf[:,j-1]
    bpf_from_hpf[:,j] = hpf[:,j]-hpf[:,j-1]
  bpf_from_lpf[:,nf-1] = 1.0-np.sum(bpf_from_lpf[:,0:nf-1], axis=1)
  bpf_from_hpf[:,nf-1] = 1.0-np.sum(bpf_from_hpf[:,0:nf-1], axis=1)
else:
  bpf_from_lpf = np.ones((nsmax+1, nf))
  bpf_from_hpf = np.ones((nsmax+1, nf))
  bpf_from_lpf[:,0] = lpf[:,0]
  bpf_from_hpf[:,0] = hpf[:,0]
  for j in range(1,nf):
    bpf_from_lpf[:,j] *= (1.0-lpf[:,0])
    bpf_from_hpf[:,j] *= (1.0-hpf[:,0])
  for i in range(1,nf-1):
    bpf_from_lpf[:,i] *= lpf[:,i]
    bpf_from_hpf[:,i] *= hpf[:,i]
    for j in range(i+1,nf):
      bpf_from_lpf[:,j] *= (1.0-lpf[:,i])
      bpf_from_hpf[:,j] *= (1.0-hpf[:,i])

# Plot
fig,ax = plt.subplots(nrows=2, figsize=(6,6))
ax[0].set_prop_cycle(color=colors[0:nf-1])
if mode == "low":
  ax[0].plot(wn, lpf, linewidth=2.0)
  ax[0].set_title("Low-pass filters", fontsize=15)
elif mode == "high":
  ax[0].plot(wn, hpf, linewidth=2.0)
  ax[0].set_title("High-pass filters", fontsize=15)
elif mode == "complementary":
  ax[0].plot(wn, lpf, linewidth=2.0)
  ax[0].plot(wn, hpf[:,::-1], linewidth=2.0, linestyle="--")
  ax[0].set_title("Low-pass and complementary high-pass filters", fontsize=15)
elif mode == "embedded":
  ax[0].plot(wn, lpf, linewidth=2.0)
  ax[0].set_title("Embedded low-pass filters", fontsize=15)
ax[0].set_xlabel("Wavenumber", fontsize=14)
ax[0].set_xlim(0, nsmax)
ax[0].set_ylim(0, 1.1)

ax[1].set_prop_cycle(color=colors)
if mode == "low":
  ax[1].plot(wn, bpf_from_lpf, linewidth=2.0)
elif mode == "high":
  ax[1].plot(wn, bpf_from_hpf, linewidth=2.0)
elif mode == "complementary":
  ax[1].plot(wn, bpf_from_lpf, linewidth=2.0)
  ax[1].plot(wn, bpf_from_hpf[:,::-1], linewidth=2.0, linestyle="--")
if mode == "embedded":
  ax[1].plot(wn, bpf_from_lpf, linewidth=2.0)
ax[1].set_title("Corresponding band-pass filters", fontsize=15)
ax[1].set_xlabel("Wavenumber", fontsize=14)
ax[1].set_xlim(0, nsmax)
ax[1].set_ylim(0, 1.1)

fig.tight_layout(pad=1.0)
filename = 'filters_equivalence_' + mode + '.pdf'
plt.savefig(filename, format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ' + filename + ' ' + filename)
