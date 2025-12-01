#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Interpolation functions
def create_interp(c, ci):
  n = len(c)
  ni = len(ci)
  interp = []
  for i in range(n):
    iim = int(i*(ni-1)/(n-1))
    iip = iim+1
    if iip == ni:
      iip = ni-1
      iim = iip-1
    rim = (ci[iip]-c[i])/(ci[iip]-ci[iim])
    rip = (c[i]-ci[iim])/(ci[iip]-ci[iim])
    interp.append([iim,iip,rim,rip])

  return interp

def apply_interp(interpx, interpy, fi, f):
  nx = f.shape[0]
  ny = f.shape[1]
  nxi = fi.shape[0]
  nyi = fi.shape[1]
  f[:,:] = 0.0

  for ix in range(nx):
    ixim = interpx[ix][0]
    ixip = interpx[ix][1]
    rxim = interpx[ix][2]
    rxip = interpx[ix][3]

    for iy in range(ny):
      iyim = interpy[iy][0]
      iyip = interpy[iy][1]
      ryim = interpy[iy][2]
      ryip = interpy[iy][3]

      # Bilinear interpolation
      f[ix,iy] = rxim*ryim*fi[ixim,iyim] + rxim*ryip*fi[ixim,iyip] + rxip*ryim*fi[ixip,iyim] + rxip*ryip*fi[ixip,iyip]

def apply_interp_ad(interpx, interpy, f, fi):
  nx = f.shape[0]
  ny = f.shape[1]
  nxi = fi.shape[0]
  nyi = fi.shape[1]
  fi[:,:] = 0.0

  for ix in range(nx):
    ixim = interpx[ix][0]
    ixip = interpx[ix][1]
    rxim = interpx[ix][2]
    rxip = interpx[ix][3]

    for iy in range(ny):
      iyim = interpy[iy][0]
      iyip = interpy[iy][1]
      ryim = interpy[iy][2]
      ryip = interpy[iy][3]

      # Bilinear interpolation adjoint
      fi[ixim,iyim] += rxim*ryim*f[ix,iy]
      fi[ixim,iyip] += rxim*ryip*f[ix,iy]
      fi[ixip,iyim] += rxip*ryim*f[ix,iy]
      fi[ixip,iyip] += rxip*ryip*f[ix,iy]

def apply_filter(interpx, interpy, norm, f, ff, fr):
  apply_interp_ad(interpx, interpy, f, ff)
  ff *= norm
  apply_interp(interpx, interpy, ff, fr)
  fr[:,:] = f[:,:]-fr[:,:]

# Dimensions
nx = 150
ny = 100
x = np.linspace(0.0, 1.0, nx)
y = np.linspace(0.0, 1.0, ny)

# Define initial field
np.random.seed(10)
rnd = np.random.random((nx, ny))
init1 = gaussian_filter(rnd, sigma=1)
init1 = (init1-np.min(init1))/(np.max(init1)-np.min(init1))
rnd = np.random.random((nx, ny))
init2 = gaussian_filter(rnd, sigma=8)
init2 = (init2-np.min(init2))/(np.max(init2)-np.min(init2))
rnd = np.random.random((nx, ny))
mask = gaussian_filter(rnd, sigma=10)
mask = (mask-np.min(mask))/(np.max(mask)-np.min(mask))
mask = mask**3
init = mask*init1+(1.0-mask)*init2
init = init-np.mean(init)

# Get min/max values
vmax = np.max(np.abs(init))
vmin = -vmax
levels = np.linspace(vmin, vmax, 31)
levels_lines = np.linspace(vmin, vmax, 16)
cmap = "coolwarm"

# Plot initial field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('Initial field (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(init), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(init), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('init.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop init.pdf init.pdf')

# SDL - filter 1
sdl_filt_1 = gaussian_filter(init, sigma=6)
sdl_res_1 = np.zeros((nx,ny))
sdl_res_1[:,:] = init[:,:]-sdl_filt_1[:,:]

# SDL - filter 2
sdl_filt_2 = gaussian_filter(sdl_res_1, sigma=1.3)
sdl_res_2 = np.zeros((nx,ny))
sdl_res_2[:,:] = sdl_res_1[:,:]-sdl_filt_2[:,:]

# SDL - reconstructed
sdl_rec = np.zeros((nx, ny))
sdl_rec += sdl_res_2
sdl_rec += sdl_filt_2
sdl_rec += sdl_filt_1

# Plot filtered field 1
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('SDL - Filtered field 1 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(sdl_filt_1), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(sdl_filt_1), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('sdl_1_filt.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop sdl_1_filt.pdf sdl_1_filt.pdf')

# Plot residual field 1
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('SDL - Residual field 1 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(sdl_res_1), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(sdl_res_1), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('sdl_1_res.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop sdl_1_res.pdf sdl_1_res.pdf')

# Plot filtered field 2
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('SDL - Filtered field 2 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(sdl_filt_2), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(sdl_filt_2), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('sdl_2_filt.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop sdl_2_filt.pdf sdl_2_filt.pdf')

# Plot residual field 2
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('SDL - Residual field 2 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(sdl_res_2), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(sdl_res_2), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('sdl_2_res.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop sdl_2_res.pdf sdl_2_res.pdf')

# Plot reconstructed field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('SDL - Reconstructed field (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(sdl_rec), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(sdl_rec), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('sdl_rec.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop sdl_rec.pdf sdl_rec.pdf')

# IFBL - interpolator 1
nx1 = 15
ny1 = 10
x1 = np.linspace(0.0, 1.0, nx1)
y1 = np.linspace(0.0, 1.0, ny1)
interp1x = create_interp(x, x1)
interp1y = create_interp(y, y1)

# Adjoint test
rnd = np.random.random((nx, ny))
rnd1 = np.random.random((nx1, ny1))
S_rnd1 = np.zeros((nx, ny))
ST_rnd = np.zeros((nx1, ny1))
apply_interp(interp1x, interp1y, rnd1, S_rnd1)
apply_interp_ad(interp1x, interp1y, rnd, ST_rnd)
if np.abs(np.sum(rnd*S_rnd1)-np.sum(rnd1*ST_rnd)) > 1.0e-12:
  print("Adjoint test error")
  exit()

# IFBL - filter 1
ones = np.ones((nx, ny))
norm1 = np.zeros((nx1, ny1))
apply_interp_ad(interp1x, interp1y, ones, norm1)
norm1 = 1.0/norm1
ifbl_filt_1 = np.zeros((nx1, ny1))
ifbl_res_1 = np.zeros((nx, ny))
apply_filter(interp1x, interp1y, norm1, init, ifbl_filt_1, ifbl_res_1)

# IFBL - interpolator 2
nx2 = nx1*4
ny2 = ny1*4
x2 = np.linspace(0.0, 1.0, nx2)
y2 = np.linspace(0.0, 1.0, ny2)
interp2x = create_interp(x, x2)
interp2y = create_interp(y, y2)

# IFBL - filter 2
ones = np.ones((nx, ny))
norm2 = np.zeros((nx2, ny2))
apply_interp_ad(interp2x, interp2y, ones, norm2)
norm2 = 1.0/norm2
ifbl_filt_2 = np.zeros((nx2, ny2))
ifbl_res_2 = np.zeros((nx, ny))
apply_filter(interp2x, interp2y, norm2, ifbl_res_1, ifbl_filt_2, ifbl_res_2)

# IFBL - reconstructed
ifbl_rec = np.zeros((nx, ny))
ifbl_rec += ifbl_res_2
ifbl_filt_interp = np.zeros((nx, ny))
apply_interp(interp2x, interp2y, ifbl_filt_2, ifbl_filt_interp)
ifbl_rec += ifbl_filt_interp
apply_interp(interp1x, interp1y, ifbl_filt_1, ifbl_filt_interp)
ifbl_rec += ifbl_filt_interp

# Plot filtered field 1
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('IFBL - Filtered field 1 (' + str(nx1) + ' x ' + str(ny1) + ')', fontsize=18)
ax.contourf(x1, y1, np.transpose(ifbl_filt_1), levels=levels, cmap=cmap)
ax.contour(x1, y1, np.transpose(ifbl_filt_1), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('ifbl_1_filt.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ifbl_1_filt.pdf ifbl_1_filt.pdf')

# Plot residual field 1
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('IFBL - Residual field 1 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(ifbl_res_1), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(ifbl_res_1), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('ifbl_1_res.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ifbl_1_res.pdf ifbl_1_res.pdf')

# Plot filtered field 2
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('IFBL - Filtered field 2 (' + str(nx2) + ' x ' + str(ny2) + ')', fontsize=18)
ax.contourf(x2, y2, np.transpose(ifbl_filt_2), levels=levels, cmap=cmap)
ax.contour(x2, y2, np.transpose(ifbl_filt_2), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('ifbl_2_filt.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ifbl_2_filt.pdf ifbl_2_filt.pdf')

# Plot residual field 2
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('IFBL - Residual field 2 (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(ifbl_res_2), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(ifbl_res_2), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('ifbl_2_res.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ifbl_2_res.pdf ifbl_2_res.pdf')

# Plot reconstructed field
fig,ax = plt.subplots(ncols=1, nrows=1,figsize=(7,5))
ax.set_title('IFBL - Reconstructed field (' + str(nx) + ' x ' + str(ny) + ')', fontsize=18)
ax.contourf(x, y, np.transpose(ifbl_rec), levels=levels, cmap=cmap)
ax.contour(x, y, np.transpose(ifbl_rec), levels=levels_lines, colors='k', linewidths=0.5)
plt.xticks([], [])
plt.yticks([], [])
plt.savefig('ifbl_rec.pdf', format='pdf', dpi=300)
plt.close()
os.system('pdfcrop ifbl_rec.pdf ifbl_rec.pdf')
