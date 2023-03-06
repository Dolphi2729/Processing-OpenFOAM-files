# file: plot_dp.py

import numpy as np
import pandas as pd
from os import system
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import own_mpl_style_module as own

own.matplotlib_header()

wdir = './'

inj_f_1 = '../injections/right_values/grids/24mln_3_p'
# inj_f_2 = '../injections/right_values/grids/5mln_3_p'
# inj_f_3 = '../injections/right_values/grids/13mln_3_p'
# inj_f_4 = '../injections/right_values/grids/24mln_3_p'
#inj_f = 'probes_with_inj/U.dat'
#inj_f = 'probes_with_inj/V.dat'

#les_f = 'probes_wout_inj/U.dat'
#les_f = 'probes_wout_inj/V.dat'
les_f = '../injections/right_values/dif_places/13mln_no_jet_p'

saveto = wdir

# LES probes files has packs of 4 probes -> fft averaging
# velocity

# pressure

#t_0, p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0, p9_0, p10_0, p11_0, p12_0, p13_0, p14_0, p15_0, p16_0, p17_0, p18_0 = np.loadtxt(wdir + les_f, unpack=True, delimiter=',')
t_0, p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0, p9_0, p10_0, p11_0, p12_0, p13_0, p14_0, p15_0, p16_0, p17_0, p18_0 = np.loadtxt(wdir + les_f, unpack=True)
#t_1, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1, p18_1 = np.loadtxt(wdir + inj_f, unpack=True, delimiter=',')
t_1, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1, p18_1 = np.loadtxt(wdir + inj_f_1, unpack=True)
# t_2, p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2, p16_2, p17_2, p18_2 = np.loadtxt(wdir + inj_f_2, unpack=True)
# t_3, p1_3, p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, p8_3, p9_3, p10_3, p11_3, p12_3, p13_3, p14_3, p15_3, p16_3, p17_3, p18_3 = np.loadtxt(wdir + inj_f_3, unpack=True)
# t_4, p1_4, p2_4, p3_4, p4_4, p5_4, p6_4, p7_4, p8_4, p9_4, p10_4, p11_4, p12_4, p13_4, p14_4, p15_4, p16_4, p17_4, p18_4 = np.loadtxt(wdir + inj_f_4, unpack=True)
# vectorization
# vectorization
p_0 = [p1_0, p2_0, p3_0, p4_0, p5_0, p6_0, p7_0, p8_0, p9_0, p10_0, p11_0, p12_0, p13_0, p14_0, p15_0, p16_0, p17_0, p18_1] 
p_1 = [p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1, p18_1]
# p_2 = [p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2, p16_2, p17_2, p18_2] 
# p_3 = [p1_3, p2_3, p3_3, p4_3, p5_3, p6_3, p7_3, p8_3, p9_3, p10_3, p11_3, p12_3, p13_3, p14_3, p15_3, p16_3, p17_3, p18_3] 
# p_4 = [p1_4, p2_4, p3_4, p4_4, p5_4, p6_4, p7_4, p8_4, p9_4, p10_4, p11_4, p12_4, p13_4, p14_4, p15_4, p16_4, p17_4, p18_4] 

# normalization
rho = 1.19  # [kg/m^3]
D = 37.63 * 1e-3    # [m]
Qc = 174.6  # [m^3/hour]

# x = [t_0, t_1, t_2, t_3, t_4]
# y = [p_0, p_1, p_2, p_3, p_4]
# c = ['r', 'b', 'black', 'g', 'orange']
x = [t_0, t_1]
y = [p_0, p_1]
c = ['r', 'b']
s = [0.1, 31400]
l = [r'Case\,2']
M = 1.25
m = 0.50
M1 = 29

for i in range(len(x)):
  #Ub = m * Qc / (np.pi * (D / 2)**2) / (60**2)  # [m/s]
  Ub = 4.99 # [m/s]
  x[i] = (x[i] - np.min(x[i])) / D * Ub
  y[i] = [y[i][j] / (rho * Ub**2.) for j in range(17)] # p normalization
  #y[i] = [y[i][j] / Ub for j in range(16)] # V normalization
  dt = 1.25e-5 / D * Ub  # only simulations

  x0 = np.linspace(start=np.min(x[i]), stop=np.max(x[i]), num=np.size(x[i]), endpoint=True)
  #for j in range(np.shape(y[i])[0]):
    # if i == 0 or (i == 1 and not PIV_SIM): ph = savgol_filter(y[i][j], 5, 3)  # only simulations
    #y[i][j] = fph(x0)
  #x[i] = x0


# ---------------------
# Layout 1 x 2
# ---------------------
fig, ax = plt.subplots(nrows=1, ncols=2)
plt.subplots_adjust(left=0.052, bottom=0.2, top=0.8, right=0.95, wspace=0.104)
# ---------------------
# Signal
# ---------------------

for i in range(len(x)):
  #ax[0].plot(x[i], y[i][4], c[i], ls='solid')
  ax[0].plot(x[i], y[i][4] - y[i][0], c[i], ls='solid')

#ax[0].set_xlim([0.0, 7])
#ax[0].set_xlim([-2.0, 0])
#ax[0].set_ylim([-12, 12])

atext = AnchoredText(r'$tU_b/D$',
                     loc='lower right',
                     pad=0.15,
                     borderpad=0.1,
                     bbox_to_anchor=(1.0, 0.0),
                     bbox_transform=ax[0].transAxes,
                     frameon=True,
                     prop = dict(size = 12))
ax[0].add_artist(atext)
ax[0].set(title = ''' Сигнал давления Сетка: 24mln
Инжекция 3\%''')
ax[0].title.set_size(10)
# atext = AnchoredText(r'$V/U_b$',
atext = AnchoredText(r'$\Delta p/(\rho U_b^2)$',
                     loc='upper left',
                     pad=0.15,
                     borderpad=0.1,
                     bbox_to_anchor=(0.0, 1.0),
                     bbox_transform=ax[0].transAxes,
                     prop = dict(size = 12),
                     frameon=True)
ax[0].set_ylim([-10,12])
ax[0].set_xlim([0,11.5])
ax[0].add_artist(atext)
# ---------------------
# FFT
# ---------------------

for i in range(len(x)):
  # fft_avg = 0.
  f = [j for j in range(len(x[i]))] / (np.max(x[i]) - np.min(x[i]))
  #y1_0 = abs(np.fft.fft(y[i][2] - y[i][0], norm='ortho'))
  #y2_0 = abs(np.fft.fft(y[i][3] - y[i][0], norm='ortho'))
  #y3_0 = abs(np.fft.fft(y[i][4] - y[i][0], norm='ortho'))
  #y4_0 = abs(np.fft.fft(y[i][5] - y[i][0], norm='ortho'))
  y1 = abs(np.fft.fft(y[i][1] - y[i][0], norm='ortho'))
  y2 = abs(np.fft.fft(y[i][2] - y[i][0], norm='ortho'))
  y3 = abs(np.fft.fft(y[i][3] - y[i][0], norm='ortho'))
  y4 = abs(np.fft.fft(y[i][4] - y[i][0], norm='ortho'))
  y5 = abs(np.fft.fft(y[i][5] - y[i][0], norm='ortho'))
  y6 = abs(np.fft.fft(y[i][6] - y[i][0], norm='ortho'))
  y7 = abs(np.fft.fft(y[i][7] - y[i][0], norm='ortho'))
  y8 = abs(np.fft.fft(y[i][8] - y[i][0], norm='ortho'))
  y9 = abs(np.fft.fft(y[i][9] - y[i][0], norm='ortho'))
  y10 = abs(np.fft.fft(y[i][10] - y[i][0], norm='ortho'))
  y11 = abs(np.fft.fft(y[i][11] - y[i][0], norm='ortho'))
  y12 = abs(np.fft.fft(y[i][12] - y[i][0], norm='ortho'))
  y13 = abs(np.fft.fft(y[i][13] - y[i][0], norm='ortho'))
  y14 = abs(np.fft.fft(y[i][14] - y[i][0], norm='ortho'))
  y15 = abs(np.fft.fft(y[i][15] - y[i][0], norm='ortho'))
  y16 = abs(np.fft.fft(y[i][16] - y[i][0], norm='ortho'))
  fft = (y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8 + y9 + y10 + y11 + y12 + y13 + y14 + y15 + y16) / 16
  #fft_1 = (y1 + y2 + y3) / 3
    # fft_avg = fft_avg + fft
  #ax[1].plot(f, fft_1,
  #           color='b',
  #           linestyle='solid',
  #           marker='',
  #           ms=5,
  #           mfc='none')
  ax[1].plot(f, fft,
             color=c[i],
             linestyle='solid',
             marker='',
             ms=5,
             mfc='none')
  ax[1].set(title = '''Спектр сигнала Сетка: 24mln
  Инжекция 3\%''')
  ax[1].title.set_size(10)        
  # find local maximum
  ind = np.argmax(fft[0:500]) # find ind corresponding to max fft in first 500 point
  print("Ub   [m/s]: {0: .2e}".format(Ub))
  print("max.f ind : {0: 2d}".format(ind))
  print("corr.fft  : {0: .2e}".format(fft[ind]))
  print("freq [1/s]: {0: .2f}".format(f[ind] * Ub / D))
  print("freq   [-]: {0: .2e}".format(f[ind]))
  print("freq of the runnner    [-]: {0: .2e}".format(40.53333333333333*D/Ub))
  print("freq of the runnner x5 [-]: {0: .2e}".format(40.53333333333333*5*D/Ub))

#ax[1].set_xlim([0.1, 6])
ax[1].set_ylim([-5., 60])
#ax[1].set_ylim([0.3, 25])
ax[1].set_xlim([0.2, 4])
atext = AnchoredText(r'$f D / U_b$',
                     loc='lower right',
                     pad=0.15,
                     borderpad=0.1,
                     bbox_to_anchor=(1.0, 0.0),
                     prop = dict(size = 12),
                     bbox_transform=ax[1].transAxes,
                     frameon=True)
ax[1].add_artist(atext)

atext = AnchoredText(r'$a.u.$',
                     loc='upper left',
                     pad=0.15,
                     borderpad=0.1,
                     bbox_to_anchor=(0.0, 1.0),
                     bbox_transform=ax[1].transAxes,
                     prop = dict(size = 12),
                     frameon=True)
ax[1].add_artist(atext)

file_pdf = saveto + 'spec' + '.pdf'
file_png = saveto + 'spec' + '.png'
plt.savefig(file_pdf, pad_inches=0)
system('convert -density 300 ' + file_pdf + ' ' + file_png)
system('file ' + file_pdf)

plt.savefig('3% (24 mln).png')
#plt.close()
