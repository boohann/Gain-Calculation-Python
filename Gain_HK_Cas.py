# Program to Calculate Gain from a Spectra based on Hakki-Paoli 'Gain spectra in GaAs double-heterostructure injection lasers' (1974) [Basil W. Hakki and Thomas L. Paoli] method and Cassidy sum/min 'Technique for measurement of gain spectra of semiconductor diode laser' (1984) [Daniel T. Cassidy].

# Written by Niall Boohan (2016) || boohann@tcd.ie

import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import csv

filename = raw_input('Enter a filename (nm,dBm spaced .txt): ')         #reads in text file of wavelenght (nm), Amp (dBm)  
L = raw_input("Please enter device length in microns: ")
R_f =  raw_input("Please enter front facet reflectivity: ")
R_r  =  raw_input("Please enter rear facet reflectivity: ")
a_i  =  raw_input("Please enter internal modal loss in cm^-1: ")



###### Convert input strings to float values ##############

L = float(L)
R_f = float(R_f)
R_r = float(R_r)
a_i = float(a_i)
L_c = L/1e4                                                             # coverts device length in microns to centimeters
L_m = L/1e6

########## Takes text files splits into arrays of floats ##### 

rawdata  = open(filename).read().splitlines()

WL = []
dBm = []
mpp = []


for x in rawdata:
    p = x.split(',')
    WL.append(float(p[0]))
    dBm.append(float(p[1]))

WL = np.array(WL)
dBm = np.array(dBm)


####### convert dBm to mW  ###############

mW = [1*10**(x/10.0) for x in dBm]



################## smooth the data using moving point average #########################



def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


mW_mo_av = movingaverage(mW, 20)

######### Refit the smoothed data points to the original spectrum ####################
# introduces ~0.25nm error to wavelength.

old_min = 0
old_max = len(mW_mo_av)
new_min = WL[0]
new_max = WL[-1]

z = np.arange(0, len(mW_mo_av))
y = [float(i) for i in z]
WL_new = [((o - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min for o in y]
WL_new = np.array(WL_new)



########## Select max and min points ########################

P = mW_mo_av[argrelextrema(mW_mo_av, np.greater)[0]]                # for local maxima of amplitude
V = mW_mo_av[argrelextrema(mW_mo_av, np.less)[0]]                   # for local minima of amplitude
WLP = WL_new[argrelextrema(mW_mo_av, np.less)[0]]                   # wavelength points between max and min
P_ind = argrelextrema(mW_mo_av, np.greater)[0]                      # index each point for cassidy measurement


max_min_data = zip(P, V, WLP, P_ind)
rm = [v for v in max_min_data if v[0] >= 1e-7] 
P_rm = [x[0] for x in rm]
V_rm = [x[1] for x in rm]
WL_rm = [x[2] for x in rm]
Avg_P = [(y + x)/2 for x,y in zip(P_rm, P_rm[1:])]
P_act_modes = [x[3] for x in rm]



################# ratio between min and averaged max ##########

r = [(x/y) for x,y in zip(Avg_P, V_rm)]
WL_rm.pop()


################## square root of ratio #######################

srr = [np.sqrt(x) for x in r]
rm_srr = zip(WL_rm, srr)
srr_data_points = [x for x in rm_srr if x[1] > 1]                   # removes erroneous gain points
srr_WL = [x[0] for x in srr_data_points]
srr_dp = [x[1] for x in srr_data_points]

################# HK gain calculation #################### 



G_HK = [-((1/L_c)*np.log((q+1)/(q-1)) + (1/L_c)*np.log(np.sqrt(R_f*R_r))) + a_i for q in srr_dp]


########## output gain data to file ###################

data_out = zip(srr_WL, G_HK)
with open('Gain_HK.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(data_out)
########### calculation of gain threshold #############

G_th = a_i + (1/(2*L_c))*np.log(1/(R_f*R_r))


########### Cassidy method of gain calculation ########


P_mode = [range(x, y) for x,y in zip(P_act_modes, P_act_modes[1:])] # select all intrapoints between maxima
power = [[mW[x] for x in y] for y in P_mode]                        # get corresponding power value
WL_C = [[WL[x] for x in y] for y in P_mode]
power_sum = [sum(a)/len(a) for a in power]                          # sum the power values
V_rm.pop()
cas_data = zip(power_sum, V_rm)                                     # combine peaks and vallies into list of tuples for cassidy comparison
p = [x/y for x,y in cas_data]                                       # compare summed/N Samples to local minima
rm_cas = zip(p, srr_WL)
p_sort= [x for x in rm_cas if x[0] > 1]
p_C = [x[0] for x in p_sort]
WL_C = [x[1] for x in p_sort]
G_C = [(1/(2*L_c))*np.log(1/(R_f*R_r)) + (1/L_c)*np.log((x-1)/(x+1)) + a_i for x in p_C]


########## output gain data to file ###################

data_out_cas = zip(WL_C, G_C)
with open('Gain_C.csv', 'w') as f:
	writer = csv.writer(f, delimiter=',')
	writer.writerows(data_out_cas)


##################### plot data ###########################

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(srr_WL, G_HK, 'b', label = 'Hakki-Paoli')
plt.plot((srr_WL[0], srr_WL[-1]), (G_th, G_th), 'r-', label = 'Threshold Calculated')
plt.title('Hakki-Paoli Gain')
plt.xlabel('Wavelength $(nm)$')
plt.ylabel('Modal Gain $(cm^{-1})$')
plt.plot(WL_C, G_C, 'g', label = 'Cassidy')
plt.plot((WL_C[0], WL_C[-1]), (G_th, G_th), 'r-')
plt.xlabel('Wavelength $(nm)$')
plt.ylabel('Modal Gain $(cm^{-1})$')
plt.title('Cassidy sum/min Gain')
plt.legend()
plt.show()
