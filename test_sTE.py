# -*- coding: utf-8 -*-
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

#%%
from my_modules.sTE import *
import numpy as np
#%%
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 28 # Font size
#%%
def main():
    fs     = 1000
    dt     = 1/fs
    td_max = 1
    t_step = 0.1
    Nsym   = 3 # Num of symbol
    #%% random case
    # Uncomment this for an example of a time series (Y) clearly anticipating values of X
    
    X = np.random.randint(10, size=5000)
    Y = np.random.randint(10, size=5000)
    sTE, t_tau = calc_sTE_all_tau(X, Y, Nsym, td_max, t_step, fs)
    
    plt.figure()
    plt.plot(t_tau, sTE[:,0], label='$sTE_{x \\rightarrow y}$')
    plt.plot(t_tau, sTE[:,1], label='$sTE_{y \\rightarrow x}$')
    plt.xlabel('delay (s)')
    plt.ylabel('sTE (bits)')
    plt.ylim(0, 4)
    plt.title('random case')
    plt.legend(bbox_to_anchor=(1.05, 1.00), loc='upper left',  borderaxespad=0, fontsize=22, frameon=True)
    plt.show()

    #%% delayed coupling case (x -> y, delay = 0.5s)
    X = np.random.randint(10, size=5000)
    Y = 0.1*np.random.randint(10, size=5000) + 0.9*np.hstack([np.zeros(int(0.5*fs)), X[:-int(0.5*fs)]])    
    
    sTE, t_tau = calc_sTE_all_tau(X, Y, Nsym, td_max, t_step, fs)
    
    plt.figure()
    plt.plot(t_tau, sTE[:,0], label='$sTE_{x \\rightarrow y}$')
    plt.plot(t_tau, sTE[:,1], label='$sTE_{y \\rightarrow x}$')
    plt.xlabel('delay (s)')
    plt.ylabel('sTE (bits)')
    plt.ylim(0, 4)
    plt.title('delayed coupling case \n($x \\rightarrow y$, delay:0.5s)')
    plt.legend(bbox_to_anchor=(1.05, 1.00), loc='upper left',  borderaxespad=0, fontsize=22, frameon=True)
    plt.show()
    
    return sTE
    #%%
if __name__ == "__main__":
    sTE= main()