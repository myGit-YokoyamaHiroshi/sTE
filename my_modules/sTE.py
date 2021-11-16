# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 11:24:50 2021

@author: yokoyama

This script is modified version for : https://github.com/mariogutierrezroig/smite

"""
# -*- coding: utf-8 -*-
from scipy import stats
import numpy as np

def calc_sTE_all_tau(x_sig, y_sig, Nsym, t_tau_max, t_step, fs):
    Nt    = len(x_sig)
    t_tau = np.arange(0, t_tau_max + t_step, t_step)
    tau   = (t_tau * fs).astype(int)
    
    sTE = np.zeros((len(tau), 2))
    win = Nt - tau.max() - 2
    
    symX  = symbolize(x_sig, Nsym)
    symY  = symbolize(y_sig, Nsym)
    
    del x_sig, y_sig
    
    cnt = 0
    for d in tau:
        x = symX[:win]
        y = symY[:win]
        X = symX[d:win+d]
        Y = symY[d:win+d]
        
        x2y = symbolic_transfer_entropy(x, y, Y)
        y2x = symbolic_transfer_entropy(y, x, X)
        
        sTE[cnt,:] = np.array([x2y, y2x])
        cnt += 1
    
    return sTE, t_tau

def symbolize(X, m):
    """
    Converts numeric values of the series to a symbolic version of it based
    on the m consecutive values.
    
    Parameters
    ----------
    X : Series to symbolize.
    m : length of the symbolic subset.
    
    Returns
    ----------
    List of symbolized X

    """
    
    X = np.array(X)

    if m >= len(X):
        raise ValueError("Length of the series must be greater than m")
    
    dummy = []
    for i in range(m):
        l = np.roll(X,-i)
        dummy.append(l[:-(m-1)])
    
    dummy = np.array(dummy)
    
    symX = []
    
    for mset in dummy.T:
        rank = stats.rankdata(mset, method="min")
        symbol = np.array2string(rank, separator="")
        symbol = symbol[1:-1]
        symX.append(symbol)
        
    return symX


def symbolic_transfer_entropy(sym_x, sym_y, sym_Y):
    """
    Computes sTE(X->Y), the transfer of entropy from symbolic series X to Y.
    
    Parameters
    ----------
    sym_x : Symbolic series X(t).
    sym_y : Symbolic series Y(t).
    sym_Y : Symbolic series Y(t + dt).
    
    Returns
    ----------
    Value for symbolic transfer entropy

    """

    if len(sym_x) != len(sym_y):
        raise ValueError('All arrays must have same length')
        
    sym_x = np.array(sym_x)
    sym_y = np.array(sym_y)
    sym_Y = np.array(sym_Y)
    
    
    jp_Yy  = symbolic_joint_probabilities(sym_Y, sym_y)
    p_y    = symbolic_probabilities(sym_y)
    
    jp_yx  = symbolic_joint_probabilities(sym_y, sym_x)
    jp_Yyx = symbolic_joint_probabilities_triple(sym_Y, sym_y, sym_x)
    
    H_Yy   = joint_entropy(jp_Yy)
    H_y    = entropy(p_y)
    
    H_yx   = joint_entropy(jp_yx)
    H_Yyx  = joint_entropy_triple(jp_Yyx)
    
    H_Y_given_y  = H_Yy - H_y
    H_Y_given_yx = H_Yyx - H_yx
    
    sTE    = H_Y_given_y - H_Y_given_yx
    
    return sTE

########### function for calculating porbability p(x)
def symbolic_probabilities(symX):
    """
    Computes the conditional probabilities where M[A][B] stands for the
    probability of getting B after A.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with conditional probabilities

    """

    symX = np.array(symX)
    
    # initialize
    p = {}
    n = len(symX)

    for xi in symX:
        if xi in p: 
            p[xi] += 1.0 / n
        else:
            p[xi] = 1.0 / n
    
    return p

########### function for calculating joint porbabilities p(x, y), p(x, y, z)
def symbolic_joint_probabilities(symX, symY):
    """
    Computes the joint probabilities where M[yi][xi] stands for the
    probability of ocurrence yi and xi.
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symbols: Collection of symbols. If "None" calculated from symX
    
    Returns
    ----------
    Matrix with joint probabilities

    """

    if len(symX) != len(symY):
        raise ValueError('All arrays must have same length')
        
    symX = np.array(symX)
    symY = np.array(symY)
    
    # initialize
    jp = {}
    n = len(symX)

    for yi, xi in zip(symY,symX):
        if yi in jp:
            if xi in jp[yi]:
                jp[yi][xi] += 1.0 / n
            else:
                jp[yi][xi] = 1.0 / n
        else:
            jp[yi] = {}
            jp[yi][xi] = 1.0 / n
    
    return jp


def symbolic_joint_probabilities_triple(symX, symY, symZ):
    """
    Computes the joint probabilities where M[y][z][x] stands for the
    probability of coocurrence y, z and x p(y,z,x).
    
    Parameters
    ----------
    symX : Symbolic series X.
    symY : Symbolic series Y.
    symZ : Symbolic series Z.
    
    Returns
    ----------
    Matrix with joint probabilities p(Y, Z, X)

    """
    
    if (len(symX) != len(symY)) or (len(symY) != len(symZ)):
        raise ValueError('All arrays must have same length')

    symX = np.array(symX)
    symY = np.array(symY)
    symZ = np.array(symZ)
    
    # initialize
    jp = {}
    n = len(symX)

    for x, y, z in zip(symX,symY,symZ):
        if y in jp:
            if z in jp[y]:
                if x in jp[y][z]:
                    jp[y][z][x] += 1.0 / n
                else:
                    jp[y][z][x] = 1.0 / n
            else:                
                jp[y][z] = {}
                jp[y][z][x] = 1.0 / n
        else:
            jp[y] = {}
            jp[y][z] = {}
            jp[y][z][x] = 1.0 / n
    
    return jp# jp[Y][Z][X]

###############################################
def entropy(p_x):
    Hx = 0
    for xi in list(p_x.keys()):
        try:
            Hx += -p_x[xi] * np.log(p_x[xi]) / np.log(2.)
        except KeyError:
            continue
    
    return Hx

def joint_entropy(p_xy):
    Hxy = 0
    
    for yi in list(p_xy.keys()):
        for xi in list(p_xy[yi].keys()):
            try:
                Hxy += -p_xy[yi][xi] * np.log(p_xy[yi][xi]) / np.log(2.)
            except KeyError:
                continue
    
    return Hxy

def joint_entropy_triple(p_xyz):
    Hxyz = 0
    
    for yi in list(p_xyz.keys()):
        for zi in list(p_xyz[yi].keys()):
            for xi in list(p_xyz[yi][zi].keys()):
                try:
                    Hxyz += -p_xyz[yi][zi][xi] * np.log(p_xyz[yi][zi][xi]) / np.log(2.)
                except KeyError:
                    continue
    
    return Hxyz