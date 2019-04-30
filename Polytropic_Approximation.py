# -*- coding: utf-8 -*-
#
# This program checks the accuracy of the approximations developed in the 
# article "A Polytropic Approximation of Compressible Flow in Pipes with
# Friction" by selecting random cases of adiabatic frictional compressible
# pipe flow, iteratively calcuating the theoretical value of the net expansion
# factor, and comparing that value to the approximate formulas developed in the
# paper.  The same approach is used to check the accuracy of the approximations
# for choking pressure drop ratios.
#
# Author:  William M. Kirkland
# Date:  January 2019
# Email:  kirklandwm@ornl.gov
#
# Copyright 2019.  All rights reserved. This work is without any warranty 
# whatsoever, express or implied.  This calculation may not be distributed, 
# duplicated, or modified without consent of the author.
# 
# Instructions:
# To perform an accuracy calculation as described in the paper, run this file 
# in Python 2.7, e.g.,
#   $ python Polytropic_Approximation.py
# Results will print to stdout

from math import acos, cos, exp, isnan, log, pi, sqrt
from numba import jit
from numpy import array
from numpy import zeros
from numpy.random import random
from scipy.interpolate import interp1d
from scipy.optimize import brentq

@jit
def Kfactor(gamma, M):
# Calculates the critical length friction loss for a given ratio of specific
# heats gamma and initial Mach number M.
# Source:
#     J.D. Anderson, *Modern Compressible Flow with Historical Perspective*,
#     3rd Edition, Mc-Graw-Hill, Boston, 2003.  Eq. (3.107), p. 114.
    return (1-M**2)/(gamma*M**2) + (gamma + 1)/(2*gamma) * log((gamma+1)*M**2
            /(2 + (gamma-1)*M**2))

@jit
def P2_over_P1(gamma, M1, M2):
# Calculates the final to initial pressure ratio for Fanno flow given the 
# ratio of specific heats gamma initial Mach number M1, and final Mach number
# M2.
# Source:  Anderson, op. cit., Eq. (3.100), p. 113.
    return M1/M2*sqrt((2+(gamma-1)*M1**2)/(2+(gamma-1)*M2**2))

@jit
def Find_M2(gamma, M1, PressRatio):
# Finds the final Mach number M2 given gamma, initial Mach number, and pressure
# ratio, using *brentq* numerical solver.
    return brentq(func_for_Find_M2, args=(gamma, M1, PressRatio), a=1.e-12, 
                  b=1000.)

@jit
def func_for_Find_M2(M2, gamma, M1, PressRatio):
# Wrapper function used by Find_M2().
    return P2_over_P1(gamma, M1, M2) - PressRatio

@jit
def Find_Choked(gamma, K):
# Calculates the choking pressure ratio given gamma and friction loss
# coefficient K, by numerically solving for an intial Mach number whose 
# critical pressure loss matches the given K.
    M1 = brentq(func_for_Find_Choked, args=(gamma, K), a=1.e-12, b=1.)
    return M1/sqrt((gamma + 1.)/(2. + (gamma - 1.)*M1**2))

@jit
def func_for_Find_Choked(M, gamma, K):
# Wrapper function used by Find_Choked().
    return Kfactor(gamma, M) - K

@jit
def Find_Y(gamma, K, DP_over_P1):
# Calculates net expansion factor Y for given gamma, K, and pressure drop ratio
# DP_over_P1.  Formula derived by solving mass flow rate equation for Y:
#     $$\rho_1 u_1 A = rho_1 A M_1 \sqrt{\gamma R T_1} = 
#        AY \sqrt{\frac{2 \Delta P \rho_1}{K}}$$
#     $$ Y = M_1 \sqrt{\frac{\gamma K P_1}{2 \Delta P} $$
    PressRatio = max(1. - DP_over_P1, Find_Choked(gamma, K))
    M1 = brentq(func_for_Find_Y, args=(gamma, PressRatio, K), a=1.e-6, b=1.)
    return M1 * sqrt(gamma*K / (2.*(1. - PressRatio)))

@jit
def func_for_Find_Y(M1, gamma, PressRatio, K):
# Wrapper function used by Find_Y().
    return Kfactor(gamma, M1) - Kfactor(gamma, Find_M2(gamma, M1, PressRatio)
        ) - K

@jit
def Find_Polytropic_Y(Poly_Eqn, gamma, K, DP_over_P1):
# Calculates the Y factor using the polytropic approximation, given by Eq. (31)
# in the current paper.
    c = Poly_Eqn(gamma, K, DP_over_P1)
    return sqrt(c/((c + 1.)*DP_over_P1)*((1. - DP_over_P1)**((c + 1.)/c) 
                - 1.)/(2./(c*K)*log(1. - DP_over_P1) - 1.))

@jit
def Isothermal_Poly_Eqn(gamma, K, DP_over_P1):
# Makes Find_Polytropic_Y() calculate the isothermal Y factor using Eq. (32)
# in the current paper
    return 1.

@jit
def Simple_Poly_Eqn(gamma, K, DP_over_P1):
# Calculates the polytropic index using Eq. (25) in the current paper.
    return 1. + 1.8*DP_over_P1*(gamma - 1.)/(K + 1.)

@jit
def Full_Poly_Eqn(gamma, K, DP_over_P1):
# Calculates the polytropic index using Eq. (24) in the current paper.
    return (1. - gamma)/(1. + 2./(K*(1. - gamma))*((1. - DP_over_P1)**(
        (gamma - 1.)/gamma) - 1.)) + gamma

@jit
def Find_Crane_Y(gamma, K, DP_over_P1):
# Calculates the net expansion factor Y by linearly interpolating from the
# critical values given in Crane Technical Paper 410, then linearly
# interpolating between the critical point and zero.  Works for gamma = 1.3 
# or 1.4 only
    if gamma == 1.3:
        ii = 0
    elif gamma == 1.4:
        ii = 1
    else:
        return 0
    
    Karray = array([1.2, 1.5, 2., 3., 4., 6., 8., 10., 15., 20., 40., 100.])
    DParray = array([[.525, .550, .593, .642, .678, .722,
                     .750, .773, .807, .831, .877, .920],
                    [.552, .576, .612, .662, .697, .737,
                     .762, .784, .818, .839, .883, .926]])
    Yarray = array([[.612, .631, .635, .658, .670, .685,
                     .698, .705, .718, .718, .718, .718],
                    [.588, .606, .622, .693, .649, .671, 
                     .685, .695, .702, .710, .710, .710]])
    
    DPchoke = interp1d(Karray, DParray[ii],bounds_error=False,
                       assume_sorted=True)
    DPchoke = DPchoke(K)
    Ychoke = interp1d(Karray, Yarray[ii],bounds_error=False, 
                      assume_sorted=True)
    Ychoke = Ychoke(K)
    
    return 1. - DP_over_P1/DPchoke*(1. - Ychoke)

@jit
def Choked_Trig(gamma, K):
# Finds the choking pressure drop ratio by solving Eq. (33) in the current 
# paper using the trigonometric cubic formula.
# Source:
#     *CRC Standard Mathematical Tables*, 19th Ed., S.M. Selby, ed., p. 104
    C1 = 1. + gamma * K
    C2 = (3. + C1**2)/9.
    C3 = (2.*C1**3 + 9.*C1 - 27.)/54.
    M = C1/3. + 2.*sqrt(C2)*cos(acos(C3/C2**1.5)/3. + 4.*pi/3.)
    return 1. - M*sqrt((2. + (gamma-1.)*M**2)/(gamma+1.))

@jit
def Choked_Best_Fit(gamma, K):
# Finds the choking pressure drop ratio using Eq. (34) in the current paper.
    return exp( -exp(0.43 - 0.6*gamma - 0.43*log(K)))

@jit
def Choked_Best_Fit2(gamma, K):
# Finds the choking pressure drop ratio using Eq. (35) in the current paper.
    return exp( -exp(0.43 - 0.6*gamma - 0.0005*K - 0.43*log(K)))

def main():    
    num_tests = 1000000
    num_entries = 5
    
    # array indexes
    theory = 0
    full_poly = 1
    simple_poly = 2
    isothermal = 3
    crane = 4
    trig = 1
    fitted = 2

    # initialize arrays
    values = zeros(num_entries)
    error = zeros(num_entries)
    sums = zeros(num_entries)
    maxerr = zeros(num_entries)
    maxK = zeros(num_entries)
    maxgam = zeros(num_entries)
    maxDP = zeros(num_entries)
    minerr = zeros(num_entries)
    minK = zeros(num_entries)
    mingam = zeros(num_entries)
    minDP = zeros(num_entries)

    # Check accuracy of approximations for mass flow rate/net expansion factor
    for ii in range(num_tests):
        gamma = random()*0.67 + 1.
        K = (1./random() - 2./3.)*3.
        DP_over_P1 = random() * (1. - Find_Choked(gamma, K))

        values[theory] = Find_Y(gamma, K, DP_over_P1)
        values[full_poly] = Find_Polytropic_Y(Full_Poly_Eqn, gamma, K, 
             DP_over_P1)
        values[simple_poly] = Find_Polytropic_Y(Simple_Poly_Eqn, gamma, K, 
              DP_over_P1)
        values[isothermal] = Find_Polytropic_Y(Isothermal_Poly_Eqn, gamma, K, 
              DP_over_P1)
        for i in range(full_poly, isothermal+1):
            error[i] = (values[i] - values[theory]) / values[theory]
            if error[i] > maxerr[i]:
                maxerr[i] = error[i]
                maxK[i] = K
                maxgam[i] = gamma
                maxDP[i] = DP_over_P1
            if error[i] < minerr[i]:
                minerr[i] = error[i]
                minK[i] = K
                mingam[i] = gamma
                minDP[i] = DP_over_P1
            if isnan(error[i]):
                ii -= 1
            else:
                sums[i] = sums[i] + error[i]**2
        if random() > 0.5:
            gamma = 1.4
        else:
            gamma = 1.3
        values[theory] = Find_Y(gamma, K, DP_over_P1)
        values[crane] = Find_Crane_Y(gamma, K, DP_over_P1)
        error[crane] = (values[crane] - values[theory])/values[theory]
        error[crane] = (values[crane] - values[theory]) / values[theory]
        if error[crane] > maxerr[crane]:
            maxerr[crane] = error[crane]
            maxK[crane] = K
            maxgam[crane] = gamma
            maxDP[crane] = DP_over_P1
        if error[crane] < minerr[crane]:
            minerr[crane] = error[crane]
            minK[crane] = K
            mingam[crane] = gamma
            minDP[crane] = DP_over_P1
        if isnan(error[crane]):
            ii -= 1
        else:
            sums[crane] = sums[crane] + error[crane]**2
 
    lables = (["Theoretical Value",
        "Polytropic Approximation:  Full Equation for Polytropic Index:",
        "Polytropic Approximation:  Simple Equation for Polytropic Index:",
        "Approximating Adiabatic Flow as Isothermal:",
        "Use of Charts in Crane Tech. Paper 410 (gamma = 1.3 or 1.4 only):"])

    for i in range(1,crane+1):
        print(lables[i])
        print("\tStandard deviation is " + str(sqrt(sums[i]/num_tests)))
        print("\tMaximum error is "+ str(maxerr[i]) + " at K = " + str(maxK[i])
            + ", gamma = " + str(maxgam[i]) + ", and DP/P1 = " + str(maxDP[i]))
        print("\tMinimum error is "+ str(minerr[i]) + " at K = " + str(minK[i])
            + ", gamma = " + str(mingam[i]) + ", and DP/P1 = " + str(minDP[i]))
    print("\n")
    
    # Reinitialize arrays for checking choking pressure
    num_entries = 4
    values = zeros(num_entries)
    error = zeros(num_entries)
    sums = zeros(num_entries)
    maxerr = zeros(num_entries)
    maxK = zeros(num_entries)
    maxgam = zeros(num_entries)
    maxDP = zeros(num_entries)
    minerr = zeros(num_entries)
    minK = zeros(num_entries)
    mingam = zeros(num_entries)
    minDP = zeros(num_entries)
    
    # Check accuracy of approximations for choking pressure
    for ii in range(num_tests):
        gamma = random()*0.67 + 1.
        K = (1./random() - 2./3.)*3.
        values[theory] = 1. - Find_Choked(gamma, K)
        values[trig] = Choked_Trig(gamma, K)
        values[fitted] = Choked_Best_Fit(gamma, K)
        values[fitted+1] = Choked_Best_Fit2(gamma, K)
        for i in range(trig, fitted+2):
            error[i] = (values[i] - values[theory]) / values[theory]
            if error[i] > maxerr[i]:
                maxerr[i] = error[i]
                maxK[i] = K
                maxgam[i] = gamma
                maxDP[i] = DP_over_P1
            if error[i] < minerr[i]:
                minerr[i] = error[i]
                minK[i] = K
                mingam[i] = gamma
                minDP[i] = DP_over_P1
            if isnan(error[i]):
                ii -= 1
            else:
                sums[i] = sums[i] + error[i]**2
        
    lables = (["Theoretical Value",
        "Trigonometric Approximation of Choking Pressure:",
        "Best Fit Equation for Choking Pressure:",
        "Modified Best Fit Equation:"])
    
    for i in range(1,fitted+2):
        print(lables[i])
        print("\tStandard deviation is " + str(sqrt(sums[i]/num_tests)))
        print("\tMaximum error is "+ str(maxerr[i]) + " at K = " + str(maxK[i])
            + " and gamma = " + str(maxgam[i]))
        print("\tMinimum error is "+ str(minerr[i]) + " at K = " + str(minK[i])
            + " and gamma = " + str(mingam[i]))

main()
