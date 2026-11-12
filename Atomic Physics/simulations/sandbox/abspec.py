from scipy.special import wofz as fadeeva
import numpy as np
import json
import physics_constants as phys

class Espectroscopia:
    @staticmethod
    def absortion_profile_fadeeva(detuning, wavelength, gamma_0, temperature, mass, c_mF, d, N):
        u = np.sqrt((2*phys.KB*temperature)/mass)
        k = 2*np.pi/wavelength
        z = (detuning + 1j*gamma_0/2) / (k*u)
        alpha = N*(c_mF**2)*(d**2)/(u*phys.HBAR*phys.EPSILON_0)*np.real(fadeeva(z))
        return alpha
    
    @staticmethod
    def absortion_profile_voigt(detuning, wavelength, gamma_0, temperature, mass, c_mF, d, N):
        return
    @staticmethod
    def absortion_profile_erfc(detuning, wavelength, gamma_0, temperature, mass, c_mF, d, N):
        return
    @staticmethod
    def absorton_profile_durham(detuning, wavelength, gamma_0, temperature, mass, c_mF, d, N):
        alpha = 2*N*np.sqrt(np.log(2)/np.pi)
        return
class Termodinamica:
    def density():
        pass