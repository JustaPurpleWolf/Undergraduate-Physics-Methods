import numpy as  np
from scipy.special import erfc, wofz
from scipy.constants import k, hbar,  epsilon_0


def voigt(x, x0, params=[1.0, 1.0, 1.0, 0.0]):
    k, u, gamma, detuning = params
    denominador = k * u
    if denominador == 0:
        denominador = 1e-15
        
    y = 2 * np.pi * (x - detuning * 1e6 - x0) / denominador
    a = gamma / denominador
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        term1 = erfc(a / 2.0 - 1j * y)
        term2 = np.exp(1j * 2.0 * a * y) *erfc(a / 2.0 + 1j * y)
        erfc_term = term1 + term2
        
    erfc_term = np.nan_to_num(erfc_term, nan=0.0, posinf=0.0, neginf=0.0)
    
    arg_espectro = 0.25 * (a - 1j * 2.0 * y)**2
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        resultado = np.abs(np.sqrt(np.pi) / (2.0 * denominador) * np.exp(arg_espectro) * erfc_term)
        
    return np.nan_to_num(resultado, nan=0.0, posinf=0.0, neginf=0.0)

class AbsortionProfiles:
    def profile_erfc(x, x0, T, N, L, Q: dict ,gamma, vals):
        T_safe = np.maximum(T, 0.1)
        N_safe = np.maximum(N, 0)
        coeff = 0
        for iso in vals['isotopos']:
            N_iso = N_safe*Q[iso['nombre']] # Q = abundancia
            u = np.sqrt((2*k*T_safe)/iso['m'])
            const = (vals['k']*N_iso*vals['d']**2)/(iso['gI']*epsilon_0*hbar)
            for i in range(len(iso['cge'])):
                params_voigt = [vals['k'], u , gamma, iso['det'][i]]
                coeff += const*voigt(x, x0, params = params_voigt)*iso['cge'][i]  #C_F^2 = Cge[i]

        exponent = -L * coeff
        exponent = np.clip(exponent, -700, 700)
    
        return np.exp(exponent)
    
    def profile_wigner(x, x0, T, N, L, Q: dict, gamma, vals):
        T_safe = np.maximum(T, 0.1)
        N_safe = np.maximum(N, 0)
        coeff = 0
        for iso in vals['isotopos']:
            N_iso = N_safe*Q[iso['nombre']] # Q = abundancia
            u = np.sqrt((2*k*T_safe)/iso['m'])
            const = (np.sqrt(np.pi)*N_iso*vals['d']**2)/(iso['gI']*epsilon_0*hbar*u)
            for i in range(len(iso['cge'])):
                doppler = vals['k']*u
                if doppler == 0: doppler = 1e-15
                detuning = 2.0 * np.pi * (x - iso['det'][i] * 1e6 - x0)
                z = (detuning+ 1j*0.5*gamma)/ doppler
                coeff += const*np.real(wofz(z))*iso['cge'][i] #C_F^2 = Cge[i]
                
        exponent = -L* coeff
        exponent = np.clip(exponent, -700, 700)

        return np.exp(exponent)