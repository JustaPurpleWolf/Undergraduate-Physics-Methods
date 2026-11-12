import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

script_dir = Path(__file__).resolve().parent
root_dir = script_dir.parent

if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.core.io_loaders import DatabaseLoader as dbl
from src.core.profiles import AbsortionProfiles as ap
from src.core.analysis import AbsortionAnalyser as abs

def atomic_density(T, line):
    database = f'../src/database/{line}.json'
    vals = dbl.load_json(database)
    pressure_params, _ = abs.get_pressure_params(T, vals)
    return abs.calculate_density(T, pressure_params)*1e-6
    

def model_comparison(line, T, gas, pressure, L, Q, with_collisons = True, with_buffer = True):
    database = f'../src/database/{line}.json'
    vals = dbl.load_json(database)
    T += 273.15

    pressure_params, _ = abs.get_pressure_params(T, vals)
    gamma_buffer = abs.get_gamma_buffer(T, gas, vals)
    N = abs.calculate_density(T, pressure_params)
    gamma = abs.calculate_gamma(T, pressure, gamma_buffer, pressure_params, vals, with_collisons, with_buffer)

    X = np.linspace(-6*1e9, 6*1e9, 1000)
    Y_erfc = []
    Y_wigner = []
    D = []
    for x in X:
        y_erfc = ap.profile_erfc(x,0, T, N, L, Q, gamma, vals = vals)
        Y_erfc.append(y_erfc)
    
        y_wigner = ap.profile_wigner(x,0, T, N, L, Q,  gamma,  vals = vals)
        Y_wigner.append(y_wigner)

        d = y_wigner - y_erfc
        D.append(d)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(X,Y_erfc, label='Perfil Erfc', color='black', alpha=0.8)
    ax1.plot(X,Y_wigner, label='Perfil Wigner', color='red', linestyle='--', linewidth=1.5)
    ax2.plot(X, D, label='Diferencia', color='purple', linestyle='-', linewidth=1.5)

    ax1.set_ylabel('Absorción')
    ax1.set_xlabel('Tiempo')
    ax1.set_title(f'Comparación de Perfiles Espectrales (T = {T - 273.15} °C)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Tiempo')
    ax2.set_ylabel('Error')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()   

def collision_effects(line, T, gas, pressure, L, Q, with_buffer = True):
    database = f'../src/database/{line}.json'
    vals = dbl.load_json(database)
    T += 273.15
    pressure_params, _ = abs.get_pressure_params(T, vals)
    gamma_buffer = abs.get_gamma_buffer(T, gas, vals)
    N = abs.calculate_density(T, pressure_params)
    gamma = abs.calculate_gamma(T, pressure, gamma_buffer, pressure_params, vals, with_collisons = True, with_buffer = with_buffer)
    gamma_no_collisions = abs.calculate_gamma(T, pressure, gamma_buffer, pressure_params, vals, with_collisons = False, with_buffer = with_buffer)
    X = np.linspace(-6*1e9, 6*1e9, 1000)
    Y_collisions = []
    Y_no_collisions = []
    D = []
    for x in X:
        y_collisions = ap.profile_wigner(x,0, T, N, L, Q, gamma, vals = vals)
        Y_collisions.append(y_collisions)
    
        y_no_collisions = ap.profile_wigner(x,0, T, N, L, Q, gamma_no_collisions, vals = vals)
        Y_no_collisions.append(y_no_collisions)

        d = y_collisions - y_no_collisions
        D.append(d)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(X, Y_collisions, label='Con Colisiones', color='blue', alpha=0.8)
    ax1.plot(X, Y_no_collisions, label='Sin Colisiones', color='orange', linestyle='--', linewidth=1.5)
    ax2.plot(X, D, label='Diferencia', color='purple', linestyle='-', linewidth=1.5)

    ax1.set_xlabel('Frecuencia Hz')
    ax1.set_ylabel('Absorción')
    ax1.set_title(f'Efecto de las Colisiones en el Perfil Espectral (T = {T - 273.15} °C)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2.set_xlabel('Frecuencia Hz')
    ax2.set_ylabel('Error')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

def buffer_effects(line, T, gas, pressure, L, Q, with_collisons = True):
    database = f'../src/database/{line}.json'
    vals = dbl.load_json(database)
    T += 273.15
    pressure_params, _ = abs.get_pressure_params(T, vals)
    gamma_buffer = abs.get_gamma_buffer(T, gas, vals)
    N = abs.calculate_density(T, pressure_params)
    gamma_with_buffer = abs.calculate_gamma(T, pressure, gamma_buffer, pressure_params, vals, with_collisons = with_collisons, with_buffer = True)
    gamma_no_buffer = abs.calculate_gamma(T, pressure, gamma_buffer, pressure_params, vals, with_collisons = with_collisons, with_buffer = False)
    X = np.linspace(-6*1e9, 6*1e9, 1000)
    Y_with_buffer = []
    Y_no_buffer = []
    D = []
    for x in X:
        y_with_buffer = ap.profile_wigner(x,0, T, N, L, Q, gamma_with_buffer, vals = vals)
        Y_with_buffer.append(y_with_buffer)
    
        y_no_buffer = ap.profile_wigner(x,0, T, N, L, Q, gamma_no_buffer, vals = vals)
        Y_no_buffer.append(y_no_buffer)

        d = y_with_buffer - y_no_buffer
        D.append(d)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(X,Y_with_buffer,label='Con Gas de Buffer', color='green', alpha=0.8)
    ax1.plot(X,Y_no_buffer,label='Sin Gas de Buffer', color='magenta', linestyle='--', linewidth=1.5)
    ax2.plot(X, D, label='Diferencia', color='purple', linestyle='-', linewidth=1.5)


    ax1.set_xlabel('Frecuencia Hz')
    ax1.set_ylabel('Absorción')
    ax1.set_title(f'Efecto del Gas de Buffer en el Perfil Espectral (T = {T - 273.15} °C)', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    ax2.set_xlabel('Frecuencia Hz')
    ax2.set_ylabel('Error')
    ax2.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()