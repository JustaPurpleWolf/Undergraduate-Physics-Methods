import numpy as np
from typing import Tuple

from scipy.constants import k
from scipy.special import lambertw

from .io_loaders import DatabaseLoader

class AbsortionAnalyser:
    def get_pressure_params(T: float, vals):
        database = DatabaseLoader.load_json('src/database/vapor_pressure.json')

        if vals['element'] not in database:
            raise ValueError(f"El elemento {vals['element']} no se encuentra en la base de datos.")
        
        element_data = database[vals['element']]
        mp = element_data['melting_point_K']

        if T < mp:
            params = element_data['solid']
        else:
            params = element_data['liquid']
        
        if T < params['t_min_K'] or T > params['t_max_K']:
            raise ValueError(f"Advertencia: La temperatura {T} K se encuentra fuera del rango recomendado para el uso de los parámetros de ajuste [{params['t_min_K']} - {params['t_max_K']}]")
        return (params['A'], params['B'], params['C'], params['D']), mp

    def get_gamma_buffer(T: float, buffer_gas: str, vals: dict):
        database = DatabaseLoader.load_json('src/database/buffer_gases.json')

        if vals['element'] not in database:
            raise ValueError(f"El elemento {vals['element']} no se encuentra en la base de datos.")
        element_data = database[vals['element']]
        if vals['line'] not in element_data:
            raise ValueError(f"La línea {vals['line']} no se encuentra en la base de datos para el elemento {vals['element']}.")
        line_data = element_data[vals['line']]
        if buffer_gas not in line_data:
            raise ValueError(f"El gas de buffer {buffer_gas} no se encuentra en la base de datos para la línea {vals['line']} del elemento {vals['element']}.")
        
        return line_data[buffer_gas]

    def calculate_density(T: float, pressure_params: Tuple[float, float, float, float]):
        p = 10**(5.006 + pressure_params[0] + pressure_params[1]/T  + pressure_params[2]*np.log10(T) 
                 + pressure_params[3]/(T**3))
        N = p/(k*T)
        return N

    def calculate_temperature(N:float, pressure_params: Tuple[float, float, float, float]):
        beta = -pressure_params[1] * np.log(10)
        T = -beta / lambertw(beta * N * k / (10 ** pressure_params[0]), k = -1).real
        return T
    
    def calculate_gamma(T: float , pressure: float, gamma_buffer: float,
                        pressure_params: Tuple[float, float, float, float], vals: dict,
                        with_collisons: bool = True, with_buffer: bool = True):
        gamma = vals['gamma']

        if with_collisons:
            N = AbsortionAnalyser.calculate_density(T, pressure_params)
            if vals['line'] == "D1": 
                beta = 2*np.pi*vals['gamma']*(vals['wavelength']/(2*np.pi))**3
            elif vals['line'] == "D2": 
                beta = 2*np.pi*vals['gamma']*np.sqrt(2)*(vals['wavelength']/(2*np.pi))**3
            else:
                beta = 0.0
            gamma += (beta*N)

        if with_buffer and pressure > 0.0:          
            gamma += (gamma_buffer * pressure)

        return gamma