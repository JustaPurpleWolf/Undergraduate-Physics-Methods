from dataclasses import dataclass
from typing import Optional, List
import numpy as np

@dataclass
class ExperimentalParameters:
    cell_length: float #[cm]
    temperature: float #[°C]
    has_buffer_gas: bool #
    buffer_gas_type: Optional[str] = None #
    pressure: Optional[float] = None #[atm]

@dataclass
class FitParameters:
    profile_variant: str # Voigt, Wigner, etc.
    fit_method: str # least squares, Levenberg-Marquart, etc.
    fit_temperature = bool
    fit_pressure = bool

@dataclass
class CalibrationParameters:
    method: str #Hyperfine-method, Fabry-Pérot cavity or Ansatz-method

    hiperfine_peaks: Optional[List[float]] = None
    hyperfine_known_freqs: Optional[List[float]] = None 


    fabry_perot_peaks: Optional[List[float]] = None
    fabyr_perot_fsr: Optional[float] = None    # free-spectral range [GHz]

    ansatz_degree: Optional[int] = 3
    ansantz_coefficients: Optional[List[float]] = None