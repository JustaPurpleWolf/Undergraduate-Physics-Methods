import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pydicom
import os

class Radiograph:
    def __init__(self, path):
        self.path = path
        self.filename = os.path.basename(path)
        self.ds = pydicom.dcmread(path)
        self.image = self.ds.pixel_array.astype(np.float64)
        self.max_bit_val = (2**(self.ds.BitsStored if 'BitsStored' in self.ds else 16)) - 1

    def invert(self):
        self.image = self.max_bit_val - self.image
        return self

    def crop(self, x1, y1, x2, y2):
        self.image = self.image[y1:y2, x1:x2]
        return self

    def measure_weighted(self, label="Muestra", thickness=0.0, sheets=1, weights=None):
        if weights is None:
            weights = np.ones_like(self.image)
        
        v1 = np.sum(weights)
        mean_w = np.sum(weights * self.image) / v1
        
        if v1 > 1:
            variance_w = np.sum(weights * (self.image - mean_w)**2) / (v1 - 1)
        else:
            variance_w = 0.0
        
        std_w = np.sqrt(variance_w)
        
        data = {
            "Filename": self.filename,
            "Label": label,
            "Thickness_mm": thickness,
            "Sheet_Count": sheets,
            "Mean_Intensity": mean_w,
            "Std_Dev": std_w,
            "Max_Bit": self.max_bit_val
        }
        
        return mean_w, std_w, data

class dicom:
    @staticmethod
    def load(path):
        return Radiograph(path)
    
    @staticmethod
    def save_to_csv(data_list, filename="calibration_thickness.csv", overwrite=False):
        df = pd.DataFrame(data_list)
        
        if overwrite:
            df.to_csv(filename, mode='w', header=True, index=False)
            print(f"Archivo '{filename}' sobrescrito con {len(df)} registros.")
        else:
            file_exists = os.path.isfile(filename) and os.path.getsize(filename) > 0
            df.to_csv(filename, mode='a', header=not file_exists, index=False)
            print(f"Se han añadido {len(df)} registros a '{filename}'.")

def plot_calibration(csv_file, unit_thickness =0.15, mu = None, MaxBit = 4095):
    plt.rcParams.update({'font.size': 6, 'font.family': 'serif', 'axes.labelsize': 8})
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(4, 3), dpi=150)
    
    x = df['Sheet_Count']
    y = df['Mean_Intensity']/MaxBit
    y_err = df['Std_Dev']/MaxBit

    ax.errorbar(x, y, yerr=y_err, fmt='ks', markersize=3, capsize=2, 
                elinewidth=0.4, markeredgecolor='black', markerfacecolor='white', 
                label='Experimental Data (Mean ± SD)', zorder=3)
    if mu:
        n = np.linspace(0, x.max(), 1000)
        t = n*unit_thickness
        I = np.exp(-mu*t)
        ax.plot(n, I, color='blue', linewidth = 0.6, label=f'Theory ($\mu$={mu})', zorder=2)

    ax.set_xlabel(r'Material Thickness [$n$ Sheets]', fontweight='bold')
    ax.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')
    ax.set_title('Figure 1. Radiographic Attenuation Calibration Curve', pad=10)

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    #ax.set_ylim(0,1)
    
    plt.tight_layout()
    plt.savefig('calibration_curve.png', bbox_inches='tight')
    plt.show()