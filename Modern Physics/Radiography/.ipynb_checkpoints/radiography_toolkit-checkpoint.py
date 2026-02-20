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

    def invert(self, fondo=0):
        att = self.max_bit_val - fondo
        self.image = self.max_bit_val - self.image + att
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

def plot_calibration(csv_file, unit_thickness =0.10, mu = None, MaxBit = 4095, xlim = False, 
                     label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df = pd.read_csv(csv_file)

    fig, bx = plt.subplots(1, figsize=(6,4), dpi = 100)
    
    x = df['Sheet_Count']
    g = x*unit_thickness
    y = df['Mean_Intensity']/MaxBit
    y_err = df['Std_Dev']/MaxBit

    if mu:
        if xlim:
            n = np.linspace(0,xlim, 1000)
        else: 
            n = np.linspace(0,x.max(), 1000)
        t = n*unit_thickness
        I = np.exp(-mu*t)
        #ax.plot(n, I, color='blue', linewidth = 0.6, label=f'Theory ($\mu$={mu})', zorder=2)
        bx.plot(t, I, color='red', linewidth = 0.8, label=f'Simulation ($\mu$={mu:.6f})', zorder=2)
        
    '''
    ax.errorbar(x, y, yerr=y_err, fmt='ks', markersize=3, capsize=2, 
                elinewidth=0.4, markeredgecolor='black', markerfacecolor='white', 
                label='Experimental Data (Mean ± SD)', zorder=3)
                
    ax.set_xlabel(r'Material Thickness [$n$ Sheets]', fontweight='bold')
    ax.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)
    '''
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)
    bx.errorbar(g, y, yerr=y_err, fmt='ks', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Experimental Data (Mean ± Std)', zorder=3)
    
    bx.set_xlabel(r'Material Thickness [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_all(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, MaxBit = 4095, 
             xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(6,4), dpi = 100)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)
    
    x1 = df_og['Sheet_Count']
    g1 = x1*unit_thickness
    y1 = df_og['Mean_Intensity']/MaxBit
    y1_err = df_og['Std_Dev']/MaxBit

    x2 = df_sf['Sheet_Count']
    g2 = x2*unit_thickness
    y2 = df_sf['Mean_Intensity']/MaxBit
    y2_err = df_sf['Std_Dev']/MaxBit

    gal = df_al['Thickness [mm]']
    gal_err = df_al['Delta [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    yal_err = df_al['StdDev [12bit]']/MaxBit
    
    '''
    ax.errorbar(x1, y1, yerr=y1_err, fmt='ks', markersize=3, capsize=2, 
                elinewidth=0.4, markeredgecolor='black', markerfacecolor='white', 
                label='Experimental Data (Mean ± Std)', zorder=3)

    ax.errorbar(x2, y2, yerr=y2_err, fmt='ro', markersize=3, capsize=2, 
                elinewidth=0.4, markeredgecolor='red', markerfacecolor='white', 
                label='Experimental Data w/o Backround(Mean ± Std)', zorder=3)

    ax.set_xlabel(r'Material Thickness [$n$ Sheets]', fontweight='bold')
    ax.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')
    '''
    bx.errorbar(g1, y1, yerr=y1_err, fmt='ks', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Paper Sheets Raw Data (Mean ± Std)', zorder=3)
    
    bx.errorbar(g2, y2, yerr=y2_err, fmt='r^', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='red', markerfacecolor='white', 
                label='Paper Sheets Background Removed (Mean ± Std)', zorder=3)
    
    bx.errorbar(gal, yal, xerr= gal_err, yerr=yal_err, fmt='bo', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='blue', markerfacecolor='white', 
                label='Aluminum Data (Mean ± Std)', zorder=3)
    
    
    if mu:
        if xlim:
            n = np.linspace(0,xlim, 1000)
        else: 
            xmax = max(x1.max(), x2.max())
            n = np.linspace(0,xmax, 1000)
        t = n*unit_thickness
        I = np.exp(-mu*t)
        #ax.plot(n, I, color='blue', linewidth = 0.6, label=f'Theory ($\mu$={mu})', zorder=2)
        bx.plot(t, I, color='black', linewidth = 0.8, label=f'Simulation ($\mu$={mu:.6f})', zorder=2)
        '''
        expA = np.exp(-mu*x2*unit_thickness)
        divA = y2/expA
        restA = expA-y2

        expB = np.exp(-mu*g2)
        divB = y2-expB
        '''
    if mu_al:
        t = np.linspace(0, gal.max(),1000)
        I = np.exp(-mu_al*t)
        bx.plot(t, I, color='blue', linewidth = 0.8, linestyle = ':', label=f'Simulation ($\mu$={mu_al:.6f})', zorder=2)

    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Material Thickness [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_diff(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, MaxBit = 4095, 
             xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(6,4), dpi = 100)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)
    
    x1 = df_og['Sheet_Count']
    g1 = x1*unit_thickness
    y1 = df_og['Mean_Intensity']/MaxBit

    x2 = df_sf['Sheet_Count']
    g2 = x2*unit_thickness
    y2 = df_sf['Mean_Intensity']/MaxBit

    gal = df_al['Thickness [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    
    I1 = np.exp(-mu*g1)
    I2 = np.exp(-mu*g2)
    Ial = np.exp(-mu_al*gal)
    
    bx.plot(g1, I1-y1, color='black', linewidth = 0.8, label=f'Paper Sheets Raw Error ($\mu$={mu:.6f})', zorder=2)   
    bx.plot(g2, I2-y2, color='red', linewidth = 0.8, linestyle = '--', label=f'Paper Sheets Background Correction Error ($\mu$={mu:.6f})', zorder=2)
    bx.plot(gal, Ial-yal, color='blue', linewidth = 0.8, linestyle = ':', label=f'Aluminum Error ($\mu$={mu_al:.6f})', zorder=2)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Material Thickness [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_quo(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, MaxBit = 4095, 
             xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(6,4), dpi = 100)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)
    
    x1 = df_og['Sheet_Count']
    g1 = x1*unit_thickness
    y1 = df_og['Mean_Intensity']/MaxBit

    x2 = df_sf['Sheet_Count']
    g2 = x2*unit_thickness
    y2 = df_sf['Mean_Intensity']/MaxBit

    gal = df_al['Thickness [mm]']
    yal = df_al['Mean [12bit]']/MaxBit
    
    I1 = np.exp(-mu*g1)
    I2 = np.exp(-mu*g2)
    Ial = np.exp(-mu_al*gal)
    
    bx.plot(g1, y1/I1, color='black', linewidth = 0.8, label=f'Paper Sheets Raw Error ($\mu$={mu:.6f})', zorder=2)   
    bx.plot(g2, y2/I2, color='red', linewidth = 0.8, linestyle = '--', label=f'Paper Sheets Background Correction Error ($\mu$={mu:.6f})', zorder=2)
    bx.plot(gal, yal/Ial, color='blue', linewidth = 0.8, linestyle = ':', label=f'Aluminum Error ($\mu$={mu_al:.6f})', zorder=2)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Material Thickness [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Relative Intensity [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()
