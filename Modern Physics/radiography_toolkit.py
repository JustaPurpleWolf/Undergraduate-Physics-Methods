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

def plot_calibration(csv_file, unit_thickness =0.10, mu = None, berger= None, MaxBit = 4095, xlim = False, 
                     label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df = pd.read_csv(csv_file)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    
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
        bx.plot(t, I, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
        if berger:
            I_berger = (1+berger*t)*np.exp(-mu*t)
            bx.plot(t, I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Aproximación de Berger  ($C$={berger/mu:.2f})', zorder=2)
        
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14,y = 0.93)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    bx.errorbar(g, y, yerr=y_err, ecolor='black', fmt='s', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Datos Experimentales', zorder=3)
    
    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.savefig('calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_all(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, berger = None, berger_al=None, 
             MaxBit = 4095, xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(8,4), dpi = 100)
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    
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
    
    bx.errorbar(g1, y1, yerr=y1_err, fmt='bs', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='blue', markerfacecolor='white', 
                label='Papel, Datos Crudos', zorder=1,alpha= 0.3)
    
    
    bx.errorbar(g2, y2, yerr=y2_err, fmt='b^', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='blue', markerfacecolor='white', 
                label='Papel, Corrección de Fondo', zorder=3)
    
    bx.errorbar(gal, yal, xerr= gal_err, yerr=yal_err, fmt='go', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='green', markerfacecolor='white', 
                label='Aluminio, Datos Crudos', zorder=3)
    
    if mu:
        if xlim:
            n = np.linspace(0,xlim, 1000)
        else: 
            xmax = max(x1.max(), x2.max())
            n = np.linspace(0,xmax, 1000)
        t = n*unit_thickness
        if berger:
            I_berger = (1+berger*t)*np.exp(-mu*t)
            bx.plot(t, I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Papel, Aproximación de Berger ($C$={berger/mu:.2f})', zorder=2)
        I = np.exp(-mu*t)
        bx.plot(t, I, color='blue', linewidth = 0.8, label=f'Papel, Ley de Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
            
    if mu_al:
        t = np.linspace(0, gal.max(),1000)
        if berger_al:
            I_berger = (1+berger_al*t)*np.exp(-mu_al*t)
            bx.plot(t, I_berger, color='green', linewidth = 0.8, linestyle = ':', label=f'Aluminio, Ley de Beer-Lambert ($\mu$={mu_al:.6f})', zorder=2)   
        I = np.exp(-mu_al*t)
        bx.plot(t, I, color='green', linewidth = 0.8, label=f'Aluminio, Aproximación de Berger ($C$={berger_al/mu_al:.2f})', zorder=2)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_diff(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, MaxBit = 4095, 
             xlim = False, label = '',figure=1, caption=''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    
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
    
    bx.plot(g1, I1-y1, color='blue', linewidth = 0.8, label=f'Papel, Datos Crudos y Beer-Lambert ($\mu$={mu:.6f})', zorder=2)   
    bx.plot(g2, I2-y2, color='blue', linewidth = 0.8, linestyle = '--', label=f'Papel, Corrección de Fondo y Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
    
    
    bx.plot(gal, Ial-yal, color='green', linewidth = 0.8, linestyle = ':', label=f'AlError ($\mu$={mu_al:.6f})', zorder=2)
    plt.suptitle(f'Radiographic Attenuation Curve.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figure {figure}. {caption}',  pad=10, fontsize = 12)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Cociente de Intensidades [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_quo(csv_file_og, csv_file_sf ,csv_file_al, unit_thickness =0.10, mu = None, mu_al= None, berger = None, berger_al= None, MaxBit = 4095,
             label = '',figure=1, caption='', var = ''):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize': 10})
    df_og = pd.read_csv(csv_file_og)
    df_sf = pd.read_csv(csv_file_sf)
    df_al = pd.read_csv(csv_file_al)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    
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
    I_berger = (1+berger*g2)*np.exp(-mu*g2)
    
    Ial = np.exp(-mu_al*gal)
    I_berger_al = (1+berger_al*gal)*np.exp(-mu_al*gal)

    if var == 'paper':
        bx.plot(g1, y1/I1, color='blue', linewidth = 0.8, label=f'Papel; Datos Crudos, Beer-Lambert ($\mu$={mu:.6f})', zorder=2)   
        bx.plot(g2, y2/I2, color='blue', linewidth = 0.8, linestyle = '--', label=f'Papel; Corrección de Fondo, Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
        bx.plot(g2, y2/I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Papel; Corrección de Fondo, Berger ($C$={berger/mu:.2f})', zorder=2)

    if var == 'al':
        bx.plot(gal, yal/Ial, color='green', linewidth = 0.8, label=f'Aluminio; Datos Crudos, Beer-Lambert ($\mu$={mu_al:.6f})', zorder=2)
        bx.plot(gal, yal/I_berger_al, color='green', linewidth = 0.8, linestyle = ':', label=f'Aluminio; Datos Crudos, Berger ($C$={berger_al/mu_al:.2f})', zorder=2)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    bx.legend(loc='upper right', frameon=True)

    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Cociente de Intensidades [AU]', fontweight='bold')
    #bx.set_ylim(0.9,1.1)

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.tight_layout()
    plt.savefig('calibration_curve_'+label+'.png', bbox_inches='tight')
    plt.show()

def plot_var(csv_file, unit_thickness =0.10, mu = None, berger= None, MaxBit = 4095, xlim = False, label = '',figure=1, caption='', var ):
    plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.labelsize':10})
    df = pd.read_csv(csv_file)

    fig, bx = plt.subplots(1, figsize=(10,4), dpi = 100)
    
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
        bx.plot(t, I, color='red', linewidth = 0.8, label=f'Ley de Beer-Lambert ($\mu$={mu:.6f})', zorder=2)
        if berger:
            I_berger = (1+berger*t)*np.exp(-mu*t)
            bx.plot(t, I_berger, color='blue', linewidth = 0.8, linestyle = ':', label=f'Aproximación de Berger  ($C$={berger/mu:.2f})', zorder=2)
        
    plt.suptitle(f'Curva de Atenuación.',fontweight = 'bold', fontsize = 14,y = 0.93)
    bx.set_title(f'Figura {figure}. {caption}',  pad=10, fontsize = 12)
    bx.errorbar(g, y, yerr=y_err, ecolor='black', fmt='s', markersize=5, capsize=3, 
                elinewidth=0.6, markeredgecolor='black', markerfacecolor='white', 
                label='Datos Experimentales', zorder=3)
    
    bx.set_xlabel(r'Grosor [$mm$]', fontweight='bold')
    bx.set_ylabel(r'Intensidad Relativa [AU]', fontweight='bold')

    bx.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(
    loc='center left', 
    bbox_to_anchor=(1.05, 0.5, 0.3, 0.2), 
    mode=None,
    fontsize=8,
    frameon=True,
    labelspacing=1.1,
    handletextpad=0.8,
    borderaxespad=0
    )

    plt.subplots_adjust(left=0.12, right=0.75, bottom=0.15, top=0.80)
    plt.savefig('calibration_curve_'+label+'.png', dpi=300, bbox_inches='tight')
    plt.show()
