# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 09:33:30 2025

@author: tifernan
"""

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import pandas as pd




import numpy as np
import math
import matplotlib.patches as mpatches

from matplotlib.colors import LinearSegmentedColormap

blue_red_cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"])

# normalize velocity (0 to 500)
norm = plt.Normalize(vmin=0, vmax=500)


legend_patches = [
    mpatches.Patch(color=(231/255, 176/255, 79/255), label='floating'),
    mpatches.Patch(color=(165/255, 220/255, 205/255), label='bouncing'),
    mpatches.Patch(color=(50/255, 120/255, 163/255), label='penetrating')
]

def get_id(id_values, particle_diam):
    unique_id = []
    idx=0;
    for i in range(len(id_values)):
      unique_id.append(idx)
      if i>0:
        if id_values[i] != id_values[i-1]:
          idx+=1
    return unique_id

def predict_fate(df):
    rho_L = 3440;
    gamma = 0.568;
    turnonv=1;
    fate = []
    print(df['ParticleYPosition m'])
    current_y = df['ParticleYPosition m'].iloc[0]; current_v = df['ParticleYVelocity m/s'].iloc[0];
    current_id = df['UniqueID'].iloc[0];

    for index, row in df.iterrows():

        prev_y = current_y; prev_v=current_v; prev_id = current_id
        current_y = row['ParticleYPosition m'];
        current_id = row['UniqueID']; current_v = row['ParticleYVelocity m/s'];

        if prev_id != current_id:
            turnonv=1;

        if current_y == -3.50525 and turnonv==1:
            mass = row['ParticleMass kg'];
            diam = row['ParticleDiameter m'];
            vel = row['ParticleYVelocity m/s'];
            rho = row['ParticleDensity kg/m3'];
            We = rho_L*vel*vel*diam/gamma;
            Bo = rho_L*9.81*diam*diam/gamma;
            unique_id = row['UniqueID'];
            turnon = row['turnon'];
            lambda_squared = (rho/rho_L)*(rho/rho_L);
            WeBo = We * math.pow(Bo*Bo*Bo,0.5);
            if WeBo >= 12/lambda_squared:
                fate_v = 2;
                print(f"penetrated! {diam}")
            elif WeBo < 12/lambda_squared and WeBo >= 6/lambda_squared:
                fate_v = 1;
                print(f"bounced! {diam}")
            else:
                fate_v = 0;
                print(f"floated! {diam}")

            #if turnon == 1:
            #    df.loc[df['UniqueID'] == unique_id, 'fate'] = fate_v
                 #df.loc[df['UniqueID'] == unique_id, 'turnon'] = 0
            fate.append(fate_v)
            turnonv=0;
             
        if current_y > prev_y and current_id==prev_id and turnonv==1:
            mass = row['ParticleMass kg'];
            diam = row['ParticleDiameter m'];
            vel = row['ParticleYVelocity m/s'];
            rho = row['ParticleDensity kg/m3'];
            We = rho_L*vel*vel*diam/gamma;
            Bo = rho_L*9.81*diam*diam/gamma;
            unique_id = row['UniqueID'];
            turnon = row['turnon'];
            fate_v = 1;
            #print(f"{unique_id} bounced!")
            #print(f"{current_id} {prev_id}")
            print(f"bounced! {diam}")
            fate.append(fate_v);
            turnonv=0;
            '''

            mass = row['ParticleMass kg'];
            diam = row['ParticleDiameter m'];
            #vel = row['ParticleYVelocity m/s'];
            rho = row['ParticleDensity kg/m3'];
            We = rho_L*prev_v*prev_v*diam/gamma;
            Bo = rho_L*9.81*diam*diam/gamma;
            unique_id = row['UniqueID'];
            turnon = row['turnon'];
            lambda_squared = (rho/rho_L)*(rho/rho_L);
            WeBo = We * math.pow(Bo*Bo*Bo,0.5);
            if WeBo >= 12/lambda_squared:
                fate_v = 2;
                #print("penetrated!")
            elif WeBo < 12/lambda_squared and WeBo >= 6/lambda_squared:

            else:
                fate_v = 0;
            '''

            #if turnon == 1:
                #id_to_fate = {'id1': 'escaped', 'id2': 'trapped', ...}
           #     df.loc[df['UniqueID'] == unique_id, 'fate'] = fate_v
           #     df.loc[df['UniqueID'] == unique_id, 'turnon'] = 0

    return fate;


def load_and_prepare_df(csv_path, num_ids=50, scale_size=2):
  print(f"reading data in {csv_path}")  
  df = pd.read_csv(csv_path, index_col=False)

  print(df)

  size_col = 'ParticleDiameter m' if 'ParticleDiameter m' in df.columns else 'ParticleDiameter m'  # adjust based on actual column
  
  str_cols = df.select_dtypes(include='object').columns
  df[str_cols] = df[str_cols].apply(lambda col: col.str.strip("'\""))
  numeric_cols = ['ParticleXPosition m', 'ParticleYPosition m', 'ParticleZPosition m', 'ParticleVelocityMagnitude m/s', 'ParticleDiameter m']  # adjust if needed
  for col in numeric_cols:
      df[col] = pd.to_numeric(df[col], errors='coerce')
  df = df[df['SurfaceID -']==8]
  
  df = df.dropna(subset=['ParticleXPosition m', 'ParticleYPosition m', 'ParticleZPosition m', 'ParticleVelocityMagnitude m/s', 'ParticleDiameter m'])
  print(f"scale_size = {scale_size}")
  if scale_size!=0:
    df['scaled_size'] = scale_size;#df[size_col] / df[size_col].max() * 10  # scale for better visual
  else:
    df['scaled_size'] = 0.3*np.sqrt(df['ParticleDiameter m']/0.0001);  
  df['turnon'] = 1;
  df['fate'] = 0;
  id_values = df['ParticleID -'].values;
  particle_diam = df['ParticleDiameter m'].values;
  unique_id = get_id(id_values, particle_diam)
  df['UniqueID'] = unique_id
  
  unique_ids = df['UniqueID'].dropna().unique()
  df = df.reset_index(drop=True)
  fate = predict_fate(df);
  id_to_color = dict(zip(unique_ids, fate))
  df['color_value'] = df['UniqueID'].map(id_to_color)
  
  x_range = df['ParticleXPosition m'].max() - df['ParticleXPosition m'].min()
  y_range = df['ParticleYPosition m'].max() - df['ParticleYPosition m'].min()
  z_range = df['ParticleZPosition m'].max() - df['ParticleZPosition m'].min()

  max_range = max(x_range, y_range, z_range)

  x_middle = (df['ParticleXPosition m'].max() + df['ParticleXPosition m'].min()) / 2
  y_middle = (df['ParticleYPosition m'].max() + df['ParticleYPosition m'].min()) / 2
  z_middle = (df['ParticleZPosition m'].max() + df['ParticleZPosition m'].min()) / 2


  #fig = go.Figure()
  num_particles = 500  

  random_ids = df['UniqueID'].dropna().unique()
  selected_ids = pd.Series(random_ids).sample(n=num_ids)

 
  
  df_subset = df[df['UniqueID'].isin(selected_ids)]
  df_subset = df_subset[df_subset['ParticleResidenceTime s']<0.2]
  df_subset = df_subset.sample(frac=0.5, random_state=42).reset_index(drop=True)
  x_min, x_max = df_subset['ParticleXPosition m'].min(), df_subset['ParticleXPosition m'].max()
  y_min, y_max = df_subset['ParticleYPosition m'].min(), df_subset['ParticleYPosition m'].max()
  z_min, z_max = df_subset['ParticleZPosition m'].min(), df_subset['ParticleZPosition m'].max()
  print(f"{x_min} {x_max}, {y_min} {y_max}, {z_min} {z_max}")
  return df, df_subset




def create_sphere(center, radius, resolution=15):
    u, v = np.mgrid[0:2*np.pi:resolution*1j, 0:np.pi:resolution*1j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    return x, y, z

def plot_df_static(df_subset):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

   
    ax.set_xlim(-4.036, 3.5989)
    ax.set_ylim(-3.5025, 0.5)
    ax.set_zlim(-3.59, 3.59)
 
    
    ax.set_xlim(1, 3)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(-2, 0)
    
    ax.set_xlim(2, 2.8)
    ax.set_ylim(0, 0.5)
    ax.set_zlim(-0.7, -0)
      
    ax.set_xlim(2.4540946, 1.92)   # X 
    ax.set_ylim(-3.1363909, -3.49) # Y (Z-position)
    ax.set_zlim(-1.1437123, -0.91) # Z (Y-position)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
   
    ax.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    
  
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
   
    ax.view_init(elev=15, azim=0)  
    ax.set_facecolor('white')
    ax.grid(False)
    
    # Draw particles
    state_info = {
        2: {'label': 'penetrating', 'color': 'blue'},
        1: {'label': 'bouncing', 'color': 'green'},
        0: {'label': 'floating', 'color': 'yellow'}
    }

    for color_val, info in state_info.items():
        df_state = df_subset[df_subset['color_value'] == color_val]

        for _, row in df_state.iterrows():
            center = [
                row['ParticleXPosition m'],
                row['ParticleZPosition m'],  # Note: Z and Y swapped 
                row['ParticleYPosition m']
            ]
            radius = row['scaled_size'] / 100  
            x, y, z = create_sphere(center, radius)

            ax.plot_surface(x, y, z, color=info['color'], alpha=0.6, linewidth=0, shade=True)
    ax.legend(handles=legend_patches, loc='upper right', title='Particle State')
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches
from matplotlib import cm
from matplotlib.colors import Normalize

def plot_df_comparison(df_subset, filename):
   
    plot_vel=0;
    df_subset['velocity_mag'] = np.sqrt(
        df_subset['ParticleXVelocity m/s']**2 +
        df_subset['ParticleYVelocity m/s']**2 +
        df_subset['ParticleZVelocity m/s']**2
    )

    
    norm = Normalize(vmin=0, vmax=500)
    cmap = cm.viridis

    
    fig = plt.figure(figsize=(18, 9))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    
    # Create circular base at z = -3.5025
    radius = 5
    Z_base = -3.5025; x_center= 2.1306057; y_center=0.98014098;
    radius = 0.1
    X = np.linspace(2.1306057-radius, 2.1306057+radius, 1000)
    Y = np.linspace(0.98014098-radius, 0.98014098+radius, 1000)
    X, Y = np.meshgrid(X, Y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    
    Z_base = -3.5025
    R = np.sqrt((X-x_center)**2 + (Y-y_center)**2)
  
    fade_zone = 0.1  # controls the softness of the edge
    alpha_map = np.clip(1 - (R - (radius - fade_zone)) / fade_zone, 0, 0.25)
    
    
    mask = R <= radius
    Xc = np.where(mask, X, np.nan)
    Yc = np.where(mask, Y, np.nan)
    Zc = np.full_like(Xc, Z_base)
    #print(Xc.shape)
    #print(Zc.shape)
    alpha_c = np.where(mask, alpha_map, np.nan)
   
    colors = np.zeros((*Xc.shape, 4))  
    colors[..., :3] = 0.5  
    colors[..., 3] = alpha_c  
    
    
    def style_ax(ax):
        xmin=2.0; xmax= 2.3
        ymin=0.9; ymax= 1.2
        zmin=-3.4; zmax= -3.1
        xr=xmax-xmin; yr = ymax-ymin; zr=zmax-zmin
        xm=(xmax+xmin)/2; ym = (ymax+ymin)/2; zm=(zmax+zmin)/2
        max_range = max(xr, yr, zr)
        ax.set_xlim(xm - max_range/2, xm + max_range/2)
        ax.set_ylim(ym - max_range/2, ym + max_range/2)
        ax.set_zlim(zm - max_range/2, zm + max_range/2)

        #ax.set_xlim(2, 2.8)
        #ax.set_ylim(0.9, 1.4)
        #ax.set_zlim(-3.5, -2.86)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.w_xaxis.line.set_color((0, 0, 0, 0))
        ax.w_yaxis.line.set_color((0, 0, 0, 0))
        ax.w_zaxis.line.set_color((0, 0, 0, 0))
        ax.set_facecolor('white')
        ax.view_init(elev=20, azim=180)

    style_ax(ax1)
    style_ax(ax2)

    # state info
    state_info = {
        2: {'label': 'penetrating', 'color': (50/255, 120/255, 163/255)},
        1: {'label': 'bouncing', 'color': (165/255, 220/255, 205/255)}, 
        0: {'label': 'floating', 'color': (231/255, 176/255, 79/255)}
    }
 
    # 1st plot: color by STATE
    if plot_vel==0:
        ax1.plot_surface(Xc, Yc, Zc, facecolors=colors, linewidth=0, antialiased=False)
        for color_val, info in state_info.items():
            df_state = df_subset[df_subset['color_value'] == color_val]
            for _, row in df_state.iterrows():
                center = [
                    row['ParticleXPosition m'],
                    row['ParticleZPosition m'],
                    row['ParticleYPosition m']
                ]
                radius = row['scaled_size'] / 100
                x, y, z = create_sphere(center, radius)
                ax1.plot_surface(x, y, z, color=info['color'], alpha=0.8, linewidth=0, antialiased=False, shade=True)
    
        legend_patches = [
            mpatches.Patch(color=info['color'], label=info['label']) for info in state_info.values()
        ]
        ax1.legend(handles=legend_patches, loc='upper right', title='Particle State', fontsize=15, title_fontsize=14)
        #ax1.set_title("Particles Colored by State", fontsize=14)
        
    if plot_vel==0:
        # 2nd plot - particle velocity
        ax2.plot_surface(Xc, Yc, Zc, facecolors=colors, linewidth=0, antialiased=False)    
        for _, row in df_subset.iterrows():
            center = [
                row['ParticleXPosition m'],
                row['ParticleZPosition m'],
                row['ParticleYPosition m']
            ]
            radius = row['scaled_size'] / 100
            x, y, z = create_sphere(center, radius)
            #facecolor = cmap(norm(row['ParticleVelocityMagnitude m/s']))
            facecolor = blue_red_cmap(norm(row['ParticleVelocityMagnitude m/s']))
            facecolors = np.tile(facecolor, (x.shape[0], x.shape[1], 1))  # shape match
            ax2.plot_surface(x, y, z, facecolors=facecolors, linewidth=0, antialiased=False, shade=True)
    
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(df_subset['ParticleVelocityMagnitude m/s'])
        sm = cm.ScalarMappable(cmap=blue_red_cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax2, fraction=0.03, pad=0.04)
        cbar.set_label('Velocity Magnitude (m/s)', fontsize=15)
        cbar.set_ticks(np.linspace(0, 500, 6), fontsize=15)
        #ax2.set_title("Particles Colored by Velocity", fontsize=14)
        
    plt.tight_layout()
    plt.savefig(filename+".png", format='png', dpi=300)
    print(f"saved {filename}.png")
    #plt.show()

import random
np.random.seed(42)  
random.seed(42)  
pd.np.random.seed(42)  


import os

def get_csv_file_paths(folder_path):
    csv_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            full_path = os.path.join(folder_path, filename)
            csv_files.append(full_path)
    return csv_files

## ENTER path to the csv files
folder = r"E:\Pressure-vm-density\data\particles" 
csv_file_paths = get_csv_file_paths(folder)
print(csv_file_paths)

for filename in csv_file_paths:
#plot_df_static(df_subset)

    #filename = r"E:\Pressure-vm-density\data\particles\400kgm3_0.15vm_1.8bar_primaryflow=0.02kgs_particlefeed=1kgs.csv"
    df, df_subset = load_and_prepare_df(filename,40,0)
    plot_df_comparison(df_subset, filename[:-4])

    #import matplotlib.pyplot as plt

    plotting_bar =0;
    if plotting_bar ==1:
        state_labels = {
            0: 'Floating',
            1: 'Bouncing',
            2: 'Penetrating'
        }
        
     
        state_counts = df['color_value'].value_counts().sort_index()
        state_colors = {
        0: (231/255, 176/255, 79/255),   # Floating
        1: (165/255, 220/255, 205/255),   # Bouncing
        2: (50/255, 120/255, 163/255)    # Penetrating
        }
        
     
        df_unique = df.drop_duplicates(subset='UniqueID')
    
       
        state_counts = df_unique['color_value'].value_counts(normalize=True).sort_index() * 100
    
      
        labels = [state_labels[i] for i in state_counts.index]
        colors = [state_colors[i] for i in state_counts.index]
    
      
        plt.figure(figsize=(6, 4))
        bars = plt.bar(labels, state_counts.values, color=colors)
        plt.xlabel("Particle State", fontsize=16)
        plt.ylabel("Percentage (%)", fontsize=16)
        plt.title("Percentage of Particles by State", fontsize=16)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        # add percentage labels above bars
        #for bar in bars:
        #    height = bar.get_height()
        #    plt.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}%', ha='center', va='bottom')
    
        plt.ylim(0, 100)
        plt.tight_layout()
        plt.savefig(filename[:-4]+"_large_particles_bar.png", format='png', dpi=300)
        plt.show()