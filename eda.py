import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from dataset import load_all_starccm_designs

# Set visual style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

DATA_DIR = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling"
OUTPUT_DIR = "EDA_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_cfd_eda():
    print("=" * 70)
    print("         CFD EXPLORATORY DATA ANALYSIS & SANITY AUDIT")
    print("=" * 70)

    # 1. LOAD RAW DATA
    dataset = load_all_starccm_designs(DATA_DIR)
    num_designs = len(dataset)
    if num_designs == 0:
        print("[ERROR] No designs found. Check DATA_DIR path.")
        return

    # =========================================================================
    # PART 1: GLOBAL DESIGN & PERFORMANCE METRICS (Level 1: Tabular)
    # =========================================================================
    scalar_records = []
    for d in dataset:
        p_d1, p_d2, p_v = d['params'].tolist()
        scalar_records.append({
            'Design': d['design_name'],
            'P_D1_r': p_d1,
            'P_D2_r': p_d2,
            'P_V': p_v,
            'Drag_N': d['true_drag'],
            'Cd': d['true_cd']
        })
    df_scalars = pd.DataFrame(scalar_records)

    print("\n[1] DESIGN PARAMETERS & REPORT TARGETS (Summary Statistics):")
    print("-" * 70)
    stats_summary = df_scalars.describe().T[['mean', 'std', 'min', '50%', 'max']]
    # Calculate Skewness
    stats_summary['skewness'] = df_scalars.skew(numeric_only=True)
    print(stats_summary.to_string())

    # Check for NaNs or Infs
    nan_count = df_scalars.isna().sum().sum()
    print(f"\n-> Missing/NaN entries in scalars: {nan_count}")
    if nan_count > 0:
        print("[WARNING] Missing values detected in scalar logs!")

    # =========================================================================
    # PART 2: FIELD-LEVEL AUDIT (Level 2: Spatial Physics across all faces)
    # =========================================================================
    all_pressures = []
    all_shears = []
    all_densities = []
    all_areas = []
    face_counts = []

    for d in dataset:
        fields = d['fields'].numpy() # [N_faces, 3] -> (P, tau, rho)
        geom = d['geom'].numpy()     # [N_faces, 7] -> (x, y, z, nx, ny, nz, A)

        all_pressures.append(fields[:, 0])
        all_shears.append(fields[:, 1])
        all_densities.append(fields[:, 2])
        all_areas.append(geom[:, 6])
        face_counts.append(len(fields))

    flat_p = np.concatenate(all_pressures)
    flat_tau = np.concatenate(all_shears)
    flat_rho = np.concatenate(all_densities)
    flat_area = np.concatenate(all_areas)

    field_df = pd.DataFrame({
        'Static_Pressure_Pa': flat_p,
        'Wall_Shear_Stress_Pa': flat_tau,
        'Density_kg_m3': flat_rho,
        'Face_Length_m': flat_area
    })

    print("\n[2] PHYSICAL FIELD DISTRIBUTIONS (Pooled Across All Surface Faces):")
    print("-" * 70)
    field_summary = field_df.describe().T[['mean', 'std', 'min', '50%', 'max']]
    field_summary['skewness'] = field_df.skew()
    print(field_summary.to_string())

    # Anomaly / Outlier Detection (Z-Score > 4)
    p_zscores = np.abs(stats.zscore(flat_p))
    extreme_p_count = np.sum(p_zscores > 4.0)
    print(f"\n-> Extreme Pressure Outliers (|z| > 4.0): {extreme_p_count} faces")
    if flat_rho.std() < 1e-4:
        print("-> Flow Regime: Incompressible (Density is virtually constant)")

    print(f"-> Mesh Consistency: Min faces = {min(face_counts)}, Max faces = {max(face_counts)}")

    # =========================================================================
    # PART 3: GENERATE VISUALIZATIONS
    # =========================================================================
    print("\n[3] Generating Diagnostic Figures...")

    # --- FIGURE 1: DESIGN SPACE COVERAGE & CORRELATIONS ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    
    # P_D1 vs P_D2 colored by Drag
    sc1 = axes[0].scatter(df_scalars['P_D1_r'], df_scalars['P_D2_r'], 
                         c=df_scalars['Drag_N'], cmap='viridis', s=80, edgecolors='k')
    axes[0].set_title("Design Space: P_D1_r vs P_D2_r")
    axes[0].set_xlabel("P_D1_r (CAD Radius 1)")
    axes[0].set_ylabel("P_D2_r (CAD Radius 2)")
    cbar1 = plt.colorbar(sc1, ax=axes[0])
    cbar1.set_label("Total Drag [N]")

    # Velocity P_V vs Drag
    sns.regplot(data=df_scalars, x='P_V', y='Drag_N', ax=axes[1], color='darkred',
                scatter_kws={'s': 60, 'edgecolor': 'k'})
    axes[1].set_title("Inlet Velocity (P_V) vs Drag")
    axes[1].set_xlabel("Inlet Velocity [m/s]")
    axes[1].set_ylabel("Total Drag [N]")

    # Correlation Matrix Heatmap
    corr = df_scalars[['P_D1_r', 'P_D2_r', 'P_V', 'Drag_N', 'Cd']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[2], vmin=-1, vmax=1)
    axes[2].set_title("Pearson Correlation Matrix")

    plt.tight_layout()
    fig1_path = os.path.join(OUTPUT_DIR, "1_design_space_correlation.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig1_path}")

    # --- FIGURE 2: PHYSICAL FIELD DISTRIBUTIONS ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Pressure
    sns.histplot(flat_p, kde=True, ax=axes[0], color='royalblue', bins=40)
    axes[0].set_title("Static Pressure Distribution")
    axes[0].set_xlabel("Pressure [Pa]")

    # Wall Shear
    sns.histplot(flat_tau, kde=True, ax=axes[1], color='crimson', bins=40)
    axes[1].set_title("Wall Shear Stress Distribution")
    axes[1].set_xlabel("Shear Stress [Pa]")

    # Density
    sns.histplot(flat_rho, kde=True, ax=axes[2], color='forestgreen', bins=20)
    axes[2].set_title("Density Distribution")
    axes[2].set_xlabel("Density [kg/m³]")

    plt.tight_layout()
    fig2_path = os.path.join(OUTPUT_DIR, "2_field_distributions.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig2_path}")

    # --- FIGURE 3: GEOMETRIC VARIABILITY (Shape Overlay) ---
    plt.figure(figsize=(9, 7))
    palette = sns.color_palette("rocket", n_colors=num_designs)

    for i, d in enumerate(dataset):
        geom = d['geom'].numpy()
        x = geom[:, 0]
        y = geom[:, 1]
        
        # Sort along y for clean continuous boundary lines
        angles = np.arctan2(y - 0.5, x - 0.5)
        sort_idx = np.argsort(angles)
        plt.plot(x[sort_idx], y[sort_idx], alpha=0.6, color=palette[i], linewidth=1.5)

    plt.title("Blunt Body Surface Geometry Variations (All Designs)")
    plt.xlabel("X Coordinate [m]")
    plt.ylabel("Y Coordinate [m]")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.5)

    fig3_path = os.path.join(OUTPUT_DIR, "3_geometric_shape_variations.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig3_path}")

    # --- FIGURE 4: PRESSURE SIGNATURE ALONG THE PERIMETER ---
    plt.figure(figsize=(11, 5))
    sorted_designs = sorted(dataset, key=lambda x: x['true_drag'])
    sample_indices = np.linspace(0, num_designs - 1, min(5, num_designs), dtype=int)

    for idx in sample_indices:
        d = sorted_designs[idx]
        geom = d['geom'].numpy()
        x = geom[:, 0]
        y = geom[:, 1]
        p = d['fields'][:, 0].numpy()
        
        # Calculate angle around the body center (-180° to +180°)
        # 180° = Leading Edge (Nose), 0° = Trailing Edge (Base)
        angles_deg = np.degrees(np.arctan2(y - 0.5, x - 0.5))
        sort_idx = np.argsort(angles_deg)
        
        plt.plot(angles_deg[sort_idx], p[sort_idx], 
                 label=f"{d['design_name']} (Drag={d['true_drag']:.2f} N)", 
                 linewidth=2)

    plt.title("Static Pressure Profile along Perimeter Angle")
    plt.xlabel("Perimeter Angle θ [degrees]  (-180° = Nose/Stagnation, 0° = Rear Base)")
    plt.ylabel("Static Pressure [Pa]")
    plt.xticks([-180, -90, 0, 90, 180], 
               ["-180° (Nose)", "-90° (Bottom)", "0° (Base)", "+90° (Top)", "+180° (Nose)"])
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.6)

    fig4_path = os.path.join(OUTPUT_DIR, "4_surface_pressure_profiles.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"  Saved: {fig4_path}")

    print("=" * 70)
    print(f"EDA COMPLETE: Review diagnostic charts in the '{OUTPUT_DIR}' directory.")
    print("=" * 70)

if __name__ == "__main__":
    run_cfd_eda()