import re

def parse_design_log(log_path):
    """
    Extracts design parameters and official STAR-CCM+ report metrics
    directly from Design_X.log.
    """
    with open(log_path, 'r', errors='ignore') as f:
        content = f.read()

    # 1. Extract Design Parameters
    p_d1 = float(re.search(r"P_D1_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_d2 = float(re.search(r"P_D2_r\s*:\s*([0-9.eE+-]+)", content).group(1))
    p_v  = float(re.search(r"P_V\s*:\s*([0-9.eE+-]+)", content).group(1))

    # 2. Extract Official STAR-CCM+ Reports
    drag = float(re.search(r"Drag\s*:\s*([0-9.eE+-]+)", content).group(1))
    cd   = float(re.search(r"Cd\s*:\s*([0-9.eE+-]+)", content).group(1))
    pdrop = float(re.search(r"PressureDrop\s*:\s*([0-9.eE+-]+)", content).group(1))

    return {
        "params": [p_d1, p_d2, p_v],
        "true_drag": drag,
        "true_cd": cd,
        "true_pdrop": pdrop
    }

# --- Test on Design 1 ---
log_file = r"AIROMMinimizingDragAroundAnAerodynamicBody_input/aero_body_3D_AI_ROM/Training_Adaptive_Sampling/Design_1/Design_1.log"
data = parse_design_log(log_file)
print("Parsed Results from Design_1.log:")
for k, v in data.items():
    print(f"  {k}: {v}")