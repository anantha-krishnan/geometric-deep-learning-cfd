import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. DATA WRANGLING: Convert STAR-CCM+ Arrays to PyTorch Graph
# =====================================================================
def build_face_graph(vertex_list, num_faces):
    """
    Finds which faces share vertices.
    vertex_list: flat array like [1,0, 2,1, 0,3, ...]
    Returns PyTorch edge_index [2, Num_Edges]
    """
    # Map vertex_id -> list of face_ids that use it
    vertex_to_faces = {}
    for face_id in range(num_faces):
        # In 2D, each face has 2 vertices
        v1, v2 = vertex_list[face_id*2], vertex_list[face_id*2 + 1]
        
        if v1 not in vertex_to_faces: vertex_to_faces[v1] = []
        if v2 not in vertex_to_faces: vertex_to_faces[v2] = []
        
        vertex_to_faces[v1].append(face_id)
        vertex_to_faces[v2].append(face_id)

    # Create edges between faces that share a vertex
    edges = []
    for v_id, faces in vertex_to_faces.items():
        if len(faces) == 2:  # Shared by exactly two faces (internal boundary)
            f_a, f_b = faces[0], faces[1]
            edges.append([f_a, f_b])
            edges.append([f_b, f_a]) # Undirected graph
            
    return torch.tensor(edges, dtype=torch.long).t()

# =====================================================================
# 2. THE AI ROM ARCHITECTURE (Parameterized GNN)
# =====================================================================
class MeshMessagePassing(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())
        self.node_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU())

    def forward(self, h, edge_index):
        src, dst = edge_index[0], edge_index[1]
        
        # Messages: concat owner and neighbor features
        msg_input = torch.cat([h[src], h[dst]], dim=-1)
        messages = self.edge_mlp(msg_input)
        
        # Aggregate messages into target faces
        aggr_msg = torch.zeros_like(h)
        aggr_msg.index_add_(0, dst, messages)
        
        # Update face features
        node_input = torch.cat([h, aggr_msg], dim=-1)
        return h + self.node_mlp(node_input)

class PhysicsAISurfaceModel(nn.Module):
    def __init__(self, geom_dim=7, design_dim=3, hidden_dim=64, out_dim=3):
        super().__init__()
        # Input = Geometry Features + Global Design Parameters
        self.encoder = nn.Linear(geom_dim + design_dim, hidden_dim)
        
        self.mp1 = MeshMessagePassing(hidden_dim)
        self.mp2 = MeshMessagePassing(hidden_dim)
        self.mp3 = MeshMessagePassing(hidden_dim)
        
        # Output = [Pressure, Shear, Density]
        self.decoder = nn.Linear(hidden_dim, out_dim)

    def forward(self, geom_features, design_params, edge_index):
        # 1. Tile the design parameters so every face knows the global design
        num_faces = geom_features.size(0)
        # Shape becomes [N_faces, 3]
        design_tiled = design_params.unsqueeze(0).repeat(num_faces, 1) 
        
        # 2. Concatenate [Geometry, Design]
        x = torch.cat([geom_features, design_tiled], dim=-1)
        
        # 3. Pass through network
        h = self.encoder(x)
        h = self.mp1(h, edge_index)
        h = self.mp2(h, edge_index)
        h = self.mp3(h, edge_index)
        
        # 4. Predict fields
        return self.decoder(h)

# =====================================================================
# 3. DRAG INTEGRATION FORMULA
# =====================================================================
def compute_drag(predicted_fields, geom_features, drag_dir=torch.tensor([1.0, 0.0, 0.0])):
    """
    predicted_fields: [N_faces, 3] -> (Pressure, Shear, Density)
    geom_features: contains Normal Vectors and Area
    """
    P_f = predicted_fields[:, 0]
    tau_f = predicted_fields[:, 1]
    
    # Assuming geom_features structure: [x, y, z, nx, ny, nz, Area]
    normals = geom_features[:, 3:6]
    area_f  = geom_features[:, 6]
    
    # 1. Pressure Force = P * A * (n dot drag_dir)
    n_dot_d = torch.matmul(normals, drag_dir)
    pressure_drag = torch.sum(P_f * area_f * n_dot_d)
    
    # 2. Viscous Force
    # Tangent vector is orthogonal to normal. In 2D (xy plane), if n=(nx, ny), t=(ny, -nx)
    tangents = torch.stack([normals[:, 1], -normals[:, 0], torch.zeros_like(normals[:, 0])], dim=1)
    t_dot_d = torch.matmul(tangents, drag_dir)
    viscous_drag = torch.sum(tau_f * area_f * t_dot_d)
    
    return pressure_drag + viscous_drag


# =====================================================================
# 4. PUTTING IT ALL TOGETHER (Mock Training Loop)
# =====================================================================
# --- Mock Data based on your STAR-CCM+ export ---
N_FACES = 88
# [x, y, z, nx, ny, nz, Area]
geom_data = torch.randn(N_FACES, 7) 

# Flat vertex list (like your export)
vertex_list = []
for i in range(N_FACES):
    vertex_list.extend([i, i+1]) # Mock connectivity
edge_index = build_face_graph(vertex_list, N_FACES)

# The Design Parameters for this specific snapshot
design_params = torch.tensor([1.2, 0.5, 30.0]) # e.g. [P_D1_r, P_D2_r, P_V]

# The Ground Truth CFD fields for this design
# [Pressure, Shear, Density]
true_fields = torch.randn(N_FACES, 3) 

# --- Initialize Model ---
model = PhysicsAISurfaceModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Step ---
model.train()
optimizer.zero_grad()

# Forward pass: Predict P, tau, rho
pred_fields = model(geom_data, design_params, edge_index)

# Loss: Multi-variable MSE (exactly as copilot described)
loss = nn.MSELoss()(pred_fields, true_fields)
loss.backward()
optimizer.step()

print(f"Field Model Loss: {loss.item():.4f}")

# --- Inference Step (What happens after training) ---
model.eval()
with torch.no_grad():
    predicted_fields = model(geom_data, design_params, edge_index)
    
    # Now WE compute the drag from the network's physics prediction!
    computed_drag = compute_drag(predicted_fields, geom_data)
    print(f"Computed Aerodynamic Drag: {computed_drag.item():.4f} N")