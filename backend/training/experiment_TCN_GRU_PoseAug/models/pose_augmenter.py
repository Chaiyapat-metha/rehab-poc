# rehab-poc/backend/training/experiment_TCN_GRU_PoseAug/models/pose_augmenter.py

import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Dict, Tuple
from scipy.spatial.transform import Rotation as R

# --- 33-Joint MediaPipe Kinematic Tree (Parents Definition) ---
# Parent index of each joint. -1 indicates the root joint (Right Hip = 24).
# This is crucial for hierarchical reconstruction.
# Index: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32
MP_PARENTS = [
    1, 2, 3, 7, 4, 5, 6, 0, 0, 10, 9, 23, 24, 11, 12, 13, 14, 15, 16, 15, 16, 15, 16, 24, -1, 24, 23, 26, 25, 28, 27, 30, 29
]

# --- Symmetric Bone Pairs (Child Indices) ---
# Pairs of bones (L/R) that must receive the same BL factor
MP_BONE_PAIRS_SYM = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22), 
    (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)
]

class PoseAugmenter(nn.Module):
    def __init__(self, parents: List[int], bone_pairs_sym: List[Tuple[int, int]], config: Dict[str, any] = None):
        super().__init__()
        
        # --- Kinematic Tree Setup ---
        self.parents = parents
        self.J = len(self.parents)
        self.child_inds = [j for j in range(self.J) if self.parents[j] != -1]
        self.parent_inds = [self.parents[j] for j in self.child_inds]
        self.K = len(self.child_inds) # Number of bones
        self.bone_pairs_sym = bone_pairs_sym or []

        # --- Config Setup (Using recommended initial values) ---
        default = {
            'bl_std': 0.12,
            'bl_clip': 0.4,
            'rt_rot_deg': 30.0,
            'rt_trans': 0.12,
        }
        self.cfg = default if config is None else {**default, **config}

    def forward(self, X: torch.Tensor):
        # X: (B,T,J,3)  -> B=Batch Size (1 in Dataloader), T=Window Size (32), J=33, C=3
        assert X.ndim == 4 and X.shape[-1] == 3
        B, T, J, C = X.shape
        device, dtype = X.device, X.dtype

        # --- BL (Bone Length) Augmentation ---
        child = torch.tensor(self.child_inds, device=device)
        parent = torch.tensor(self.parent_inds, device=device)

        X_child = X[..., child, :]
        X_parent = X[..., parent, :]
        Bvec = X_child - X_parent                               # Bone Vectors (B,T,K,3)

        eps = 1e-8
        lengths = torch.norm(Bvec, dim=-1, keepdim=True)        # Bone Lengths (B,T,K,1)
        B_hat = Bvec / (lengths + eps)                          # Unit Bone Vectors (B,T,K,3)

        # 1. Generate scaling factors (constant across time for simplicity: B, 1, K, 1)
        bl_std = float(self.cfg['bl_std'])
        bl_factors = torch.randn(B, 1, self.K, 1, device=device, dtype=dtype) * bl_std + 1.0
        clip = float(self.cfg.get('bl_clip', 0.4))
        if clip and clip > 0:
            bl_factors = torch.clamp(bl_factors, 1.0 - clip, 1.0 + clip)

        # 2. Enforce Symmetry (Average factor for L/R bones)
        for (ka, kb) in self.bone_pairs_sym:
            # FIX: Only proceed if BOTH joints are actual children (i.e., not the root)
            if ka in self.child_inds and kb in self.child_inds: 
                a_idx = self.child_inds.index(ka)
                b_idx = self.child_inds.index(kb)

                a = bl_factors[..., a_idx:a_idx+1, :]
                b = bl_factors[..., b_idx:b_idx+1, :]
                mean = 0.5 * (a + b)
                
                bl_factors[..., a_idx:a_idx+1, :] = mean
                bl_factors[..., b_idx:b_idx+1, :] = mean 

        # 3. Apply scaling to reconstruct the bone vectors
        lengths_prime = lengths * bl_factors
        Bvec_prime = B_hat * lengths_prime                      # New Bone Vectors

        # 4. Hierarchical Reconstruction (H-inverse)
        root_idx = self.parents.index(-1)
        Xp = torch.zeros_like(X, device=device, dtype=dtype)
        Xp[..., root_idx, :] = X[..., root_idx, :] # Root remains fixed
        
        # We assume parents are ordered before children (top-down traversal)
        processed = {root_idx}
        child2bone = {c: idx for idx, c in enumerate(self.child_inds)}
        
        # Traverse the tree to reconstruct skeleton
        for c in self.child_inds: # Iterating through children
            p = self.parents[c]
            if p in processed:
                k = child2bone[c]
                Xp[..., c, :] = Xp[..., p, :] + Bvec_prime[..., k, :]
                processed.add(c)
        X = Xp # X now holds the BL-augmented pose

        # X คือ BL-augmented pose data (B, T, J, 3)
        # ----------------------------------------------------
        # --- RT (Rigid Transform) Augmentation ---
        # ----------------------------------------------------
        rot_deg = float(self.cfg.get('rt_rot_deg', 30.0))
        trans_mag = float(self.cfg.get('rt_trans', 0.12))
        
        # 1. Center the Pose around the Root Joint (24)
        root_idx = self.parents.index(-1)
        root_pos = X[..., root_idx:root_idx+1, :]  # (B,T,1,3)
        Xc = X - root_pos                          # Centered Pose (B, T, J, 3)
        
        # 2. Generate Rotation Matrix (R_mat) - [B, 3, 3]
        rot_angles = (torch.rand(B, 1, 3, device=device, dtype=dtype) * 2.0 - 1.0) * (rot_deg * math.pi / 180.0)
        
        # ใช้ scipy.spatial.transform.Rotation (R) ในการสร้าง R_mat [B, 3, 3]
        rot_angles_np = rot_angles.squeeze(1).cpu().numpy()
        R_mat_np = R.from_euler('xyz', rot_angles_np, degrees=False).as_matrix()
        R_mat = torch.tensor(R_mat_np, device=device, dtype=dtype) # R_mat shape: (B, 3, 3)

        # 3. Apply Rotation using Batched Matrix Multiplication (BMM)
        
        # a. เตรียม Centered Pose: รวม Batch และ Time เข้าด้วยกัน
        # Xc_bmm shape: (B * T, J, 3)
        Xc_bmm = Xc.view(B * T, J, 3)
        
        # b. เตรียม Rotation Matrix: ต้องทำซ้ำ (repeat) R_mat T ครั้ง
        # R_mat_bmm shape: (B * T, 3, 3)
        R_mat_bmm = R_mat.unsqueeze(1).repeat(1, T, 1, 1).view(B * T, 3, 3)
        
        # c. ทำ BMM: [ (B*T), J, 3 ] @ [ (B*T), 3, 3 ] -> [ (B*T), J, 3 ]
        rotated_bmm = torch.bmm(Xc_bmm, R_mat_bmm) 
        
        # d. ปรับรูปร่างกลับเป็น (B, T, J, 3)
        rotated = rotated_bmm.view(B, T, J, 3)
        
        # 4. Apply Translation
        trans_mag = float(self.cfg.get('rt_trans', 0.12))
        trans_vecs = (torch.rand(B, 1, 3, device=device, dtype=dtype) * 2.0 - 1.0) * trans_mag
        
        Xp_rot = rotated + root_pos                 # Re-add root position
        Xp_trans = Xp_rot + trans_vecs.unsqueeze(2) # Add translation (broadcasts across T and J)
        
        return Xp_trans.contiguous()

# The Augmenter requires initialization with the actual structure
def initialize_pose_augmenter(config: Dict[str, any]):
    """Helper function to create the PoseAugmenter instance."""
    # NOTE: You MUST ensure scipy and torch work together for 3D rotation.
    # The parent indices for 33 joints must be fully listed for the model to work correctly.
    
    # We use the simplified structure defined globally
    return PoseAugmenter(MP_PARENTS, MP_BONE_PAIRS_SYM, config['augmentation']['poseaug'])