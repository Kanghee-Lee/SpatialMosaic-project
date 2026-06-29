import os
import sys

# Get the directory containing the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

def build_spatial_tower(spatial_tower_cfg, **kwargs):
    spatial_tower = getattr(spatial_tower_cfg, "mm_spatial_tower", getattr(spatial_tower_cfg, "spatial_tower", 'spann3r'))
    if spatial_tower == "spann3r":
        spann3r_path = os.path.join(repo_root, 'spann3r')
        if spann3r_path not in sys.path:
             sys.path.append(spann3r_path)
        # Use relative import for the encoder wrapper/adapter file
        from .spann3r_spatial_encoder import Spann3rSpatialTower
        return Spann3rSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    elif spatial_tower == "vggt":
        vggt_path = os.path.join(repo_root, 'vggt')
        if vggt_path not in sys.path:
            sys.path.append(vggt_path)
        # Use relative import for the encoder wrapper/adapter file
        from .vggt_spatial_encoder import VGGTSpatialTower
        return VGGTSpatialTower(spatial_tower, spatial_tower_cfg=spatial_tower_cfg, **kwargs)
    raise ValueError(f"Unknown vision tower: {spatial_tower}")
