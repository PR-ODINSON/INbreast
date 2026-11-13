# Convenience wrapper for importing the vendored GridAttentionBlock2D
import os
import sys

# Add third party directory to sys.path
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_repo_dir = os.path.join(_curr_dir, "Attention-Gated-Networks")
if _repo_dir not in sys.path:
    sys.path.insert(0, _repo_dir)

from models.layers.grid_attention_layer import GridAttentionBlock2D

__all__ = ["GridAttentionBlock2D"]

