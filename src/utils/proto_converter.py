"""
Utility for converting between JSON API schemas and internal tensor formats.
Keeps the API layer clean and the network layer focused on tensors.
"""

import numpy as np
from typing import Optional


# Board constants (must match Java engine)
TOTAL_NODES = 126
NODES_PER_LAYER = 18
LAYERS = 7
ROWS = 3
COLUMNS = 6
CHANNELS = 8


def node_to_coords(node_id: int) -> tuple[int, int, int]:
    """Convert flat node ID to (layer, row, column)."""
    layer = node_id // NODES_PER_LAYER
    offset = node_id % NODES_PER_LAYER
    row = offset // COLUMNS
    col = offset % COLUMNS
    return layer, row, col


def coords_to_node(layer: int, row: int, col: int) -> int:
    """Convert (layer, row, column) to flat node ID."""
    return layer * NODES_PER_LAYER + row * COLUMNS + col


def state_to_tensor_3d(flat_state: list[float]) -> np.ndarray:
    """
    Reshape flat 1008-float state into [7, 3, 6, 8] tensor
    for convolutional processing.
    """
    arr = np.array(flat_state, dtype=np.float32)
    # flat layout: node * CHANNELS + channel
    reshaped = arr.reshape(TOTAL_NODES, CHANNELS)

    tensor = np.zeros((LAYERS, ROWS, COLUMNS, CHANNELS), dtype=np.float32)
    for node_id in range(TOTAL_NODES):
        l, r, c = node_to_coords(node_id)
        tensor[l, r, c, :] = reshaped[node_id, :]

    return tensor
