import numpy as np
def get_depth(x, y, depth_frame):
    """
    Mengambil kedalaman dari titik (x, y) pada frame kedalaman.

    Parameters:
    x (int): Koordinat X pada frame.
    y (int): Koordinat Y pada frame.
    depth_frame (ndarray): Frame kedalaman yang didapat dari sensor kedalaman.

    Returns:
    float: Nilai kedalaman pada titik (x, y).
    """
    # Pastikan koordinat berada dalam batas frame
    if x < 0 or x >= depth_frame.shape[1] or y < 0 or y >= depth_frame.shape[0]:
        return None

    # Ambil kedalaman di titik (x, y)
    depth_value = depth_frame[y, x]

    # Jika nilai kedalaman tidak valid, return None
    if depth_value == 0:
        return None

    return depth_value

# Contoh fungsi ini dipanggil dengan frame kedalaman dari sensor:
# depth_frame = sensor.get_depth_frame()  # Contoh dari sensor kedalaman
# depth = get_depth(100, 150, depth_frame)
