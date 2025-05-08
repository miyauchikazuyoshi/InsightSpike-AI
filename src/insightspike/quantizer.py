import numpy as np

class Quantizer:
    def __init__(self, codebook: np.ndarray):
        self.codebook = codebook
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        # TODO: ベクトルをコードブックに量子化
        return np.zeros_like(x)

    def decode(self, indices: np.ndarray) -> np.ndarray:
        # TODO: 量子化インデックスから復元
        return np.zeros((len(indices), self.codebook.shape[1]))