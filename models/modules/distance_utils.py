import numpy as np


def get_cosine_distance(source_vecs: np.ndarray,
                        target_vecs: np.ndarray,
                        for_all_combinations: bool = True) -> np.ndarray:
    src_normalized = source_vecs / np.linalg.norm(source_vecs, axis=1)[:, np.newaxis]
    tgt_normalized = target_vecs / np.linalg.norm(target_vecs, axis=1)[:, np.newaxis]

    if for_all_combinations:
        similarity = np.matmul(src_normalized, tgt_normalized.T)
    else:
        similarity = (src_normalized * tgt_normalized).sum(dim=1)

    # convert cosine distance [-1, 1] to distance [2, 0]
    return 1 - similarity


def get_euclid_distance(source_vecs: np.ndarray,
                        target_vecs: np.ndarray,
                        for_all_combinations: bool = True) -> np.ndarray:
    if for_all_combinations:
        distances = np.linalg.norm(source_vecs[:, np.newaxis] - target_vecs, axis=2)
    else:
        distances = np.linalg.norm(source_vecs - target_vecs, axis=1)
    return distances
