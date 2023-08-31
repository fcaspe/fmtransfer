"""
Audio utility functions
"""

import numpy as np

def chunked_follower(audio, sr, attack_factor, release_factor, chunk_size):
    """
    Envelope Follower

    Args:
        audio (np.array): Audio Samples
        sr (int): Sample Rate
        attack_factor
        release_factor
        chunk_sizr (int): block size over which compute the result
    """
    envelope = [0]
    indices = [0]
    alpha_attack = attack_factor / sr
    alpha_release = release_factor / sr

    while indices[-1] + chunk_size < audio.shape[0]:
        new_index = indices[-1] + chunk_size
        chunk_power = np.mean(audio[indices[-1] : new_index] ** 2)
        chunked_attack_factor = alpha_attack * chunk_size
        chunked_release_factor = alpha_release * chunk_size
        new_envelope = max(
            envelope[-1] * (1 - chunked_attack_factor)
            + chunk_power * chunked_attack_factor,
            envelope[-1] * (1 - chunked_release_factor)
            + chunk_power * chunked_release_factor,
        )
        indices.append(new_index)
        envelope.append(new_envelope)
    return indices, envelope