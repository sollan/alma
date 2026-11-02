"""Explicit tracked versus analysis coordinates for diagnostic skeletons.

Visible low-confidence tracking is shown as such, without making it eligible
for parameter calculation. Filter availability never removes tracked points.
"""
import numpy as np

from Functions.TreadmillValidation import BODY_PARTS

MISSING, OBSERVED, LOW_CONFIDENCE, RECONSTRUCTED = range(4)


def skeleton_coordinates(coordinates, source='tracked', confidence_threshold=.6):
    """Return (frame, joint, xy) coordinates and per-joint display states.

    ``tracked`` uses untouched input positions, including finite low-confidence
    samples marked LOW_CONFIDENCE. ``analysis`` uses the selected filtered
    positions and reconstruction flags. Neither view fills a missing point.
    """
    if source not in ('tracked', 'analysis'):
        raise ValueError('Skeleton source must be tracked or analysis.')
    suffix = ' raw' if source == 'tracked' else ''
    names = [f'{bp}{suffix} {axis}' for bp in BODY_PARTS for axis in ('x', 'y')]
    if any(name not in coordinates for name in names):
        raise ValueError(f'{source.capitalize()} skeleton coordinates are unavailable. Re-export coordinates or select the analysis view.')
    xy = coordinates[names].to_numpy(dtype=float).reshape(len(coordinates), len(BODY_PARTS), 2)
    finite = np.isfinite(xy).all(axis=2)
    states = np.zeros(finite.shape, dtype=np.int8)
    for i, bp in enumerate(BODY_PARTS):
        likelihood = coordinates[f'{bp} likelihood'].to_numpy(dtype=float)
        reliable = np.isfinite(likelihood) & (likelihood >= confidence_threshold)
        states[:, i] = np.where(finite[:, i], np.where(reliable, OBSERVED, LOW_CONFIDENCE), MISSING)
        if source == 'analysis' and f'{bp} metric interpolated' in coordinates:
            reconstructed = coordinates[f'{bp} metric interpolated'].to_numpy(dtype=bool)
            states[finite[:, i] & reconstructed, i] = RECONSTRUCTED
    return xy, states
