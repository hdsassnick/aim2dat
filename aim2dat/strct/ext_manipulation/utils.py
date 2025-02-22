

import itertools
from typing import List, Union

from aim2dat.strct import Structure


# TODO Add doc strings.
def _check_distances(
    structure: Structure, indices: List[int], dist_threshold: Union[float, None], silent: bool
):
    if dist_threshold is None:
        return True

    other_indices = [i for i in range(len(structure)) if i not in indices]
    if len(other_indices) == 0:
        indices1, indices2 = zip(*itertools.combinations(indices, 2))
    else:
        indices1, indices2 = zip(*itertools.product(other_indices, indices))
    dists = structure.calculate_distance(
        list(indices1), list(indices2), backfold_positions=True
    )
    if any(d0 < dist_threshold for d0 in dists.values()):
        if not silent:
            raise ValueError("Atoms are too close to each other.")
        return False
    return True
