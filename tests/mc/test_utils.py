"""Test the aim2dat.mc.utils module."""

# Third party library imports
import pytest
import numpy as np

# Internal library imports
from aim2dat.mc.utils import calc_deletion_acceptance, calc_insertion_acceptance


def test_calc_deletion_acceptance():
    """Test calc_deletion_acceptance function."""
    energies = [0.35285122, 0.1766219, 0.38483953, 0.38260086]
    volume = [5311.37931439, 5312.37931439, 5313.37931439, 5320.37931439]
    n_molecules = [10, 10, 18, 18]
    pressure = [1000.0, 2000.0]
    temperature = 298.15
    ref = [
        [0.008411831860420297, 1.0, 0.004357979910218697, 0.004748482087491492],
        [0.004205915930210149, 1.0, 0.0021789899551093485, 0.002374241043745746],
    ]
    acc = calc_deletion_acceptance(energies, n_molecules, volume, pressure, temperature)
    assert np.allclose(acc, ref)
    acc = calc_deletion_acceptance(energies[0], n_molecules[0], volume[0], pressure, temperature)
    assert np.allclose(acc, [ref[0][0], ref[1][0]])
    with pytest.raises(ValueError) as error:
        calc_deletion_acceptance(energies[0], n_molecules, volume[0], pressure, temperature)
    assert (
        str(error.value) == "`energies`, `n_molecules` and `volumes` need to have the same length."
    )


def test_calc_insertion_acceptance():
    """Test calc_insertion_acceptance function."""
    energies = -0.15963919
    volume = 5311.37931439
    n_molecules = 10
    pressure = 1000.0
    temperature = 298.15
    acc = calc_insertion_acceptance(energies, n_molecules, volume, pressure, temperature)
    assert abs(acc - 0.0585810) < 1e-6
