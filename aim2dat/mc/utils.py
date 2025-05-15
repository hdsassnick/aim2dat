"""Utility methods related to Monte Carlo simulations."""

# Standard library imports
from typing import Union

# Third party library imports
import numpy as np

# Internal library imports
import aim2dat.utils.units as a2d_units


def _validate_inputs(energies: Union[float, list], pressure: Union[float, list], *args) -> list:
    """Validate and transform input parameters."""
    is_float = False
    if isinstance(energies, (int, float)):
        is_float = True
        energies = [energies]
    energies = np.array(energies)
    if isinstance(pressure, (int, float)):
        pressure = [pressure]
    pressure = np.array(pressure)

    args = list(args)
    for i, arg in enumerate(args):
        if not isinstance(arg, (float, int)):
            args[i] = np.array(arg)
            if len(energies) != len(arg):
                raise ValueError(
                    "`energies`, `n_molecules` and `volumes` need to have the same length."
                )

    return [energies, pressure] + args + [is_float]


def _get_acceptance_output(
    acc: np.array, pressure: np.array, is_float: bool
) -> Union[float, list]:
    """Process output."""
    acc = acc * pressure[:, np.newaxis]
    acc = np.where(acc < 1.0, acc, 1.0).tolist()
    if is_float:
        acc = [v[0] for v in acc]
    if len(pressure) == 1:
        acc = acc[0]
    return acc


def calc_deletion_acceptance(
    energies: Union[float, list],
    n_molecules: Union[int, list],
    volume: Union[float, list],
    pressure: Union[float, list],
    temperature: float = 298.15,
) -> Union[float, list]:
    """
    Calculate acceptance probability of deletion moves.

    Parameters
    ----------
    energies : float or list
        Energy difference(s) between the (n - 1) and n configuration in eV.
    n_molecules : int or list
        Number of molecules.
    volume : float or list
        Volume of structure in angstrom.
    pressure : float or list
        Pressure in Pa.
    temperature : float (optional)
        Temperature in K.

    Returns
    -------
    float or list
        Acceptance probability. If ``energies`` is a number, the output is of type float or a 1d
        list if ``pressure`` is a list. If ``energies`` is given as a list, the output is a list
        as well (if ``pressure`` is a list, the output is a nested list
        ``len(pressure)xlen(energies)``).

    Raises
    ------
    ValueError
        ``energies``, ``n_molecules`` and ``volumes`` need to have the same length.
    """
    energies, pressure, n_molecules, volume, is_float = _validate_inputs(
        energies, pressure, n_molecules, volume
    )
    beta = 1.0 / (temperature * a2d_units.constants.kb * a2d_units.energy.j)
    pressure = 1.0 / (pressure * a2d_units.pressure.pascal)
    acc = np.tile(
        n_molecules / (volume * beta) * np.exp(-1.0 * beta * energies), (len(pressure), 1)
    )
    return _get_acceptance_output(acc, pressure, is_float)


def calc_insertion_acceptance(
    energies: Union[float, list],
    n_molecules: Union[int, list],
    volume: Union[float, list],
    pressure: Union[float, list],
    temperature: float = 298.15,
) -> Union[float, list]:
    """
    Calculate acceptance probability of insertion moves.

    Parameters
    ----------
    energies : float or list
        Energy difference(s) between the (n + 1) and n configuration in eV.
    n_molecules : int or list
        Number of molecules.
    volume : float or list
        Volume of structure in angstrom.
    pressure : float or list
        Pressure in Pa.
    temperature : float (optional)
        Temperature in K.

    Returns
    -------
    float or list
        Acceptance probability. If ``energies`` is a number, the output is of type float or a 1d
        list if ``pressure`` is a list. If ``energies`` is given as a list, the output is a list
        as well (if ``pressure`` is a list, the output is a nested list
        ``len(pressure)xlen(energies)``).

    Raises
    ------
    ValueError
        ``energies``, ``n_molecules`` and ``volumes`` need to have the same length.
    """
    energies, pressure, n_molecules, volume, is_float = _validate_inputs(
        energies, pressure, n_molecules, volume
    )
    beta = 1.0 / (temperature * a2d_units.constants.kb * a2d_units.energy.j)
    pressure = pressure * a2d_units.pressure.pascal
    acc = np.tile(
        volume * beta / (n_molecules + 1) * np.exp(-1.0 * beta * energies), (len(pressure), 1)
    )
    return _get_acceptance_output(acc, pressure, is_float)
