"""Interface to the ase calculator."""


def get_potential_energy(structure, calculator):
    """get potential energy from an ase calculator."""
    ase_atoms = structure.to_ase_atoms()
    ase_atoms.calc = calculator
    return ase_atoms.get_potential_energy()
