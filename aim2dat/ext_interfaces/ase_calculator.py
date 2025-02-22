


def get_potential_energy(structure, calculator):
    ase_atoms = structure.to_ase_atoms()
    ase_atoms.calc = calculator
    return ase_atoms.get_potential_energy()
