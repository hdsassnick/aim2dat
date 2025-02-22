
import abc

import numpy as np


from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.ext_manipulation import add_structure_position, translate_structure, rotate_structure
from aim2dat.utils.units import energy, constants

class BaseMove(abc.ABC):
    n_rand_nrs = 3

    def __init__(self, structure, component, ase_calculator):
        self.structure = structure.copy()
        self.component = component.copy()
        self.new_structure = None

        self.ase_calculator = ase_calculator


    @abc.abstractmethod
    def perform_move(self, rand_nrs):
        pass

    def accept_move(self, rand_nr, temperature):
        # TODO add presssure/fugacity
        self.get_energy(self.new_structure)
        e_diff = self.new_structure.get_attribute("ref_energy") - self.structure.get_attribute("ref_energy")
        # TODO check here if energies are available?
        print(e_diff, self.new_structure.get_attribute("ref_energy"), self.structure.get_attribute("ref_energy"))
        print(rand_nr < min(1.0, np.exp(-1.0 * e_diff / (temperature * constants.kb * energy.j))))
        return rand_nr < min(1.0, np.exp(-1.0 * e_diff / (temperature * constants.kb * energy.j)))

    def get_energy(self, structure):
        # TODO implement lammps and openmm backends and make backend module
        if self.ase_calculator is not None:
            backend_module = _return_ext_interface_modules("ase_calculator")
            structure.set_attribute("ref_energy", backend_module.get_potential_energy(structure, self.ase_calculator))
        else:
            raise ValueError("No viable backend found to evaluate the energy.")

    def get_random_component_indices(self, rand_nr):
        indices = {}
        for idx, kind in enumerate(self.structure.kinds):
            if kind is None:
                continue
            if self.component.label in kind:
                indices.setdefault(kind, []).append(idx)
        kind = list(indices.keys())[int(rand_nr * len(indices))]
        return kind, indices[kind]


class RotateComponent(BaseMove):
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        comp_label, comp_indices = self.get_random_component_indices(rand_nrs[0])
        try:
            self.new_structure = rotate_structure(
                self.structure, angles=rand_nrs[1] * 360.0, vector=rand_nrs[2:5],
                site_indices=comp_indices, change_label=False, dist_threshold=1.25
            )
        except ValueError:
            return False
        return True


class TranslateComponent(BaseMove):
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        comp_label, comp_indices = self.get_random_component_indices(rand_nrs[0])
        v = np.array(rand_nrs[1:4]) - 0.5
        v *= rand_nrs[4] * np.linalg.norm(v)
        try:
            self.new_structure = translate_structure(self.structure, site_indices=comp_indices, vector=v, dist_threshold=1.25, change_label=False)
        except ValueError:
            return False
        return True


class ReinsertComponent(BaseMove):
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        comp_label, comp_indices = self.get_random_component_indices(rand_nrs[0])
        new_strct = self.structure.delete_atoms(site_indices=comp_indices, change_label=False)
        component = rotate_structure(self.component, angles=rand_nrs[1] * 360.0, vector=rand_nrs[2:5], change_label=False)
        component.kinds = [comp_label] * len(component)
        pos = (np.array(self.structure.cell).T).dot(np.array(rand_nrs[5:8]))
        try:
            self.new_structure = add_structure_position(
                new_strct, guest_structure=component, position=pos, dist_threshold=1.25, change_label=False
            )
        except ValueError:
            return False
        return True



