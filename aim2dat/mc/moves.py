
import abc

import numpy as np


from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct import SamePositionsError
from aim2dat.strct.ext_manipulation import add_structure_random, translate_structure, rotate_structure, DistanceThresholdError
from aim2dat.utils.units import energy, constants


# TODO deal with the case that two atoms have the same position.

class BaseMove(abc.ABC):
    n_rand_nrs = 3

    def __init__(self, structure, components, component_key, component_indices, dist_threshold, ase_calculator):
        self.structure = structure
        self.components = components
        self.component_key = component_key
        self.component_indices = component_indices
        self.dist_threshold = dist_threshold
        self.ase_calculator = ase_calculator

        self.new_structure = None


    @abc.abstractmethod
    def perform_move(self, rand_nrs):
        pass

    def accept_move(self, rand_nr, temperature):
        # TODO add presssure/fugacity
        e = self.get_energy()
        if e is None:
            return False

        e_diff = e - self.structure.get_attribute("ref_energy")
        if e_diff < 0:
            return True
        else:
            return rand_nr < min(1.0, np.exp(-1.0 * e_diff / (temperature * constants.kb * energy.j)))

    def get_energy(self):
        if self.new_structure is None:
            return None

        # TODO implement lammps and openmm backends and make backend module
        if self.ase_calculator is not None:
            backend_module = _return_ext_interface_modules("ase_calculator")
            energy = backend_module.get_potential_energy(self.new_structure, self.ase_calculator)
            self.new_structure.set_attribute("ref_energy", energy)
            return energy
        else:
            raise ValueError("No viable backend found to evaluate the energy.")

    def _insert_component(self, structure, rand_nrs):
        new_comp_idx = int(rand_nrs[0] * len(self.components[self.component_key[0]]))
        new_key = (self.component_key[0], new_comp_idx, self.component_key[2])
        new_comp = self.components[self.component_key[0]][new_comp_idx].copy()
        new_comp.kinds = ["_".join(str(v) for v in new_key)] * len(new_comp)
        try:
            new_structure = add_structure_random(
                structure, guest_structure=new_comp, random_nrs=rand_nrs[1:], max_tries=1, dist_threshold=self.dist_threshold, change_label=False
            )
        except (DistanceThresholdError, SamePositionsError):
            return None

        #print(self.component_indices)
        #del self.component_indices[self.component_key]
        self.new_component_key = new_key
        self.component_indices[new_key] = list(range(len(structure), len(structure) + len(new_comp)))
        return new_structure

    def _remove_component(self, structure):
        indices =  self.component_indices.pop(self.component_key)
        # Shift indices to match the new structure:
        for key, val in self.component_indices.items():
            self.component_indices[key] = [v - sum(1 if v > i else 0 for i in indices) for v in val]
        return structure.delete_atoms(site_indices=indices, change_label=False)


class RotateComponent(BaseMove):
    n_rand_nrs = 4

    def perform_move(self, rand_nrs):
        #_, site_indices = self.get_random_component_indices(rand_nrs[0])
        try:
            self.new_structure = rotate_structure(
                self.structure, angles=rand_nrs[0] * 360.0, vector=rand_nrs[1:4],
                site_indices=self.component_indices[self.component_key], dist_threshold=self.dist_threshold, change_label=False
            )
        except (DistanceThresholdError, SamePositionsError):
            pass


class TranslateComponent(BaseMove):
    n_rand_nrs = 4

    def perform_move(self, rand_nrs):
        v = np.array(rand_nrs[0:3]) - 0.5
        v *= rand_nrs[1] * np.linalg.norm(v)
        try:
            self.new_structure = translate_structure(
                self.structure, site_indices=self.component_indices[self.component_key], vector=v,
                dist_threshold=self.dist_threshold, change_label=False
            )
        except (DistanceThresholdError, SamePositionsError):
            pass


class RemoveComponent(BaseMove):
    n_rand_nrs = 0

    def perform_move(self, _):
        self.new_structure = self._remove_component(self.structure)


class InsertComponent(BaseMove):
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        self.new_structure = self._insert_component(self.structure, rand_nrs)


class ReinsertComponent(BaseMove):
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        self.new_structure = self._insert_component(self._remove_component(self.structure), rand_nrs)




