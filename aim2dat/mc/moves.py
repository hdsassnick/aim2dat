
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
        e = self.get_energy()
        if e is None:
            return False

        e_diff = e - self.structure.get_attribute("ref_energy")
        # TODO check here if energies are available?
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

    def get_all_component_indices(self):
        indices = {}
        for idx, kind in enumerate(self.structure.kinds):
            if kind is None:
                continue
            if self.component.label in kind:
                indices.setdefault(kind, []).append(idx)
        return indices


    def get_random_component_indices(self, rand_nr):
        indices = self.get_all_component_indices()
        kind = list(indices.keys())[int(rand_nr * len(indices))]
        return kind, indices[kind]


def _insert_component(structure, component, component_label, rand_nrs):
    # TODO use add_structure_random and adjust interface to do so.
    component = rotate_structure(component, angles=rand_nrs[0] * 360.0, vector=rand_nrs[1:4], change_label=False)
    component.kinds = [component_label] * len(component)
    pos = (np.array(structure.cell).T).dot(np.array(rand_nrs[4:7]))
    try:
        new_structure = add_structure_position(
            structure, guest_structure=component, position=pos, dist_threshold=1.25, change_label=False
        )
    except ValueError: # TODO be more specific on error message.
        return None
    return new_structure


class RotateComponent(BaseMove):
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        _, site_indices = self.get_random_component_indices(rand_nrs[0])
        try:
            self.new_structure = rotate_structure(
                self.structure, angles=rand_nrs[1] * 360.0, vector=rand_nrs[2:5],
                site_indices=site_indices, change_label=False, dist_threshold=1.25
            )
        except ValueError: # TODO be more specific on error message.
            pass


class TranslateComponent(BaseMove):
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        _, site_indices = self.get_random_component_indices(rand_nrs[0])
        #comp_indices = self.get_component_indices()
        #kind = list(comp_indices.keys())[int(rand_nrs[0] * len(comp_indices))]
        v = np.array(rand_nrs[1:4]) - 0.5
        v *= rand_nrs[4] * np.linalg.norm(v)
        try:
            self.new_structure = translate_structure(self.structure, site_indices=site_indices, vector=v, dist_threshold=1.25, change_label=False)
        except ValueError: #TODO be more specific on error message or add new error for AtomsTooClose.
            pass


class RemoveComponent(BaseMove):
    n_rand_nrs = 1

    def perform_move(self, rand_nrs):
        _, site_indices = self.get_random_component_indices(rand_nrs[0])
        self.new_structure = self.structure.delete_atoms(site_indices=site_indices, change_label=False)


class InsertComponent(BaseMove):
    n_rand_nrs = 7

    def perform_move(self, rand_nrs):
        comp_indices = self.get_all_component_indices()
        ind_numbers = [int(v.split("_")[-1]) for v in comp_indices]
        max_idx = max(ind_numbers) if len(ind_numbers) > 0 else 0
        new_kind = f"{self.component.label}_{max_idx + 1}"
        self.new_structure = _insert_component(self.structure, self.component, new_kind, rand_nrs)


class ReinsertComponent(BaseMove):
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        label, site_indices = self.get_random_component_indices(rand_nrs[0])
        new_strct = self.structure.delete_atoms(site_indices=site_indices, change_label=False)
        self.new_structure = _insert_component(new_strct, self.component, label, rand_nrs[1:])



