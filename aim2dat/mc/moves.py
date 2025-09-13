"""Monte Carlo move classes."""

# Standard library imports
import abc
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct import Structure, SamePositionsError
from aim2dat.strct.ext_manipulation import (
    add_structure_random,
    add_structure_coord,
    translate_structure,
    rotate_structure,
    DistanceThresholdError,
)
import aim2dat.utils.units as a2d_units
from aim2dat.utils.element_properties import get_atomic_radius


class BaseMove(abc.ABC):
    """Base class for Monte Carlo moves."""

    name = "MC Move"
    n_rand_nrs = 1
    n_change = 0

    def __init__(
        self,
        structure,
        components,
        component_indices,
        dist_threshold,
        ase_calculator,
        openmm_potential,
    ):
        """Initialize class."""
        self.structure = structure
        self.components = components
        self.component_indices = copy.deepcopy(component_indices)
        self.dist_threshold = dist_threshold
        self.ase_calculator = ase_calculator
        self.openmm_potential = openmm_potential

        self.new_structure = None

    @abc.abstractmethod
    def perform_move(self, rand_nrs):
        """Perform move."""
        pass

    def accept_move(self, rand_nr, temperature, pressure=1.0, fugacity_coeff=1.0):
        """
        Evaluate accpetance criterium.

        Parameters
        ----------
        rand_nr : float
            Random number.
        temperature : float
            Temperature in kelvin.

        Returns
        -------
        bool
            Whether to accept the move.
        """
        # TODO add presssure/fugacity
        if self.new_structure is None:
            self.energy_difference = 1.0  # TODO find better value?
            self.acceptance = 0.0
            return False

        e_diff = self.get_energy(self.new_structure) - self.get_energy(self.structure)
        if self.n_change == 0 and e_diff < 0:
            self.energy_difference = e_diff
            self.acceptance = 1.0
            return True

        beta = 1.0 / (temperature * a2d_units.constants.kb * a2d_units.energy.j)
        prefactor = 1.0
        n_molecules = len(self.component_indices[self.component_index[0]])
        if self.n_change == -1:
            e_comp = self.get_energy(
                self.components[self.component_index[0]]["structure"],
                self.components[self.component_index[0]],
            )
            e_diff += e_comp
            prefactor = (n_molecules + 1) / (
                self.structure.cell_volume
                * beta
                * fugacity_coeff
                * pressure
                * a2d_units.pressure.pascal
            )
        elif self.n_change == 1:
            e_comp = self.get_energy(
                self.components[self.component_index[0]]["structure"],
                self.components[self.component_index[0]],
            )
            e_diff -= e_comp
            prefactor = (
                self.structure.cell_volume
                * beta
                * fugacity_coeff
                * pressure
                * a2d_units.pressure.pascal
            ) / n_molecules
        self.energy_difference = e_diff
        self.acceptance = min(1.0, prefactor * np.exp(-beta * e_diff))
        return rand_nr < self.acceptance

    def get_energy(self, structure, component=None):
        """
        Get energy.

        Parameters
        ----------
        structure : aim2dat.strct.Structure
            Structure.
        component : dict
            Component dictionary containing the ase calculator or openmm potential.

        Returns
        -------
        float
            Total energy.

        Raises
        ------
        ValueError
            If no backend is found.
        """
        energy = structure.attributes.get("ref_energy", None)
        if energy is not None:
            return energy

        component = {} if component is None else component
        ase_calculator = component.get("ase_calculator", self.ase_calculator)
        # openmm_potential = component.get("openmm_potential", self.openmm_potential)
        # TODO implement openmm backends
        if ase_calculator is not None:
            backend_module = _return_ext_interface_modules("ase_calculator")
            energy = backend_module.get_potential_energy(structure, ase_calculator)
            structure.attributes["ref_energy"] = energy
            return energy
        else:
            raise ValueError("No viable backend found to evaluate the energy.")

    def set_random_mol_index(self, rand_nr):
        """
        Select a random molecule and set ``component_index``.

        Parameters
        ----------
        rand_nr : float
            Random number.
        """
        idx = int(rand_nr * sum([len(comp) for comp in self.component_indices]))
        n_prev_comps = 0
        for comp_idx, comp in enumerate(self.component_indices):
            if n_prev_comps + len(comp) > idx:
                break
            n_prev_comps += len(comp)
        idx -= n_prev_comps
        self.component_index = (comp_idx, idx)

    def _insert_component(self, structure, rand_nrs):
        new_mol = self.components[self.component_index[0]]["structure"]
        try:
            new_structure = add_structure_random(
                structure,
                guest_structure=new_mol,
                random_nrs=rand_nrs,
                max_tries=1,
                dist_threshold=self.dist_threshold,
                change_label=False,
            )
            new_structure.attributes["ref_energy"] = None
        except (DistanceThresholdError, SamePositionsError):
            return None

        indices = list(range(len(structure), len(structure) + len(new_mol)))
        self.component_indices[self.component_index[0]].append(indices)
        return new_structure

    def _insert_component_coord(self, structure, rand_nrs):
        new_mol = self.components[self.component_index[0]]["structure"]
        host_index = int(rand_nrs[0] * len(self.structure))
        guest_index = int(rand_nrs[1] * len(new_mol))
        bond_length = (1.0 + 2.0 * rand_nrs[2]) * (
            get_atomic_radius(self.structure.elements[host_index], radius_type="chen_manz")
            + get_atomic_radius(new_mol.elements[guest_index], radius_type="chen_manz")
        )
        indices = list(range(len(self.structure), len(self.structure) + len(new_mol)))
        try:
            new_structure = add_structure_coord(
                self.structure,
                host_indices=host_index,
                guest_indices=guest_index,
                guest_structure=new_mol,
                # rotate_guest=True,
                bond_length=bond_length,
                method="atomic_radius",
                radius_type="chen_manz",
                atomic_radius_delta=0.1,
                dist_threshold=self.dist_threshold,
                change_label=False,
            )
            new_structure = rotate_structure(
                new_structure,
                angles=rand_nrs[3] * 5.0,
                vector=rand_nrs[4:7],
                site_indices=indices,
                dist_threshold=self.dist_threshold,
                change_label=False,
            )
            new_structure.attributes["ref_energy"] = None
        except (DistanceThresholdError, SamePositionsError):
            return None

        self.component_indices[self.component_index[0]].append(indices)
        return new_structure

    def _delete_component(self, structure, rand_nr):
        self.set_random_mol_index(rand_nr)
        if len(self.component_indices[self.component_index[0]]) > 0:
            site_indices = self.component_indices[self.component_index[0]].pop(
                self.component_index[1]
            )
            # Shift indices to match the new structure:
            for comp_idx0, comp in enumerate(self.component_indices):
                for idx, val in enumerate(comp):
                    self.component_indices[comp_idx0][idx] = [
                        v - sum(1 if v > i else 0 for i in site_indices) for v in val
                    ]
            new_structure = structure.delete_atoms(site_indices=site_indices, change_label=False)
            new_structure.attributes["ref_energy"] = None
            self.deleted_structure = Structure(
                elements=[structure.elements[idx] for idx in site_indices],
                positions=[structure.positions[idx] for idx in site_indices],
                is_cartesian=True,
                pbc=False,
            )
            return new_structure


class RotateComponent(BaseMove):
    """Move class to rotate component."""

    name = "Rot."
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.set_random_mol_index(rand_nrs[0])
        if len(self.component_indices[self.component_index[0]]) > 0:
            site_indices = self.component_indices[self.component_index[0]][self.component_index[1]]
            try:
                self.new_structure = rotate_structure(
                    self.structure,
                    angles=rand_nrs[1] * 360.0,
                    vector=rand_nrs[2:5],
                    site_indices=site_indices,
                    dist_threshold=self.dist_threshold,
                    change_label=False,
                )
                self.new_structure.attributes["ref_energy"] = None
            except (DistanceThresholdError, SamePositionsError):
                pass


class TranslateComponent(BaseMove):
    """Move class to translate component."""

    name = "Tra."
    n_rand_nrs = 5

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.set_random_mol_index(rand_nrs[0])
        if len(self.component_indices[self.component_index[0]]) > 0:
            site_indices = self.component_indices[self.component_index[0]][self.component_index[1]]
            v = np.array(rand_nrs[1:4]) - 0.5
            v *= 0.5 * rand_nrs[4] / np.linalg.norm(v)
            try:
                self.new_structure = translate_structure(
                    self.structure,
                    site_indices=site_indices,
                    vector=v,
                    dist_threshold=self.dist_threshold,
                    change_label=False,
                )
                self.new_structure.attributes["ref_energy"] = None
            except (DistanceThresholdError, SamePositionsError):
                pass


class DeleteComponent(BaseMove):
    """Move class to delete component."""

    name = "Del."
    n_change = -1

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.new_structure = self._delete_component(self.structure, rand_nrs[0])


class InsertComponent(BaseMove):
    """Move class to insert component."""

    name = "Ins."
    n_rand_nrs = 8
    n_change = 1

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.component_index = (int(rand_nrs[0] * len(self.component_indices)), None)
        self.new_structure = self._insert_component(self.structure, rand_nrs[1:])


class InsertComponentCoord(BaseMove):
    """Move class to insert a component coordinated to another site."""

    name = "CoI."
    n_rand_nrs = 8
    n_change = 1

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.component_index = (int(rand_nrs[0] * len(self.component_indices)), None)
        self.new_structure = self._insert_component_coord(self.structure, rand_nrs[1:])


class ReinsertComponent(BaseMove):
    """Move class to reinsert component."""

    name = "Rei."
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.new_structure = self._insert_component(
            self._delete_component(self.structure, rand_nrs[0]), rand_nrs[1:]
        )


class ReinsertComponentCoord(BaseMove):
    """Move class to re-insert a component coordinated to another site."""

    name = "CoR."
    n_rand_nrs = 8

    def perform_move(self, rand_nrs):
        """
        Perform move.

        Parameters
        ----------
        rand_nrs : list
            List of random numbers.
        """
        self.new_structure = self._insert_component_coord(
            self._delete_component(self.structure, rand_nrs[0]), rand_nrs[1:]
        )
