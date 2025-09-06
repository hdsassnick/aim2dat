"""Monte Carlo engines."""

# Standard library imports
from typing import Union
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct import Structure
from aim2dat.mc.moves import InsertComponent, DeleteComponent
from aim2dat.strct.ext_manipulation import add_structure_random, DistanceThresholdError
from aim2dat.utils.print import _print_dict


class _BaseMonteCarlo:

    def __init__(
        self, structure, components, dist_threshold, ase_calculator, openmm_potential, random_seed
    ):
        self.structure = structure
        self.dist_threshold = dist_threshold
        self.ase_calculator = ase_calculator
        self.openmm_potential = openmm_potential
        self.rng = np.random.default_rng(random_seed)

        self.structures = []

        self._components = []
        self._component_indices = []
        if components is not None:
            components = [components] if isinstance(components, dict) else components
            for comp in components:
                self.add_component(**comp)

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        if isinstance(value, str):
            self._structure = Structure.from_file(value)
        elif isinstance(value, Structure):
            self._structure = value
        else:
            raise TypeError(
                "`structure` needs to of type `Structure` or represent a path to a structure file."
            )

    @property
    def n_molecules(self):
        # TODO doc strings
        return tuple([c["n_molecules"] for c in self._components])

    @property
    def component_labels(self):
        # TODO doc-strings
        return tuple(c["structure"].label for c in self._components)

    @property
    def component_indices(self):
        return copy.deepcopy(self._component_indices)

    def add_component(
        self,
        structure,
        n_molecules,
        label=None,
        indices=None,
        ref_energy=None,
    ):
        # 1) check if several structures are given
        # 2) check for labels.
        # 3) Check for indices.
        # 4) Check ref energy.
        if len(self._components) == 1:
            raise ValueError("Multi-component simulations are not yet implemented.")
        if isinstance(structure, str):
            # TODO handle different file support for structures.
            # TODO Handle list of structures
            structure = Structure.from_file(structure)
        if label is not None:
            structure.label = label

        # Handle reference energies:
        if ref_energy is not None:
            structure.attributes["ref_energy"] = ref_energy

        comp_dict = {
            "n_molecules": n_molecules,
            "structure": structure,
        }

        # Handle indices:
        if indices is not None:
            if isinstance(indices[0], int):
                indices = [indices]
            comp_dict["indices"] = indices
            if any(len(ind) != len(structure) for ind in indices):
                raise ValueError(
                    "Each element of `indices` need to have the same length as `structure`."
                )

        # add component
        self._components.append(comp_dict)

    def _prepare_structure(self):

        # Add/remove components.
        for idx, comp in enumerate(self._components):
            comp_indices = comp.pop("indices", [])
            if len(self._component_indices) > idx:
                comp_indices += self._component_indices[idx]
            if comp["n_molecules"] - len(comp_indices) > 0:
                print(
                    f"Adding {comp['n_molecules'] - len(comp_indices)} "
                    + f"molecules of component {comp['structure'].label}.",
                    flush=True,
                )
                for i in range(len(comp_indices), comp["n_molecules"]):
                    for j in range(1000):
                        try:
                            self.structure = add_structure_random(
                                self.structure,
                                guest_structure=comp["structure"].copy(),
                                random_nrs=[self.rng.random() for _ in range(7)],
                                max_tries=1,
                                dist_threshold=self.dist_threshold,
                                change_label=False,
                            )
                        except DistanceThresholdError:
                            if j == 999:
                                raise ValueError(
                                    f"Could not add {comp['n_molecules']} molecules"
                                    + f" of component {comp['structure'].label}."
                                )
                            continue
                        break
                    comp_indices.append(
                        list(
                            range(
                                len(self.structure) - len(comp["structure"]), len(self.structure)
                            )
                        )
                    )
            if comp["n_molecules"] - len(comp_indices) < 0:
                print(
                    f"Removing {comp['n_molecules'] - len(comp_indices)} molecules"
                    + f" of component {comp['structure'].label}.",
                    flush=True,
                )
                for i in range(len(comp_indices) - 1, comp["n_molecules"] - 1, -1):
                    # TODO figure out if indices are not sorted.
                    indices = comp_indices.pop(i)
                    self.structure = self.structure.delete_atoms(
                        site_indices=indices, change_label=False
                    )
                self.structure.attributes.pop("ref_energy", None)
            if idx < len(self._component_indices):
                self._component_indices[idx] = comp_indices
            else:
                self._component_indices.append(comp_indices)

    def _postprocess_step(self, step, output_dict, n_steps, n_print, n_store, structure=None):
        store_interval = int(n_steps / n_store)
        print_interval = int(n_steps / n_print)
        if store_interval == 0 or (step + 1) % store_interval == 0:
            structure = self.structure if structure is None else structure
            self.structures.append(structure)
        if print_interval == 0 or (step + 1) % print_interval == 0:
            step_str = " ".join([""] * (8 - len(str(step + 1)))) + str(step + 1)
            print(
                _print_dict(f"Step {step_str} |", output_dict, inline=True, float_precision=5),
                flush=True,
            )


class MonteCarlo(_BaseMonteCarlo):
    """Conventional Monte Carlo simulation engine with a fixed number of molecules."""

    def __init__(
        self,
        structure: Structure,
        moves,  # TODO typehint
        temperature=298.15,
        pressure=1.0,
        fugacity_coeff=1.0,
        components: Union[list, dict] = None,
        dist_threshold: Union[dict, list, float, int, str, None] = 0.8,
        ase_calculator=None,  # TODO typehint
        openmm_potential=None,  # TODO typehint
        random_seed: int = None,
    ):
        """Initialize class."""
        _BaseMonteCarlo.__init__(
            self,
            structure=structure,
            components=components,
            dist_threshold=dist_threshold,
            ase_calculator=ase_calculator,
            openmm_potential=openmm_potential,
            random_seed=random_seed,
        )
        self.moves = moves
        self.temperature = temperature
        self.pressure = pressure
        self.fugacity_coeff = fugacity_coeff

        self.move_statistics = [[0, 0] for _ in range(len(moves))]
        self.uptake = []

    def run(self, n_steps: int, n_print: int = 10, n_store: int = 10):
        """
        Run simulation.

        Parameters
        ----------
        n_steps : int
            Number of steps.
        n_print : int
            Number of print statements.
        n_store : int
            Number of stored structures.
        """
        self._prepare_structure()
        step = 0
        while step < n_steps:
            move_idx = int(self.rng.random() * len(self.moves))
            move = self.moves[move_idx](
                structure=self.structure,
                components=self._components,
                component_indices=self._component_indices,
                dist_threshold=self.dist_threshold,
                ase_calculator=self.ase_calculator,
                openmm_potential=self.openmm_potential,
            )
            move.perform_move([self.rng.random() for _ in range(self.moves[move_idx].n_rand_nrs)])
            if move.new_structure is None:
                continue

            accepted = False
            if move.accept_move(
                self.rng.random(), self.temperature, self.pressure, self.fugacity_coeff
            ):
                self.structure = move.new_structure
                self._component_indices = move.component_indices
                self.move_statistics[move_idx][0] += 1
                accepted = True

            self.move_statistics[move_idx][1] += 1
            self.uptake.append(len(self._component_indices[0]))
            self._postprocess_step(
                step,
                {
                    "Energy": self.structure.attributes.get("ref_energy", 0.0),
                    "Delta E": move.energy_difference,
                    "n_comp": len(self._component_indices[0]),
                    "Move": move.name,
                    "Accepted": accepted,
                },
                n_steps,
                n_print,
                n_store,
            )
            step += 1


class TransitionMatrixMonteCarlo(_BaseMonteCarlo):
    """Transition Matrix Monte Carlo simulation engine."""

    def __init__(
        self,
        structure,
        components: Union[list, dict] = None,
        temperature: float = 298.15,
        dist_threshold: Union[dict, list, float, int, str, None] = 0.8,
        ase_calculator=None,
        openmm_potential=None,
        random_seed: int = None,
        use_molecule_geometry: bool = False,
    ):
        """Initialize class."""
        _BaseMonteCarlo.__init__(
            self,
            structure=structure,
            components=components,
            dist_threshold=dist_threshold,
            ase_calculator=ase_calculator,
            openmm_potential=openmm_potential,
            random_seed=random_seed,
        )
        self.use_molecule_geometry = use_molecule_geometry
        self.structures = []  # TODO move to base class.
        self.insertion_energies = []
        self.deletion_energies = []
        self.volumes = []

        self._max_n_molecules = None
        self._deletion_energies_hist = {}

    def run(self, n_steps: int, n_print: int = 10, n_store: int = 10):
        """
        Run simulation.

        Parameters
        ----------
        n_steps : int
            Number of steps.
        n_print : int
            Number of print statements.
        n_store : int
            Number of stored structures.
        """
        self._prepare_structure()

        for step in range(n_steps):
            output_dict = {}

            # Calculate deletion energy:
            if sum(self.n_molecules) > 0:
                del_move = DeleteComponent(
                    structure=self.structure,
                    components=self._components,
                    component_indices=self._component_indices,
                    dist_threshold=self.dist_threshold,
                    ase_calculator=self.ase_calculator,
                    openmm_potential=self.openmm_potential,
                )
                rand_nrs = [self.rng.random() for _ in range(DeleteComponent.n_rand_nrs)]
                del_move.set_random_mol_index(rand_nrs[0])
                del_indices = del_move.component_indices[del_move.component_index[0]][
                    del_move.component_index[1]
                ]
                if del_move.component_index in self._deletion_energies_hist:
                    prev_move_idx = self._deletion_energies_hist[del_move.component_index]
                    e_del = self.deletion_energies[prev_move_idx]
                else:
                    del_move.perform_move(rand_nrs)
                    e_new = del_move.get_energy(del_move.new_structure)
                    e_old = del_move.get_energy(del_move.structure)
                    component = self._components[del_move.component_index[0]]
                    mol_structure = (
                        del_move.deleted_structure
                        if self.use_molecule_geometry
                        else component["structure"]
                    )
                    e_comp = del_move.get_energy(mol_structure, component)
                    e_del = e_new - e_old + e_comp
                    self._deletion_energies_hist[del_move.component_index] = len(
                        self.deletion_energies
                    )
                self.deletion_energies.append(e_del)
                output_dict["deletion energy"] = e_del

            # Calculate insertion energy:
            for i in range(1000):
                ins_move = InsertComponent(
                    structure=self.structure,
                    components=self._components,
                    component_indices=self._component_indices,
                    dist_threshold=self.dist_threshold,
                    ase_calculator=self.ase_calculator,
                    openmm_potential=self.openmm_potential,
                )
                rand_nrs = [self.rng.random() for _ in range(InsertComponent.n_rand_nrs)]
                ins_move.perform_move(rand_nrs)
                if ins_move.new_structure is not None:
                    break
            if ins_move.new_structure is None:
                raise ValueError("Could not insert molecule, structure seems too aggregated.")
            ins_indices = ins_move.component_indices[ins_move.component_index[0]][-1]
            e_new = ins_move.get_energy(ins_move.new_structure)
            e_old = ins_move.get_energy(ins_move.structure)
            component = self._components[ins_move.component_index[0]]
            e_comp = ins_move.get_energy(component["structure"], component)
            e_ins = e_new - e_old - e_comp

            # Prepare structure to be stored.
            strct2store = ins_move.new_structure
            strct2store.attributes["insertion_energy"] = e_ins
            strct2store.attributes["insertion_indices"] = ins_indices
            if sum(self.n_molecules) > 0:
                strct2store.attributes["deletion_energy"] = e_del
                strct2store.attributes["deletion_indices"] = del_indices
            self.insertion_energies.append(e_ins)
            output_dict["insertion energy"] = e_ins
            self.volumes.append(self.structure.cell_volume)
            self._postprocess_step(
                step, output_dict, n_steps, n_print, n_store, structure=strct2store
            )
