"""Monte Carlo engines."""

# Standard library imports
from typing import Union
import copy

# Third party library imports
import numpy as np

# Internal library imports
from aim2dat.strct import Structure, StructureCollection
from aim2dat.mc.moves import InsertComponent, DeleteComponent
from aim2dat.mc.utils import calc_transition_probabilities
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
        self._component_indices = {}
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
        # TODO doc-strings
        return tuple(
            [c["n_molecules"] for c in self._components]
        )  # TODO take component indices instead?

    @property
    def component_labels(self):
        # TODO doc-strings
        return tuple(tuple(label for label in c["strct_c"].labels) for c in self._components)

    @property
    def component_indices(self):
        return copy.deepcopy(self._component_indices)

    def add_component(
        self,
        structure,
        n_molecules,
        label=None,
        ref_energy=None,
        ase_calculator=None,
        openmm_potential=None,
    ):
        # 1) check if several structures are given
        # 2) check for labels.
        # 3) Check ref energy
        # 4) Check that either ref energies or calculator or openmm potential are present.
        # Handle structures and make sure it is either a list or StructureCollection:
        if len(self._components) == 1:
            raise ValueError("Multi-component simulations are not yet implemented.")
        if isinstance(structure, str):
            # TODO handle different file support for structures.
            # TODO Handle list of structures
            structure = Structure.from_file(structure)
        structure = [structure] if isinstance(structure, Structure) else structure

        # Handle labels:
        if isinstance(label, str):
            if len(structure) > 1:
                label = [f"{label}_{i}" for i in range(len(structure))]
            else:
                label = [label]
        if label is not None and len(label) != len(structure):
            raise ValueError("`label` must have the same length/dimension as `structure`.")

        # Handle reference energies:
        if isinstance(ref_energy, (int, float)):
            ref_energy = [ref_energy] * len(structure)
        if ref_energy is not None and len(ref_energy) != len(structure):
            raise ValueError("`ref_energy` must have the same length/dimension as `structure`.")

        # Obtain label or ref_energy from structures and store all information in the
        # StructureCollection.
        strct_c = StructureCollection()
        for idx, strct in enumerate(structure):
            strct = strct.copy()
            strct_label = strct.label if label is None else label[idx]
            if strct_label is None:
                raise ValueError(f"Could not set `label` of structure {idx}.")
            if strct_label in self.component_labels:
                raise ValueError(f"Label '{strct_label}' of structure {idx} is already used.")
            ref_e = strct.get_attribute("ref_energy") if ref_energy is None else ref_energy[idx]
            if ref_e is not None:
                strct.set_attribute("ref_energy", ref_e)
            strct_c.append_structure(strct, label=strct_label)

        # add component and set calculator/potential
        comp_dict = {
            "n_molecules": n_molecules,
            "strct_c": strct_c,
        }
        if ase_calculator is not None:
            comp_dict["ase_calculator"] = ase_calculator
        elif openmm_potential is not None:
            comp_dict["openmm_potential"] = openmm_potential
        self._components.append(comp_dict)

    def _prepare_structure(self):
        # Get component indices.
        self._component_indices = [{} for _ in range(len(self._components))]
        if self.structure.kinds is None:
            self.structure.kinds = [None] * len(self.structure)
        else:
            for idx, kind in enumerate(self.structure.kinds):
                if not isinstance(kind, str):
                    continue
                kind_sp = kind.split("_")
                if len(kind_sp) < 2:
                    continue
                kind_label = "_".join(kind_sp[:-1])
                try:
                    comp_idx = int(kind_sp[-1])
                except ValueError:
                    continue
                for comp_labels, comp_dict in zip(self.component_labels, self._component_indices):
                    if kind_label in comp_labels:
                        comp_dict.setdefault((kind_label, comp_idx), []).append(idx)

        # Add/remove components.
        for idx, (comp, comp_indices) in enumerate(zip(self._components, self._component_indices)):
            if comp["n_molecules"] - len(comp_indices) > 0:
                print(
                    f"Adding {comp['n_molecules'] - len(comp_indices)} "
                    + f"molecules of component {idx}.",
                    flush=True,
                )
                for i in range(len(comp_indices), comp["n_molecules"]):
                    strct_idx = int(self.rng.random() * len(comp["strct_c"]))
                    comp_strct = comp["strct_c"][strct_idx].copy()
                    comp_strct.kinds = [f"{comp_strct.label}_{i}"] * len(comp_strct)
                    for j in range(1000):
                        try:
                            self.structure = add_structure_random(
                                self.structure,
                                guest_structure=comp_strct,
                                random_nrs=[self.rng.random() for _ in range(7)],
                                max_tries=1,
                                dist_threshold=self.dist_threshold,
                                change_label=False,
                            )
                        except DistanceThresholdError:
                            if j == 999:
                                raise ValueError(
                                    f"Could not add {comp['n_molecules']} molecules"
                                    + f" of component {idx}."
                                )
                            continue
                        break
                    comp_indices[(comp_strct.label, i)] = list(
                        range(len(self.structure) - len(comp_strct), len(self.structure))
                    )
            if comp["n_molecules"] - len(comp_indices) < 0:
                print(
                    f"Removing {comp['n_molecules'] - len(comp_indices)} molecules"
                    + f" of component {idx}.",
                    flush=True,
                )
                for i in range(comp["n_molecules"], len(comp_indices)):
                    indices = comp_indices.pop(list(comp_indices.keys())[-1])
                    self.structure = self.structure.delete_atoms(
                        site_indices=indices, change_label=False
                    )
                # TODO change this once attributes are properly implemented..
                attribs = self.structure.attributes
                if "ref_energy" in attribs:
                    del attribs["ref_energy"]
                    self.structure._attributes = attribs

    def _postprocess_step(self, step, output_dict, n_steps, n_print, n_store):
        store_interval = int(n_steps / n_store)
        print_interval = int(n_steps / n_print)
        if store_interval == 0 or (step + 1) % store_interval == 0:
            self.structures.append(self.structure)
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
            move_cls = self.moves[int(self.rng.random() * len(self.moves))]
            move = move_cls(
                structure=self.structure,
                components=self._components,
                component_indices=self._component_indices,
                dist_threshold=self.dist_threshold,
                ase_calculator=self.ase_calculator,
                openmm_potential=self.openmm_potential,
            )
            move.perform_move([self.rng.random() for _ in range(move_cls.n_rand_nrs)])
            if move.accept_move(
                self.rng.random(), self.temperature
            ):  # TODO fix optional parameters.
                self.structure = move.new_structure
                self._component_indices = move.component_indices
            self._postprocess_step(
                step,
                {"energy": self.structure.get_attribute("ref_energy")},
                n_steps,
                n_print,
                n_store,
            )


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
        energy_penalty: float = 1.0e5,
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
        self.temperature = temperature  # TODO can be moved to base class.
        self.energy_penalty = energy_penalty
        self.use_molecule_geometry = use_molecule_geometry
        self.insertion_structures = []
        self.deletion_structures = []
        self.insertion_energies = []
        self.deletion_energies = []
        self.volumes = []

        self._max_n_molecules = None
        self._deletion_energies_hist = {}

    def set_md(
        self,
        initial_steps: int,
        steps: int,
        openmm_integrator=None,
        openmmm_add_forces=None,
        openmm_reporters=None,
    ):
        """Set hybrid MD."""
        self.md_initial_steps = initial_steps
        self.md_steps = steps
        self.openmm_integrator = openmm_integrator
        self.openmmm_add_forces = [] if openmmm_add_forces is None else openmmm_add_forces
        self.openmm_reporters = [] if openmm_reporters is None else openmm_reporters

    def set_tmmc_convergence(
        self, initial_steps, interval_steps, pressures, convergence_threshold
    ):
        """Set TMMC convergence parameters."""
        self.initial_steps = initial_steps
        self.interval_steps = interval_steps
        self.pressures = pressures
        self.convergence_threshold = convergence_threshold
        self.macro_state_transitions = []

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
        if hasattr(self, "md_steps"):
            self.openmm_simulation = self.structure.to_openmm_simulation(
                potential=self.openmm_potential,
                integrator=self.openmm_integrator,
                # potential_kwargs=openmm_potential_kwargs,
                # device="cpu"
            )  # TODO expose device
            for add_force in self.openmmm_add_forces:
                self.openmm_simulation.system.addForce(add_force)
            self.openmm_simulation.context.reinitialize(preserveState=True)
            for rep in self.openmm_reporters:
                self.openmm_simulation.reporters.append(rep)

        # self._n_max_molecules = sum(self.n_molecules) if self._n_max_molecules is None
        for step in range(n_steps):
            output_dict = {}

            # Run MD simulation if set:
            if hasattr(self, "md_steps"):
                md_steps = self.md_initial_steps if step == 0 else self.md_steps
                if (
                    hasattr(self, "convergence_threshold")
                    and len(self.macro_state_transitions) > 0
                ):
                    if self.macro_state_transitions[-1] == step:
                        md_steps = self.md_initial_steps
                self.openmm_simulation.step(md_steps)
                self.structure = Structure.from_openmm_simulation(self.openmm_simulation)
                self._deletion_energies_hist = {}

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
                del_move.set_random_mol_indices(rand_nrs[0])
                if (
                    del_move.component_index,
                    del_move.component_key,
                ) in self._deletion_energies_hist:
                    e_diff = self.deletion_energies[
                        self._deletion_energies_hist[
                            (del_move.component_index, del_move.component_key)
                        ]
                    ]
                    self.deletion_structures.append(
                        self.deletion_structures[
                            self._deletion_energies_hist[
                                (del_move.component_index, del_move.component_key)
                            ]
                        ]
                    )
                    self.deletion_energies.append(e_diff)
                else:
                    del_move.perform_move(rand_nrs)
                    e_new = del_move.get_energy(del_move.new_structure)
                    e_old = del_move.get_energy(del_move.structure)
                    component = self._components[del_move.component_index]
                    mol_structure = (
                        del_move.deleted_structure
                        if self.use_molecule_geometry
                        else component["strct_c"][del_move.component_key[0]]
                    )
                    e_comp = del_move.get_energy(mol_structure, component)
                    e_diff = e_new - e_old + e_comp
                    self._deletion_energies_hist[
                        (del_move.component_index, del_move.component_key)
                    ] = len(self.deletion_energies)
                    del_move.new_structure.set_attribute("deletion_energy", e_diff)
                    del_move.new_structure.set_attribute(
                        "component_indices", del_move.component_indices
                    )
                    self.deletion_structures.append(del_move.new_structure)
                    self.deletion_energies.append(e_diff)
                output_dict["deletion energy"] = e_diff

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
                if ins_move.new_structure is not None or self.energy_penalty is not None:
                    break
            if ins_move.new_structure is None:
                if self.energy_penalty is None:
                    raise ValueError("Could not insert molecule, structure seems too aggregated.")
                else:
                    e_diff = self.energy_penalty
            else:
                e_new = ins_move.get_energy(ins_move.new_structure)
                e_old = ins_move.get_energy(ins_move.structure)
                component = self._components[ins_move.component_index]
                e_comp = ins_move.get_energy(
                    component["strct_c"][ins_move.component_key[0]],
                    self._components[ins_move.component_index],
                )
                e_diff = e_new - e_old - e_comp
                ins_move.new_structure.set_attribute("insertion_energy", e_diff)
                ins_move.new_structure.set_attribute(
                    "component_indices", ins_move.component_indices
                )
                self.insertion_structures.append(ins_move.new_structure)
            self.insertion_energies.append(e_diff)
            output_dict["insertion energy"] = e_diff

            self.volumes.append(self.structure.cell_volume)

            should_break = False
            if hasattr(self, "convergence_threshold"):
                should_break, outp_dict0 = self._check_transition_probability(step)
                output_dict.update(outp_dict0)
            self._postprocess_step(step, output_dict, n_steps, n_print, n_store)
            if should_break:
                break

    def _check_transition_probability(self, step):
        def get_error(A, B):
            return np.square(np.subtract(A, B) / B).mean()

        if step < self.initial_steps or (step - self.initial_steps + 1) % self.interval_steps != 0:
            return False, {}

        old_probs = calc_transition_probabilities(
            self.deletion_energies[: -self.interval_steps],
            self.insertion_energies[: -self.interval_steps],
            sum(self.n_molecules),
            self.volumes[: -self.interval_steps],
            self.pressures,
        )
        new_probs = calc_transition_probabilities(
            self.deletion_energies,
            self.insertion_energies,
            sum(self.n_molecules),
            self.volumes,
            self.pressures,
        )
        err_minus = 0.0 if sum(self.n_molecules) == 0 else get_error(new_probs[0], old_probs[0])
        err_plus = get_error(new_probs[1], old_probs[1])
        return err_minus < self.convergence_threshold and err_plus < self.convergence_threshold, {
            "Error n-1": err_minus,
            "Errror n+1": err_plus,
        }
