


import numpy as np

from aim2dat.strct import Structure
from aim2dat.mc.moves import InsertComponent, RemoveComponent
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.ext_manipulation import add_structure_random, DistanceThresholdError


# TODO implement general logging function.
# TODO implement restart file (binary?)
class _BaseMonteCarlo:

    def __init__(self, structure, components, n_components, dist_threshold, n_steps, n_print, ase_calculator, random_seed):
        # TODO add type checks for structure/components and input validation.
        self.structure = structure.copy()

        #self.components = [comp.copy() for comp in components]
        #self.n_components = n_components

        self.dist_threshold = dist_threshold

        self.n_steps = n_steps # TODO implement logic with steps and cycles.
        self.n_print = n_print

        self.ase_calculator = ase_calculator

        self.rng = np.random.default_rng(random_seed)

        self._set_components(components, n_components)
        self._prepare_structure()

    def _set_components(self, components, n_components):
        comp_labels = []
        # TODO implement components as StructureCollections?
        # TODO how to handle different cases???
        if isinstance(components, Structure):
            comp_dict = {comp.get("label", "comp"): [components.copy()]}
        elif isinstance(components, dict):
            comp_dict = components
        else:
            comp_dict = {}
            for i, comp in enumerate(components):
                if not isinstance(comp, Structure):
                    raise ValueError("`components` need to contain `Structure` objects.")

                comp_strct = comp.copy()
                if comp_strct.label is None:
                    dict_key = "comp"
                else:
                    dict_key = "_".join(comp_strct.label.split("_")[:-1])
                comp_dict.setdefault(dict_key, []).append(comp_strct)
        self.components = comp_dict
        self.n_components = n_components
        # TODO check n_components??



    def _prepare_structure(self):
        # TODO handle multiple components.
        if len(self.components) > 1:
            raise ValueError("Simulations with multiple components are not yet supported.")

        comp_label, comp = list(self.components.items())[0]
        n_comp = self.n_components[0]
        #if comp.label is None:
        #    comp.label = "comp1"

        # Get component indices.
        # TODO make component_indices keys here tuples
        self._component_indices = {}
        if self.structure.kinds is None:
            self.structure.kinds = [None] * len(structure)
        else:
            for idx, kind in enumerate(self.structure.kinds):
                if kind is None:
                    continue
                # TODO check structure label as well.
                if comp_label in kind:
                    comp_key = tuple(int(v) if i > 0 else v for i, v in enumerate(kind.split("_")))
                    self._component_indices.setdefault(comp_key, []).append(idx)

        # Add/remove components.
        if n_comp - len(self._component_indices) > 0:
            print(f"Adding {n_comp - len(self._component_indices)} molecules of component {comp_label}.", flush=True)
            for i in range(len(self._component_indices), n_comp):
                strct_idx = int(self.rng.random() * len(comp))
                label = (comp_label, strct_idx, i)#f"{comp_label}_{strct_idx}_{i}"
                comp_strct = comp[strct_idx].copy()
                comp_strct.kinds = ["_".join(str(v) for v in label)] * len(comp_strct)
                for j in range(1000): # TODO Expose this value?
                    try:
                        self.structure = add_structure_random(
                            self.structure, guest_structure=comp_strct, random_nrs=[self.rng.random() for _ in range(7)], max_tries=1,
                            dist_threshold=self.dist_threshold, change_label=False
                        )
                    except DistanceThresholdError:
                        if j == 999:
                            raise ValueError(f"Could not add {n_comp} molecules of component {comp_label}.")
                        continue
                    break
                self._component_indices[label] = list(range(len(self.structure) - len(comp_strct), len(self.structure)))

        elif n_comp - len(self._component_indices) < 0:
            print(f"Removing {len(self._component_indices) - n_comp} molecules of component {comp_label}.", flush=True)
            for i in range(n_comp, len(self._component_indices)):
                indices = self._component_indices.pop(list(self._component_indices.keys())[len(self._component_indices) - i - 1])
                self.structure = self.structure.delete_atoms(site_indices=indices, change_label=False)


    def _get_energy(self, structure):
        if "ref_energy" not in structure.attributes:
            if self.ase_calculator is not None:
                backend_module = _return_ext_interface_modules("ase_calculator")
                # TODO check if ref_energy is a good key, maybe mc_energy better?
                structure.set_attribute("ref_energy", backend_module.get_potential_energy(structure, self.ase_calculator))
            else:
                raise ValueError("No viable backend found to evaluate the energy.") # TODO make error message more informative
        return structure.attributes["ref_energy"]

    def _print_stdout(self, step, energy):
        if (step + 1) % int(self.n_steps / self.n_print) == 0:
            # TODO rounding of numbers...
            print(f"Step: {step + 1}, energy: {energy}", flush=True)




class MonteCarlo(_BaseMonteCarlo):

    def __init__(
        self, structure, components, n_components, moves, temperature: float,
        n_steps: int, n_print: int = 10, n_store: int = 10,
        dist_threshold: float = 1.25, ase_calculator = None, random_seed: int = None
    ):
        _BaseMonteCarlo.__init__(
            self,
            structure=structure, components=components, n_components=n_components,
            dist_threshold=dist_threshold, n_steps=n_steps, n_print=n_print,
            ase_calculator=ase_calculator, random_seed=random_seed
        )
        # TODO Distinguish between canonical and grand canonical MC
        # TODO implement custom probability weights?
        self.moves = moves
        self.structures = []

        self.temperature = temperature
        self.n_store = n_store

    def run(self):
        self._get_energy(self.structure)

        for i in range(self.n_steps):
            move_cls = self.moves[int(self.rng.random() * len(self.moves))]
            comp_key = list(self._component_indices.keys())[int(self.rng.random() * len(self._component_indices))]
            move = move_cls(
                structure=self.structure,
                components=self.components,
                component_key=comp_key,
                component_indices=self._component_indices.copy(), # TODO use deepcopy?
                dist_threshold=self.dist_threshold,
                ase_calculator=self.ase_calculator,
            )
            move.perform_move([self.rng.random() for _ in range(move_cls.n_rand_nrs)])
            if move.accept_move(self.rng.random(), self.temperature): # TODO fix optional parameters.
                self.structure = move.new_structure
                self._component_indices = move.component_indices
            # TODO move this to a logger function.
            if i % int(self.n_steps / self.n_store) == 0:
                self.structures.append(self.structure)
            self._print_stdout(i, self.structure.attributes["ref_energy"])



class TransitionMatrixMonteCarlo(_BaseMonteCarlo):
    def __init__(self, structure, components, n_components, n_steps: int, n_print: int = 10, dist_threshold: float = 1.25, ase_calculator = None, random_seed: int = None):
        _BaseMonteCarlo.__init__(
            self,
            structure=structure, components=components, n_components=n_components,
            dist_threshold=dist_threshold, n_steps=n_steps, n_print=n_print,
            ase_calculator=ase_calculator, random_seed=random_seed
        )
        self.energy_penalty = 1.0e5 # TODO remove and add acceptance criteria instead.
        self.insert_energies = []
        self.remove_energies = []

        self._remove_energies_hist = {}

    def run(self):
        e_structure = self._get_energy(self.structure)
        for i in range(self.n_steps):
            comp_key = ""
            e_new, e_comp = self._move_wrapper(InsertComponent, (list(self.components.keys())[0], None, self.n_components[0])) # TODO Generalize to multi-component case...
            self.insert_energies.append(e_new - e_structure - e_comp)
            if self.n_components[0] > 0:
                comp_key = list(self._component_indices.keys())[int(self.rng.random() * len(self._component_indices))]
                if comp_key in self._remove_energies_hist:
                    self.remove_energies.append(self._remove_energies_hist[comp_key])
                else:
                    e_new, e_comp = self._move_wrapper(RemoveComponent, comp_key)
                    self.remove_energies.append(e_new - e_structure + e_comp)
                    self._remove_energies_hist[comp_key] = self.remove_energies[-1]
            self._print_stdout(i, f"{self._remove_energies_hist.get(comp_key, 0.0)}/{self.insert_energies[-1]}")

    def _move_wrapper(self, move_cls, comp_key):
        move = move_cls(
            structure=self.structure,
                components=self.components,
                component_key=comp_key,
                component_indices=self._component_indices.copy(), # TODO use deepcopy?
                dist_threshold=self.dist_threshold,
                ase_calculator=self.ase_calculator,
        )
        move.perform_move([self.rng.random() for _ in range(move_cls.n_rand_nrs)])
        e_new = move.get_energy()
        if e_new is None:
            return self.energy_penalty, 0.0
        elif isinstance(move, InsertComponent):
            return e_new, self._get_energy(self.components[move.new_component_key[0]][move.new_component_key[1]])
        else:
            return e_new, self._get_energy(self.components[comp_key[0]][comp_key[1]])














