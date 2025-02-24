


import numpy as np


from aim2dat.mc.moves import InsertComponent, RemoveComponent
from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.ext_manipulation import add_structure_random


# TODO implement general logging function.
class _BaseMonteCarlo:

    def _prepare_structure(self):
        if len(self.components) > 1:
            raise ValueError("Simulations with multiple components are not yet supported.")

        comp = self.components[0]
        n_comp = self.n_components[0]
        if comp.label is None: # TODO set label internally.
            raise ValueError(f"Component label neeeds to be set.")

        # TODO expose comp_labels to be used in moves or store in structure??
        comp_labels = []
        if self.structure.kinds is None:
            self.structure.kinds = [None] * len(structure)
        else:
            for idx, kind in enumerate(self.structure.kinds):
                if kind is None:
                    continue
                if comp.label in kind and kind not in comp_labels:
                    comp_labels.append(kind)

        print(f"Adding {n_comp - len(comp_labels)} molecules of component {comp.label}.")
        for i in range(len(comp_labels), n_comp):
            comp.kinds = [f"{comp.label}_{i}"] * len(comp)
            self.structure = add_structure_random(self.structure, guest_structure=comp, dist_threshold=1.2, change_label=False)

    def _get_energy(self, structure):
        # TODO outsource into external function..
        if "ref_energy" not in structure.attributes:
            if self.ase_calculator is not None:
                backend_module = _return_ext_interface_modules("ase_calculator")
                # TODO check if ref_energy is a good key, maybe mc_energy better?
                structure.set_attribute("ref_energy", backend_module.get_potential_energy(structure, self.ase_calculator))
            else:
                raise ValueError("No viable backend found to evaluate the energy.") # TODO make error message more informative
        return structure.attributes["ref_energy"]

    def _print_stdout(self, step, energy):
        if step % 1000 == 0:
            print(f"Step: {step}, energy: {energy}")




class MonteCarlo(_BaseMonteCarlo):

    def __init__(self, structure, components, n_components, moves, temperature: float, n_steps: int, n_store: int = None, random_seed: int = None, ase_calculator = None):
        # TODO Distinguish between canonical and grand canonical MC
        # TODO Add properties to validate input parameters.
        self.structure = structure
        self.components = components
        self.n_components = n_components
        # TODO implement custom probability weights?
        self.moves = moves
        self.structures = []

        self.temperature = temperature
        self.n_steps = n_steps # TODO implement logic with steps and cycles.
        self.n_store = 10 if n_store is None else n_store

        self.random_seed = random_seed
        self.ase_calculator = ase_calculator

        self._prepare_structure()

    def run(self):
        # TODO outsource into external function..
        self._get_energy(self.structure)

        rng = np.random.default_rng(seed=self.random_seed)
        for i in range(self.n_steps):
            move_cls = self.moves[int(rng.random() * len(self.moves))] # TODO change this to implement
            rand_nrs = [rng.random() for _ in range(move_cls.n_rand_nrs + 1)]
            move = move_cls(
                structure=self.structure,
                component=self.components[0],
                ase_calculator=self.ase_calculator,
            )
            move.perform_move(rand_nrs[:-1])
            if move.accept_move(rand_nrs[-1], self.temperature): # TODO fix optional parameters.
                self.structure = move.new_structure

            # TODO move this to a logger function.
            if i % int(self.n_steps / self.n_store) == 0:
                self.structures.append(self.structure)
            self._print_stdout(i, self.structure.attributes["ref_energy"])



class TransitionMatrixMonteCarlo(_BaseMonteCarlo):
    def __init__(self, structure, components, n_components, n_steps: int, random_seed: int = None, ase_calculator = None):
        # TODO Distinguish between canonical and grand canonical MC
        # TODO Add properties to validate input parameters.
        self.structure = structure
        self.components = components
        self.n_components = n_components
        # TODO implement custom probability weights?

        self.n_steps = n_steps # TODO implement logic with steps and cycles.
        self.energy_penalty = 1.0e5
        self.insert_energies = []
        self.remove_energies = []


        self.random_seed = random_seed
        self.ase_calculator = ase_calculator

        self._prepare_structure()

    def run(self):
        rng = np.random.default_rng(seed=self.random_seed)
        # TODO calculate energy if not given.
        e_structure = self._get_energy(self.structure)
        e_component = self._get_energy(self.components[0])
        for i in range(self.n_steps):
            self.insert_energies.append(self._move_wrapper(InsertComponent, rng) - e_structure - e_component)
            e_rm = 0.0
            if self.n_components[0] > 0:
                self.remove_energies.append(self._move_wrapper(RemoveComponent, rng) - e_structure + e_component)
                e_rm = self.remove_energies[-1]
            self._print_stdout(i, f"{e_rm}/{self.insert_energies[-1]}")

    def _move_wrapper(self, move_cls, rng):
        rand_nrs = [rng.random() for _ in range(move_cls.n_rand_nrs)]
        move = move_cls(
            structure=self.structure,
            component=self.components[0],
            ase_calculator=self.ase_calculator,
        )
        move.perform_move(rand_nrs)
        e = move.get_energy()
        if e is None:
            return self.energy_penalty
        return e














