

# TODO move this to ext_interfaces.
#from mpi4py import MPI
import numpy as np


from aim2dat.ext_interfaces import _return_ext_interface_modules
from aim2dat.strct.ext_manipulation import add_structure_random


class MonteCarloSimulation:

    def __init__(self, structure, components, n_components, moves, temperature: float, n_steps: int, random_seed: int = None, ase_calculator = None):
        # TODO Distinguish between canonical and grand canonical MC
        # TODO Add properties to validate input parameters.
        self.structure = structure
        self.components = components
        self.n_components = n_components
        # TODO implement custom probability weights?
        self.moves = moves

        self.temperature = temperature
        self.pressures = None # TODO implement this?
        self.n_steps = n_steps # TODO implement logic with steps and cycles.
        self.eos = None

        self.random_seed = random_seed
        self.ase_calculator = ase_calculator


        self._prepare_structures()

    def run(self):
        # TODO outsource into external function..
        if self.ase_calculator is not None:
            backend_module = _return_ext_interface_modules("ase_calculator")
            self.structure.set_attribute("ref_energy", backend_module.get_potential_energy(self.structure, self.ase_calculator))
        else:
            raise ValueError("No viable backend found to evaluate the energy.") # TODO make error message more informative

        rng = np.random.default_rng(seed=self.random_seed)

        for i in range(self.n_steps):
            print("step:", i)
            move_cls = self.moves[int(rng.random() * len(self.moves))] # TODO change this to implement
            print(move_cls)
            rand_nrs = [rng.random() for _ in range(move_cls.n_rand_nrs + 1)]
            move = move_cls(
                structure=self.structure,
                component=self.components[0],
                ase_calculator=self.ase_calculator,
            )
            valid_move = move.perform_move(rand_nrs[:-1])
            if not valid_move:
                continue

            if move.accept_move(rand_nrs[-1], self.temperature): # TODO fix optional parameters.
                self.structure = move.new_structure

    def _prepare_structures(self):
        if len(self.components) > 1:
            raise ValueError("Multipe compontents simulations are not yet supported")

        if self.components[0].label is None:
            raise ValueError(f"Component label neeeds to be set.")

        # Add n component:
        component = self.components[0]
        component_label = self.components[0].label
        n_component = self.n_components[0]
        comp_labels = []
        if self.structure.kinds is None:
            self.structure.kinds = [None] * len(self.structure)
        else:
            for idx, kind in enumerate(self.structure.kinds):
                if kind is None:
                    continue
                if component_label in kind and kind not in comp_labels:
                    comp_labels.append(kind)

        for i in range(len(comp_labels), n_component):
            component.kinds = [f"{component_label}_{i}"] * len(component)
            self.structure = add_structure_random(self.structure, guest_structure=component, dist_threshold=1.2, change_label=False)












