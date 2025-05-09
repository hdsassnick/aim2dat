
# Third party library imports
import numpy as np
from openmm import Platform, LangevinMiddleIntegrator
from openmm.app import Topology, Simulation, Element
import openmm.unit as unit

# Internal library imports
import aim2dat.utils.units as a2d_units


def get_potential_energy(structure, potential, energy_type="energy", dtype="float64", device="cpu", get_forces=False): # TODO add kwarg for different energy units?
    simulation = create_openmm_simulation(structure, potential, energy_type=energy_type, dtype=dtype, device=device)
    state = simulation.context.getState(getEnergy=True, getForces=get_forces)
    energy = float(state.getPotentialEnergy().value_in_unit(unit.kilojoule/unit.mole)) * 1000.0 * a2d_units.energy.joule / a2d_units.constants.na
    if get_forces:
        forces  = [[float(v) * 100.0 * a2d_units.energy.joule / a2d_units.constants.na for v in val] for val in state.getForces().value_in_unit(unit.kilojoule/unit.mole/unit.nanometer)]
        return energy, forces
    else:
        return energy


def create_openmm_simulation(structure, potential, temperature=300.0, friction_coeff=1000.0, step_size=0.5, energy_type="energy", dtype="float64", device="cpu"):
    topology = create_openmm_topology(structure)
    system = potential.createSystem(topology, dtype=dtype, device=device, returnEnergyType=energy_type)
    integrator = LangevinMiddleIntegrator(temperature * unit.kelvin, friction_coeff / unit.femtosecond, step_size * unit.femtosecond)
    simulation = Simulation(topology, system, integrator, platform=Platform.getPlatformByName(device.upper()))
    simulation.context.setPositions(np.array(structure.positions) * unit.angstrom)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
    return simulation


def create_openmm_topology(structure):
    label = "aim2dat_structure" if structure.label is None else structure.label
    topology = Topology()
    if structure.cell is not None:
        #TODO take care about rotation?
        topology.setPeriodicBoxVectors(np.array(structure.cell) * unit.angstrom)
    chain = topology.addChain()
    residue = topology.addResidue(label, chain)
    for el, kind in structure.iter_sites(get_kind=True):
        kind = el if kind is None else kind
        topology.addAtom(kind, Element.getBySymbol(el), residue)
    return topology


def extract_structure_from_openmm(simulation):
    state = simulation.context.getState(getPositions=True)
    strct_dict = {
        "elements": [],
        "kinds": [],
        "positions": [],
        "pbc": False,
    }

    if state.getPeriodicBoxVectors():
        strct_dict["cell"] = [
            [v.value_in_unit(unit.nanometer) * 10.0 for v in vec] for vec in state.getPeriodicBoxVectors()
        ]
        strct_dict["pbc"] = True

    for pos, at in zip(state.getPositions(), simulation.topology.atoms()):
        el = at.element.symbol
        kind = at.name if at.name != el else None
        strct_dict["elements"].append(el)
        strct_dict["kinds"].append(kind)
        strct_dict["positions"].append([v.value_in_unit(unit.nanometer) * 10.0 for v in pos])
    return strct_dict
