

import os

import numpy as np
from mace.calculators import MACECalculator


from aim2dat.strct import Structure
from aim2dat.mc.engines import MonteCarlo, TransitionMatrixMonteCarlo
from aim2dat.mc.moves import ReinsertComponent, TranslateComponent, RotateComponent
from aim2dat.io.yaml import load_yaml_file

STRUCTURES_PATH = os.path.dirname(__file__) + "/../strct/structures/"
MLPS_PATH = os.path.dirname(__file__) + "/mlps/"
REF_PATH = os.path.dirname(__file__) + "/ref/"

def test_mc(structure_comparison):
    ref = load_yaml_file(REF_PATH + "mc.yaml")
    mof_strct = Structure.from_file(STRUCTURES_PATH + "MOF-303.xsf")
    water_strct = Structure(**dict(load_yaml_file(STRUCTURES_PATH + "H2O.yaml")))
    calculator = MACECalculator(model_paths=MLPS_PATH + "mace_MOF-303_compiled.model", default_dtype="float32")
    mc_sim = MonteCarlo(
        structure=mof_strct, components=water_strct,
        n_components=[5],
        moves=[ReinsertComponent, TranslateComponent, RotateComponent],
        temperature=298.15,
        n_steps=5,
        n_store=5,
        dist_threshold="chen_manz+25",
        ase_calculator=calculator,
        random_seed=111,
    )
    mc_sim.run()
    assert mc_sim._component_indices == ref["component_indices"]
    assert len(mc_sim.structures) == 5
    for strct, ref_strct in zip(mc_sim.structures, ref["structures"]):
        structure_comparison(strct, ref_strct)
    # from aim2dat.io.yaml import store_in_yaml_file
    # store_in_yaml_file(
    #     "ref_mc.yaml",
    #     {"structures": [strct.to_dict() for strct in mc_sim.structures], "component_indices": mc_sim._component_indices,}
    # )

def test_tmmc(structure_comparison):
    ref = load_yaml_file(REF_PATH + "tmmc.yaml")
    mof_strct = Structure.from_file(STRUCTURES_PATH + "MOF-303.xsf")
    water_dict = dict(load_yaml_file(STRUCTURES_PATH + "H2O.yaml"))
    water_dict["attributes"]["ref_energy"] = -467.837350
    water_strct = Structure(**water_dict)
    calculator = MACECalculator(model_paths=MLPS_PATH + "mace_MOF-303_compiled.model", default_dtype="float32")
    tmmc_sim = TransitionMatrixMonteCarlo(
        structure=mof_strct, components=water_strct,
        n_components=[5],
        n_steps=5,
        energy_penalty=None,
        dist_threshold="chen_manz+25",
        ase_calculator=calculator,
        random_seed=222,
    )
    tmmc_sim.run()
    # from aim2dat.io.yaml import store_in_yaml_file
    # store_in_yaml_file(
    #     "tmmc.yaml",
    #     {
    #         "structure": tmmc_sim.structure.to_dict(), "component_indices": tmmc_sim._component_indices,
    #         "insert_energies": tmmc_sim.insert_energies,
    #         "remove_energies": tmmc_sim.remove_energies,
    #     }
    # )
    assert tmmc_sim._component_indices == ref["component_indices"]
    structure_comparison(tmmc_sim.structure, ref["structure"])
    np.testing.assert_allclose(tmmc_sim.insert_energies, ref["insert_energies"], atol=1.0e-5)
    np.testing.assert_allclose(tmmc_sim.remove_energies, ref["remove_energies"], atol=1.0e-5)

