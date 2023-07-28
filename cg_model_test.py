#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging
import numpy as np
import shutil
import sys
import pathlib
import json
import stk
import itertools
import os
from openmm import openmm
from rdkit import RDLogger

from cgexplore.ensembles import Ensemble
from cgexplore.generation_utilities import (
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_near_models,
    yield_shifted_models,
)
from cgexplore.geom import GeomMeasure
from cgexplore.beads import bead_library_check, produce_bead_library
from cgexplore.generation_utilities import build_building_block
from cgexplore.molecule_construction.topologies import ThreeC1Arm, TwoC1Arm
from cgexplore.utilities import check_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def optimise_cage(
    molecule,
    name,
    output_dir,
    bead_set,
    custom_torsion_set,
    custom_vdw_set,
):
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")
    if os.path.exists(fina_mol_file):
        ensemble = Ensemble(
            base_molecule=molecule,
            base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
            conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
            data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
            overwrite=False,
        )
        return ensemble.get_lowest_e_conformer()

    ensemble = Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )

    molecule = run_constrained_optimisation(
        molecule=molecule,
        bead_set=bead_set,
        name=name,
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
    )

    logging.info(f"optimisation of {name}")
    conformer = run_optimisation(
        molecule=molecule,
        bead_set=bead_set,
        name=name,
        file_suffix="opt1",
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        custom_torsion_set=custom_torsion_set,
        bonds=True,
        angles=True,
        torsions=False,
        vdw_bond_cutoff=2,
        # max_iterations=50,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in yield_shifted_models(molecule, bead_set):
        conformer = run_optimisation(
            molecule=test_molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="sopt",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
            # max_iterations=50,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    # Collect and optimise structures nearby in phase space.
    logging.info(f"optimisation of nearby structures of {name}")
    for test_molecule in yield_near_models(
        molecule=molecule,
        name=name,
        bead_set=bead_set,
        output_dir=output_dir,
    ):
        conformer = run_optimisation(
            molecule=test_molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="nopt",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
            # max_iterations=50,
        )
        ensemble.add_conformer(conformer=conformer, source="nearby_opt")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        molecule=ensemble.get_lowest_e_conformer().molecule,
        bead_set=bead_set,
        ensemble=ensemble,
        output_dir=output_dir,
        custom_vdw_set=custom_vdw_set,
        custom_torsion_set=None,
        bonds=True,
        angles=True,
        torsions=False,
        vdw_bond_cutoff=2,
        suffix="smd",
        bond_ff_scale=10,
        angle_ff_scale=10,
        temperature=300 * openmm.unit.kelvin,
        num_steps=num_steps,
        time_step=0.5 * openmm.unit.femtoseconds,
        friction=1.0 / openmm.unit.picosecond,
        reporting_freq=traj_freq,
        traj_freq=traj_freq,
    )
    if soft_md_trajectory is None:
        logging.info(f"!!!!! {name} MD exploded !!!!!")
        # md_exploded = True
        raise ValueError("OpenMM Exception")

    soft_md_data = soft_md_trajectory.get_data()
    logging.info(f"collected trajectory {len(soft_md_data)} confs long")
    # Check that the trajectory is as long as it should be.
    if len(soft_md_data) != num_steps / traj_freq:
        logging.info(f"!!!!! {name} MD failed !!!!!")
        # md_failed = True
        raise ValueError()

    # Go through each conformer from soft MD.
    # Optimise them all.
    for md_conformer in soft_md_trajectory.yield_conformers():
        conformer = run_optimisation(
            molecule=md_conformer.molecule,
            bead_set=bead_set,
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            custom_vdw_set=custom_vdw_set,
            custom_torsion_set=custom_torsion_set,
            bonds=True,
            angles=True,
            torsions=False,
            vdw_bond_cutoff=2,
            # max_iterations=50,
        )
        ensemble.add_conformer(conformer=conformer, source="smd")
    ensemble.write_conformers_to_file()

    min_energy_conformer = ensemble.get_lowest_e_conformer()
    min_energy_conformerid = min_energy_conformer.conformer_id
    min_energy = min_energy_conformer.energy_decomposition["total energy"][0]
    logging.info(
        f"Min. energy conformer: {min_energy_conformerid} from "
        f"{min_energy_conformer.source}"
        f" with energy: {min_energy} kJ.mol-1"
    )

    min_energy_conformer.molecule.write(fina_mol_file)
    return min_energy_conformer


def target_torsions(bead_set, custom_torsion_option):
    try:
        (t_key_1,) = (i for i in bead_set if i[0] == "a")
    except ValueError:
        # For when 3+4 cages are being built - there are no target
        # torsions.
        return None

    (c_key,) = (i for i in bead_set if i[0] == "c")
    (t_key_2,) = (i for i in bead_set if i[0] == "b")
    custom_torsion_set = {
        (
            t_key_2,
            t_key_1,
            c_key,
            t_key_1,
            t_key_2,
        ): custom_torsion_option,
    }
    return custom_torsion_set


def collect_custom_torsion(
    bb2_bead_set,
    custom_torsion_options,
    custom_torsion,
    bead_set,
):
    if custom_torsion_options[custom_torsion] is None:
        custom_torsion_set = None
    else:
        tors_option = custom_torsion_options[custom_torsion]
        custom_torsion_set = target_torsions(
            bead_set=bead_set,
            custom_torsion_option=tors_option,
        )

    return custom_torsion_set


def bond_k():
    return 1e5


def angle_k():
    return 1e2


def core_2c_beads():
    return produce_bead_library(
        type_prefix="c",
        element_string="Ag",
        angles=(180,),
        bond_rs=(2,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def arm_2c_beads():
    return produce_bead_library(
        type_prefix="a",
        element_string="Ba",
        bond_rs=(1,),
        angles=(125, 160, 175),  # 180),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def binder_beads():
    return produce_bead_library(
        type_prefix="b",
        element_string="Pb",
        bond_rs=(1,),
        angles=(180,),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=2,
    )


def beads_3c():
    return produce_bead_library(
        type_prefix="n",
        element_string="C",
        bond_rs=(2,),
        angles=(70, 90, 120),  # 60),
        bond_ks=(bond_k(),),
        angle_ks=(angle_k(),),
        sigma=1,
        epsilon=10.0,
        coordination=3,
    )


def get_final_energy(path):
    if ".out" in str(path):
        with open(path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "total energy:" in line:
                    return float(line.rstrip().split()[2])

    elif ".json" in str(path):
        with open(path, "r") as f:
            data = json.load(f)
        min_e = 1e10
        min_e_i = 0
        for i in data:
            ey = data[i]["total energy"][0]
            if ey < min_e:
                min_e = ey
                min_e_i = i
        return min_e, min_e_i


def compare_final_energies(path1, path2):
    if ".out" in str(path1):
        e1 = get_final_energy(path1)
        e2 = get_final_energy(path2)
        print(path1.name, path2.name, e1, e2)
        # assert np.isclose(e1, e2, atol=1e-2, rtol=0)
    elif ".json" in str(path1):
        e1, id1 = get_final_energy(path1)
        e2, id2 = get_final_energy(path2)
        print(path1.name, path2.name, e1, e2, id1, id2)
        # assert np.isclose(e1, e2, atol=1e-2, rtol=0)
        # assert id1 == id2


def main():
    first_line = f"Usage: {__file__}.py"
    if not len(sys.argv) == 1:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        pass

    struct_output = pathlib.Path().absolute() / "structures"
    check_directory(struct_output)
    calculation_output = pathlib.Path().absolute() / "calculations"
    check_directory(calculation_output)
    ligand_output = pathlib.Path().absolute() / "ligands"
    check_directory(ligand_output)

    struct_done = pathlib.Path().absolute() / "structures_done"
    calculation_done = pathlib.Path().absolute() / "calculations_done"
    ligand_done = pathlib.Path().absolute() / "ligands_done"

    # Define bead libraries.
    beads_core_2c_lib = core_2c_beads()
    beads_3c_lib = beads_3c()
    beads_arm_2c_lib = arm_2c_beads()
    beads_binder_lib = binder_beads()
    full_bead_library = (
        list(beads_3c_lib.values())
        + list(beads_arm_2c_lib.values())
        + list(beads_core_2c_lib.values())
        + list(beads_binder_lib.values())
    )
    bead_library_check(full_bead_library)

    logging.info("building building blocks")
    c2_blocks = build_building_block(
        topology=TwoC1Arm,
        option1_lib=beads_core_2c_lib,
        option2_lib=beads_arm_2c_lib,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )
    c3_blocks = build_building_block(
        topology=ThreeC1Arm,
        option1_lib=beads_3c_lib,
        option2_lib=beads_binder_lib,
        calculation_output=calculation_output,
        ligand_output=ligand_output,
    )

    logging.info(
        f"there are {len(c2_blocks)} 2-C and "
        f"{len(c3_blocks)} 3-C and building blocks."
    )

    # Define list of topology functions.
    cage_3p2_topologies = {"4P6": stk.cage.FourPlusSix}

    populations = {
        "2p3": {
            "t": cage_3p2_topologies,
            "c2": c2_blocks,
            "cl": c3_blocks,
        },
    }
    custom_torsion_options = {"ton": (180, 50), "toff": None}
    custom_vdw_options = {"von": True}

    cages = []
    for popn in populations:
        popn_iterator = itertools.product(
            populations[popn]["t"],
            populations[popn]["c2"],
            populations[popn]["cl"],
            custom_torsion_options,
            custom_vdw_options,
        )
        for iteration in popn_iterator:
            (
                cage_topo_str,
                bb2_str,
                bbl_str,
                custom_torsion,
                custom_vdw,
            ) = iteration

            bb2, bb2_bead_set = populations[popn]["c2"][bb2_str]
            bbl, bbl_bead_set = populations[popn]["cl"][bbl_str]

            bead_set = bb2_bead_set.copy()
            bead_set.update(bbl_bead_set)

            custom_torsion_set = collect_custom_torsion(
                bb2_bead_set=bb2_bead_set,
                custom_torsion_options=(custom_torsion_options),
                custom_torsion=custom_torsion,
                bead_set=bead_set,
            )

            custom_vdw_set = custom_vdw_options[custom_vdw]

            for run in range(1):
                name = (
                    f"{cage_topo_str}_{bbl_str}_{bb2_str}_"
                    f"{custom_torsion}_{custom_vdw}_{run}"
                )

                logging.info(f"building {name}")
                cage = stk.ConstructedMolecule(
                    topology_graph=populations[popn]["t"][cage_topo_str](
                        building_blocks=(bb2, bbl),
                    ),
                )

                conformer = optimise_cage(
                    molecule=cage,
                    name=name,
                    output_dir=calculation_output,
                    bead_set=bead_set,
                    custom_torsion_set=custom_torsion_set,
                    custom_vdw_set=custom_vdw_set,
                )

                if conformer is not None:
                    conformer.molecule.write(
                        str(struct_output / f"{name}_optc.mol")
                    )
                cages.append(name)

    for i in cages:
        if "a00" in i:
            old = i.replace("a00", "a07")
        elif "a01" in i:
            old = i.replace("a01", "a014")
        elif "a02" in i:
            old = i.replace("a02", "a017")

        if "n00" in i:
            old = old.replace("n00", "n02")
        elif "n01" in i:
            old = old.replace("n01", "n04")
        elif "n02" in i:
            old = old.replace("n02", "n07")
        # compare_final_energies(
        #     path1=calculation_done / f"{old}_opt1_omm.out",
        #     path2=calculation_output / f"{i}_opt1_omm.out",
        # )
        compare_final_energies(
            path1=calculation_done / f"{old}_ensemble.json",
            path2=calculation_output / f"{i}_ensemble.json",
        )
    raise SystemExit()

    # for i in c2_blocks:
    #     old = i.replace("a00", "a07").replace("a01", "a018")
    #     compare_final_energies(
    #         path1=calculation_done / f"{old}_omm.out",
    #         path2=calculation_output / f"{i}_omm.out",
    #     )

    # for i in c3_blocks:
    #     print(i)
    #     old = i.replace("n01", "n07").replace("n00", "n01")
    #     print(old)
    #     compare_final_energies(
    #         path1=calculation_done / f"{old}_omm.out",
    #         path2=calculation_output / f"{i}_omm.out",
    #     )
    # raise SystemExit()

    # g_measure = GeomMeasure(custom_torsion_atoms)
    # bond_data = g_measure.calculate_bonds(conformer.molecule)
    # angle_data = g_measure.calculate_angles(conformer.molecule)
    # dihedral_data = g_measure.calculate_torsions(
    #     molecule=conformer.molecule,
    #     absolute=True,
    #     path_length=5,
    # )
    # min_b2b_distance = g_measure.calculate_minb2b(conformer.molecule)
    # radius_gyration = g_measure.calculate_radius_gyration(
    #     molecule=conformer.molecule,
    # )
    # max_diameter = g_measure.calculate_max_diameter(conformer.molecule)
    # if radius_gyration > max_diameter:
    #     raise ValueError(
    #         f"{name} Rg ({radius_gyration}) > maxD ({max_diameter})"
    #     )

    shutil.rmtree(calculation_output)
    shutil.rmtree(struct_output)
    shutil.rmtree(ligand_output)


if __name__ == "__main__":
    main()
