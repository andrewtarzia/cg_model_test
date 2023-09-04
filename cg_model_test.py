#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging

# import shutil
import sys
import pathlib
import json
import stk
import itertools
import os
from openmm import openmm
from rdkit import RDLogger
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cgexplore.bonds import TargetBondRange
from cgexplore.angles import TargetAngleRange
from cgexplore.torsions import TargetTorsionRange, TargetTorsion
from cgexplore.nonbonded import TargetNonbondedRange
from cgexplore.ensembles import Ensemble
from cgexplore.forcefield import ForceFieldLibrary
from cgexplore.generation_utilities import (
    run_constrained_optimisation,
    run_optimisation,
    run_soft_md_cycle,
    yield_shifted_models,
    optimise_ligand,
)
from cgexplore.geom import GeomMeasure
from cgexplore.beads import bead_library_check, CgBead
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
    force_field,
    platform,
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
        name=name,
        output_dir=output_dir,
        force_field=force_field,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    logging.info(f"optimisation of {name}")
    conformer = run_optimisation(
        molecule=molecule,
        name=name,
        file_suffix="opt1",
        output_dir=output_dir,
        force_field=force_field,
        # max_iterations=50,
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in yield_shifted_models(conformer.molecule, force_field):
        conformer = run_optimisation(
            molecule=test_molecule,
            name=name,
            file_suffix="sopt",
            output_dir=output_dir,
            force_field=force_field,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = run_soft_md_cycle(
        name=name,
        molecule=ensemble.get_lowest_e_conformer().molecule,
        output_dir=output_dir,
        force_field=force_field,
        suffix="smd",
        bond_ff_scale=10,
        angle_ff_scale=10,
        temperature=300 * openmm.unit.kelvin,
        num_steps=num_steps,
        time_step=0.5 * openmm.unit.femtoseconds,
        friction=1.0 / openmm.unit.picosecond,
        reporting_freq=traj_freq,
        traj_freq=traj_freq,
        platform=platform,
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
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
            force_field=force_field,
            # max_iterations=50,
            platform=platform,
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
        # print(path1.name, path2.name, e1, e2)
        # assert np.isclose(e1, e2, atol=1e-2, rtol=0)
        return e1, e2
    elif ".json" in str(path1):
        e1, id1 = get_final_energy(path1)
        e2, id2 = get_final_energy(path2)
        print(path1.name, path2.name, e1, e2, id1, id2)
        # try:
        #     assert np.isclose(e1, e2, atol=1e-1, rtol=0)
        # except AssertionError:
        #     assert e1 > 5 and e2 > 5
        # assert id1 == id2
        return e1, e2


def define_forcefield_library(full_bead_library, calculation_output, prefix):
    forcefieldlibrary = ForceFieldLibrary(
        bead_library=full_bead_library,
        vdw_bond_cutoff=2,
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a",
            class2="c",
            eclass1="Ba",
            eclass2="Ag",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="a",
            class2="b",
            eclass1="Ba",
            eclass2="Pb",
            bond_rs=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_bond_range(
        TargetBondRange(
            class1="b",
            class2="n",
            eclass1="Pb",
            eclass2="C",
            bond_rs=(
                openmm.unit.Quantity(value=1.5, unit=openmm.unit.angstrom),
            ),
            bond_ks=(
                openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="a",
            class2="c",
            class3="a",
            eclass1="Ba",
            eclass2="Ag",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b",
            class2="a",
            class3="c",
            eclass1="Pb",
            eclass2="Ba",
            eclass3="Ag",
            angles=(
                openmm.unit.Quantity(value=125, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=160, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=175, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="n",
            class2="b",
            class3="a",
            eclass1="C",
            eclass2="Pb",
            eclass3="Ba",
            angles=(
                openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )
    forcefieldlibrary.add_angle_range(
        TargetAngleRange(
            class1="b",
            class2="n",
            class3="b",
            eclass1="Pb",
            eclass2="C",
            eclass3="Pb",
            angles=(
                openmm.unit.Quantity(value=70, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
                openmm.unit.Quantity(value=120, unit=openmm.unit.degrees),
            ),
            angle_ks=(
                openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        )
    )

    forcefieldlibrary.add_torsion_range(
        TargetTorsionRange(
            search_string=("b", "a", "c", "a", "b"),
            search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
            measured_atom_ids=[0, 1, 3, 4],
            phi0s=(openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),),
            torsion_ks=(
                openmm.unit.Quantity(
                    value=50,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
                openmm.unit.Quantity(
                    value=0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
            ),
            torsion_ns=(1,),
        )
    )

    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            "a",
            "Ba",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            "c",
            "Ag",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            "n",
            "C",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
        )
    )
    forcefieldlibrary.add_nonbonded_range(
        TargetNonbondedRange(
            "b",
            "Pb",
            epsilons=(
                openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
            ),
            sigmas=(
                openmm.unit.Quantity(value=1.0, unit=openmm.unit.angstrom),
            ),
        )
    )

    count = 0
    for force_field in forcefieldlibrary.yield_forcefields(
        prefix=prefix, output_path=calculation_output
    ):
        force_field.write_xml_file()
        count += 1

    logging.info(f"there are {count} forcefields")
    return forcefieldlibrary


def main():
    first_line = f"Usage: {__file__}.py path "
    if not len(sys.argv) == 2:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        path = sys.argv[1]

    prefix = "cg_model_test"

    struct_output = pathlib.Path().absolute() / path / "structures"
    check_directory(struct_output)
    calculation_output = pathlib.Path().absolute() / path / "calculations"
    check_directory(calculation_output)
    ligand_output = pathlib.Path().absolute() / path / "ligands"
    check_directory(ligand_output)

    struct_done = pathlib.Path().absolute() / path / "structures_done"
    calculation_done = pathlib.Path().absolute() / path / "calculations_done"

    # Define bead libraries.
    core_bead = CgBead(
        element_string="Ag",
        bead_type="c",
        coordination=2,
    )
    arm_bead = CgBead(
        element_string="Ba",
        bead_type="a",
        coordination=2,
    )
    binder_bead = CgBead(
        element_string="Pb",
        bead_type="b",
        coordination=2,
    )
    trigonal_bead = CgBead(
        element_string="C",
        bead_type="n",
        coordination=2,
    )
    full_bead_library = (core_bead, arm_bead, binder_bead, trigonal_bead)
    bead_library_check(full_bead_library)

    logging.info(f"defining force field for {prefix}")
    forcefieldlibrary = define_forcefield_library(
        full_bead_library=full_bead_library,
        calculation_output=calculation_output,
        prefix=prefix,
    )

    logging.info("building building blocks")
    ditopic = TwoC1Arm(bead=core_bead, abead1=arm_bead)
    tritopic = ThreeC1Arm(bead=trigonal_bead, abead1=binder_bead)
    for force_field in forcefieldlibrary.yield_forcefields(
        prefix=prefix, output_path=calculation_output
    ):
        for bb in (ditopic, tritopic):
            temp_name = f"{bb.get_name()}_f{force_field.get_identifier()}"
            opt_bb = optimise_ligand(
                molecule=bb.get_building_block(),
                name=temp_name,
                output_dir=calculation_output,
                force_field=force_field,
                platform="CPU",
            )
            opt_bb.write(str(ligand_output / f"{temp_name}_optl.mol"))

    # Define list of topology functions.
    cage_3p2_topologies = {"4P6": stk.cage.FourPlusSix}

    cages = []
    popn_iterator = itertools.product(
        cage_3p2_topologies,
        tuple(
            forcefieldlibrary.yield_forcefields(
                prefix=prefix, output_path=calculation_output
            )
        ),
    )
    for cage_topo_str, force_field in popn_iterator:
        name = (
            f"{cage_topo_str}_{tritopic.get_name()}_"
            f"{ditopic.get_name()}_"
            f"f{force_field.get_identifier()}"
        )

        logging.info(f"building {name}")
        cage = stk.ConstructedMolecule(
            topology_graph=cage_3p2_topologies[cage_topo_str](
                building_blocks=(
                    tritopic.get_building_block(),
                    ditopic.get_building_block(),
                ),
            ),
        )

        conformer = optimise_cage(
            molecule=cage,
            name=name,
            output_dir=calculation_output,
            force_field=force_field,
            # platform="CPU",
            # platform="CUDA",
            platform=None,
        )

        if conformer is not None:
            conformer.molecule.write(str(struct_output / f"{name}_optc.mol"))
        cages.append(name)

    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))

    ax = axs[0][0]
    ax_bond = axs[0][1]
    ax_angle = axs[0][2]
    ax_tors = axs[1][0]
    ax_rg = axs[1][1]
    ax_md = axs[1][2]

    for i in cages:
        print(i)
        raise SystemExit()
        if "ton" in i:
            c = "r"
        elif "toff" in i:
            c = "gray"

        if "a00" in i:
            old = i.replace("a00", "a07")
        elif "a01" in i:
            old = i.replace("a01", "a014")
        elif "a02" in i:
            old = i.replace("a02", "a017")
        elif "a03" in i:
            old = i.replace("a03", "a018")

        if "n00" in i:
            old = old.replace("n00", "n02")
        elif "n01" in i:
            old = old.replace("n01", "n04")
        elif "n02" in i:
            old = old.replace("n02", "n07")
        elif "n03" in i:
            old = old.replace("n03", "n01")
        # compare_final_energies(
        #     path1=calculation_done / f"{old}_opt1_omm.out",
        #     path2=calculation_output / f"{i}_opt1_omm.out",
        # )
        e1, e2 = compare_final_energies(
            path1=calculation_done / f"{old}_ensemble.json",
            path2=calculation_output / f"{i}_ensemble.json",
        )

        ax.scatter(
            e1,
            e2,
            c=c,
            edgecolor="none",
            s=100,
            alpha=1.0,
        )

        new_struct = stk.BuildingBlock.init_from_file(
            str(struct_output / f"{i}_optc.mol")
        )
        old_struct = stk.BuildingBlock.init_from_file(
            str(struct_done / f"{old}_optc.mol")
        )

        assert new_struct.get_num_atoms() == old_struct.get_num_atoms()
        assert new_struct.get_num_bonds() == old_struct.get_num_bonds()

        g_measure = GeomMeasure(
            target_torsions=(
                TargetTorsion(
                    search_string=("b", "a", "c", "a", "b"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(
                        value=180, unit=openmm.unit.degrees
                    ),
                    torsion_k=openmm.unit.Quantity(
                        value=50,
                        unit=openmm.unit.kilojoule / openmm.unit.mole,
                    ),
                    torsion_n=1,
                ),
            )
        )
        bond_data1 = g_measure.calculate_bonds(old_struct)
        bond_data2 = g_measure.calculate_bonds(new_struct)
        for i in bond_data1:
            assert i in bond_data2
            assert len(bond_data1[i]) == len(bond_data2[i])
            for bd1, bd2 in zip(bond_data1[i], bond_data2[i]):
                # print(bd1, bd2)
                # assert np.isclose(bd1, bd2, atol=1e-1, rtol=0)
                ax_bond.scatter(
                    bd1,
                    bd2,
                    c=c,
                    edgecolor="none",
                    s=100,
                    alpha=1.0,
                )

        angle_data1 = g_measure.calculate_angles(old_struct)
        angle_data2 = g_measure.calculate_angles(new_struct)
        for i in angle_data1:
            assert i in angle_data2
            assert len(angle_data1[i]) == len(angle_data2[i])
            for bd1, bd2 in zip(angle_data1[i], angle_data2[i]):
                # print(bd1, bd2)
                # assert np.isclose(bd1, bd2, atol=1, rtol=0)
                ax_angle.scatter(
                    bd1,
                    bd2,
                    c=c,
                    edgecolor="none",
                    s=100,
                    alpha=1.0,
                )

        dihedral_data1 = g_measure.calculate_torsions(
            molecule=old_struct,
            absolute=True,
        )
        dihedral_data2 = g_measure.calculate_torsions(
            molecule=new_struct,
            absolute=True,
        )
        for i in dihedral_data1:
            assert i in dihedral_data2
            assert len(dihedral_data1[i]) == len(dihedral_data2[i])
            for bd1, bd2 in zip(dihedral_data1[i], dihedral_data2[i]):
                # assert np.isclose(bd1, bd2, atol=1, rtol=0)
                ax_tors.scatter(
                    bd1,
                    bd2,
                    c=c,
                    edgecolor="none",
                    s=100,
                    alpha=1.0,
                )

        max_diameter1 = g_measure.calculate_max_diameter(old_struct)
        max_diameter2 = g_measure.calculate_max_diameter(new_struct)
        # print(max_diameter1, max_diameter2)
        # assert np.isclose(bd1, bd2, atol=1, rtol=0)
        ax_md.scatter(
            max_diameter1,
            max_diameter2,
            c=c,
            edgecolor="none",
            s=100,
            alpha=1.0,
        )

        radius_gyration1 = g_measure.calculate_radius_gyration(old_struct)
        radius_gyration2 = g_measure.calculate_radius_gyration(new_struct)
        # print(radius_gyration1, radius_gyration2)
        # assert np.isclose(bd1, bd2, atol=1, rtol=0)
        ax_rg.scatter(
            radius_gyration1,
            radius_gyration2,
            c=c,
            edgecolor="none",
            s=100,
            alpha=1.0,
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("first run energy", fontsize=16)
    ax.set_ylabel("second run energy", fontsize=16)
    ax.set_xlim(0.001, 100)
    ax.set_ylim(0.001, 100)
    ax.plot([0, 100], [0, 100], c="k", ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax_bond.tick_params(axis="both", which="major", labelsize=16)
    ax_bond.set_xlabel("first run bonds", fontsize=16)
    ax_bond.set_ylabel("second run bonds", fontsize=16)
    ax_bond.set_xlim(0.9, 1.6)
    ax_bond.set_ylim(0.9, 1.6)
    ax_bond.plot([0.9, 1.6], [0.9, 1.6], c="k", ls="--")

    ax_angle.tick_params(axis="both", which="major", labelsize=16)
    ax_angle.set_xlabel("first run angles", fontsize=16)
    ax_angle.set_ylabel("second run angles", fontsize=16)
    ax_angle.set_xlim(0, 180)
    ax_angle.set_ylim(0, 180)
    ax_angle.plot([0, 180], [0, 180], c="k", ls="--")

    ax_tors.tick_params(axis="both", which="major", labelsize=16)
    ax_tors.set_xlabel("first run tors", fontsize=16)
    ax_tors.set_ylabel("second run tors", fontsize=16)
    ax_tors.set_xlim(0, 180)
    ax_tors.set_ylim(0, 180)
    ax_tors.plot([0, 180], [0, 180], c="k", ls="--")

    ax_rg.tick_params(axis="both", which="major", labelsize=16)
    ax_rg.set_xlabel("first run rgs", fontsize=16)
    ax_rg.set_ylabel("second run rgs", fontsize=16)
    ax_rg.set_xlim(0, 10)
    ax_rg.set_ylim(0, 10)
    ax_rg.plot([0, 10], [0, 10], c="k", ls="--")

    ax_md.tick_params(axis="both", which="major", labelsize=16)
    ax_md.set_xlabel("first run max diams", fontsize=16)
    ax_md.set_ylabel("second run max diams", fontsize=16)
    ax_md.set_xlim(5, 15)
    ax_md.set_ylim(5, 15)
    ax_md.plot([5, 15], [5, 15], c="k", ls="--")

    legend_elements = []
    for tstr in {"ton": "r", "toff": "gray"}:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=tstr,
                markerfacecolor={"ton": "r", "toff": "gray"}[tstr],
                markersize=7,
                markeredgecolor="none",
                alpha=1.0,
            )
        )
    ax.legend(handles=legend_elements, fontsize=16, ncol=1)

    fig.tight_layout()
    fig.savefig(
        "parity.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()

    # shutil.rmtree(calculation_output)
    # shutil.rmtree(struct_output)
    # shutil.rmtree(ligand_output)
    raise SystemExit()

    raise SystemExit("define bond, angle, torsion, vdw objects")
    raise SystemExit(
        "these provide a way to set the force field in python code - "
        "smarts, values"
    )
    raise SystemExit("use openFF to use the FF xml file written by this code")
    raise SystemExit(
        "rewrite the optimiser classes to handle this and remove the "
        "default behaviour"
    )
    raise SystemExit()


if __name__ == "__main__":
    main()
