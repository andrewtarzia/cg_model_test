#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to generate and optimise CG models.

Author: Andrew Tarzia

"""

import logging
import sys
import pathlib
import json
import stk
import itertools
import os
from openmm import openmm
from rdkit import RDLogger
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import cgexplore as cgx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def optimise_cage(
    molecule,
    name,
    output_dir,
    forcefield,
    platform,
    database,
):
    fina_mol_file = os.path.join(output_dir, f"{name}_final.mol")
    # Do not rerun if database entry exists.
    if database.has_molecule(key=name):
        final_molecule = database.get_molecule(key=name)
        final_molecule.write(fina_mol_file)
        return cgx.molecular.Conformer(
            molecule=final_molecule,
            energy_decomposition=database.get_property(
                key=name,
                property_key="energy_decomposition",
                property_type=dict,
            ),
        )

    # Do not rerun if final mol exists.
    if os.path.exists(fina_mol_file):
        ensemble = cgx.molecular.Ensemble(
            base_molecule=molecule,
            base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
            conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
            data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
            overwrite=False,
        )
        conformer = ensemble.get_lowest_e_conformer()
        database.add_molecule(molecule=conformer.molecule, key=name)
        database.add_properties(
            key=name,
            property_dict={
                "energy_decomposition": conformer.energy_decomposition,
                "source": conformer.source,
                "optimised": True,
            },
        )
        return ensemble.get_lowest_e_conformer()

    assigned_system = forcefield.assign_terms(molecule, name, output_dir)

    ensemble = cgx.molecular.Ensemble(
        base_molecule=molecule,
        base_mol_path=os.path.join(output_dir, f"{name}_base.mol"),
        conformer_xyz=os.path.join(output_dir, f"{name}_ensemble.xyz"),
        data_json=os.path.join(output_dir, f"{name}_ensemble.json"),
        overwrite=True,
    )
    temp_molecule = cgx.utilities.run_constrained_optimisation(
        assigned_system=assigned_system,
        name=name,
        output_dir=output_dir,
        bond_ff_scale=10,
        angle_ff_scale=10,
        max_iterations=20,
        platform=platform,
    )

    logging.info(f"optimisation of {name}")
    conformer = cgx.utilities.run_optimisation(
        assigned_system=cgx.forcefields.AssignedSystem(
            molecule=temp_molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        name=name,
        file_suffix="opt1",
        output_dir=output_dir,
        # max_iterations=50,
        platform=platform,
    )
    ensemble.add_conformer(conformer=conformer, source="opt1")

    # Run optimisations of series of conformers with shifted out
    # building blocks.
    logging.info(f"optimisation of shifted structures of {name}")
    for test_molecule in cgx.utilities.yield_shifted_models(
        temp_molecule, forcefield, kicks=(1, 2, 3, 4)
    ):
        conformer = cgx.utilities.run_optimisation(
            assigned_system=cgx.forcefields.AssignedSystem(
                molecule=test_molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="sopt",
            output_dir=output_dir,
            # max_iterations=50,
            platform=platform,
        )
        ensemble.add_conformer(conformer=conformer, source="shifted")

    logging.info(f"soft MD run of {name}")
    num_steps = 20000
    traj_freq = 500
    soft_md_trajectory = cgx.utilities.run_soft_md_cycle(
        name=name,
        assigned_system=cgx.forcefields.AssignedSystem(
            molecule=ensemble.get_lowest_e_conformer().molecule,
            forcefield_terms=assigned_system.forcefield_terms,
            system_xml=assigned_system.system_xml,
            topology_xml=assigned_system.topology_xml,
            bead_set=assigned_system.bead_set,
            vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
        ),
        output_dir=output_dir,
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
        conformer = cgx.utilities.run_optimisation(
            assigned_system=cgx.forcefields.AssignedSystem(
                molecule=md_conformer.molecule,
                forcefield_terms=assigned_system.forcefield_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            ),
            name=name,
            file_suffix="smd_mdc",
            output_dir=output_dir,
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

    # Add to atomlite database.
    database.add_molecule(molecule=min_energy_conformer.molecule, key=name)
    database.add_properties(
        key=name,
        property_dict={
            "energy_decomposition": min_energy_conformer.energy_decomposition,
            "source": min_energy_conformer.source,
            "optimised": True,
        },
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
    if ".json" in str(path1):
        e1, id1 = get_final_energy(path1)
        e2, id2 = get_final_energy(path2)
    try:
        assert np.isclose(e1, e2, atol=1e-1, rtol=0)
    except AssertionError:
        # assert e1 > 5 and e2 > 5
        print("!!!!!!!!!!!!!!!!")
        print(e1, id1, e2, id2)
        print(path1.name, path2.name)
        print("!!!!!!!!!!!!!!!!")
    return e1, e2


def define_forcefield_library(present_beads, prefix):
    forcefieldlibrary = cgx.forcefields.ForceFieldLibrary(
        present_beads=present_beads,
        vdw_bond_cutoff=2,
        prefix=prefix,
    )

    if "2p3" in prefix:
        bonds = (
            ("a1", "c1", "Ba", "Ag", (1.5,), (1e5,)),
            ("a1", "b1", "Ba", "Pb", (1.0,), (1e5,)),
            ("b1", "n1", "Pb", "C", (1.5,), (1e5,)),
        )
    elif "2p4" in prefix:
        bonds = (
            ("a1", "c1", "Ba", "Ag", (1.5,), (1e5,)),
            ("a1", "b1", "Ba", "Pb", (1.0,), (1e5,)),
            ("b1", "m1", "Pb", "Pd", (1.5,), (1e5,)),
        )
    for bond in bonds:
        r_range = tuple(
            openmm.unit.Quantity(value=b, unit=openmm.unit.angstrom) for b in bond[4]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule
                / openmm.unit.mole
                / openmm.unit.nanometer**2,
            )
            for k in bond[5]
        )
        forcefieldlibrary.add_bond_range(
            cgx.terms.TargetBondRange(
                type1=bond[0],
                type2=bond[1],
                element1=bond[2],
                element2=bond[3],
                bond_rs=r_range,
                bond_ks=k_range,
            )
        )

    if "2p3" in prefix:
        angles = (
            ("a1", "c1", "a1", "Ba", "Ag", "Ba", (180,), (1e2,)),
            ("b1", "a1", "c1", "Pb", "Ba", "Ag", (125, 160, 175), (1e2,)),
            ("n1", "b1", "a1", "C", "Pb", "Ba", (180,), (1e2,)),
            ("b1", "n1", "b1", "Pb", "C", "Pb", (70, 90, 120), (1e2,)),
        )
        pyramids = ()
    elif "2p4" in prefix:
        angles = (
            ("a1", "c1", "a1", "Ba", "Ag", "Ba", (180,), (1e2,)),
            ("b1", "a1", "c1", "Pb", "Ba", "Ag", (135, 160), (1e2,)),
            ("m1", "b1", "a1", "Pd", "Pb", "Ba", (180,), (1e2,)),
        )
        pyramids = (("b1", "m1", "b1", "Pb", "Pd", "Pb", (80, 90), (1e2,)),)
    for angle in angles:
        a_range = tuple(
            openmm.unit.Quantity(value=a, unit=openmm.unit.degrees) for a in angle[6]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2,
            )
            for k in angle[7]
        )
        forcefieldlibrary.add_angle_range(
            cgx.terms.TargetAngleRange(
                type1=angle[0],
                type2=angle[1],
                type3=angle[2],
                element1=angle[3],
                element2=angle[4],
                element3=angle[5],
                angles=a_range,
                angle_ks=k_range,
            )
        )
    for pyramid in pyramids:
        a_range = tuple(
            openmm.unit.Quantity(value=a, unit=openmm.unit.degrees) for a in pyramid[6]
        )
        k_range = tuple(
            openmm.unit.Quantity(
                value=k,
                unit=openmm.unit.kilojoule / openmm.unit.mole / openmm.unit.radian**2,
            )
            for k in pyramid[7]
        )
        forcefieldlibrary.add_angle_range(
            cgx.terms.PyramidAngleRange(
                type1=pyramid[0],
                type2=pyramid[1],
                type3=pyramid[2],
                element1=pyramid[3],
                element2=pyramid[4],
                element3=pyramid[5],
                angles=a_range,
                angle_ks=k_range,
            )
        )

    forcefieldlibrary.add_torsion_range(
        cgx.terms.TargetTorsionRange(
            search_string=("b1", "a1", "c1", "a1", "b1"),
            search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
            measured_atom_ids=[0, 1, 3, 4],
            phi0s=(openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),),
            torsion_ks=(
                openmm.unit.Quantity(
                    value=50,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
                openmm.unit.Quantity(
                    value=0,
                    unit=openmm.unit.kilojoules_per_mole,
                ),
            ),
            torsion_ns=(1,),
        )
    )

    if "2p3" in prefix:
        nonbondeds = (
            ("a", "Ba", 10.0, 1.0),
            ("c", "Ag", 10.0, 1.0),
            ("b", "Pb", 10.0, 1.0),
            ("n", "C", 10.0, 1.0),
        )
    elif "2p4" in prefix:
        nonbondeds = (
            ("a", "Ba", 10.0, 1.0),
            ("c", "Ag", 10.0, 1.0),
            ("b", "Pb", 10.0, 1.0),
            ("m", "Pd", 10.0, 1.0),
        )

    for nb in nonbondeds:
        forcefieldlibrary.add_nonbonded_range(
            cgx.terms.TargetNonbondedRange(
                bead_class=nb[0],
                bead_element=nb[1],
                epsilons=(
                    openmm.unit.Quantity(
                        value=nb[2], unit=openmm.unit.kilojoules_per_mole
                    ),
                ),
                sigmas=(openmm.unit.Quantity(value=nb[3], unit=openmm.unit.angstrom),),
                force="custom-excl-vol",
            )
        )

    return forcefieldlibrary


def analysis(
    cages,
    struct_output,
    struct_done,
    calculation_output,
    calculation_done,
    database,
):
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(16, 8))

    ax = axs[0][0]
    ax_bond = axs[0][1]
    ax_angle = axs[0][2]
    ax_tors = axs[1][0]
    ax_rg = axs[1][1]
    ax_md = axs[1][2]
    ff_map = {
        "2p3": {
            "0": {"bac": 125, "bnb": 70, "tors": 50},
            "1": {"bac": 125, "bnb": 70, "tors": 0},
            "2": {"bac": 125, "bnb": 90, "tors": 50},
            "3": {"bac": 125, "bnb": 90, "tors": 0},
            "4": {"bac": 125, "bnb": 120, "tors": 50},
            "5": {"bac": 125, "bnb": 120, "tors": 0},
            "6": {"bac": 160, "bnb": 70, "tors": 50},
            "7": {"bac": 160, "bnb": 70, "tors": 0},
            "8": {"bac": 160, "bnb": 90, "tors": 50},
            "9": {"bac": 160, "bnb": 90, "tors": 0},
            "10": {"bac": 160, "bnb": 120, "tors": 50},
            "11": {"bac": 160, "bnb": 120, "tors": 0},
            "12": {"bac": 175, "bnb": 70, "tors": 50},
            "13": {"bac": 175, "bnb": 70, "tors": 0},
            "14": {"bac": 175, "bnb": 90, "tors": 50},
            "15": {"bac": 175, "bnb": 90, "tors": 0},
            "16": {"bac": 175, "bnb": 120, "tors": 50},
            "17": {"bac": 175, "bnb": 120, "tors": 0},
        },
        "2p4": {
            "0": {"bac": 135, "bmb": 80, "tors": 50},
            "1": {"bac": 135, "bmb": 80, "tors": 0},
            "2": {"bac": 135, "bmb": 90, "tors": 50},
            "3": {"bac": 135, "bmb": 90, "tors": 0},
            "4": {"bac": 160, "bmb": 80, "tors": 50},
            "5": {"bac": 160, "bmb": 80, "tors": 0},
            "6": {"bac": 160, "bmb": 90, "tors": 50},
            "7": {"bac": 160, "bmb": 90, "tors": 0},
        },
    }

    bac_map = {
        125: "2C1c0000a0700",
        135: "2C1c0000a0900",
        160: "2C1c0000a01400",
        175: "2C1c0000a01700",
    }
    bnb_map = {
        70: "3C1n0200b0000",
        90: "3C1n0400b0000",
        120: "3C1n0700b0000",
    }
    bmb_map = {
        80: "4C1m0300b0000",
        90: "4C1m0400b0000",
    }

    alpha = 1.0
    m = "o"

    structure_comparisons = []
    old_cage_suffix = "_von_0"
    for cage_name in cages:
        ff_name = cage_name.split("_f")[1]
        t_str = cage_name.split("_")[0]
        if "4P6" in cage_name:
            ff_values = ff_map["2p3"][ff_name]
            bb1name = bnb_map[ff_values["bnb"]]
        elif "6P12" in cage_name:
            ff_values = ff_map["2p4"][ff_name]
            bb1name = bmb_map[ff_values["bmb"]]
        torsion = "ton" if ff_values["tors"] == 50 else "toff"
        bb2name = bac_map[ff_values["bac"]]
        old_cage = f"{t_str}_{bb1name}_{bb2name}_{torsion}"
        print(f"comparing {cage_name} with {old_cage}")

        if "4P6" in old_cage:
            if "ton" in old_cage:
                c = "r"
            elif "toff" in old_cage:
                c = "gray"
        elif "6P12" in old_cage:
            if "ton" in old_cage:
                c = "skyblue"
            elif "toff" in old_cage:
                c = "gold"

        # compare_final_energies(
        #     path1=calculation_done / f"{old}_opt1_omm.out",
        #     path2=calculation_output / f"{i}_opt1_omm.out",
        # )
        e1, e2 = compare_final_energies(
            path1=(calculation_done / f"{old_cage}{old_cage_suffix}_ensemble.json"),
            path2=calculation_output / f"{cage_name}_ensemble.json",
        )

        ax.scatter(
            e1,
            e2,
            c=c,
            marker=m,
            edgecolor="none",
            s=100,
            alpha=alpha,
        )

        new_struct = database.get_molecule(cage_name)
        old_struct = stk.BuildingBlock.init_from_file(
            str(struct_done / f"{old_cage}{old_cage_suffix}_optc.mol")
        )
        structure_comparisons.append(
            f'{str(calculation_output / f"{cage_name}_final.mol")} '
            f'{str(struct_done / f"{old_cage}{old_cage_suffix}_optc.mol")}'
        )

        assert new_struct.get_num_atoms() == old_struct.get_num_atoms()
        assert new_struct.get_num_bonds() == old_struct.get_num_bonds()

        g_measure = cgx.analysis.GeomMeasure(
            target_torsions=(
                cgx.terms.TargetTorsion(
                    search_string=("b1", "a1", "c1", "a1", "b1"),
                    search_estring=("Pb", "Ba", "Ag", "Ba", "Pb"),
                    measured_atom_ids=[0, 1, 3, 4],
                    phi0=openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
                    torsion_k=openmm.unit.Quantity(
                        value=50,
                        unit=openmm.unit.kilojoules_per_mole,
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
                # print("bonds", bd1, bd2, abs(bd1 - bd2))
                # assert np.isclose(bd1, bd2, atol=1e-1, rtol=0)
                ax_bond.scatter(
                    bd1,
                    bd2,
                    c=c,
                    marker=m,
                    edgecolor="none",
                    s=100,
                    alpha=alpha,
                )

        angle_data1 = g_measure.calculate_angles(old_struct)
        angle_data2 = g_measure.calculate_angles(new_struct)
        for i in angle_data1:
            assert i in angle_data2
            assert len(angle_data1[i]) == len(angle_data2[i])
            for bd1, bd2 in zip(angle_data1[i], angle_data2[i]):
                # print(
                #     "angles",
                #     round(bd1, 2),
                #     round(bd2, 2),
                #     round(abs(bd1 - bd2), 2),
                #     round((1 / 2) * 1e2 * (bd1 - bd2) ** 2, 2),
                # )
                # assert np.isclose(bd1, bd2, atol=1, rtol=0)
                if not np.isclose(bd1, bd2, atol=1, rtol=0):
                    print("angles", round(bd1, 2), round(bd2, 2))
                ax_angle.scatter(
                    bd1,
                    bd2,
                    c=c,
                    marker=m,
                    edgecolor="none",
                    s=100,
                    alpha=alpha,
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
                if not np.isclose(bd1, bd2, atol=1, rtol=0):
                    print("torsions", round(bd1, 2), round(bd2, 2))
                ax_tors.scatter(
                    bd1,
                    bd2,
                    c=c,
                    marker=m,
                    edgecolor="none",
                    s=100,
                    alpha=alpha,
                )

        max_diameter1 = g_measure.calculate_max_diameter(old_struct)
        max_diameter2 = g_measure.calculate_max_diameter(new_struct)
        # assert np.isclose(bd1, bd2, atol=1, rtol=0)
        ax_md.scatter(
            max_diameter1,
            max_diameter2,
            c=c,
            marker=m,
            edgecolor="none",
            s=100,
            alpha=alpha,
        )

        radius_gyration1 = g_measure.calculate_radius_gyration(old_struct)
        radius_gyration2 = g_measure.calculate_radius_gyration(new_struct)
        # assert np.isclose(bd1, bd2, atol=1, rtol=0)
        ax_rg.scatter(
            radius_gyration1,
            radius_gyration2,
            c=c,
            marker=m,
            edgecolor="none",
            s=100,
            alpha=alpha,
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
    cmap = {
        ("4P6", "ton"): "r",
        ("4P6", "toff"): "gray",
        ("6P12", "ton"): "skyblue",
        ("6P12", "toff"): "gold",
    }
    for tstr, tors in cmap:
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker=m,
                color="w",
                label=f"{tstr},{tors}",
                markerfacecolor=cmap[(tstr, tors)],
                markersize=7,
                markeredgecolor="none",
                alpha=alpha,
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


def main() -> None:
    """Run script."""
    first_line = f"Usage: {__file__}.py path "
    if not len(sys.argv) == 2:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        path = sys.argv[1]

    prefix = "cg_model_test"

    struct_output = pathlib.Path().absolute() / path / "structures"
    cgx.utilities.check_directory(struct_output)
    calculation_output = pathlib.Path().absolute() / path / "calculations"
    cgx.utilities.check_directory(calculation_output)
    ligand_output = pathlib.Path().absolute() / path / "ligands"
    cgx.utilities.check_directory(ligand_output)

    struct_done = pathlib.Path().absolute() / path / "old_structures"
    calculation_done = pathlib.Path().absolute() / path / "old_calculations"

    # Define bead libraries.
    core_bead = cgx.molecular.CgBead(
        element_string="Ag",
        bead_class="c",
        bead_type="c1",
        coordination=2,
    )
    arm_bead = cgx.molecular.CgBead(
        element_string="Ba",
        bead_class="a",
        bead_type="a1",
        coordination=2,
    )
    binder_bead = cgx.molecular.CgBead(
        element_string="Pb",
        bead_class="b",
        bead_type="b1",
        coordination=2,
    )
    trigonal_bead = cgx.molecular.CgBead(
        element_string="C",
        bead_class="n",
        bead_type="n1",
        coordination=3,
    )
    tetragonal_bead = cgx.molecular.CgBead(
        element_string="Pd",
        bead_class="m",
        bead_type="m1",
        coordination=4,
    )
    present_beads = (
        core_bead,
        arm_bead,
        binder_bead,
        trigonal_bead,
        tetragonal_bead,
    )

    logging.info(f"defining force field for {prefix}")
    forcefieldlibrary_2p3 = define_forcefield_library(
        present_beads=present_beads,
        prefix=prefix + "_2p3",
    )
    forcefieldlibrary_2p4 = define_forcefield_library(
        present_beads=present_beads,
        prefix=prefix + "_2p4",
    )

    logging.info("defining building blocks")
    ditopic = cgx.molecular.TwoC1Arm(bead=core_bead, abead1=arm_bead)
    tritopic = cgx.molecular.ThreeC1Arm(bead=trigonal_bead, abead1=binder_bead)
    tetratopic = cgx.molecular.FourC1Arm(bead=tetragonal_bead, abead1=binder_bead)

    # Define list of topology functions.
    cage_2p3_topologies = {"4P6": stk.cage.FourPlusSix}
    cage_2p4_topologies = {"6P12": stk.cage.M6L12Cube}

    populations = {
        "2p4": {
            "topologies": cage_2p4_topologies,
            "c2": ditopic,
            "cl": tetratopic,
            "fflibrary": forcefieldlibrary_2p4,
        },
        "2p3": {
            "topologies": cage_2p3_topologies,
            "c2": ditopic,
            "cl": tritopic,
            "fflibrary": forcefieldlibrary_2p3,
        },
    }

    database = cgx.utilities.AtomliteDatabase(
        db_file=struct_output / "cg_model_test.db"
    )

    cages = []
    for population in populations:
        logging.info(f"running population {population}")
        popn_dict = populations[population]
        popn_iterator = itertools.product(
            popn_dict["topologies"],
            tuple(popn_dict["fflibrary"].yield_forcefields()),
        )
        for cage_topo_str, forcefield in popn_iterator:
            c2_precursor = popn_dict["c2"]
            cl_precursor = popn_dict["cl"]
            name = (
                f"{cage_topo_str}_{cl_precursor.get_name()}_"
                f"{c2_precursor.get_name()}_"
                f"f{forcefield.get_identifier()}"
            )

            # Optimise building blocks.
            c2_name = f"{c2_precursor.get_name()}_f{forcefield.get_identifier()}"
            c2_building_block = cgx.utilities.optimise_ligand(
                molecule=c2_precursor.get_building_block(),
                name=c2_name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=None,
            )
            c2_building_block.write(str(ligand_output / f"{c2_name}_optl.mol"))

            cl_name = f"{cl_precursor.get_name()}_f{forcefield.get_identifier()}"
            cl_building_block = cgx.utilities.optimise_ligand(
                molecule=cl_precursor.get_building_block(),
                name=cl_name,
                output_dir=calculation_output,
                forcefield=forcefield,
                platform=None,
            )
            cl_building_block.write(str(ligand_output / f"{cl_name}_optl.mol"))

            logging.info(f"building {name}")
            cage = stk.ConstructedMolecule(
                topology_graph=popn_dict["topologies"][cage_topo_str](
                    building_blocks=(c2_building_block, cl_building_block),
                ),
            )

            conformer = optimise_cage(
                molecule=cage,
                name=name,
                output_dir=calculation_output,
                forcefield=forcefield,
                # platform="CPU",
                # platform="CUDA",
                platform=None,
                database=database,
            )
            if conformer is not None:
                conformer.molecule.write(str(struct_output / f"{name}_optc.mol"))

            cages.append(name)

    analysis(
        cages=cages,
        struct_output=struct_output,
        struct_done=struct_done,
        calculation_output=calculation_output,
        calculation_done=calculation_done,
        database=database,
    )


if __name__ == "__main__":
    main()
