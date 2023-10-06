#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Distributed under the terms of the MIT License.

"""
Script to run OpenMM tests.

Author: Andrew Tarzia

"""

import logging
import sys
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import stk
from cgexplore.assigned_system import AssignedSystem
from cgexplore.beads import bead_library_check, CgBead
from cgexplore.forcefield import Forcefield
from cgexplore.openmm_optimizer import CGOMMDynamics, CGOMMOptimizer
from cgexplore.utilities import (
    angle_between,
    check_directory,
    get_dihedral,
)
from cgexplore.angles import TargetAngle
from cgexplore.bonds import TargetBond
from cgexplore.nonbonded import TargetNonbonded
from cgexplore.torsions import TargetTorsion
from openmm import openmm
from rdkit import RDLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
RDLogger.DisableLog("rdApp.*")


def bond_function(x, k, r0):
    return (1 / 2) * k * (x - r0) ** 2


def angle_function(x, k, theta0):
    return (1 / 2) * k * (x - theta0) ** 2


def torsion_function(x, k, theta0, n):
    return k * (1 + np.cos(n * x - theta0))


def nonbond_function(x, epsilon1, sigma1, epsilon2, sigma2):
    return np.sqrt(epsilon1 * epsilon2) * ((sigma1 + sigma2) / (2 * x)) ** 12


def points_in_circum(r, n=100):
    return [
        (
            np.cos(2 * np.pi / n * x) * r,
            np.sin(2 * np.pi / n * x) * r,
        )
        for x in range(0, n + 1)
    ]


def define_force_field(c_bead, calculation_output):
    force_field = Forcefield(
        identifier=0,
        prefix="omm",
        present_beads=(c_bead,),
        bond_targets=(
            TargetBond(
                class1="c",
                class2="c",
                eclass1="Ag",
                eclass2="Ag",
                bond_r=openmm.unit.Quantity(
                    value=2.0, unit=openmm.unit.angstrom
                ),
                bond_k=openmm.unit.Quantity(
                    value=1e5,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.nanometer**2,
                ),
            ),
        ),
        angle_targets=(
            TargetAngle(
                class1="c",
                class2="c",
                class3="c",
                eclass1="Ag",
                eclass2="Ag",
                eclass3="Ag",
                angle=openmm.unit.Quantity(value=90, unit=openmm.unit.degrees),
                angle_k=openmm.unit.Quantity(
                    value=1e2,
                    unit=openmm.unit.kilojoule
                    / openmm.unit.mole
                    / openmm.unit.radian**2,
                ),
            ),
        ),
        torsion_targets=(
            TargetTorsion(
                search_string=("c", "c", "c", "c"),
                search_estring=("Ag", "Ag", "Ag", "Ag"),
                measured_atom_ids=[0, 1, 2, 3],
                phi0=openmm.unit.Quantity(value=180, unit=openmm.unit.degrees),
                torsion_k=openmm.unit.Quantity(
                    value=50,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
                torsion_n=1,
            ),
        ),
        nonbonded_targets=(
            TargetNonbonded(
                bead_class="c",
                bead_element="Ag",
                epsilon=openmm.unit.Quantity(
                    value=10.0,
                    unit=openmm.unit.kilojoule / openmm.unit.mole,
                ),
                sigma=openmm.unit.Quantity(
                    value=1.0, unit=openmm.unit.angstrom
                ),
                force="custom-excl-vol",
            ),
        ),
        vdw_bond_cutoff=2,
    )
    return force_field


def random_test(c_bead, force_field, calculation_output):
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{c_bead.element_string}]([{c_bead.element_string}])"
            f"[{c_bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [1, 0, 0]],
    )

    temperature = 10

    runs = {
        0: (None, 1000, "-", "CPU"),
        1: ("k", 1000, "-", "CPU"),
        2: ("green", 2000, "-", "CPU"),
        3: ("r", 2000, "--", "CPU"),
        4: ("b", None, "-", "CPU"),
        5: ("gold", None, "--", "CPU"),
        6: ("gray", 2000, "--", None),
    }

    assigned_system = force_field.assign_terms(
        molecule=linear_bb,
        name="rt",
        output_dir=calculation_output,
    )

    tdict = {}
    for run in runs:
        tdict[run] = {}

        logging.info(f"running MD random test; {run}")
        opt = CGOMMDynamics(
            fileprefix=f"mdr_{run}",
            output_dir=calculation_output,
            temperature=temperature,
            random_seed=runs[run][1],
            num_steps=2000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
            platform=runs[run][3],
        )
        trajectory = opt.run_dynamics(assigned_system)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            tdict[run][timestep] = (meas_temp, pot_energy, posmat)

    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(8, 8))

    for run in tdict:
        if run == 0:
            continue
        data = tdict[run]
        zero_data = tdict[0]
        xs = []
        tdata = []
        rdata = []
        edata = []
        for timestep in data:
            xs.append(timestep)
            m_temp = data[timestep][0]
            z_temp = zero_data[timestep][0]
            tdata.append(m_temp - z_temp)
            m_pe = data[timestep][1]
            z_pe = zero_data[timestep][1]
            edata.append(m_pe - z_pe)
            m_posmat = data[timestep][2]
            z_posmat = zero_data[timestep][2]
            rdata.append(
                np.sqrt(np.sum((m_posmat - z_posmat) ** 2) / len(m_posmat))
            )

        axs[0].plot(
            xs,
            rdata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )
        axs[1].plot(
            xs,
            tdata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )
        axs[2].plot(
            xs,
            edata,
            c=runs[run][0],
            lw=2,
            linestyle=runs[run][2],
            # s=30,
            # edgecolor="none",
            alpha=1.0,
            label=f"run {run}",
        )

    # ax.axhline(y=0, c="k", lw=2, linestyle="--")
    # ax.axvline(x=bead.angle_centered, c="k", lw=2)

    axs[0].tick_params(axis="both", which="major", labelsize=16)
    axs[0].set_xlabel("timestep [s]", fontsize=16)
    axs[0].set_ylabel("RMSD [A]", fontsize=16)

    axs[1].tick_params(axis="both", which="major", labelsize=16)
    axs[1].set_ylabel("deltaT [K]", fontsize=16)

    axs[2].tick_params(axis="both", which="major", labelsize=16)
    axs[2].set_ylabel("deltaE [kJmol-1]", fontsize=16)
    axs[2].legend(fontsize=16)

    fig.tight_layout()
    fig.savefig(
        "random_test.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()
    raise SystemExit()


def test1(c_bead, force_field, calculation_output):
    linear_bb = stk.BuildingBlock(
        smiles=f"[{c_bead.element_string}][{c_bead.element_string}]",
        position_matrix=[[0, 0, 0], [2, 0, 0]],
    )
    bond_k = 1e5
    bond_r = 2.0
    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    assigned_system = force_field.assign_terms(
        molecule=linear_bb,
        name="t1",
        output_dir=calculation_output,
    )

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test1; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl1_{temp}",
            output_dir=calculation_output,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
            platform=None,
        )

        trajectory = opt.run_dynamics(assigned_system)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            distance = np.linalg.norm(posmat[1] - posmat[0])
            tdict[temp][timestep] = (meas_temp, pot_energy, distance)

    coords = np.linspace(0, 5, 20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l1_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[1] = np.array([1, 0, 0]) * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om1",
            output_dir=calculation_output,
            platform=None,
        )
        energy = opt.calculate_energy(
            AssignedSystem(
                molecule=new_bb,
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            )
        )
        distance = np.linalg.norm(new_posmat[1] - new_posmat[0])
        xys.append(
            (
                distance,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"{bond_r} A, {bond_k} kJ/mol/nm2", fontsize=16.0)

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        bond_function(x / 10, bond_k, bond_r / 10),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        "l1.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test3(c_bead, force_field, calculation_output):
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{c_bead.element_string}]([{c_bead.element_string}])"
            f"[{c_bead.element_string}]"
        ),
        position_matrix=[[0, 0, 0], [2, 0, 0], [1, 0, 0]],
    )
    angle_k = 1e2
    angle_c = 90
    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    assigned_system = force_field.assign_terms(
        molecule=linear_bb,
        name="t3",
        output_dir=calculation_output,
    )

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test3; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl3_{temp}",
            output_dir=calculation_output,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
            platform=None,
        )

        trajectory = opt.run_dynamics(assigned_system)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            vector1 = posmat[1] - posmat[0]
            vector2 = posmat[2] - posmat[0]
            angle = np.degrees(angle_between(vector1, vector2))
            tdict[temp][timestep] = (meas_temp, pot_energy, angle)

    coords = points_in_circum(r=2, n=100)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l3_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[2] = np.array([coord[0], coord[1], 0])
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om3",
            output_dir=calculation_output,
            platform=None,
        )
        energy = opt.calculate_energy(
            AssignedSystem(
                molecule=new_bb,
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            )
        )
        pos_mat = new_bb.get_position_matrix()
        vector1 = pos_mat[1] - pos_mat[0]
        vector2 = pos_mat[2] - pos_mat[0]
        angle = np.degrees(angle_between(vector1, vector2))
        xys.append(
            (
                angle,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"{angle_c} [deg], {angle_k}" " kJ/mol/radian2", fontsize=16.0
    )

    x = np.linspace(0, 180, 100)
    ax.plot(
        x,
        angle_function(np.radians(x), angle_k, np.radians(angle_c)),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    ax.axvline(x=angle_c, c="k", lw=2)

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("angle [theta]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        "l3.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test4(c_bead, force_field, calculation_output):
    linear_bb = stk.BuildingBlock(
        smiles=(
            f"[{c_bead.element_string}][{c_bead.element_string}]"
            f"[{c_bead.element_string}][{c_bead.element_string}]"
        ),
        position_matrix=[[0, 2, 0], [0, 0, 0], [2, 0, 0], [2, 2, 0]],
    )

    torsion_k = 50
    phi0 = 180
    tcol = {
        700: "k",
        300: "gold",
        100: "orange",
        10: "green",
    }

    assigned_system = force_field.assign_terms(
        molecule=linear_bb,
        name="t4",
        output_dir=calculation_output,
    )

    tdict = {}
    for temp in tcol:
        tdict[temp] = {}
        logging.info(f"running MD test4; {temp}")
        opt = CGOMMDynamics(
            fileprefix=f"mdl4_{temp}",
            output_dir=calculation_output,
            temperature=temp,
            random_seed=1000,
            num_steps=10000,
            time_step=1 * openmm.unit.femtoseconds,
            friction=1.0 / openmm.unit.picosecond,
            reporting_freq=100,
            traj_freq=100,
            platform=None,
        )

        trajectory = opt.run_dynamics(assigned_system)

        traj_log = trajectory.get_data()
        for conformer in trajectory.yield_conformers():
            timestep = conformer.timestep
            row = traj_log[traj_log['#"Step"'] == timestep].iloc[0]
            meas_temp = float(row["Temperature (K)"])
            pot_energy = float(row["Potential Energy (kJ/mole)"])
            posmat = conformer.molecule.get_position_matrix()
            torsion = get_dihedral(
                pt1=posmat[0],
                pt2=posmat[1],
                pt3=posmat[2],
                pt4=posmat[3],
            )
            tdict[temp][timestep] = (meas_temp, pot_energy, torsion)

    coords = points_in_circum(r=2, n=20)
    xys = []
    for i, coord in enumerate(coords):
        name = f"l4_{i}"
        new_posmat = linear_bb.get_position_matrix()
        new_posmat[3] = np.array([2, coord[0], coord[1]])
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om4",
            output_dir=calculation_output,
            platform=None,
        )
        energy = opt.calculate_energy(
            AssignedSystem(
                molecule=new_bb,
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            )
        )
        pos_mat = new_bb.get_position_matrix()
        torsion = get_dihedral(
            pt1=pos_mat[0],
            pt2=pos_mat[1],
            pt3=pos_mat[2],
            pt4=pos_mat[3],
        )
        xys.append(
            (
                torsion,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"n: 1, k={torsion_k} [kJ/mol], phi0={phi0} [deg]",
        fontsize=16.0,
    )

    torsions = [i[0] for i in xys]
    x = np.linspace(min(torsions), max(torsions), 100)
    ax.plot(
        x,
        torsion_function(np.radians(x), torsion_k, np.radians(phi0), 1),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )
    ax.axhline(y=0.0, c="k", lw=2, linestyle="--")
    ax.axvline(x=0.0, c="k", lw=2)

    for temp in tdict:
        data = tdict[temp]
        ax.scatter(
            [data[i][2] for i in data],
            [data[i][1] for i in data],
            c=tcol[temp],
            s=30,
            edgecolor="none",
            alpha=1.0,
            label=f"{temp} K",
        )

    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("torsion [theta]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        "l4.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def test5(c_bead, force_field, calculation_output):
    linear_bb = stk.BuildingBlock(
        smiles=f"[{c_bead.element_string}].[{c_bead.element_string}]",
        position_matrix=[[0, 0, 0], [1, 0, 0]],
    )

    sigma = 1.0
    epsilon = 10

    assigned_system = force_field.assign_terms(
        molecule=linear_bb,
        name="t5",
        output_dir=calculation_output,
    )

    coords = list(np.linspace(sigma, 2, 20)) + list(np.linspace(3, 10, 7))
    xys = []
    for i, coord in enumerate(coords):
        name = f"l5_{i}"
        new_posmat = linear_bb.get_position_matrix() * coord
        new_bb = linear_bb.with_position_matrix(new_posmat)
        new_bb.write(str(calculation_output / f"{name}.mol"))
        logging.info(f"evaluating {name}")
        opt = CGOMMOptimizer(
            fileprefix=f"{name}_om5",
            output_dir=calculation_output,
            platform=None,
        )
        energy = opt.calculate_energy(
            AssignedSystem(
                molecule=new_bb,
                force_field_terms=assigned_system.force_field_terms,
                system_xml=assigned_system.system_xml,
                topology_xml=assigned_system.topology_xml,
                bead_set=assigned_system.bead_set,
                vdw_bond_cutoff=assigned_system.vdw_bond_cutoff,
            )
        )
        distance = np.linalg.norm(new_posmat[1] - new_posmat[0])
        xys.append(
            (
                distance,
                energy.value_in_unit(openmm.unit.kilojoules_per_mole),
            )
        )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        f"sigma: {sigma} A, epsilon: {epsilon} kJ/mol",
        fontsize=16.0,
    )

    distances = [i[0] for i in xys]
    x = np.linspace(min(distances), max(distances), 100)
    ax.plot(
        x,
        nonbond_function(
            x / 10,
            epsilon,
            sigma / 10,
            epsilon,
            sigma / 10,
        ),
        c="r",
        lw=2,
        label="analytical",
    )

    ax.scatter(
        [i[0] for i in xys],
        [i[1] for i in xys],
        c="skyblue",
        s=120,
        edgecolor="k",
        alpha=1.0,
        label="numerical",
    )
    ax.axhline(y=0, c="k", lw=2, linestyle="--")
    # ax.axvline(x=rmin, c="gray", lw=2, linestyle="--")
    ax.axvline(x=sigma, lw=2, linestyle="--", c="k")
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.set_xlabel("distance [A]", fontsize=16)
    ax.set_ylabel("energy [kJmol-1]", fontsize=16)
    ax.legend(fontsize=16)
    fig.tight_layout()
    fig.savefig(
        "l5.png",
        dpi=720,
        bbox_inches="tight",
    )
    plt.close()


def main():
    first_line = f"Usage: {__file__}.py path "
    if not len(sys.argv) == 2:
        logging.info(f"{first_line}")
        sys.exit()
    else:
        path = sys.argv[1]

    struct_output = pathlib.Path().absolute() / path / "ommtest"
    check_directory(struct_output)
    calculation_output = pathlib.Path().absolute() / path / "ommcalculations"
    check_directory(calculation_output)
    ligand_output = pathlib.Path().absolute() / path / "ligands"
    check_directory(ligand_output)

    # Define bead libraries.
    c_bead = CgBead(
        element_string="Ag",
        bead_class="c",
        bead_type="c1",
        coordination=2,
    )
    full_bead_library = (c_bead,)
    bead_library_check(full_bead_library)
    force_field = define_force_field(c_bead, calculation_output)

    test1(c_bead, force_field, calculation_output)
    test3(c_bead, force_field, calculation_output)
    test4(c_bead, force_field, calculation_output)
    test5(c_bead, force_field, calculation_output)
    random_test(c_bead, force_field, calculation_output)


if __name__ == "__main__":
    main()
