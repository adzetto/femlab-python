from __future__ import annotations

import math

import numpy as np

from .. import init, kbar, qbar, solve_lag_general


def _k_truss(area: float, modulus: float, length: float, alpha: float):
    c = math.cos(alpha)
    s = math.sin(alpha)
    transform = np.array(
        [
            [c, s, 0.0, 0.0],
            [-s, c, 0.0, 0.0],
            [0.0, 0.0, c, s],
            [0.0, 0.0, -s, c],
        ],
        dtype=float,
    )
    stiffness = area * modulus / length * np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return stiffness, transform


def ex_lag_mult_data():
    areas = np.array([1.0, 1.0, 1.0], dtype=float)
    moduli = np.array([64.0, 64.0, 64.0], dtype=float)
    lengths = np.array([4.0, 4.0, 6.0], dtype=float)
    angles = np.array([math.acos(3.0 / 4.0), -math.acos(3.0 / 4.0), 0.0], dtype=float)

    # Node layout implied by the original Dvec/L/A/alpha definition.
    coordinates = np.array(
        [
            [0.0, 0.0],
            [6.0, 0.0],
            [3.0, math.sqrt(7.0)],
        ],
        dtype=float,
    )
    topology = np.array([[1, 3, 1], [3, 2, 2], [1, 2, 3]], dtype=int)
    dof_map = np.array([[1, 2, 5, 6], [5, 6, 3, 4], [1, 2, 3, 4]], dtype=int)
    materials = np.column_stack([areas, moduli])

    constraint_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.0,
                -math.sin(math.radians(60.0)),
                math.cos(math.radians(60.0)),
                0.0,
                0.0,
            ],
        ],
        dtype=float,
    )
    constraint_rhs = np.zeros((constraint_matrix.shape[0], 1), dtype=float)

    load = np.zeros((6, 1), dtype=float)
    load[5, 0] = -10.0

    return {
        "A": areas,
        "E_modulus": moduli,
        "L": lengths,
        "alpha": angles,
        "X": coordinates,
        "T": topology,
        "Dvec": dof_map,
        "G": materials,
        "constraint_matrix": constraint_matrix,
        "constraint_rhs": constraint_rhs,
        "P": load,
        "dof": 2,
    }


def run_ex_lag_mult():
    data = ex_lag_mult_data()
    stiffness, _, q = init(data["X"].shape[0], data["dof"], use_sparse=False)
    stiffness = kbar(stiffness, data["T"], data["X"], data["G"])
    displacement, lagrange = solve_lag_general(
        stiffness,
        data["P"],
        data["constraint_matrix"],
        data["constraint_rhs"],
        return_lagrange=True,
    )
    q, stress, strain = qbar(q, data["T"], data["X"], data["G"], displacement)

    member_forces = np.zeros((4, data["T"].shape[0]), dtype=float)
    local_displacements = np.zeros_like(member_forces)
    for e in range(data["T"].shape[0]):
        k_local, transform = _k_truss(
            float(data["A"][e]),
            float(data["E_modulus"][e]),
            float(data["L"][e]),
            float(data["alpha"][e]),
        )
        ueg = displacement[data["Dvec"][e] - 1]
        ue_local = transform @ ueg
        local_displacements[:, e] = ue_local[:, 0]
        member_forces[:, e] = (k_local @ ue_local)[:, 0]

    reactions = stiffness[np.array([0, 1, 2, 3]), :] @ displacement
    constraint_residual = (
        data["constraint_matrix"] @ displacement - data["constraint_rhs"]
    )

    return {
        "U": displacement,
        "Lag": lagrange,
        "R": reactions,
        "member_forces": member_forces,
        "local_displacements": local_displacements,
        "constraint_residual": constraint_residual,
        "K": stiffness,
        "q": q,
        "S": stress,
        "E": strain,
        "data": data,
    }


__all__ = ["ex_lag_mult_data", "run_ex_lag_mult"]
