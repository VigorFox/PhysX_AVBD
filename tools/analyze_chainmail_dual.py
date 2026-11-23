#!/usr/bin/env python3
"""Linear counterexample for pose-only Chebyshev plus an inexact AL dual.

This analyzes a pinned scalar chain, not a second physical solver or a
replacement for the native Chainmail gates. A GS sweep is the exact
lower-triangular splitting of M/h^2 + rho J^T J. The multiplier is updated
before the pose-only extrapolation, matching the backend's scheduling.
"""
import json
import numpy as np


def spectral_radius(dual_step, decay, rho=1.0e4, count=12):
    mass = 4.0 / 3.0 * np.pi * 0.12**3 * 3.0
    mass_h2 = mass * 60.0**2
    jacobian = np.zeros((count + 1, count))
    jacobian[0, 0] = jacobian[-1, -1] = 1.0
    for index in range(1, count):
        jacobian[index, index] = 1.0
        jacobian[index, index - 1] = -1.0
    metric = mass_h2 * np.eye(count) + rho * jacobian.T @ jacobian
    lower = np.tril(metric)
    gs_pose = np.linalg.solve(lower, -np.triu(metric, 1))
    gs_dual = np.linalg.solve(lower, -jacobian.T)
    omega = 2.0 / (1.0 + np.sqrt(1.0 - 0.92**2))
    recurrence = np.block([
        [omega * gs_pose, (1.0 - omega) * np.eye(count), omega * gs_dual],
        [np.eye(count), np.zeros((count, count)), np.zeros((count, count + 1))],
        [dual_step * jacobian @ gs_pose, np.zeros((count + 1, count)),
         decay * np.eye(count + 1) + dual_step * jacobian @ gs_dual],
    ])
    return float(np.max(np.abs(np.linalg.eigvals(recurrence))))


if __name__ == "__main__":
    mass_h2 = 4.0 / 3.0 * np.pi * 0.12**3 * 3.0 * 60.0**2
    bounded_step = min(mass_h2, 1.0e4**2 / (1.0e4 + mass_h2))
    old = spectral_radius(1.0e4, 1.0)
    bounded = spectral_radius(bounded_step, 0.99)
    print(json.dumps(dict(mass_over_h_squared=mass_h2,
                         full_dual_spectral_radius=old,
                         bounded_dual_spectral_radius=bounded), indent=2))
    assert old > 1.0 and bounded < 1.0
