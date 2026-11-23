"""Algebra checks for the compact frame-material design, not a solver benchmark.

Compare the existing 18-coordinate potential with a seven-value step reference
and a closed rotational quadratic on unit quaternions. Directional derivatives
are checked against finite differences of the independent coordinate sum.
"""
import argparse
import json
import math
from pathlib import Path
import random


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    scale = math.sqrt(dot(v, v))
    return [x / scale for x in v]


def mul(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return [w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z]


def conjugate(q):
    return [-q[0], -q[1], -q[2], q[3]]


def rotate_inverse(q, v):
    return mul(mul(conjugate(q), list(v)+[0.0]), q)[:3]


def rotational_coordinates(q):
    return [2*q[i]*q[j] for i in range(3) for j in range(4)]


def reference_rotation(q, q0, k, dh):
    c, c0 = rotational_coordinates(q), rotational_coordinates(q0)
    return .5*sum(k[i//4]*x*x + dh[i//4]*(x-c0[i])**2 for i, x in enumerate(c))


def compact_rotation(q, q0, k, dh):
    # Align only for stable subtraction; the underlying products are sign-even.
    ref = q0 if dot(q, q0) >= 0 else [-x for x in q0]
    delta = [q[i]-ref[i] for i in range(4)]
    damping = sum(dh[i]*sum((delta[i]*q[j]+ref[i]*delta[j])**2
                            for j in range(4)) for i in range(3))
    return 2*sum(k[i]*q[i]**2 for i in range(3)) + 2*damping


def quadratic_derivatives(q, q0, k, dh, axis):
    v = [dh[i]*q0[i] for i in range(3)]+[0.0]
    diagonal = [k[i]+dh[i] for i in range(3)]+[0.0]
    def apply(x):
        return [diagonal[i]*x[i]-q0[i]*dot(v, x)-v[i]*dot(q0, x) for i in range(4)]
    gradient = [4*x for x in apply(q)]
    dq = [.5*x for x in mul(list(axis)+[0.0], q)]
    ddq = [-.25*dot(axis, axis)*x for x in q]
    return dot(gradient, dq), 4*dot(dq, apply(dq))+dot(gradient, ddq)


def increment(q, axis, angle):
    return unit(mul([x*math.sin(angle*.5) for x in axis]+[math.cos(angle*.5)], q))


def validate(samples):
    rng = random.Random(20261113)
    peak_energy = peak_gradient = peak_curvature = peak_sign = 0.0
    near_rest = near_pi = 0
    for n in range(samples):
        a, b, q0 = (unit([rng.gauss(0, 1) for _ in range(4)]) for _ in range(3))
        axis = unit([rng.gauss(0, 1) for _ in range(3)])
        if n % 8 in (0, 1, 2):
            q = increment(q0, axis, (0.0, 1.e-9, 1.e-5)[n % 8])
            b = unit(mul(a, q))
            near_rest += 1
        elif n % 8 == 3:
            q = [x*math.sin((math.pi-1.e-9)*.5) for x in axis]+[math.cos((math.pi-1.e-9)*.5)]
            b = unit(mul(a, q))
            near_pi += 1
        q = unit(mul(conjugate(a), b))
        k = [10**rng.uniform(-3, 6) for _ in range(6)]
        dh = [10**rng.uniform(-4, 5) for _ in range(6)]
        if n % 9 == 0:
            dh = [0.0]*6
        d, u0 = ([rng.uniform(-2, 2) for _ in range(3)] for _ in range(2))
        u, v, v0 = rotate_inverse(a, d), rotate_inverse(b, d), rotate_inverse(q0, u0)
        old = .5*sum(k[i%3]*(x/math.sqrt(2))**2 +
                        dh[i%3]*((x-y)/math.sqrt(2))**2
                        for i, (x, y) in enumerate(zip(u+v, u0+v0)))
        old += reference_rotation(q, q0, k[3:], dh[3:])
        new = .25*sum(k[i]*(u[i]**2+v[i]**2) + dh[i]*((u[i]-u0[i])**2+(v[i]-v0[i])**2)
                        for i in range(3)) + compact_rotation(q, q0, k[3:], dh[3:])
        peak_energy = max(peak_energy, abs(old-new)/max(1.0, old, new))
        er = compact_rotation(q, q0, k[3:], dh[3:])
        for sign_q, sign_ref in ((-1, 1), (1, -1), (-1, -1)):
            other = compact_rotation([sign_q*x for x in q], [sign_ref*x for x in q0], k[3:], dh[3:])
            peak_sign = max(peak_sign, abs(er-other)/max(1.0, er, other))
        grad, curvature = quadratic_derivatives(q, q0, k[3:], dh[3:], axis)
        h = .002
        f = [reference_rotation(increment(q, axis, t*h), q0, k[3:], dh[3:]) for t in (-2, -1, 0, 1, 2)]
        fd_grad = (f[0]-8*f[1]+8*f[3]-f[4])/(12*h)
        fd_curvature = (-f[0]+16*f[1]-30*f[2]+16*f[3]-f[4])/(12*h*h)
        scale = max(1.0, *[k[i]+dh[i] for i in range(3, 6)])
        peak_gradient = max(peak_gradient, abs(grad-fd_grad)/scale)
        peak_curvature = max(peak_curvature, abs(curvature-fd_curvature)/scale)
        if not all(math.isfinite(x) for x in (old, new, grad, curvature)) or new < 0:
            raise AssertionError("Invalid compact potential")
    checks = {"energyIdentity": peak_energy < 5.e-12, "signInvariance": peak_sign < 5.e-12,
              "angularGradient": peak_gradient < 1.e-7, "angularCurvature": peak_curvature < 1.e-6}
    return {"kind": "design_algebra_only", "samples": samples, "nearRestSamples": near_rest,
            "nearPiSamples": near_pi, "relativeEnergyError": peak_energy, "signError": peak_sign,
            "coefficientScaledGradientError": peak_gradient, "coefficientScaledCurvatureError": peak_curvature,
            "checks": checks, "pass": all(checks.values()),
            "limitations": "No full endpoint/cross-block derivative, production integration, physical quality or speed qualification."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    report = validate(args.samples)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["pass"] else 1)
