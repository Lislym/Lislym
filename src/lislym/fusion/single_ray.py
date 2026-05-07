"""K=1 fallback: single-camera ray solved against kinematic constraints.

When only one camera sees a marker, a single 2D pixel observation does
*not* uniquely determine the 3D point — it only constrains it to a ray
through the camera centre. We resolve the remaining degree of freedom by
intersecting the ray with whichever of the following constraints is most
appropriate for the marker:

* **Distance to a previous position** — works for any marker the tracker
  has seen recently. Picks the depth along the ray that puts the marker
  closest to where it just was.
* **Distance to an anchor** — for the ankle markers we know
  ``|hip - ankle| ≈ thigh + shin`` from the personal calibration. Given
  the hip's freshly triangulated 3D position and the ankle's single ray,
  we solve for the depth that satisfies the bone-length constraint.

Each constraint is reduced to a quadratic ``a*t^2 + b*t + c = 0`` along
the ray parameter ``t``. We pick the positive root closer to the
prediction; if neither root is real, we return the minimum-distance point
on the ray to the prior — effectively projecting the prior onto the ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Ray:
    """A 3D ray parametrised as ``origin + t * direction`` with ``t >= 0``."""

    origin: np.ndarray
    direction: np.ndarray

    def at(self, t: float) -> np.ndarray:
        return self.origin + t * self.direction


def _project_onto_ray(ray: Ray, point: np.ndarray) -> tuple[np.ndarray, float]:
    """Closest point on ``ray`` to ``point`` (always defined; unbounded ``t``)."""
    t = float(np.dot(point - ray.origin, ray.direction))
    return ray.at(t), t


def solve_with_distance_to_anchor(
    ray: Ray,
    anchor: np.ndarray,
    target_distance_m: float,
    prior: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Find the point on ``ray`` at distance ``target_distance_m`` from ``anchor``.

    Returns the point and a one-word reason code:

    * ``"intersect"`` — the sphere of radius ``target_distance_m`` around
      the anchor crosses the ray; the root closer to ``prior`` is chosen.
    * ``"tangent"`` — the sphere just kisses the ray; the unique solution
      is returned.
    * ``"projected"`` — the sphere does not intersect; we return the
      projection of ``prior`` onto the ray, which is the best the K=1
      fallback can offer until another camera sees the marker.
    """
    o = ray.origin - anchor
    d = ray.direction
    a = float(np.dot(d, d))
    b = 2.0 * float(np.dot(d, o))
    c = float(np.dot(o, o) - target_distance_m ** 2)
    disc = b * b - 4.0 * a * c

    if disc < 0:
        proj, _ = _project_onto_ray(ray, prior)
        return proj, "projected"
    if disc == 0:
        t = -b / (2.0 * a)
        return ray.at(t), "tangent"

    sqrt_disc = float(np.sqrt(disc))
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    p1 = ray.at(t1)
    p2 = ray.at(t2)
    d1 = float(np.linalg.norm(p1 - prior))
    d2 = float(np.linalg.norm(p2 - prior))
    return (p1, "intersect") if d1 <= d2 else (p2, "intersect")


def solve_with_distance_to_prior(
    ray: Ray,
    prior: np.ndarray,
    max_step_m: float,
) -> np.ndarray:
    """Pick the point on ``ray`` closest to ``prior`` within ``max_step_m``.

    Returns a position on the ray. If the projection is within
    ``max_step_m`` of the prior, that's the answer; otherwise we clamp
    the move so the tracker doesn't teleport when the K=1 ray happens to
    pass far from the prior.
    """
    proj, t = _project_onto_ray(ray, prior)
    delta = proj - prior
    norm = float(np.linalg.norm(delta))
    if norm <= max_step_m:
        # ``t`` may be negative if the prior is behind the camera; the
        # caller should have flagged that case before getting here.
        return proj
    return prior + delta * (max_step_m / norm)
