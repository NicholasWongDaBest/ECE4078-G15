"""
path_planner.py -- RRT* motion planning for the fruit-searching robot (Milestone 3).

Deliberately robot-independent: it works on plain (x, y) obstacle circles and
arena bounds, and returns a list of (x, y) waypoints. That means it can be
unit-tested off-robot (see the __main__ block below, which plans a full route
through search_list.txt against truemap.txt and saves a plot) before ever being
wired into auto_fruit_search.py's drive_to_point()/get_robot_pose().

Coordinate convention matches the rest of the codebase: world-frame (x, y) in
metres. Robot heading is handled separately by drive_to_point() (turn-then-drive)
-- this module only reasons about the robot as a point with an inflated
collision radius, which is the standard simplification for this kind of planner.
"""

import csv
import math
import random

import numpy as np

# Arena size + boundary rule, confirmed with the team 2026-09-08:
#   - truemap_generator/config.yaml's 2.5m is measured to the INNER edge of the
#     boundary tape (the "playable" square).
#   - The tape itself is 2cm wide, and the team is allowed up to half the
#     robot's footprint on top of the tape before it counts as out-of-bounds --
#     so the true legal limit for the robot's CENTRE is the tape's OUTER edge
#     plus one robot radius (at that point exactly half the footprint sits
#     past the tape).
ARENA_INNER_SIZE = 2.5   # m, inner-edge-to-inner-edge of the boundary tape
TAPE_WIDTH = 0.02        # m
ROBOT_RADIUS = 0.095     # m, measured centre-to-widest-point of the PiBot chassis

_outer_tape_half_extent = ARENA_INNER_SIZE / 2 + TAPE_WIDTH     # 1.27m
_center_half_extent = _outer_tape_half_extent + ROBOT_RADIUS    # 1.365m
ARENA_BOUNDS = ((-_center_half_extent, _center_half_extent),
                 (-_center_half_extent, _center_half_extent))

# ArUco marker block size is 0.06m (config.yaml) -> ~half-diagonal as a circle.
DEFAULT_MARKER_RADIUS = 0.05


# ---------------------------------------------------------------------------
# Obstacle model
# ---------------------------------------------------------------------------

def load_object_radii(csv_path="object_list.csv"):
    """
    Per-fruit collision radius, derived from object_list.csv's length/width
    measurements (half the diagonal of the footprint), instead of one generic
    radius for every object type.
    """
    radii = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            length = float(row["length(m)"])
            width = float(row["width(m)"])
            radii[row["object"]] = math.hypot(length, width) / 2
    return radii


def build_obstacles(aruco_positions, object_positions, robot_radius,
                     marker_radius=DEFAULT_MARKER_RADIUS, object_radii=None,
                     default_object_radius=0.08, safety_margin=0.05, exclude=None):
    """
    Build the list of inflated circular obstacles the planner must avoid.

    @param aruco_positions: (10, 2) array of ArUco marker (x, y) positions
    @param object_positions: dict {object_name: (x, y)} for every object to
        treat as an obstacle. Do NOT include the current navigation target here
        -- leave it out with `exclude` so the goal doesn't block its own approach.
    @param robot_radius: robot footprint radius (m), centre to widest point
    @param marker_radius: physical half-size of an ArUco marker block (m)
    @param object_radii: optional {object_name: radius} from load_object_radii();
        falls back to `default_object_radius` for any name not in the dict
    @param safety_margin: extra clearance on top of the geometric sum (m), to
        absorb SLAM/pose noise -- e.g. your M2 fruit-pose RMSE
    @param exclude: object name to leave out (typically the current target)
    @return: (N, 3) array of [x, y, inflated_radius]
    """
    object_radii = object_radii or {}
    circles = []

    for ax, ay in aruco_positions:
        circles.append([ax, ay, robot_radius + marker_radius + safety_margin])

    for name, (ox, oy) in object_positions.items():
        if exclude is not None and name == exclude:
            continue
        obj_r = object_radii.get(name, default_object_radius)
        circles.append([ox, oy, robot_radius + obj_r + safety_margin])

    return np.array(circles, dtype=float) if circles else np.empty((0, 3))


def point_in_collision(point, obstacles):
    """True if `point` (x, y) lies inside (or on) any inflated obstacle circle."""
    if len(obstacles) == 0:
        return False
    d = np.hypot(obstacles[:, 0] - point[0], obstacles[:, 1] - point[1])
    return bool(np.any(d <= obstacles[:, 2]))


def segment_in_collision(p1, p2, obstacles):
    """
    True if the straight segment p1->p2 comes within any obstacle's inflated
    radius. Uses the exact closest-point-on-segment-to-centre distance for each
    obstacle (vectorised), rather than marching samples along the segment.
    """
    if len(obstacles) == 0:
        return False
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    seg = p2 - p1
    seg_len2 = float(seg @ seg)

    centres = obstacles[:, :2]
    radii = obstacles[:, 2]

    if seg_len2 < 1e-12:
        d = np.hypot(centres[:, 0] - p1[0], centres[:, 1] - p1[1])
        return bool(np.any(d <= radii))

    t = np.clip(((centres - p1) @ seg) / seg_len2, 0.0, 1.0)
    closest = p1 + np.outer(t, seg)
    d = np.hypot(closest[:, 0] - centres[:, 0], closest[:, 1] - centres[:, 1])
    return bool(np.any(d <= radii))


def dist_between(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


# ---------------------------------------------------------------------------
# RRT*
# ---------------------------------------------------------------------------

class _Tree:
    """Plain parallel-array RRT* tree -- simple to reason about and to explain in the viva."""

    def __init__(self, root):
        self.positions = [np.asarray(root, dtype=float)]
        self.parents = [-1]
        self.costs = [0.0]

    def nearest(self, point):
        pts = np.array(self.positions)
        d = np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])
        idx = int(np.argmin(d))
        return idx, float(d[idx])

    def near(self, point, radius):
        pts = np.array(self.positions)
        d = np.hypot(pts[:, 0] - point[0], pts[:, 1] - point[1])
        return list(np.where(d <= radius)[0])

    def add(self, point, parent_idx, cost):
        self.positions.append(np.asarray(point, dtype=float))
        self.parents.append(parent_idx)
        self.costs.append(cost)
        return len(self.positions) - 1

    def path_to_root(self, idx):
        path = []
        while idx != -1:
            path.append(self.positions[idx])
            idx = self.parents[idx]
        path.reverse()
        return path


def rrt_star(start, goal, obstacles, bounds=ARENA_BOUNDS, step_size=0.15, goal_bias=0.1,
             rewire_radius=0.3, max_iter=2000, goal_tolerance=0.05, rng=None):
    """
    Plan a collision-free path from `start` to `goal` with RRT*.

    @param start, goal: (x, y) tuples/arrays, metres
    @param obstacles: (N, 3) array of [x, y, radius] from build_obstacles()
    @param bounds: ((xmin, xmax), (ymin, ymax)) arena limits, metres
    @param step_size: max distance to extend the tree per iteration (m)
    @param goal_bias: probability of sampling the goal directly each iteration
    @param rewire_radius: neighbourhood radius considered for RRT*'s rewiring (m)
    @param max_iter: sampling budget
    @param goal_tolerance: how close a node must get to `goal` to be accepted (m)
    @param rng: optional random.Random instance, for reproducible tests
    @return: list of (x, y) waypoints start -> goal (inclusive), or None if no
        collision-free path was found within the iteration budget
    """
    rng = rng or random
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    (xmin, xmax), (ymin, ymax) = bounds

    if point_in_collision(start, obstacles):
        raise ValueError("start point is inside an inflated obstacle -- check robot pose/obstacle radii")
    if point_in_collision(goal, obstacles):
        raise ValueError("goal point is inside an inflated obstacle -- pick a different standoff point")

    tree = _Tree(start)
    best_goal_idx = None
    best_goal_cost = math.inf

    for _ in range(max_iter):
        sample = goal if rng.random() < goal_bias else np.array(
            [rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])

        nearest_idx, _ = tree.nearest(sample)
        nearest_pt = tree.positions[nearest_idx]
        direction = sample - nearest_pt
        dist = float(np.hypot(*direction))
        if dist < 1e-9:
            continue
        new_pt = nearest_pt + direction / dist * min(step_size, dist)

        if not (xmin <= new_pt[0] <= xmax and ymin <= new_pt[1] <= ymax):
            continue
        if point_in_collision(new_pt, obstacles) or segment_in_collision(nearest_pt, new_pt, obstacles):
            continue

        # choose the cheapest collision-free parent among nearby nodes (RRT*, part 1)
        near_idxs = tree.near(new_pt, rewire_radius)
        best_parent = nearest_idx
        best_cost = tree.costs[nearest_idx] + dist_between(nearest_pt, new_pt)
        for idx in near_idxs:
            candidate_cost = tree.costs[idx] + dist_between(tree.positions[idx], new_pt)
            if candidate_cost < best_cost and not segment_in_collision(tree.positions[idx], new_pt, obstacles):
                best_parent = idx
                best_cost = candidate_cost

        new_idx = tree.add(new_pt, best_parent, best_cost)

        # rewire nearby nodes through the new node if that's now cheaper (RRT*, part 2)
        for idx in near_idxs:
            if idx == best_parent:
                continue
            candidate_cost = best_cost + dist_between(new_pt, tree.positions[idx])
            if candidate_cost < tree.costs[idx] and not segment_in_collision(new_pt, tree.positions[idx], obstacles):
                tree.parents[idx] = new_idx
                tree.costs[idx] = candidate_cost

        if dist_between(new_pt, goal) <= goal_tolerance and best_cost < best_goal_cost:
            best_goal_idx = new_idx
            best_goal_cost = best_cost

    if best_goal_idx is None:
        return None

    path = tree.path_to_root(best_goal_idx)
    path.append(goal)
    return [tuple(p) for p in path]


# ---------------------------------------------------------------------------
# Path smoothing
# ---------------------------------------------------------------------------

def smooth_path(path, obstacles, iterations=100, rng=None):
    """
    Shortcut-smooth a path: repeatedly try connecting two non-adjacent waypoints
    directly, keeping the shortcut if the straight line is collision-free. This
    is what turns RRT*'s slightly wiggly output into a small number of straight
    segments the robot can actually execute cleanly -- fewer turns means less
    opportunity for encoder drift to compound (see M1 notes on swerving from
    stacked corrections).
    """
    rng = rng or random
    path = list(path)
    for _ in range(iterations):
        if len(path) <= 2:
            break
        i, j = sorted(rng.sample(range(len(path)), 2))
        if j - i < 2:
            continue
        if not segment_in_collision(path[i], path[j], obstacles):
            path = path[:i + 1] + path[j:]
    return path


# ---------------------------------------------------------------------------
# Standoff point (goal selection near a target object)
# ---------------------------------------------------------------------------

def standoff_point(target, obstacles, bounds=ARENA_BOUNDS, standoff_dist=0.3,
                    reference=None, n_directions=16):
    """
    Pick a point `standoff_dist` from `target` that is collision-free and inside
    the arena bounds, to use as the RRT* goal -- NOT the target's own coordinate,
    which sits inside its own inflated obstacle circle and can never be reached
    directly. standoff_dist=0.3 leaves margin inside the 0.4m qualifying radius.

    Tries `n_directions` candidate points around the target and, if `reference`
    (typically the robot's current position) is given, prefers whichever
    candidate is closest to it, so the resulting approach path tends to be shorter.
    """
    (xmin, xmax), (ymin, ymax) = bounds
    target = np.asarray(target, dtype=float)
    candidates = []
    for k in range(n_directions):
        theta = 2 * math.pi * k / n_directions
        pt = target + standoff_dist * np.array([math.cos(theta), math.sin(theta)])
        if not (xmin <= pt[0] <= xmax and ymin <= pt[1] <= ymax):
            continue
        if point_in_collision(pt, obstacles):
            continue
        candidates.append(pt)

    if not candidates:
        return None

    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        candidates.sort(key=lambda p: dist_between(p, reference))

    return tuple(candidates[0])


# ---------------------------------------------------------------------------
# Off-robot verification (no robot connection needed for this)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Off-robot RRT* planner test")
    parser.add_argument("--map", type=str, default="truemap.txt")
    parser.add_argument("--search-list", type=str, default="search_list.txt")
    parser.add_argument("--object-list", type=str, default="object_list.csv")
    parser.add_argument("--robot-radius", type=float, default=ROBOT_RADIUS,
                         help="PiBot footprint radius, centre to widest point (measured)")
    parser.add_argument("--out", type=str, default="lab_output/m3_planner_test.png")
    args = parser.parse_args()

    with open(args.map, "r") as f:
        gt = json.load(f)
    aruco_positions = np.array([[v["x"], v["y"]] for k, v in gt.items() if k.startswith("aruco")])
    object_positions = {k[:-2]: (v["x"], v["y"]) for k, v in gt.items() if not k.startswith("aruco")}
    object_radii = load_object_radii(args.object_list)

    with open(args.search_list, "r") as f:
        search_list = [line.strip() for line in f if line.strip()]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(*ARENA_BOUNDS[0])
    ax.set_ylim(*ARENA_BOUNDS[1])
    ax.set_aspect("equal")
    ax.set_title(f"M3 Level 1 -- RRT* route (robot_radius={args.robot_radius}m)")

    current = (0.0, 0.0)
    colors = plt.cm.tab10.colors
    all_ok = True
    for i, target_name in enumerate(search_list):
        target = object_positions[target_name]
        obstacles = build_obstacles(aruco_positions, object_positions, args.robot_radius,
                                     object_radii=object_radii, exclude=target_name)
        goal = standoff_point(target, obstacles, reference=current)
        if goal is None:
            print(f"[{target_name}] FAILED -- no collision-free standoff point found")
            all_ok = False
            continue
        path = rrt_star(current, goal, obstacles)
        if path is None:
            print(f"[{target_name}] FAILED -- RRT* found no path within the iteration budget")
            all_ok = False
            continue
        path = smooth_path(path, obstacles)

        xs, ys = zip(*path)
        ax.plot(xs, ys, "-o", color=colors[i % 10], label=f"{i+1}. {target_name}", markersize=3)
        ax.plot(*target, "*", color=colors[i % 10], markersize=15)
        print(f"[{target_name}] OK -- {len(path)} waypoints after smoothing, "
              f"final point is {dist_between(path[-1], target):.3f}m from target")
        current = path[-1]

    for cx, cy in aruco_positions:
        ax.add_patch(plt.Circle((cx, cy), DEFAULT_MARKER_RADIUS, color="grey", alpha=0.6))
    for name, (ox, oy) in object_positions.items():
        r = object_radii.get(name, 0.08)
        style = dict(color="orange", alpha=0.35) if name in search_list else dict(color="red", alpha=0.35)
        ax.add_patch(plt.Circle((ox, oy), r, **style))

    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {args.out}")
    print("ALL TARGETS REACHED" if all_ok else "SOME TARGETS FAILED -- see above")
