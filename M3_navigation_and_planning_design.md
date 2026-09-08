# ECE4078 Milestone 3 — Navigation & Planning: Design Plan

Group 15 · fruit-searching PiBot · drafted before any M3 code is written

## 1. Where things stand

Checked the repo (`ECE4078 Lab Student`) against the M3 spec in the lab manual. M1 (SLAM) and M2 (object detection/pose) are both well past skeleton stage — `slam/ekf.py` already has an `IEKF`, a `FruitEKF` for fusing fruit detections, RMSE tooling, and — usefully for M3 — a `load_true_map()` method that **freezes marker positions and only localises the robot**, plus a `recover_from_pause()` that re-anchors pose once ≥2 landmarks are back in view. `cv/detector.py` has a working YOLO `ObjectDetector`, and `object_pose_est.py`'s `estimate_pose()` converts a bounding box + robot pose into a world (x, y).

`auto_fruit_search.py`, on the other hand, is the untouched skeleton the manual ships: `drive_to_point()` and `get_robot_pose()` are both `pass`, and there's no path-planning code anywhere in the repo yet (`git log` shows only M1/M2 commits). So M3 is a clean start, but it gets to stand on a lot of working infrastructure.

**Worth a quick team check-in:** since this is a shared repo, confirm nobody else has M3 work sitting on a branch before starting.

## 2. What M3 actually requires (recap)

Given `search_list.txt` (4 fruits) and a map, navigate to each in order, stopping within **0.4 m** of the object, printing evidence to console, and waiting for a single demonstrator keypress before continuing — hands off the keyboard once the script starts. Three levels, each a strictly harder mapping assumption:

| Level | Map given | Points | Order enforced? |
|---|---|---|---|
| 1 | Full (10 markers + 7 objects) | 70 | Yes |
| 2 | Partial (10 markers + 4 search-list objects; other 3 objects are hidden obstacles anywhere; listed objects may be shifted ≤0.2m or **swapped** with each other) | 85 | Yes |
| 3 | Minimal (10 markers only) | 100 | No (but can't hand-pick an order either) |

Two rules that shape the whole design: the robot must use **SLAM for pose, not just vision** (explicit in the manual), and collisions/out-of-bounds cost -5pts *each occurrence*, so the planner needs real clearance margins, not just point-to-point lines.

## 3. Shared infrastructure (needed for all three levels)

**Arena bounds.** `truemap_generator/config.yaml` sets `arena_size: 2.5m × 2.5m`, centered at the origin (consistent with the marker coordinates in `truemap.txt`, which range roughly ±1.0m). I'll use bounds `x, y ∈ [-1.25, 1.25]` — **worth confirming against the physical arena tape** before it's hardcoded, since going out of bounds is a penalty.

**Obstacle model.** Every ArUco marker and every non-target object is a circular obstacle: `radius = robot_footprint_radius + object_radius + safety_margin`.
- Object radius: pull from `object_list.csv` (already has length/width/height per fruit, e.g. capsicum 0.070×0.070×0.11m, mango 0.122×0.067×0.061m) — use half the diagonal of length/width as a conservative circle.
- ArUco markers: 0.06m blocks (from `config.yaml`) → ~0.05m radius.
- **Robot footprint radius isn't in the code anywhere** — needs a physical measurement (ruler across the chassis, center to the widest point). I'd guess ~0.10–0.12m from the photos in the manual, but don't want to hardcode a guess — flagging this as the one physical-world number to measure before tuning collision margins.
- Safety margin: recommend +0.03m on top of the geometric sum, to absorb SLAM/pose uncertainty.

**Robot pose (`get_robot_pose`).** Call `ekf.load_true_map(map_file)` once at startup — this already exists and does exactly what's needed (freeze landmarks, localise-only). At each pose check: grab a frame, detect ArUco markers, build a `DriveMeasurement` from the encoder delta since the last predict, call `ekf.predict()` then `ekf.update()`, return `ekf.robot.state` as `[x, y, theta]`. If zero markers are visible, fall back to dead-reckoning and do a small in-place scan rotation before committing to the next leg — this is exactly what `recover_from_pause()`'s "≥2 landmarks" logic already does, so it's a reuse, not a new component.

**Waypoint execution (`drive_to_point`).** Turn-then-drive, closed-loop at the waypoint level (matches the manual's own recommendation to "correct pose after every waypoint step"):
1. Compute bearing to the waypoint from the current SLAM pose, normalize to `[-π, π]`, turn in place using `move_auto_encoder` (baseline = 0.1386m, already calibrated in `calibration/param/baseline.txt`).
2. Drive straight the remaining distance using `move_auto_encoder` (reuses the PID-tuned straight-line driving from Checkpoint 1).
3. Re-run `get_robot_pose()` and use the *actual* resulting pose — not the commanded one — as the start of the next leg's turn/drive calculation. This is what keeps error from compounding across a multi-waypoint path instead of open-looping the whole route.

## 4. Level 1 — known map, RRT*

You picked RRT* over grid A* — reasonable, since it scales more naturally into Level 3's unknown-space exploration later, and it's a genuinely continuous planner (no grid resolution to tune against the 0.4m tolerance).

**Design:**
- **State space:** planning is over `(x, y)` only — a point-robot with inflated obstacles. Robot orientation is handled entirely by `drive_to_point`'s turn-then-drive execution, which is the standard way to decouple planning from kinematics here and keeps the planner simple.
- **Goal handling:** the target itself is an obstacle (can't drive into fruit), so the planner's goal isn't the object's coordinate — it's a **standoff point** ~0.3m from the object center (comfortably inside the 0.4m budget), picked along a collision-free line from the object outward, then verified against the obstacle set. Simpler and more reliable than planning to a goal *region*.
- **Sampling:** uniform random in arena bounds, ~10% goal-biased.
- **Steering:** extend toward the sample by a fixed step (~0.15m).
- **Rewiring:** standard RRT* rewiring within a fixed radius (~0.3m is reasonable for a 2.5×2.5m arena) — this is what RRT* buys over plain RRT: straighter, lower-cost paths, which matters here because every extra turn is another place for encoder drift to creep in (a lesson your M1 notes already flagged — swerving from stacked corrections).
- **Budget:** 1500–3000 iterations is plenty at this arena scale; test off-robot first (see §7) to tune this without burning demo time.
- **Post-processing:** shortcut-smooth the raw RRT* path (try connecting non-adjacent nodes directly, keep it if collision-free) before handing it to `drive_to_point` — fewer, straighter segments to execute.

**Per-target loop** (`search_list.txt` order, since Level 1 enforces order):
```
ekf.load_true_map(known_map)
obstacles = build_obstacles(aruco_true_pos, object_true_pos)   # all of them
for target in search_list:
    target_pos = object_true_pos[target]
    goal = standoff_point(target_pos, obstacles)
    leg_obstacles = obstacles - {target}                        # don't block the goal with itself
    path = smooth(rrt_star(current_xy, goal, leg_obstacles, ARENA_BOUNDS))
    for wp in path:
        drive_to_point(wp)                                      # re-anchors pose each leg
    assert dist(get_robot_pose()[:2], target_pos) <= 0.4
    print(f"Found {target} at {target_pos}")
    wait_for_single_keypress()                                  # demonstrator verification
```

## 5. Level 2 — partial map, obstacle detection, re-planning

Adds two real wrinkles on top of Level 1's planner: **3 completely unlisted objects** (position unknown, anywhere) and the 4 known ones possibly **shifted ≤0.2m or swapped with each other**.

- **Live detection during driving:** periodically (each leg, or at fixed intervals along a long leg) run `ObjectDetector` on the current frame; for each box, compute world position via `estimate_pose(get_robot_pose(), box, ...)` — both already built in M2's code. If a detection doesn't match a currently-known obstacle within tolerance, register it as newly discovered.
- **Re-plan trigger:** if a new/updated obstacle now intersects the remaining path (within `robot_radius + obstacle_radius` of any upcoming segment), abort the rest of the current path and re-run RRT* from the current pose with the updated obstacle set. This is the only new *planning* logic needed — the RRT* planner from Level 1 is reused as-is, just re-invoked more often.
- **Swap handling matters for correctness, not just collision-avoidance:** because two listed objects can swap places, arriving at the *nominal* coordinate for "orange" doesn't guarantee it's actually the orange there. Confirm identity via the YOLO class label on approach before counting a target as reached — if the label doesn't match, the planner needs to re-target the object that's actually at that location and go find the real "orange" from wherever the swap put it.
- **Position tolerance for listed objects:** inflate their obstacle circles by the extra 0.2m shift radius (rather than trusting the exact partial-map coordinate) until confirmed by detection.

## 6. Level 3 — minimal map, exploration

Only markers are known; all 7 objects are undiscovered. Order isn't enforced, but the search list also can't be edited/reordered by hand — it has to be genuinely autonomous discovery-and-go.

Given the arena is small (2.5×2.5m) and the demo window is tight (15 min), I'd avoid a full frontier/occupancy-grid exploration — recommend a simpler, more reliable pattern:

- **Systematic coverage sweep:** generate a boustrophedon (lawnmower) pattern of waypoints across the arena, connected via the same RRT* planner (obstacle set = markers only, initially).
- **Opportunistic detection:** run the detector at each sweep stop (same mechanism as Level 2). The moment a detected object's label matches an unfound search-list target, break off the sweep, RRT*-plan directly to it (reusing the Level 1/2 per-target loop), visit it, then **resume the sweep from the current position** over whatever's left uncovered — no need to replan the whole sweep from scratch.
- **Termination:** stop once all `search_list` targets are found and visited, or the sweep is exhausted.
- This reuses literally every other piece (planner, obstacle manager, drive_to_point, get_robot_pose) — the only new code is the sweep-pattern generator and the "found vs. still searching" bookkeeping.

## 7. Proposed file layout

- **`path_planner.py`** (new) — RRT*, collision checking, path smoothing, obstacle-circle helpers. Deliberately robot-independent so it's unit-testable off-robot with `matplotlib` against `truemap.txt` before ever touching the physical PiBot.
- **`obstacle_manager.py`** (new, for Level 2/3) — tracks known + discovered obstacles, handles the shift/swap identity resolution, feeds `path_planner`.
- **`auto_fruit_search.py`** (fill in) — `drive_to_point`, `get_robot_pose`, and a `--level {1,2,3}` main loop. A single script with a level flag (rather than 3 separate scripts) keeps one coherent submission and consistent console-evidence formatting, while still satisfying the manual's note that you can submit variants per level if that ends up cleaner.
- **Unchanged, reused directly:** `slam/ekf.py`, `slam/robot.py`, `slam/aruco_sensor.py`, `botconnect.py`, `cv/detector.py`, `object_pose_est.py`.

## 8. Test plan (before demo day)

1. **Off-robot:** unit-test `path_planner.py` against `truemap.txt` with a matplotlib plot — verify goal-reaching, obstacle clearance, and smoothing, with zero hardware time spent.
2. **Level 1 on the bench:** static known map, verify `drive_to_point` + `get_robot_pose` hit the 0.4m tolerance consistently across a few legs; this is where turn/drive gains get tuned.
3. **Level 2:** stage the 3 unlisted objects and a shifted/swapped listed object, confirm detection triggers a genuine re-plan (not just detection logging).
4. **Level 3:** full sweep + opportunistic pickup, timed against the real 15-minute demo budget — this is the one most worth a dry run before Week 9.

## 9. Open items before coding starts

- Confirm the arena's physical bounds match the `2.5m × 2.5m` in `config.yaml`.
- Measure the robot's footprint radius (nothing in the codebase defines this today).
- Quick sync with the team — no M3 commits exist yet, worth checking nobody's started this in parallel.
- Given the time budget, it may be worth aiming to get Level 2 solid before stretching for Level 3, since Level 3 depends on everything below it working reliably first.

---

Once this looks right, next step is `path_planner.py` (RRT* + smoothing, testable off-robot first), then wiring `drive_to_point`/`get_robot_pose` into `auto_fruit_search.py` for Level 1.
