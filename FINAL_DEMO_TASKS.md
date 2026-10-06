# Final Demo (Checkpoint 2) — Task Tracker

Source of truth for the requirements: `FInalDemo.txt`. Agents working in this repo should read this file first, update the checkboxes as work lands, and add notes under **Log** at the bottom with the date.

Status key: `[ ]` todo · `[~]` in progress · `[x]` done · `[?]` needs verification on the real robot

---

## 1. What the demo requires (summary)

- Build a map of **10 ArUco markers (M1)** and **7 objects (M2)**, then do **object navigation (M3)** to the fruits in `search_list.txt`.
- **No true map is given.** `truemap.txt` cannot be used at the demo. Everything must come from our own mapping run.
- The estimated map may be **offset from (0,0)**. Relative positions are still correct, but the planner's bounds must cope with a shifted map.
- **Marks:** SLAM map 25, object map 25, navigation 50. The result is scaled by the viva factor.
- **Final score** is the better of these two:
  - best map (manual or auto) + best Level 1/2 navigation run
  - best single Level 3 attempt (its own auto map + its own navigation)

### Levels

| Level | Mapping | Navigation | Order | Twist |
|---|---|---|---|---|
| 1 | Teleop | Autonomous | Search-list order | — |
| 2 | Teleop | Autonomous | Search-list order | After mapping: non-targets can move **anywhere**. Targets can shift ≤0.2 m or be **swapped** with each other |
| 3 | Autonomous, done during navigation | Autonomous | Any order (but the order can't be hand-picked after seeing the arena) | One combined task |

### Rules that apply to every level

- Each collision with a marker or object costs a penalty. So does each time the robot goes out of bounds.
- Hands off the keyboard after the script starts. At each object:
  - **print which object was found**
  - wait for the demonstrator
  - **ONE keypress** to continue
- A run only qualifies if the robot is **≤0.4 m** from the object, centre to centre, and **at least 2 objects** are reached.
- Submission file names (index increments for each attempt):
  - Level 1/2: `slam_manual_{i}.txt`, `objects_manual_{i}.txt`
  - Level 3: `slam_auto_{i}.txt`, `objects_auto_{i}.txt`

---

## 2. Current state of the code (as of 2026-10-06, branch `brandon_final_demo`)

- `auto_fruit_search.py` has `run_level1`, `run_level3` and `run_manual`. The `--level` flag accepts `auto|1|3`. **There is no explicit Level 2 mode.**
- `--map` defaults to `truemap.txt`. Level 1 freezes markers with `ekf.load_true_map()`. For the demo, that file has to be **our own generated map**.
- `path_planner.ARENA_BOUNDS` is hardcoded and centred on the origin (±~1.25 m). **This doesn't handle the offset-map case.**
- `operate.py` **deletes `lab_output/` on startup** (`shutil.rmtree`). A second mapping run would wipe the first run's files.
- `run_level3` writes its fruit map to `lab_output/m3_fruit_map.txt`. It's unclear whether it also writes a marker `slam` file in the submission format.

---

## 3. Tasks

### A. Mapping pipeline (M1 + M2): Levels 1/2 (teleop)
- [ ] A1. Confirm `operate.py` teleop produces both a marker map (slam) and `objects.txt` in the marker-evaluation format.
- [ ] A2. Stop `operate.py` from wiping earlier attempts. Archive `lab_output/` to a timestamped folder instead of `rmtree`, or prompt before deleting.
- [ ] A3. Add a save or rename step that writes `slam_manual_{i}.txt` / `objects_manual_{i}.txt`. Pick the next free index automatically.
- [ ] A4. Add a script or utility that converts a slam + objects pair into the map file `auto_fruit_search.py --map` expects (truemap JSON format).
- [?] A5. Mapping-quality rehearsal: teleop map → `eval.py` RMSE against a known layout. Target a marker RMSE of ___ and an object RMSE of ___.

### B. Navigation from our own map: Level 1
- [ ] B1. Run Level 1 with `--map <generated map>` instead of `truemap.txt`. Remove every place that silently falls back to `truemap.txt`. Check `--compare-map` too.
- [ ] B2. **Offset map:** derive planner bounds from the map itself. Options:
  - compute a bounding box of the markers plus a margin
  - re-centre the map so the start pose / map centroid sits at (0,0)

  Make sure the edge margin and out-of-bounds checks use the new bounds.
- [ ] B3. Check the start pose. For L1/2 the robot is placed "in the middle" for navigation. Make sure the EKF's initial pose agrees with the frame the map was built in, or relocalise from markers at startup.
- [ ] B4. Console proof at each object: print the object name and the distance. Wait for exactly one keypress. Don't respond to stray input before or after.
- [?] B5. Full Level 1 rehearsal on the robot: 4 targets, zero collisions, all ≤0.4 m.

### C. Level 2: objects shifted or swapped after mapping
- [ ] C1. Add a `--level 2` mode, or document that L1 code already handles it.
- [ ] C2. Treat mapped **non-target** positions as unreliable. Use live YOLO detections as obstacles and re-plan when a new obstacle appears on the path.
- [ ] C3. For **targets**:
  - search within 0.2 m of the mapped position
  - **verify the label with YOLO** before declaring "found"
  - if the label doesn't match (swap), use the current detection and update the map for the remaining targets
- [?] C4. Rehearsal: shift and swap fruits after mapping, then run.

### D. Level 3: simultaneous mapping and navigation
- [?] D1. Make sure `run_level3` does not rely on `truemap.txt` markers. Done in `final_demo_l3.py`:
  - phase 0 maps the markers with live SLAM, then locks them
  - late markers are added and locked after 3 consistent sightings
  - simulation only so far: needs a real run
- [x] D2. Write the `slam_auto_{i}.txt` / `objects_auto_{i}.txt` outputs. They go to `submission/`, using the next free index. Markers are saved after phase 0, and both files are saved again in `finally` (so also on ESC or Ctrl-C).
- [x] D3. Make sure the search-list order is used as given: no hand-picked order. `search_list.txt` is used as is; the order is automatic (`optimise_order`), or `--keep-order`.
- [x] D4. Bounds for an offset map in L3. Decided: the robot starts at the centre, square to a wall. The map frame is the start pose, so the existing ±1.25 m bounds hold. **Placement matters.**
- [?] D5. Full Level 3 rehearsal: mapping score plus at least 2 objects reached.

### E. Robustness and safety (all levels)
- [ ] E1. Collision margins: check `--safety-margin`, `--fruit-margin` and `--clearance` against the measured robot footprint.
- [ ] E2. Out of bounds: keep `--edge-margin` and test near the tape.
- [ ] E3. Graceful behaviour when markers are lost. The scan / relocalise paths exist; test them in a sparse corner.
- [ ] E4. Freeze calibration files (`calibration/param/*`) the day before the demo. Re-check `turn_scale` on the demo floor.
- [ ] E5. Far-landmark navigation (bookmarked earlier, **not built**). Decide whether it's needed.

### F. Demo-day logistics
- [ ] F1. Write a one-page run sheet with exact commands for: manual map → L1/L2 nav → manual map → L3. Include file names.
- [ ] F2. Pre-flight checklist:
  - robot IP
  - battery
  - YOLO weights path `cv/model/best.pt`
  - `search_list.txt` matches what's given on the day
  - `lab_output/` archived
- [ ] F3. Decide the attempt strategy, for example: map ×1, L1 ×2, L3 ×1. Remember L3 mapping and nav must come from the **same** attempt.
- [ ] F4. Collect submission files and double-check the naming convention.

### G. Viva prep (each member individually)
- [ ] G1. Each member lists the modules they own and can explain them end to end. Existing walkthroughs live in `Claude outputs/`.
- [ ] G2. Write a short overview of how M1 (EKF SLAM), M2 (YOLO + pose estimation + FruitEKF) and M3 (planner + navigator) connect.
- [ ] G3. Go through likely questions:
  - EKF predict and update equations
  - why markers are frozen in L1
  - A* vs RRT
  - how standoff points are chosen
  - how the offset map is handled

---

## 4. Open questions
- Physical arena size and tape: confirm the inner edge is 2.5 m.
- Does the demonstrator place the robot at a known pose for L1/L2 navigation, or only "in the middle"?
- Exact format the markers expect for slam / objects files: check against the lab's `eval.py` / `SLAM_eval`.

---

## 5. Log
- 2026-10-06: Tracker created from `FInalDemo.txt` and a quick audit of `auto_fruit_search.py`, `path_planner.py` and `operate.py`.
- 2026-10-06: Added `final_demo_l3.py` (a fork of `auto_fruit_search.py`, Level 3 only).
  - Phase 0: live-SLAM marker mapping (360° sweep plus up to 2 extra viewpoints), then lock.
  - Then the unchanged M3 L3 fruit exploration and parking.
  - Writes `submission/slam_auto_{i}.txt` and `objects_auto_{i}.txt`.
  - `m3_display.py` now skips NaN (unmapped) markers.
  - Tested in simulation (phase 0 / lock / late markers / save; no YOLO): about 1–1.5 cm aligned marker RMSE.
  - Not yet run on the robot.
- 2026-10-06: `eval.py` can now score final-demo files: `--run auto_1` / `--run all` (from `submission/`, with a summary table). Plots are saved as `eval_<run>.png`. The scoring maths is unchanged.
