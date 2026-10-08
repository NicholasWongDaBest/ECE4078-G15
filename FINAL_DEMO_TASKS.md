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
- [?] D6. ArUco faces in `final_demo_l3.py`: heading from the marker faces in phase 0, after the lock and in the re-fit (see the Log). Simulation only so far: needs a real run. Check first that the blocks sit square to the arena.

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
- 2026-10-07: Accuracy and logging changes in `final_demo_l3.py`:
  - Applies the marker distance correction (`CorrectedArucoSensor`; `--no-dist-correction` turns it off). Readings were 6 cm short within 1.2 m. Sim: aligned marker RMSE 5.9 cm -> 0.9-2.1 cm.
  - `m3_display.py`: the T overlay now lines the true map up to the robot's frame before drawing it, adds true markers and a live RMSE readout, and draws marker uncertainty ellipses or sd-at-lock circles.
  - Run log: `run_logs/<date-time>/` (meta, events, console, final). Summarise with `python analyze_run.py [--truemap truemap.txt]`.
  - Next: R5 (noise model from real logs), R6 (end-of-run marker re-fit).
- 2026-10-07 (from run_logs/20261007-134902):
  - Run result: 4/4 parks. Submitted marker map 7.0 cm aligned RMSE.
  - Cause: phase-0 turn drift. 2-3 tick turns come up 10-15% short, which rotated the locked map about 12 degrees (and the planning bounds with it).
  - Added the in-run marker re-fit (`Navigator.refit_markers`, `solve_pose_graph`):
    - Runs after each exploration viewpoint.
    - Moves the markers, the robot pose, each fruit (with the frames that saw it) and stored positions.
    - Safety gate: refused if the whole map turns more than 20 degrees, any marker moves more than 30 cm relative to the others, or the robot moves more than 40 cm / 20 degrees beyond the map change.
    - `--no-refit` turns it off.
    - Replayed on this log: 7.0 cm -> 3.4-3.7 cm map from viewpoint 3 onward, fruits 5.8 -> 4.7 cm, 0.5-2.4 s per re-fit.
  - Distance correction is now OFF by default (`--dist-correction` turns it on). It over-corrected on this run.
  - Not done yet: turn model (affine), drive overshoot (~21%, 6 samples).
- 2026-10-07 (from run_logs/20261007-143413):
  - Results: 4/4 parks. Markers 2.6 cm aligned RMSE (was 7.0). Fruits: targets 3.4 cm, others 5.2 cm (capsicum 7.5 cm from 2 views).
  - Re-fit: all 6 accepted. The first turned the map -7.7 degrees, with one marker moving 28 cm relative to the others (gate limit 30).
  - Turns: turn_scale_fast learned 0.565 (file says 0.6125). Slow turns still overshoot by a fixed ~+1.9 degrees per turn, so the turn model is still open. The user will recalibrate fast turns themselves.
  - Changes:
    - Run starts on ENTER.
    - Every fruit is held to the search-list bar for the object map (`FruitMapper.score_labels`, `needs_work()`). Search-list fruits keep priority. Unseen non-target fruits get a fixed-viewpoint 360.
- 2026-10-07: Fixed late-marker registration (run_logs/20261007-170534: marker 7 was missed in phase 0 and seen 52 times after the lock, but never added):
  - Bug fix: the single-marker heading shortcut returned before the late-marker check. Now it runs the check.
  - Live rule no longer needs a known marker in the same frame (pose converged + 3 sightings agreeing within 10 cm + range up to 1.5 m).
  - The re-fit now adds a marker it can place that isn't registered yet (6+ readings, 2+ frames, inside the arena).
  - Late markers are left out of the re-fit safety gate.
  - Replay of this log: marker 7 added at t=127 s (22 cm off), then refined by 8 accepted re-fits to 7.0 cm.
- 2026-10-07: Fruit re-solve and phase merge:
  - Re-fit now re-solves each fruit from its stored sightings, through the re-fitted poses (`refit_fruit_positions`):
    - Bearing weighted hard, range weakly.
    - Mislabel gate: a sighting more than 0.5 m from that fruit's current estimate is not stored.
    - Fewer than 3 sightings, or a change over 35 cm: the fruit is moved with the map instead.
    - Replay of 3 runs: fruits 5.8->3.6, 4.2->4.5, 5.4->3.0 cm.
  - Tested fruits as pose landmarks too. No gain on any run (±0.3 cm), so not used.
  - Phase merge:
    - Phase-0 sweeps feed the real fruit map.
    - A re-fit runs right after the lock (sim with turns 12% short: aligned 0.8-1.9 -> 0.4-0.5 cm, and the map's 8-9 degree drift rotation is corrected).
    - Level 3 skips its centre 360 (~40 s).
  - Caveat: in sim, a biased camera combined with --dist-correction made the phase-0 re-fit worse. Keep the correction off (the default) unless re-validated.
- 2026-10-07: Phase-0 sweep step changed 20 -> 10 degrees (`--p0-step`). The camera FOV is 31 degrees, so 20-degree steps left about 1.6 stops per object.
- 2026-10-07 (run_logs/20261007-174140, capsicum 20.7 cm off):
  - Cause: the fruit re-solve gave each sighting its own graph node, linked by commanded moves. Pan sightings are handed over after the pan, so that link included the rest of the pan, and the camera heading came out up to 20 degrees wrong. The re-solve then moved capsicum 29 cm away from its sightings.
  - Fix: each sighting is attached to its nearest marker node through the filter's own pose offset.
  - Replay of 4 runs: fruit mean 4.2 / 4.1 / 3.1 / 5.0 cm (was 3.6 / 4.5 / 3.0 / 7.7). Capsicum on the latest run: 29.8 -> 5.9 cm.
- 2026-10-07: ArUco faces in `final_demo_l3.py` (`slam/aruco_faces.py`, ported from the M3 branch; `--no-faces` turns all of it off):
  - Every block sits square to the arena, so a face's direction is a heading measurement that needs no map. Readings also become block centres (the face is half a block in front of the centre the marked map holds).
  - Phase 0: each frame's face heading is fused BEFORE the marker update, so new markers are placed with the corrected heading instead of the drifted one. The start heading gets +/-3 deg (`--start-heading-sd`), so the faces also square the map to the arena when the robot was placed a few degrees off (arena axes at 0 in the start frame, `--arena-axes`).
  - The same readings teach turn_scale from the first sweep on (only frames whose face picks don't depend on the heading estimate; at most 15% per sample).
  - A big face correction (over 3 sd and 6 deg) waits until another block agrees, then gets through even if the filter was too sure of its heading. Heading from faces switches itself off if more than half of 40+ frames disagree (blocks not square).
  - After the lock: faces fused after each update. Pan / scan / refine / arrival lines end with `| faces ...`.
  - Re-fit: every usable face reading is a heading residual (`solve_pose_graph(faces=...)`, with a small tilt per block), so the re-fitted poses, and the fruits re-solved through them, carry the faces' heading. Positions not tied to a node now move with the markers' rigid change (the nodes of a sweep on the spot give no usable rotation).
  - `m3_display.py` reads markers through `Navigator.detect_markers()`. `slam/aruco_sensor.py` only adds face data when given a faces config, so `operate.py` and `auto_fruit_search.py` behave exactly as before. `slam/ekf.py` untouched.
  - Run log: `frame` events carry `faces` (measurement, sd, innovation, fused / held / rejected); `phase0_faces` after each sweep; `faces_off` if the guard trips; meta.json has the faces settings.
  - Sim (fake faces; slow turns 12% short, fast 5% long; 30 seeds, current code vs faces): marker map 7.7 -> 0.7 cm aligned RMSE, fruits 11.3 -> 1.3 cm, heading error at drives 14 -> 3.8 deg, map rotation vs arena 10 -> 0.3 deg, parks 3.6 -> 4 of 4. The current code does worse in this sim than on the robot (some runs lose 30+ deg in phase 0), so read these as relative. Robot placed 4 deg off: map lined up with the arena to 0.4 deg. Blocks turned ~8 deg each: still 0.9 cm / 3.3 cm.
- 2026-10-07: Merged `ck` (YOLO with a `marker` class) into `brandon_final_demo`, and added obstructions:
  - Conflicts: `eval.py` keeps both sides (final-demo `--run` scoring + CK's object RMSE and grade); `m3_display.py` keeps both; `slam/aruco_faces.py` and `slam/aruco_sensor.py` are the final-demo versions (ck's were the old M3 copies).
  - ck's "stop tracking PNG files" commit deleted `ui/8bit/*.png`, `ui/gui_mask.png` and `ui/loading.png`, which `slam/ekf.py`, `operate.py` and `m3_display.py` load at start-up. Restored; `.gitignore` now keeps `ui/**/*.png`.
  - `best.pt` was renamed on ck: `final_demo_l3.py` and `auto_fruit_search.py` now default to `cv/model/include_marker_model.pt` (old fruit-only model: `cv/model/M3_best.pt`).
  - Obstructions (`ObstructionMap`, `--no-obstructions` turns them off): a marker block the CNN sees that no ArUco reading or mapped marker explains, seen in 2+ frames, is avoided like a marker (plus its uncertainty) until a marker or fruit is mapped there, which replaces it. Dropped after 3 looks from different spots that had it in view and saw nothing; one-frame detections are forgotten after 30 frames. Before each drive, any unexplained marker block in the robot's path stops it short (`fruit_ahead`), in phase 0 too. Where the margins leave no route, phase 0 and parking fall back to marker-sized circles, then to none.
  - Range from the box height; a box cut at the bottom of the frame uses its width (or 0.2 m if a side is cut too). On the 81 labelled photos in `cv/marker`: box-height range within 2% of the ArUco depth; on this robot no tag nearer than ~0.31 m decodes (it runs off the bottom of the frame), and 67 of the 160 marker boxes are cut at the bottom.
  - Sim (3 maps x 10 seeds, on vs off): normal conditions, marker contacts 2 -> 0, all parks, time within +/-3%; tags read only within 1.2 m and 30% read failures, marker contacts 16 -> 4, time +3% to +20%; a false marker box in 5% of frames, all parks, ~4% slower.
  - Console: `obstruction: a marker block with no readable tag at [...]`, `... replaced by marker N`, `camera: marker block in the way`, and an `Obstructions: ...` summary at the end. The run log has `obstruction` events and the leftovers in final.json.
  - Not yet run on the robot. CK's `cv/detector.py` prints `marker cut off at: [...]` for every partial marker box.
- 2026-10-07 (run_logs/20261007-213921, first robot run with obstructions): 4/4 parks, 8 obstructions noted, 7 replaced by markers. One misfire at [-0.12, +0.77] (marker 4 seen cut off at the frame edge, placed ~0.3 m off) stayed to the end. Changes:
  - Once every arena marker (1-10) is mapped, all obstructions are cleared and no new ones are noted: every marker block is then on the map, so a leftover can only be a misplaced copy or a false detection. The look-ahead before each drive still runs.
  - Each obstruction keeps the rays it was seen along; a mapped marker that lies along all of them (inside the box's bearing span, at a fitting range) replaces it even when the positions are further apart than the replace radius.
  - The window's detector panel now updates on every YOLO pass (phase 0's sweeps and drives too); before it only updated from Level 3's frames.
  - `--compare-map` defaults to `truemap.txt` when the file exists: the true map is shown in the window from the start (T hides / shows it) and scored at the end. Display and report only. `--no-compare` turns it off.
- 2026-10-07 (auto_3 / run_logs/20261007-222402 scored 91.7 / 72.8 vs auto_1's 100 / 95.6): the run itself was as good (final map 2.2 cm aligned, vs 1.8 cm), but the program was closed at the "Run finished. ESC to exit" prompt, so the end-of-run save never ran and `submission/*_auto_3.txt` still held the after-phase-0 snapshot (3.0 cm markers, phase-0 fruits: mango 18 cm). Now the files are saved after every re-fit and once more as soon as Level 3 finishes, before the ESC prompt.
  - ticks_per_meter 193.3 -> 171.7 (changed by hand between the two runs) checks out: the first marker update after each drive moved the pose forward by a median 12% of the drive in run 1 and 2% in run 3 (9 drives each).
- 2026-10-08: Drive roll-on (tape test with verify_motion.py, 12 drives of 5-70 cm). The wheels roll on ~14 ticks (~7 cm) after the Pi stops them, whatever the drive length, so the error was +40% on a 10 cm drive and 0% on 70 cm -- no single ticks_per_meter can fix that (171.7 hid it on long drives only). Counting the roll-on, every drive of 15 cm+ gives 190-197 ticks/m.
  - `final_demo_l3.py`: drives (`Navigator._drive`) are commanded short by the expected roll-on, so the robot lands where planned, and the EKF is told the ticks the encoders counted once the wheels stopped (no tick for 0.15 s, at most 1 s). Counts that look wrong fall back to the command plus the expected roll-on. The roll-on is re-learned during the run from drives of 25+ ticks (median of the last 12).
  - Calibration in `calibration/param/drive_motion.json` (ticks_per_meter 194.2, roll-on 15.3 ticks, ramp 14 ticks: fitted from that test); `init_ekf()` reads ticks_per_meter from it (default 193.3). `--no-count-ticks` = the old way (still with the file's ticks_per_meter), `--coast-ticks N` overrides the roll-on. Start-up prints the drive settings; the end of the run prints how many drives were counted and the run's own roll-on; move events in the run log carry `ticks_cmd` / `ticks_counted`.
  - On the test data, counted ticks / 193.3 matched the tape to +0.2 cm mean, +/-0.5 cm (was up to 6 cm off). Sim with a 14-tick roll-on (3 maps x 6 seeds): position error at drives 4.3 -> 2.9 cm, fruits 2.3 -> 1.6 cm, markers 0.87 -> 0.65 cm; with no roll-on it learns 0 and still works.
  - Turns are unchanged; turn_scale*.txt were calibrated with 193.3 ticks/m, which is what the demo uses again.
  - `verify_motion.py` rewritten to match: `d` = drive the way the demo does, `r` = raw drive for calibrating, `t` = turn; `sum` fits ticks_per_meter and the roll-on, `save` writes drive_motion.json.
- 2026-10-08: The run window's true-map compare (T) now shows what `eval.py` prints for the files saved at that moment (`m3_display.eval_like_score`). Each fruit's error follows `eval_object`'s rule: the smaller of its raw and its aligned error ("raw" or "al" in the panel and on the map), and the M1 / M2 grades use eval.py's default constants. The panel's top line is `M1 <grade> <marker cm>  M2 <grade> <object cm>`, and the map caption has the same. The end-of-run `[compare]` lists raw / aligned / eval per fruit, and final.json's `compare` gains an `eval` section. Checked on auto_11: 0.02784 m / 97.0 for objects and 2.1 cm / 99.4 for markers, the same as eval.py. Display only; nothing reaches the robot.
- 2026-10-08: Fruits had to converge a little more before exploration moves on (defaults; the flags still override): `--target-sightings` 6 -> 8, `--target-views` 4 -> 5, `--target-spread` 60 -> 75 deg, `--min-sightings` 4 -> 5. They apply to every scored fruit. Expect a somewhat longer phase 1. Earlier, `--target-spread 120` cost ~4 min of scanning for no gain.
- 2026-10-08: Turn roll-on. In verify_motion turn tests (41 trials), the same tick count repeated within about +/-1-2 deg, but every turn came out n x (one tick) + a fixed extra: fast (0.35) 7.2 deg/tick + 5.8 deg, slow (0.25) 5.8 deg/tick + 1.6 deg. That fixed extra is the wheels rolling on after the Pi stops them. A turn_scale alone cannot fit it: small fast turns came out +15-30% and big ones +2%, which is why tuning turn_scale kept moving the error from one turn size to another.
  - `final_demo_l3.py`: `Navigator._turn_plan()` picks the tick count whose result (n x tick + roll-on) is nearest the request, and tells the EKF that result. `calibration/param/turn_motion.json` holds turn_scale, turn_scale_fast and the roll-on per speed (offset_deg, offset_fast_deg). When the file is there it takes precedence over turn_scale*.txt. Without it, or with `--no-turn-offset`, turns work as before. `--turn-offset` / `--turn-offset-fast` override the roll-on. Start-up prints the turn settings, and meta.json records them.
  - Delivered turn_motion.json, fitted from those trials: turn_scale 0.7348 + 1.55 deg, turn_scale_fast 0.5928 + 5.81 deg. On that data, what the EKF is told about a 5-180 deg turn is off by at most 1.4 deg (mean -0.4), against up to 5.5 deg (mean +4.2) with the old model.
  - `verify_motion.py`: `t` turns the way the demo does, `tr` is a raw turn. `sum` lists, per speed, the EKF-told vs measured error and the repeatability per tick count, and fits deg/tick + roll-on (needs 4+ turns over 2+ tick counts). `save` writes turn_motion.json and turn_scale*.txt. `--small-turn-deg` should match the demo's.
  - Sim with a 5 deg / 1.3 deg roll-on (3 maps x 20 seeds), old vs new turns: no out-of-bounds (233 events on map A with the old turns, where some runs lost their heading), markers 1.59 -> 0.75 cm, fruits 2.36 -> 1.71 cm, heading error at drives 4.6 -> 3.6 deg, about 4% faster; contacts 10 / 10.

## Never give up on a fruit: last-resort smaller collision zones (2026-10-08)
Run 20261008-213510 skipped redapple AND orange ("no collision-free standoff point"): after lime and lemon the robot was in the top-left pocket (lime, marker 1, marker 9, lemon, marker 8 -- it got in along the top edge past capsicum, whose estimate then moved up 10 cm and sealed that way). Every way out was closed even with the tight margins. Exploration had ended at 03:59 for the same reason (11 viewpoints "sealed off").
- Parking (run_level1): when no parking spot is reachable, the margins now step down: normal -> tight -> no obstructions -> squeeze 4/3 cm (marker/fruit) -> 2.5/2 cm -> 1.5/1 cm, the last also allowing spots 0.35/0.38 m out (still inside 0.4 m). While squeezing: 15 cm drives, no open legs; once the robot is through, back to the lowest rung that still reaches the spot. A target that still fails is retried once after the others. Options: `squeeze_margins`, `squeeze_leg`, `last_resort_rings`, `retry_failed` (run_level1 arguments).
- Exploration (run_level3): if even the tight margins find no viewpoint because the robot is boxed in, or `_go_to` finds no route, it squeezes out with 2.5/2 cm margins and 15 cm drives instead of ending exploration.
- Offline replay of the run's map from where the robot stood: old code skips both; new code parks at both (redapple via the marker 1/9 gap, closest body 4.8 cm).
- Sim (12 seeds x 4 maps, D = this run's layout): skipped targets 13 -> 1 (D), 0 -> 0 elsewhere; contacts 11 -> 11; mean clock 603 -> 673 s (more fruits reached, longer routes).
- 2026-10-08: Room to turn + double drive through gaps (run 20261008-220626: squeezing out of a pocket used 2.5/2 cm margins and 15 cm legs for the WHOLE route, so the robot stopped 6 times inside the marker 5/8 gap and re-localised there, turning up to ~110 deg next to blocks).
  - `turn_room(p)`: gap between the robot body and the nearest marker/fruit/obstruction body. A spot is turn-safe with at least `--turn-clearance` (default 0.05 m); `--turn-radius` sets the radius the body sweeps when turning (default the planner's 0.095 m; measure axle midpoint to furthest chassis point).
  - `_gap_plan()` (parking and exploration legs): a leg that would end where it cannot turn either stops 3 cm before the tight stretch (pose check there), or, from the mouth/inside, drives straight through in one go to the first turn-safe point (up to 0.8 m, past the squeeze leg cap), or, if there is no straight way out, ends with no pan and no facing turn. No pre-drive pose-check pan from a spot without room to turn.
  - Parking spots with room to turn are preferred over ones without.
  - Exploration squeezes only until a route exists with the tight or normal margins, then drives the rest normally (`escaping()` in run_level3).
  - Sim (map D, 8 seeds, short check only): all 4 targets every run both before and after; contacts 3 -> 5 (capsicum mapping, one lime, one marker 6). The sim turns the robot about its centre, so it cannot show turn swing hits; the real test is on the robot.
