# How the M3 Level 1 code works: a walkthrough from scratch

This covers the two files behind Milestone 3 Level 1: `path_planner.py` and `auto_fruit_search.py`. It assumes no prior background. It describes the code as it is on disk right now, and checks every number in it against that code.

> **Read section 7 before your next robot run.** While tracing through the code for this walkthrough, I found five bugs. Three of them would stop Level 1 working on the real robot: it would crash on the first target, and the robot would barely turn. They're in code I wrote or kept unchanged, and each one comes with a fix.

---

## 1. The job, in one paragraph

The robot starts at the centre of a 2.5 m × 2.5 m arena. You give it a map (`truemap.txt`) with the positions of 10 ArUco markers (the black-and-white square tags) and 7 fruits. You also give it a shopping list (`search_list.txt`: redapple, greenapple, orange). It has to drive to each fruit on the list in order and stop within 0.4 m of it without hitting anything. It then prints proof on the console and waits for the demonstrator to press a key before moving on to the next fruit.

The work is split in two, like a sat-nav and a driver:

- **`path_planner.py` is the sat-nav.** You give it "I'm here, I want to get there, these are the things in the way", and it gives back a list of points to drive through. It knows nothing about the robot, wheels or camera, so you can test it on a laptop.
- **`auto_fruit_search.py` is the driver.** It talks to the robot, works out where the robot is (using SLAM), and drives from point to point. It asks the planner for a route whenever it needs one.

---

## 2. Five ideas you need first

| Idea | What it means here | Where it lives |
|---|---|---|
| **Pose** | Where the robot is and which way it's facing: `(x, y, θ)`. The origin `(0, 0)` is the arena centre. `x` points the way the robot faces at the start, and `y` points to its left. θ is the heading in radians, and it goes up as the robot turns left (anticlockwise). | `ekf.robot.state` |
| **Map** | `truemap.txt` is a JSON file of `{name: {x, y}}` for every marker and fruit. | read by `read_true_map()` and `ekf.load_true_map()` |
| **Encoder ticks** | Each wheel has a sensor that counts "ticks" as the wheel turns. Your calibration says **172.5 ticks = 1 metre** of wheel travel, so 1 tick ≈ 5.8 mm. To drive a set distance, you ask the robot for that many ticks. | `ticks_per_meter` |
| **SLAM / EKF** | There are two ways to know where you are. The first is counting wheel ticks (*dead reckoning*): it's smooth, but small errors add up over time. The second is spotting markers whose map positions you know: this corrects the drift. The EKF (Extended Kalman Filter) blends the two, weighting each by how much it trusts it. It keeps that trust in a matrix called **P** (the uncertainty). In M3 the marker positions are frozen at their true values, so the only thing the EKF estimates is the robot's pose. | `slam/ekf.py` |
| **Waypoints** | A route is a list of `(x, y)` points. The robot drives in straight lines between them: turn to face the next point, drive straight, repeat. | `drive_to_point()` |

---

## 3. What happens when you run it

Command: `python auto_fruit_search.py --ip <robot-ip>`
(Optional flags: `--map` defaults to `truemap.txt`, `--calib-dir` defaults to `calibration/param/`, and `--manual` lets you type waypoints by hand instead.)

The `if __name__ == "__main__":` block at the bottom runs these steps in order:

1. **Read the command-line options** (`argparse`).
2. **Connect to the robot.** `BotConnect(args.ip)` opens two background connections: one sends wheel commands and receives encoder counts, and the other streams camera images. After a 1-second pause for those to connect, `set_pid(kp=2, ki=0.04, kd=0.29)` sends the same wheel-speed controller gains that `operate.py` uses.
3. **Build the SLAM filter.** `init_ekf()` loads your calibration files: `intrinsic.txt` (camera focal length and centre), `distCoeffs.txt` (lens distortion), `scale.txt`, and `baseline.txt` (distance between the wheels). It builds a `Robot` with `ticks_per_meter=172.5` and wraps it in an `EKF`. The robot's pose starts at `(0, 0, 0)` by default.
4. **Freeze the map.** `ekf.load_true_map("truemap.txt")` puts the 10 markers into the filter at their true positions and marks them as fixed, so they can never move. From then on, the EKF only corrects the robot's pose.
5. **Set up the marker detector.** `ArucoSensor(..., marker_length=0.06)` finds ArUco tags in a camera image. For each one it works out how far ahead and how far to the side the tag is, relative to the robot.
6. **Create the `Navigator`.** This is one object that holds the robot connection, the EKF and the detector, and provides "where am I?", "turn by this much" and "drive this far".
7. **Read the map and shopping list again for the planner.** `read_true_map()` returns fruit names, fruit positions and a 10×2 array of marker positions. `read_search_list()` reads the target order. `print_object_pos()` prints them.
8. **Go:** `run_level1(...)`, or `run_manual(...)` if you passed `--manual`.

---

## 4. The Level 1 loop (`run_level1`)

This happens for each fruit on the shopping list, in order:

1. **Look up the fruit's position** in the map. If it isn't there, print a warning and skip it.
2. **Ask "where am I?"** with `nav.get_robot_pose()` (explained in section 6.2).
3. **Build the obstacle list.** Every marker and every fruit *except the current target* becomes a circle to avoid. The target is left out so that it doesn't block its own approach (section 5.1).
4. **Pick a parking spot next to the target.** You can't drive *to* the fruit, because you'd hit it. `standoff_point()` finds a clear spot 0.3 m from it, which leaves a 0.1 m margin inside the 0.4 m rule (section 5.3).
5. **Plan a route** from where you are to that spot with `rrt_star()` (section 5.4).
6. **Straighten the route** with `smooth_path()` (section 5.5).
7. **Drive it.** For each waypoint, `drive_to_point()` does turn, check, drive, check (section 6.4).
8. **Check and report.** The code measures the distance from the robot's SLAM pose to the fruit. It prints `=== Found redapple at [...] (robot is 0.3xx m away -- OK) ===`, or `OUT OF TOLERANCE` if the distance is over 0.4 m. Then it waits on `input()` until the demonstrator presses ENTER.

If there's no parking spot or no route, the code prints why and skips to the next fruit. It isn't supposed to crash, but section 7 covers two cases where it currently does.

---

## 5. Inside `path_planner.py` (the sat-nav)

### 5.1 Turning everything into circles (`build_obstacles`)

Planning is much easier if the robot is treated as a **single dot**. The trick is to shrink the robot down to its centre point and grow every obstacle by the robot's radius to make up for it. If the dot stays outside every grown circle, the real robot's body can't touch anything.

Each obstacle's circle radius is: **robot radius + obstacle's own radius + safety margin**

- Robot radius: `ROBOT_RADIUS = 0.095` m (you measured this).
- Marker: `0.05` m. The marker blocks are 6 cm, so a 5 cm circle covers the corners.
- Fruit: half the diagonal of its footprint, from `object_list.csv`. A redapple is 0.074 × 0.074 m, giving 0.052 m. `load_object_radii()` does this for every fruit.
- Safety margin: `0.05` m, to absorb pose and map error.

So a marker becomes a circle of radius **0.195 m**. Most fruits come out at **0.19–0.20 m**, and mango, the longest fruit, at **0.215 m**. The result is an array with one row per obstacle: `[x, y, radius]`.

The **arena edge** is handled separately by `ARENA_BOUNDS`. The robot's centre may go anywhere in ±1.365 m. That number is half the 2.5 m inner-tape size, plus the 2 cm tape, plus one robot radius, which matches the rule that half the robot may sit on the tape.

### 5.2 "Would I hit something?" (collision checks)

- `point_in_collision(p, obstacles)` measures the distance from `p` to every circle centre at once. If any distance is ≤ that circle's radius, the answer is yes.
- `segment_in_collision(p1, p2, obstacles)` asks the same for a straight line from `p1` to `p2`. For each circle it finds the **point on the line closest to the circle's centre**. It does this by projecting the centre onto the line (`t`) and clamping it between the two ends (0 to 1). If that closest point is inside the circle, the line is blocked. This gives an exact answer, not a check at a few sample points.

### 5.3 Picking a parking spot (`standoff_point`)

Picture a circle of radius 0.3 m around the fruit. The code tries **16 points evenly spaced around it** (every 22.5°). It throws away any that are outside the arena or inside another obstacle's grown circle. From the ones left, it picks the **closest to the robot**, so the drive is short. If none survive, it returns `None`.

### 5.4 Finding a route: RRT* (`rrt_star`)

**The idea:** grow a tree of reachable points outward from the robot's position, like roots spreading through soil. Each new branch must be collision-free. Once a branch reaches the parking spot, follow the branches back to the start, and that's your route.

**How the tree is stored (`_Tree`):** three lists, all indexed the same way:
- `positions[i]`: where point *i* is.
- `parents[i]`: which point it branched off from. The start has parent `-1`.
- `costs[i]`: the total distance from the start to point *i* along the tree.

**One round of the loop**, repeated `max_iter = 2000` times:

1. **Pick a random target point** somewhere in the arena. 10% of the time (`goal_bias = 0.1`), pick the parking spot itself instead, which pulls the tree toward the goal.
2. **Find the closest existing tree point** to that random target (`nearest`).
3. **Step toward it by at most 0.15 m** (`step_size`). That gives the new candidate point.
4. **Reject it** if it's outside the arena, inside an obstacle, or if the line from the closest point to it crosses an obstacle.
5. **Choose the best parent (the "*" in RRT\*, part 1).** Look at every tree point within 0.3 m (`rewire_radius`). Connect the new point to whichever one gives the **shortest total distance from the start**, as long as that link is also collision-free.
6. **Rewire (the "*", part 2).** For each of those nearby points, check whether going *through the new point* would give it a shorter route from the start. If so, and the link is clear, switch its parent to the new point.
7. **Did we arrive?** If the new point is within 0.05 m of the parking spot (`goal_tolerance`) and is the cheapest arrival so far, remember it.

After all 2000 rounds (the loop keeps improving the route even after first reaching the goal), `path_to_root` follows the parent links backward from the best arrival point, reverses the list, and adds the exact parking spot at the end. If nothing ever arrived, the function returns `None`.

**Why RRT\* and not plain RRT?** Plain RRT stops at the first route it finds, which is usually zig-zaggy. Steps 5 and 6 keep reshaping the tree toward shorter routes. Fewer, straighter segments means fewer turns, and every turn is a chance for the wheels to slip.

**Safety check at the start:** if the start or the goal is already inside an obstacle, `rrt_star` raises a `ValueError` instead of planning. Section 7 explains why this matters.

### 5.5 Straightening the route (`smooth_path`)

RRT\* routes are still a bit wobbly, because they're made of many 0.15 m steps. The smoother tries 100 times to cut corners. Each time, it picks two random points on the route with at least one point between them. If a straight line between them is collision-free, it deletes everything in between. In testing, routes shrank to just 2–4 points.

### 5.6 Testing it without the robot

Run `python path_planner.py` from the repo folder. It plans a full route through `search_list.txt` on `truemap.txt` and saves a picture to `lab_output/m3_planner_test.png`. Grey dots are markers, orange circles are target fruits, and red circles are other fruits. It prints OK or FAILED for each fruit.

---

## 6. Inside `auto_fruit_search.py` (the driver)

### 6.1 `init_ekf()`
This loads the calibration files and builds `Robot` → `EKF`, exactly as `operate.py` does. It returns the EKF and the baseline value from `baseline.txt`, which `turn()` uses. See bug 3 in section 7.

### 6.2 "Where am I?" (`Navigator.get_robot_pose`)

The same function behaves differently on the first call and on every call after that:

**First call** (before the robot has moved):
The EKF has just had its map frozen, and its uncertainty **P for the robot is exactly zero**. A filter that is 100% sure of itself ignores new evidence, so a normal EKF update would change nothing. Instead, the code grabs a camera frame, finds the markers, and calls `ekf.recover_from_pause()`. That function solves for the pose directly: "given where these markers appear to me and where the map says they are, where must I be standing?" This is a best-fit alignment called Umeyama. It needs **at least 2 markers** in view. If fewer are visible, it prints a warning and the pose stays at the default `(0, 0, 0)`. That's why starting at the arena centre, facing +x, matters: the default happens to be correct.

**Every later call:**
1. **Predict:** "Since I last asked, the wheels turned this many ticks, so I've probably moved here." `_drive_measurement_since_last_call()` reads the wheel counters, subtracts the previous reading, and packs the difference into a `DriveMeasurement`. `ekf.predict()` then moves the pose estimate and increases P (it gets a bit less sure).
2. **Correct:** grab a frame, detect markers, `ekf.update()`. Each marker's seen position is compared with where it *should* appear from the predicted pose, and the pose is nudged to reduce the difference. Because P is no longer zero, the nudge actually does something.
3. Return `[x, y, θ]`.

`_drive_measurement_since_last_call()` also has a guard copied from `operate.py`: if either counter went *down* by more than 10 ticks, it assumes the counters were reset and treats the movement as zero. See bug 4 in section 7.

### 6.3 Moving (`turn`, `drive_forward`, `_wait_for_move`)

- **`drive_forward(d)`**: ticks = `d × 172.5`, so 0.5 m is 86 ticks. It sends both wheels forward at speed 0.4 with that tick target (`move_auto_encoder`). The robot counts the ticks itself and stops.
- **`turn(dθ)`**: spinning on the spot, each wheel travels an arc of `(baseline / 2) × |dθ|`. With a 0.1386 m baseline, a 90° turn is 0.109 m per wheel, which is 19 ticks. The wheels spin in opposite directions: `[-0.35, +0.35]` turns left (the same as operate.py's left arrow key), and `[+0.35, -0.35]` turns right.
- **`_wait_for_move()`**: waits for the robot to report "done" (`autonomous_done`), checking every 20 ms. If a move takes longer than 15 s, it force-stops the robot and prints a warning.

### 6.4 Going to one waypoint (`drive_to_point`)

1. Get the pose. Work out the compass direction to the waypoint (`atan2`), subtract the current heading, and wrap the result into −180°…+180° (`_normalize_angle`), so the robot never turns 270° left when 90° right would do. Then **turn**.
2. Get the pose **again**, because the turn is never perfect. Measure the straight-line distance to the waypoint from where the robot *actually* is now, then **drive** it.
3. Get the pose a third time, print it, and return it. The next waypoint starts from this corrected pose, so errors get fixed at each waypoint instead of piling up.

### 6.5 `run_manual()`
With `--manual`, you type an x and y, the robot drives there with `drive_to_point`, and it asks whether you want another. This is useful for testing the driving on its own.

---

## 7. Bugs I found while writing this (fix before the next robot run)

I checked these against the real files on disk, plus a simulated robot that uses your real `slam/ekf.py` and calibration files.

**Bug 1 (would crash on the first target): `read_true_map()` puts the markers in the wrong rows.**
It stores marker *N* in row `int(key[5])`, so aruco1 goes in row 1 and aruco9 in row 9. Then aruco10 is also written to row 9, which overwrites aruco9. Row 0 is never filled. It comes from `np.empty`, so it holds whatever junk was in memory; in my test that was ≈ `(0, 0)`, the robot's starting spot. The result: a phantom marker at the start, and marker 9 at (−0.95, 1.05) missing from the obstacle list. In my test, `rrt_star` immediately raised `ValueError: start point is inside an inflated obstacle` on redapple.
*Fix:* `marker_id = int(key[5:].split('_')[0]) - 1` (this works for 1–10), and use `np.zeros` rather than `np.empty`. I kept this function from the manual's template without checking it. That's on me.

**Bug 2 (shifts obstacles by up to 7 cm): `read_true_map()` rounds positions to 0.1 m.**
Your map has positions like −0.920. Rounding moves things by up to 0.068 m, which is more than the 0.05 m safety margin. It also means the planner and the SLAM filter disagree about where the markers are, because the filter reads the exact values. The parking spot was 0.264 m from greenapple and 0.324 m from orange instead of 0.300 m.
*Fix:* remove the `np.round(..., 1)` so the exact values are used.

**Bug 3 (the robot barely turns): the calibrated baseline is negative.**
`calibration/param/baseline.txt` contains **−0.1386**. The minus sign comes from the calibration formula using the −0.5 wheel speed. `turn()` computes ticks from `baseline / 2`, so the result is negative, and `max(1, …)` turns it into **1 tick** for every turn, about 5°. In simulation, asking for (0, 0.5) from the start made the robot turn 4.8°, drive along x instead, and finish 0.68 m from the waypoint.
*Fix:* use `abs(baseline)` in `turn()`. Leave the file alone: the SLAM filter has been working with the negative value since M1, so the counters and the filter are presumably already consistent with it.

**Bug 4 (the robot loses track of its moves): encoder differences are taken across whole moves.**
`operate.py` reads the counters on every pass of its main loop, so each difference is small. My `Navigator` reads them only before and after a whole turn or drive, and that breaks in two ways:
- In an on-the-spot turn, one wheel's counter goes *down*. Any turn bigger than about 50° drops it by more than 10 ticks, so the "counter reset" guard throws the whole turn away.
- Your own `calibrate_encoder.py` notes that **the Pi resets its counters to zero the moment the robot stops**. If that also happens in encoder mode, as it very likely does, the counters read ≈0 before and after every move, and predict thinks the robot never moved.

In the simulation with counters reset on stop, the robot really drove 0.4 m, but SLAM still said `(0.00, 0.00)`. Marker updates alone didn't rescue it.
*Fix:* straight after each `turn()` / `drive_forward()` finishes, feed the EKF the move you *commanded*. Build a `DriveMeasurement` from the requested ticks, written in the EKF's own left/right convention, so it works whatever the counters do. Then drop the before/after counter reading.

**Bug 5 (could crash mid-run): `rrt_star`'s `ValueError` isn't caught.**
Parking spots can sit right at the edge of a neighbour's grown circle. If the SLAM pose lands 1–2 cm inside that edge after arriving, the next plan raises `ValueError` and the whole script stops, mid-demo.
*Fix:* in `run_level1`, catch it and plan from the nearest free point instead, for example by stepping the start point straight out of the circle it's in.

**Quick way to see bugs 3 and 4 on the real robot:** put the robot at the centre facing +x. Run `python auto_fruit_search.py --ip <ip> --manual` and enter x = 0, y = 0.5. It *should* turn left 90° and drive 0.5 m. With the current code, it will barely turn, and the printed "pose now" won't match where the robot actually is.
