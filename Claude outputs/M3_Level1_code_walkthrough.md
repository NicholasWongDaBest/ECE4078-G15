# How the M3 Level 1 code works: a walkthrough from scratch

This covers the two files behind Milestone 3 Level 1: `path_planner.py` and `auto_fruit_search.py`. It assumes no prior background. It describes the code as it is on disk right now, and checks every number in it against that code.

> **Updated 22 September 2026.** The first version of this walkthrough reported five bugs, three of which would have stopped Level 1 working on the robot. They're now fixed, and sections 4 and 6 describe the fixed code. Section 7 lists what changed, the simulation results, and what to check on the real robot.

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
(Optional flags: `--map` defaults to `truemap.txt`, and `--calib-dir` defaults to `calibration/param/`. `--manual` lets you choose waypoints yourself: click the map in the window, or type them in the terminal with `--no-display`. `--no-display` runs without the window, as it did before section 8 was added.)

It opens a window like `operate.py`'s, with the camera, the map and a setup step for pointing the robot at markers before it starts. Section 8 covers the window.

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
4. **Make sure the start is clear.** After parking next to something, pose noise can put the estimate a centimetre or two inside a neighbour's safety circle, and the planner refuses to start from inside one. `_escape_obstacles()` finds the nearest clear point, and the route starts from there.
5. **Pick a parking spot next to the target.** You can't drive *to* the fruit, because you'd hit it. `standoff_point()` finds a clear spot 0.3 m from it, which leaves a 0.1 m margin inside the 0.4 m rule (section 5.3).
6. **Plan a route** to that spot with `rrt_star()` (section 5.4). If it finds nothing in 2000 rounds, it tries once more with 4000.
7. **Straighten the route** with `smooth_path()` (section 5.5).
8. **Drive it.** The route's first point is where the robot already is, so it's dropped (unless step 4 moved the start). Each remaining waypoint goes to `drive_to_point()` (section 6.4).
9. **Check and report.** The code measures the distance from the robot's SLAM pose to the fruit. It prints `=== Found redapple at [...] (robot is 0.3xx m away -- OK) ===`, or `OUT OF TOLERANCE` if the distance is over 0.4 m. Then it waits on `input()` until the demonstrator presses ENTER.

If there's still no parking spot or route, the code prints why and skips to the next fruit. Reaching fruits out of order scores nothing for that run, so if you see a skip, stop and restart rather than letting it carry on.

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

**Safety check at the start:** if the start or the goal is already inside an obstacle, `rrt_star` raises a `ValueError` instead of planning. `run_level1` avoids that by moving the start to the nearest clear point first (section 4, step 4).

### 5.5 Straightening the route (`smooth_path`)

RRT\* routes are still a bit wobbly, because they're made of many 0.15 m steps. The smoother tries 100 times to cut corners. Each time, it picks two random points on the route with at least one point between them. If a straight line between them is collision-free, it deletes everything in between. In testing, routes shrank to just 2–4 points.

### 5.6 Testing it without the robot

Run `python path_planner.py` from the repo folder. It plans a full route through `search_list.txt` on `truemap.txt` and saves a picture to `lab_output/m3_planner_test.png`. Grey dots are markers, orange circles are target fruits, and red circles are other fruits. It prints OK or FAILED for each fruit.

---

## 6. Inside `auto_fruit_search.py` (the driver)

### 6.1 `init_ekf()`
This loads the calibration files and builds `Robot` → `EKF`, exactly as `operate.py` does. It returns the EKF and the value in `baseline.txt`. That value is stored as −0.1386 (the minus sign comes from the calibration formula), so the `Navigator` uses its size, 0.1386 m, for its turning maths and leaves the EKF's copy exactly as M1 has always used it.

### 6.2 "Where am I?" (`Navigator.get_robot_pose`)

Every call grabs the latest camera frame and finds the markers in it. At startup, it first waits (up to 5 s) for the camera to deliver its first frame. What happens next depends on whether the robot has moved yet.

**Before the first move:** the EKF has just had its map frozen, and its uncertainty **P for the robot is exactly zero**. A filter that is 100% sure of itself ignores new evidence, so a normal update would change nothing. Instead, it calls `ekf.recover_from_pause()`, which solves for the pose directly: "given where these markers appear to me and where the map says they are, where must I be standing?" This is a best-fit alignment (Umeyama). It needs **at least 2 known markers** in view, and prints `Initial pose from N markers: ...` when it works. If it doesn't, it prints a warning once, and the pose stays at the default `(0, 0, 0)`: the arena centre, facing the map's +x. That's why starting there, facing +x, is the safe choice.

**After that:** `turn()` and `drive_forward()` have already told the EKF about the move (section 6.3), so this only needs the **correct** step. `ekf.update()` compares where each marker appears with where it *should* appear from the predicted pose, and nudges the pose to close the gap. If 2 or more markers are in view but the update rejects every one of them as too far from what it expected, the prediction has gone badly wrong. It then re-anchors straight from the markers with `recover_from_pause()`, and prints that it did.

**How much to trust a marker.** `slam/ekf.py` was tuned for M1 mapping, and it assumes each marker reading could be 15–30 cm out. That's so wide that an update fixes only about 2% of a position error. With the map frozen, there are no marker estimates to protect, so the `Navigator` gives this EKF a tighter noise model: 3 cm + 3 cm per metre of distance (`marker_noise`). Nothing changes in `ekf.py` or `operate.py`.

### 6.3 Moving (`turn`, `drive_forward`, `_wait_for_move`)

- **`drive_forward(d)`**: ticks = `d × 172.5`, so 0.5 m is 86 ticks. It sends both wheels forward at speed 0.4 with that tick target (`move_auto_encoder`). The robot counts the ticks itself and stops.
- **`turn(dθ)`**: spinning on the spot, each wheel travels an arc of `(0.1386 / 2) × |dθ|`, so a 90° turn is 0.109 m per wheel, which is 19 ticks. The wheels spin in opposite directions: `[-0.35, +0.35]` turns left (the same as operate.py's left arrow key), and `[+0.35, -0.35]` turns right. One tick is the smallest possible turn (about 4.8°), so a turn under half a tick is skipped. If turns come up consistently short or long on the real floor, `turn_scale` sends proportionally more or fewer ticks (section 7).
- **Telling the EKF about the move.** Straight after each move, `_predict_commanded_motion()` calls `ekf.predict()` with the move that was *commanded*: the tick counts sent to the robot. It doesn't read the wheel counters, because the Pi zeroes them whenever the robot stops. A commanded move is less certain than a measured one, so it also widens the pose uncertainty by 10% of each turn (heading) and 5% of each drive (position). That's what lets the next marker update pull the pose back when a move came up short or long.
- **`_wait_for_move()`**: waits for the robot to report "done" (`autonomous_done`), checking every 20 ms. If a move takes longer than 15 s, it force-stops the robot, prints a warning and widens the uncertainty a lot. After every move, it waits 0.4 s so that the next pose check uses a camera frame taken after the robot stopped, not a blurred one from mid-move.

### 6.4 Going to one waypoint (`drive_to_point`)

1. Get the pose. If the waypoint is under 2 cm away, it's already there, so return.
2. Work out the compass direction to the waypoint (`atan2`), subtract the current heading, and wrap the result into −180°…+180° (`_normalize_angle`), so the robot never turns 270° left when 90° right would do. Then **turn**.
3. Get the pose again. If the heading is still more than 5° off (the turn slipped, or the markers showed it isn't where it thought), turn again, up to 2 extra times. A 10° error would put the robot 17 cm off course after 1 m.
4. Measure the straight-line distance from where the robot *actually* is now, and **drive** it.
5. Get the pose once more, print it, and return it. The next waypoint starts from this corrected pose, so errors get fixed at each waypoint instead of piling up.

### 6.5 `run_manual()`
With `--manual`, you type an x and y, the robot drives there with `drive_to_point`, and it asks whether you want another. This is useful for testing the driving on its own.

---

## 7. What was fixed on 22 September, and what to check on the robot

Tracing the code for the first version of this walkthrough turned up five bugs. All five are now fixed in `auto_fruit_search.py`. `path_planner.py`, `slam/ekf.py` and `operate.py` were not changed.

| # | Bug | What it did | Fix |
|---|---|---|---|
| 1 | `read_true_map()` put marker *N* in row *N* | aruco10 overwrote aruco9, and row 0 was left as junk (≈ the start point), so the planner crashed on the first fruit | Marker *N* goes in row *N*−1, and the array starts as zeros |
| 2 | `read_true_map()` rounded positions to 0.1 m | Obstacles moved by up to 6.8 cm, more than the safety margin | Uses the exact values |
| 3 | `baseline.txt` holds −0.1386 | Every turn was sent as 1 tick (about 5°) | Turning uses the size, 0.1386 m |
| 4 | Wheel counters read before and after whole moves | The Pi zeroes them when it stops, so SLAM never saw the robot move | Each move is fed to the EKF as commanded, with extra uncertainty (6.3) |
| 5 | `rrt_star`'s `ValueError` not caught | A pose estimate just inside a safety circle would crash the run | Plans from the nearest clear point, and retries with 4000 rounds before giving up (section 4) |

Testing those fixes turned up four more problems, also fixed:

- The route's first waypoint was the robot's own position. Aiming at a point 0 m away gives a random heading, so the robot made a pointless turn at the start of every route.
- The first pose check could run before the camera had sent a frame.
- M1's marker noise model stopped updates from correcting position (6.2).
- `drive_to_point` drove off even when a turn had clearly slipped (6.4).

`print_object_pos()` also only checked the first few objects in the map.

**Simulated test.** I ran the fixed code 20 times per scenario on a fake robot that uses your real `ekf.py`, map, calibration and planner, with the Pi zeroing its counters at every stop. The fake camera sees markers within ±30° and 2 m. "Clean" markers have 1 cm + 1 cm/m of noise. "Noisy" markers have 3 cm + 3 cm/m, with every range reading 2% long.

| Wheels | Clean markers: fruits over 0.4 m / runs with a collision | Noisy markers: fruits over 0.4 m / runs with a collision |
|---|---|---|
| Turns and drives accurate | 0/60 · 0/20 | 0/60 · 1/20 |
| Turns 10% short, drives 3% short | 0/60 · 0/20 | 0/60 · 1/20 |
| Turns 20% short, drives 5% long | 0/60 · 7/20 | 1/60 · 10/20 |
| Same, with `turn_scale = 1.25` | 0/60 · 0/20 | 0/60 · 0/20 |

The table used the earlier `truemap.txt`. On the updated map (changed 22 September), results were similar: no collisions with accurate or 10%-slip turns, and 2–3 runs in 20 with a collision at 20% slip.

For comparison, the same fixes with M1's marker noise model gave 2 fruits out of tolerance and 6/20 runs with a collision in the 10%-slip case. The simulation is only as good as its guesses about slip and marker noise. It shows the logic works, but not that the real robot will score. The row that matters is how much your robot's turns actually slip.

**What to check on the real robot, in order:**

1. Run `python path_planner.py` to plan and plot the routes without the robot.
2. Put the robot at the centre facing +x, run `python auto_fruit_search.py --ip <ip> --manual`, lock the pose in the setup step, then click the map at (0, 0.5): the grid lines are every 0.5 m, so that's one line straight up from the centre. It should turn left about 90° and drive 0.5 m. Watch where the *first* turn stops, before any correction turns. If turns are consistently short or long, set `turn_scale` in the `Navigator(...)` line of the `__main__` block. For example, if it reaches 80° when asked for 90°, use `turn_scale=1.125`.
3. After a few waypoints, compare the printed "pose now" with a tape measure. More than about 5 cm apart means the pose tracking needs tuning (`marker_noise`, `turn_noise_frac`, `drive_noise_frac`).
4. Do a full run: `python auto_fruit_search.py --ip <ip> --map truemap.txt`.

---

## 8. The live window (`m3_display.py`, added 22 September)

The window uses `operate.py`'s layout, font and marker and robot artwork:

- **Robot Cam** (top left) shows the camera with detected markers outlined, plus a count of how many known markers are in view. The count is green at 2 or more, orange at 1 and red at 0.
- **Status** (bottom left) shows the markers in view, the robot's pose, and its uncertainty (SD). SD turns orange and then red as the robot gets less sure where it is. It also shows the pose the visible markers imply (FIT), each target with its result, and the run time.
- **Map** (right) is centred on the arena, with +x to the right, +y up and grid lines every 0.5 m:
  - The thick square is the boundary tape, and the thin red square is where the robot's centre is allowed to go.
  - Markers seen in the current frame get a green ring. Fruits are drawn in their colour, and targets are numbered in search order.
  - Grey circles are the safety circles for the current leg. The robot's *centre* must stay outside them, so the robot's drawn body overlapping a circle is normal.
  - The green circle around the current target is its 0.4 m success zone.
  - The blue line is the planned route and the cross is the parking spot.
  - The yellow lines show what the camera can see, and the small ellipse is the pose uncertainty.

**Setup step (before anything drives).** Put the robot at the centre. Then:

- **← / →** turn it 15° on the spot (5° with SHIFT).
- **ENTER with 2+ markers in view** locks the full pose from them.
- **ENTER with 1 marker in view** locks the heading from that marker, assuming the robot is at the centre. The window warns you if the marker's measured distance is more than 15 cm off its map distance, which usually means the robot isn't at the centre. Press ENTER again to accept it anyway.
- **ENTER with no markers in view** (pressed twice) starts from the assumed pose: the centre, facing +x, plus any turns you made.

Until the pose is locked, the map labels the robot "assumed" and draws the camera's view from the marker fit. That way you can see which way the robot really faces, even if it wasn't placed facing +x.

**Your camera's view is narrow.** `intrinsic.txt` gives a focal length of about 1073 px, which is only about ±16° of view (32° total). From the centre, two markers rarely fit in view at once. That's why the one-marker heading lock exists, and in simulation it's what you'll use most of the time. The same narrow view means markers are seen less often while driving, so the pose relies more on the wheels.

Re-running the simulation with the narrow view (20 runs per scenario, noisy markers):

| Wheels | Fruits over 0.4 m | Runs with a collision |
|---|---|---|
| Turns and drives accurate | 0/60 | 0/20 |
| Turns 10% short, drives 3% short | 0/60 | 0/20 |
| Turns 20% short, drives 5% long | 4/60 | 7/20 |

In the 10% case the closest call was 0.399 m, so calibrating `turn_scale` matters more with this camera.

**During the run:** the window keeps updating while the robot moves. After each fruit, press **ENTER or SPACE in the window** (not the terminal) once the demonstrator has checked it.

**Any time:** **ESC**, or closing the window, stops the robot and quits.

**With `--manual`:** click anywhere on the map and the robot drives there (turn, check, drive, check). ← / → still turn it on the spot.

