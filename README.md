# ECE4078-G15
ECE4078 Intelligent Robotics Group 15

Here is the breakdown of the achievable tasks and key deliverables required for **Checkpoint 1** as well as **Milestones 1, 2, and 3** based on the lab manual:

---

## 1. Checkpoint 1: Setting Up & Teleoperating PiBot

* **Keyboard Teleoperation:**
* Complete the `TODO` section in the `update_keyboard` function inside `operate.py` to drive the robot (forward, backward, left, right, stop) via keyboard controls.

* **PID Controller & Drive Tuning:**
* Tune the PID controller parameters ($K_p$, $K_i$, $K_d$) in the `__init__` function of `operate.py` to ensure the robot moves in a reasonably straight line using encoder feedback.

* **Demonstration:**
* Demonstrate basic teleoperation (driving forward/backward and turning left/right) to a demonstrator during the lab. (Note: Ungraded)

---

## 2. Milestone 1: Simultaneous Localization and Mapping (SLAM)

In this milestone, you will implement an Extended Kalman Filter (EKF) to construct a map of 10 ArUco markers while localizing the robot.

### Achievables & Tasks

* **Calibration:**
* **Wheel Calibration:** Complete `calibration/wheel_calibration.py` to calculate wheel parameters (`scale` and `baseline`).

* **Camera Calibration:** Take photos of a checkerboard with `take_pic.py` and run `camera_calibration.py` to produce intrinsic matrix and distortion files (`param/intrinsic.txt`, `param/distCoeffs.txt`).

* **EKF SLAM Implementation:**
* Complete `slam/robot.py` by writing the derivatives and covariance computations for the motion model.
* Complete `slam/ekf.py` by implementing predicted and updated robot states.

* **Map Generation & Evaluation:**
* Drive the robot around the arena during SLAM, press `s` to save the generated map (`lab_output/slam.txt`), and evaluate it against a generated ground truth map using `eval.py`.

* **Deliverables / Submissions (Week 4/5):**
* **M1: Code:** Submit `M1.zip` (zipped code folder, excluding `venv`) to Moodle.
* **M1: Output:** Submit the generated `lab_output/slam.txt` after your live demo.





---

## 3. Milestone 2: Object Recognition and Localisation

This milestone involves training a YOLOv8 model to recognize fruits and estimating their $x\text{-}y$ coordinates in the arena.

### Achievables & Tasks

* **Data Collection & Annotation:**
* Measure target fruits and update `object_list.csv` with length, width, and height values.
* Capture pictures of fruits and arena backgrounds, generate a synthetic dataset, and annotate bounding boxes using Roboflow.

* **Model Training & Integration:**
* Train a YOLOv8 object detector model using `YOLOv8_training_notebook.ipynb` and save the trained weights (`.pt` file) into `cv/model`.

* **Pose Estimation & Merging:**
* Modify SLAM/script logic so loaded ground truth coordinates can be imported into `self.markers` in `ekf.py` without being overwritten during running.

* Complete the `estimate_pose` function in `object_pose_est.py` to compute $x\text{-}y$ object coordinates using camera detections and robot poses.

* Complete the `merge_estimations` function in `object_pose_est.py` to combine multiple image estimations into a single prediction per fruit type.

* Verify accuracy against true maps using `eval.py`.

* **Deliverables / Submissions (Week 7/8):**
* **M2: Code:** Submit `M2.zip` (zipped code folder, excluding `venv` and datasets) to Moodle.

* **M2: Output:** Submit `lab_output/objects.txt` right after the live demo.





---

## 4. Milestone 3: Navigation and Planning

This milestone focuses on autonomously guiding the robot to target fruits in order while avoiding obstacles.

### Achievables & Tasks

* **Core Modules Implementation (`auto_fruit_search.py`):**
* **Waypoint Navigation:** Complete `drive_to_point` to navigate to given coordinates and `get_robot_pose` using SLAM state feedback.
* **Path Planning (Known Map - Level 1):** Implement a path planner (e.g., A* or RRT) to generate obstacle-free paths given full map details.
* **Dynamic Re-planning (Partial Map - Level 2):** Integrate obstacle detection to detect unlisted fruits/obstacles and re-plan trajectories in real time.
* **Exploration & Searching (Minimal Map - Level 3):** Implement an exploration algorithm to search an unknown area given marker locations only and navigate to target fruits in any order.
* **Target Criteria & Execution:**
* Read target fruits from `search_list.txt`.
* Stop within $0.4\text{ m}$ radius of each target fruit and output console confirmation before moving to the next.


* **Deliverables / Submissions (Week 9/10):**
* **M3: Code:** Submit `M3.zip` containing all scripts and code variants to Moodle.