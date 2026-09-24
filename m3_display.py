"""
m3_display.py -- live window for M3 (auto_fruit_search.py), laid out like operate.py.

    +-------------+---------------------------+-------------+
    | Robot Cam   |                           | Detector    |
    | (markers    |   Map: arena, markers,    | (last YOLO  |
    |  outlined)  |   fruits, safety circles, |  frame+boxes)|
    +-------------+   route, robot + camera   +-------------+
    | Status      |   view, pose uncertainty, | Fruit map   |
    | (markers in |   fruit estimates with    | (estimate,  |
    |  view, pose,|   their uncertainty       |  sigma,     |
    |  targets)   |   ellipses (Level 3)      |  views)     |
    +-------------+---------------------------+-------------+
    | notification / key help                               |

Phases:
    setup  -- before anything drives. <- / -> turn the robot on the spot (15 deg,
              or 5 deg with SHIFT) so you can get 2+ markers in view. The map
              shows the pose those markers imply ("marker fit"). ENTER locks the
              pose from the markers and starts.
    run    -- Level 1 route driving. The window keeps updating during moves.
    wait   -- after each fruit: ENTER or SPACE in this window continues.
    manual -- (--manual) click anywhere on the map to drive there.
ESC or closing the window stops the robot and quits, in any phase.

Everything here is display and keyboard handling only -- the pose estimate,
planning and driving all stay in auto_fruit_search.py / path_planner.py.
Single-threaded on purpose: the navigation code calls idle()/refresh() while it
waits, so pygame never runs alongside the EKF in another thread.
"""

import math
import time

import cv2
import numpy as np
import pygame

import path_planner


def _wrap_deg(theta):
    """Radians -> degrees in (-180, 180]."""
    return math.degrees(math.atan2(math.sin(theta), math.cos(theta)))


class M3Abort(Exception):
    """Raised when the user presses ESC or closes the window."""


class NullDisplay:
    """Used with --no-display: same calls as M3Display, terminal behaviour."""
    active = False

    def idle(self, seconds):
        time.sleep(seconds)

    def refresh(self, force=False):
        pass

    def notify(self, text):
        pass

    def begin_target(self, index, name, target_xy, obstacles):
        pass

    def set_route(self, goal, waypoints):
        pass

    def set_waypoint(self, waypoint):
        pass

    def target_done(self, name, distance, ok):
        pass

    def wait_for_key(self, prompt):
        input(prompt)

    def finish(self, text):
        pass


# Display colours (RGB) -- chosen to read like the real fruit on a light-grey map.
FRUIT_RGB = {
    'redapple':   (175, 20, 30),
    'greenapple': (70, 200, 40),
    'orange':     (255, 140, 0),
    'mango':      (255, 195, 90),
    'capsicum':   (0, 110, 30),
    'lemon':      (240, 225, 0),
    'lime':       (150, 215, 40),
}
DEFAULT_FRUIT_RGB = (0, 170, 220)

BG = (213, 213, 213)          # same map background as operate.py / draw_slam_state
GRID = (198, 198, 198)
AXIS = (170, 170, 170)
TAPE = (60, 60, 60)
LIMIT = (205, 90, 90)
SAFETY = (160, 160, 160)
ROUTE = (0, 90, 210)
FOV = (215, 180, 60)
ROBOT = (30, 30, 30)
POSE_COV = (0, 30, 56)
GHOST = (0, 150, 150)
SEEN = (0, 160, 0)
ZONE = (0, 140, 0)

TEXT = (220, 220, 220)
GOOD = (120, 220, 120)
WARN = (240, 180, 60)
BAD = (240, 90, 80)
DIM = (140, 140, 140)


class M3Display:
    active = True

    WIDTH, HEIGHT = 1180, 760
    MAP_RES = 520
    MAP_POS = (360, 40)
    CAM_POS = (20, 40)
    CAM_SIZE = (320, 240)
    INFO_POS = (20, 320)
    INFO_SIZE = (320, 240)
    DET_POS = (900, 40)         # last frame the fruit detector ran on, boxes + labels
    DET_SIZE = (260, 195)
    FRUIT_POS = (900, 250)      # per-fruit estimate / uncertainty / coverage readout
    FRUIT_SIZE = (260, 310)
    VIEW_HALF = 1.45            # metres shown either side of the arena centre
    TURN_STEP = math.radians(15)
    FINE_STEP = math.radians(5)

    def __init__(self, nav, aruco_true_pos, object_positions, search_list,
                 object_radii=None, fps=15):
        self.nav = nav
        self.ekf = nav.ekf
        self.aruco_true_pos = np.asarray(aruco_true_pos, dtype=float)
        self.object_positions = dict(object_positions)
        self.search_list = list(search_list)
        self.object_radii = object_radii or {}
        self.min_period = 1.0 / fps

        pygame.init()
        self.canvas = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('ECE4078 M3')
        try:
            pygame.display.set_icon(pygame.image.load('ui/8bit/pibot5.png'))
        except (pygame.error, FileNotFoundError):
            pass
        try:
            self.bg = pygame.image.load('ui/gui_mask.jpg')
        except (pygame.error, FileNotFoundError):
            self.bg = None
        self.title_font = self._font(35)
        self.text_font = self._font(40)
        self.small_font = self._font(30)
        self.panel_font = self._font(26)

        self.scale = self.MAP_RES / (2 * self.VIEW_HALF)   # pixels per metre
        K = self.ekf.robot.camera_matrix
        self.fov_half = math.atan2(K[0][2], K[0][0])       # half the camera's horizontal view

        self.phase = 'setup'
        self.notification = ''
        self.help_lines = []
        self._events = []
        self._last_draw = 0.0
        self.run_start = None

        self.cam_img = None
        self.visible_tags = []
        self.last_measurement = []
        self.ghost = None          # (x, y, theta, resid, n) from the visible markers

        self.target_index = None
        self.current_target = None
        self.target_xy = None
        self.obstacles = None
        self.goal = None
        self.route = []
        self.active_wp = None
        self.target_status = {}    # name -> (distance or None, ok)
        self.click_point = None

    @staticmethod
    def _font(size):
        try:
            return pygame.font.Font('ui/8-BitMadness.ttf', size)
        except (pygame.error, FileNotFoundError, OSError):
            return pygame.font.Font(None, int(size * 0.8))

    # ------------------------------------------------------------------
    # Calls used by the navigation code
    # ------------------------------------------------------------------

    def notify(self, text):
        self.notification = text

    def begin_target(self, index, name, target_xy, obstacles):
        self.phase = 'run'
        if self.run_start is None:
            self.run_start = time.time()
        self.target_index = index
        self.current_target = name
        self.target_xy = tuple(target_xy)
        self.obstacles = obstacles
        self.goal = None
        self.route = []
        self.active_wp = None
        self.help_lines = ["ESC  stop the robot and quit"]

    def set_route(self, goal, waypoints):
        self.goal = tuple(goal)
        self.route = [tuple(w) for w in waypoints]

    def set_waypoint(self, waypoint):
        self.active_wp = tuple(waypoint)

    def target_done(self, name, distance, ok):
        self.target_status[name] = (distance, ok)
        self.active_wp = None

    def idle(self, seconds):
        """Keep the window alive for `seconds` (used while the robot moves)."""
        end = time.time() + seconds
        while True:
            self.refresh()
            remaining = end - time.time()
            if remaining <= 0:
                break
            time.sleep(min(0.02, remaining))

    def pop_events(self):
        events, self._events = self._events, []
        return events

    def wait_for_key(self, prompt):
        """Wait for ENTER/SPACE in the window (the demonstrator's keypress)."""
        previous_phase = self.phase
        self.phase = 'wait'
        self.notification = prompt
        self.help_lines = ["ENTER / SPACE  continue", "ESC  quit"]
        self.refresh(force=True)
        self.pop_events()
        while True:
            self.refresh()
            for ev in self.pop_events():
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                    self.phase = previous_phase
                    return
            time.sleep(0.02)

    def finish(self, text):
        self.phase = 'done'
        self.current_target = None
        self.obstacles = None
        self.route = []
        self.goal = None
        self.notification = text
        self.help_lines = ["ESC or close the window to exit"]
        try:
            while True:
                self.refresh()
                time.sleep(0.02)
        except M3Abort:
            return

    # ------------------------------------------------------------------
    # Interactive phases
    # ------------------------------------------------------------------

    def run_setup(self):
        """Before the run: turn the robot until 2+ markers are in view, then
        ENTER locks the pose from them. With fewer than 2, a second ENTER
        starts anyway from the assumed/dead-reckoned pose."""
        self.phase = 'setup'
        self.help_lines = ["<- ->  turn 15 deg   (SHIFT: 5 deg)",
                           "ENTER  lock pose and start    ESC  quit"]
        pending = None   # a warning waiting for a second ENTER to confirm
        while True:
            self.refresh()
            if pending is None:
                self.notification = self._setup_hint()
            for ev in self.pop_events():
                if ev.type != pygame.KEYDOWN:
                    continue
                if ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self._turn_from_key(ev)
                    pending = None
                elif ev.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                    self.refresh(force=True)   # decide on the newest frame
                    n = len(self.visible_tags)
                    if n >= 2 and self.ekf.recover_from_pause(self.last_measurement):
                        return self._locked("Pose locked from {} markers".format(n))
                    if n == 1 and self.ghost is not None:
                        range_error = self.ghost[3]
                        if range_error <= 0.15 or pending == 'range':
                            self._lock_one_marker_heading()
                            return self._locked("Heading locked from marker {}".format(self.visible_tags[0]))
                        pending = 'range'
                        self.notification = ("Marker {} is {:.2f} m off its map distance. Not at the "
                                             "centre? ENTER again to accept".format(self.visible_tags[0], range_error))
                        continue
                    if pending == 'none':
                        return self._locked("Starting WITHOUT a marker fix")
                    pending = 'none'
                    self.notification = "No markers in view. ENTER again to start from the assumed pose"
            time.sleep(0.01)

    def _setup_hint(self):
        n = len(self.visible_tags)
        if n >= 2:
            return "{} markers in view: ENTER locks the pose".format(n)
        if n == 1:
            return "Marker {} in view: ENTER locks heading (robot at centre)".format(self.visible_tags[0])
        return "No markers in view: turn with <- -> to find one"

    def _locked(self, what):
        self.nav._localised = True
        s = self.ekf.robot.state
        msg = "{}: [{:.2f}, {:.2f}, {:.0f} deg]".format(what, s[0, 0], s[1, 0], _wrap_deg(s[2, 0]))
        print(msg)
        self.notification = msg
        self.refresh(force=True)

    def run_manual(self, drive_to):
        """--manual with the window: click the map to drive to that point."""
        self.phase = 'manual'
        self.notification = 'Click the map to drive there'
        self.help_lines = ["CLICK MAP  drive there    <- ->  turn",
                           "ESC  quit"]
        while True:
            self.refresh()
            for ev in self.pop_events():
                if ev.type == pygame.KEYDOWN and ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                    self._turn_from_key(ev)
                elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                    wp = self._click_to_world(ev.pos)
                    if wp is None:
                        continue
                    self.click_point = wp
                    self.notification = "Driving to [{:.2f}, {:.2f}]".format(*wp)
                    self.refresh(force=True)
                    pose = drive_to(wp)
                    err = math.hypot(pose[0] - wp[0], pose[1] - wp[1])
                    self.notification = "Arrived: pose [{:.2f}, {:.2f}], {:.2f} m from click".format(
                        pose[0], pose[1], err)
                    self.pop_events()   # ignore clicks made while driving
            time.sleep(0.01)

    def _turn_from_key(self, ev):
        step = self.FINE_STEP if (ev.mod & pygame.KMOD_SHIFT) else self.TURN_STEP
        if ev.key == pygame.K_RIGHT:
            step = -step
        self.notification = "Turning {} {:.0f} deg".format('left' if step > 0 else 'right', abs(math.degrees(step)))
        self.nav.turn(step)
        self.idle(0.1)
        self.pop_events()   # drop key presses queued up during the turn
        self.notification = "{} marker{} in view".format(
            len(self.visible_tags), '' if len(self.visible_tags) == 1 else 's')
        self.refresh(force=True)

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def refresh(self, force=False):
        now = time.time()
        if not force and now - self._last_draw < self.min_period:
            return
        self._last_draw = now
        self._pump_events()
        self._sense()
        self._draw()
        pygame.display.update()

    def _pump_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                raise M3Abort()
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                raise M3Abort()
            if ev.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                self._events.append(ev)

    def _sense(self):
        img = self.nav.botconnect.get_image()
        measurement, aruco_img = self.nav.aruco_sensor.detect_marker_positions(img)
        self.cam_img = aruco_img if aruco_img is not None else img
        self.last_measurement = measurement
        self.visible_tags = sorted({int(lm.tag) for lm in measurement if lm.tag in self.ekf.taglist})

        self.ghost = None
        n = len(self.visible_tags)
        if n >= 2:   # every phase: during the run this is the check that the pose still agrees with the markers
            try:
                _, lm_new, lm_prev, _ = self.ekf._match_known_landmarks(measurement)
                fit = self.ekf._fit_rigid_pose(lm_new, lm_prev)
            except Exception:
                fit = None
            if fit is not None:
                R, t, resid = fit
                t = np.asarray(t).reshape(-1)
                self.ghost = (float(t[0]), float(t[1]), math.atan2(R[1][0], R[0][0]), resid, n)
        elif self.phase == 'setup' and n == 1:
            self.ghost = self._one_marker_fit(measurement)

    def _one_marker_fit(self, measurement):
        """Heading from a single marker, ASSUMING the robot is where the EKF
        thinks it is -- true in setup, where the robot has only turned on the
        spot since being placed at the centre. The marker's map bearing minus
        its bearing in the camera gives the heading. Returns
        (x, y, theta, range_error, 1): range_error is how far the measured
        distance to the marker is from the map distance, a check on the
        assumed position."""
        s = self.ekf.robot.state
        x, y = float(s[0, 0]), float(s[1, 0])
        for lm in measurement:
            if lm.tag not in self.ekf.taglist:
                continue
            f, l = (float(v) for v in np.asarray(lm.position).reshape(-1)[:2])
            mx, my = self.ekf.markers[:, self.ekf.taglist.index(lm.tag)]
            theta = math.atan2(my - y, mx - x) - math.atan2(l, f)
            theta = math.atan2(math.sin(theta), math.cos(theta))
            range_error = abs(math.hypot(f, l) - math.hypot(mx - x, my - y))
            return (x, y, theta, range_error, 1)
        return None

    def _lock_one_marker_heading(self):
        """Set the EKF heading from the single-marker fit and give the pose an
        honest uncertainty: 3 cm for where the robot was placed, and the heading
        error that placement (plus marker noise) causes at this range."""
        x, y, theta, _, _ = self.ghost
        tag = self.visible_tags[0]
        mx, my = self.ekf.markers[:, self.ekf.taglist.index(tag)]
        r = max(math.hypot(mx - x, my - y), 0.3)
        self.ekf.robot.state[2, 0] = theta
        self.ekf.P[0:3, :] = 0.0
        self.ekf.P[:, 0:3] = 0.0
        self.ekf.P[0, 0] = self.ekf.P[1, 1] = 0.03 ** 2
        self.ekf.P[2, 2] = math.atan2(0.03 + 0.02 * r, r) ** 2

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self):
        c = self.canvas
        c.fill((0, 0, 0))
        if self.bg is not None:
            c.blit(self.bg, (0, 0))

        self._draw_camera()
        self._draw_info()
        c.blit(self._draw_map(), self.MAP_POS)
        self._draw_detector()
        self._draw_fruit_panel()

        self._caption('Robot Cam', self.CAM_POS)
        self._caption('Status', self.INFO_POS)
        self._caption('Map  (+x right, +y up)', self.MAP_POS)
        self._caption('Detector', self.DET_POS)
        self._caption('Fruit map', self.FRUIT_POS)

        # notification + key help, in a bar across the whole window
        bar = pygame.Rect(self.CAM_POS[0], 578, self.WIDTH - 2 * self.CAM_POS[0], 166)
        pygame.draw.rect(c, (55, 55, 62), bar)
        pygame.draw.rect(c, (90, 90, 110), bar, 1)
        x0 = bar.x + 12
        c.blit(self._fit_text(self.notification, bar.w - 24, TEXT), (x0, bar.y + 12))
        for k, line in enumerate(self.help_lines[:3]):
            c.blit(self._fit_text(line, bar.w - 24, (170, 170, 170), small=True), (x0, bar.y + 58 + 30 * k))

    def _fit_text(self, text, max_width, colour, small=False):
        """Render with the 8-bit font, dropping to smaller sizes if it would overflow."""
        fonts = [self.small_font, self.panel_font] if small else [self.text_font, self.small_font, self.panel_font]
        for font in fonts:
            surface = font.render(text, False, colour)
            if surface.get_width() <= max_width:
                return surface
        return surface

    def _caption(self, text, pos):
        self.canvas.blit(self.title_font.render(text, False, (200, 200, 200)), (pos[0], pos[1] - 25))

    def _draw_camera(self):
        w, h = self.CAM_SIZE
        if self.cam_img is None:
            view = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            view = cv2.resize(np.asarray(self.cam_img, dtype=np.uint8), (w, h))
        view = np.ascontiguousarray(view)
        surface = pygame.image.frombuffer(view.tobytes(), (w, h), 'RGB')
        self.canvas.blit(surface, self.CAM_POS)

        n = len(self.visible_tags)
        colour = GOOD if n >= 2 else (WARN if n == 1 else BAD)
        label = "{} marker{} in view".format(n, '' if n == 1 else 's')
        text = self.small_font.render(label, False, colour)
        box = pygame.Surface((text.get_width() + 12, text.get_height() + 4))
        box.set_alpha(170)
        box.fill((0, 0, 0))
        self.canvas.blit(box, (self.CAM_POS[0] + 4, self.CAM_POS[1] + 4))
        self.canvas.blit(text, (self.CAM_POS[0] + 10, self.CAM_POS[1] + 6))

    def _draw_detector(self):
        """The last frame the fruit detector ran on, with its boxes and class
        labels drawn the way operate.py's detector view draws them. The
        navigation code decides when YOLO runs (every pan step, scan step and
        pose check); this panel only shows what it saw, and how long ago."""
        x, y = self.DET_POS
        w, h = self.DET_SIZE
        nav = self.nav
        img = getattr(nav, '_last_detect_img', None)
        if img is None:
            view = np.zeros((h, w, 3), dtype=np.uint8)
            note = "no detector loaded" if getattr(nav, 'detector', None) is None else "detector not run yet"
            colour = DIM
        else:
            frame = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)).copy()
            for label, box in getattr(nav, '_last_boxes', []):
                xc, yc, bw, bh = [float(v) for v in box]
                x1, y1 = int(round(xc - bw / 2)), int(round(yc - bh / 2))
                x2, y2 = int(round(xc + bw / 2)), int(round(yc + bh / 2))
                col = FRUIT_RGB.get(label, DEFAULT_FRUIT_RGB)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                cv2.putText(frame, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
            view = cv2.resize(frame, (w, h))
            age = time.time() - getattr(nav, '_last_detect_time', 0.0)
            names = ", ".join(label for label, _ in nav._last_boxes) or "nothing"
            note = "YOLO {:.1f}s ago: {}".format(age, names)
            colour = GOOD if nav._last_boxes else DIM
        view = np.ascontiguousarray(view)
        surface = pygame.image.frombuffer(view.tobytes(), (w, h), 'RGB')
        self.canvas.blit(surface, (x, y))
        pygame.draw.rect(self.canvas, (90, 90, 110), (x, y, w, h), 1)
        text = self.panel_font.render(note[:34], False, colour)
        box = pygame.Surface((min(w - 8, text.get_width() + 12), text.get_height() + 4))
        box.set_alpha(170)
        box.fill((0, 0, 0))
        self.canvas.blit(box, (x + 4, y + h - text.get_height() - 8))
        self.canvas.blit(text, (x + 10, y + h - text.get_height() - 6))

    def _draw_fruit_panel(self):
        """Per-fruit readout, like M2's coverage view: estimate, sigma, how
        many distinct bearings it has been seen from and over what arc, and
        whether that is enough to trust it (see FruitMapper.well_mapped)."""
        x, y = self.FRUIT_POS
        w, h = self.FRUIT_SIZE
        panel = pygame.Surface((w, h))
        panel.fill((25, 25, 35))
        self.canvas.blit(panel, (x, y))
        pygame.draw.rect(self.canvas, (90, 90, 110), (x, y, w, h), 1)

        lines = []
        mapper = getattr(self.nav, 'fruit_mapper', None)
        if mapper is None:
            lines.append(("positions from the map file", DIM))
            for k, name in enumerate(self.search_list, start=1):
                if name in self.object_positions:
                    ox, oy = self.object_positions[name]
                    lines.append(("{}:{:9s} {:+.2f} {:+.2f}".format(k, name[:9], ox, oy), TEXT))
            for name, (ox, oy) in self.object_positions.items():
                if name not in self.search_list:
                    lines.append(("  {:9s} {:+.2f} {:+.2f}".format(name[:9], ox, oy), DIM))
        else:
            fe = mapper.fruit_ekf
            locked = getattr(mapper, 'frozen', False)
            lines.append(("YOLO {} frames  acc {}  rej {}{}".format(
                getattr(self.nav, '_n_detect_frames', 0), mapper.n_accepted, mapper.n_rejected,
                "  LOCKED" if locked else ""), GOOD if locked else DIM))
            # Unseen search-list fruits first: the panel only has room for
            # 13 lines, and with six fruits mapped a "not seen yet" line at
            # the bottom was cut off -- the one line that says the run is
            # about to skip a target.
            for k, label in enumerate(self.search_list, start=1):
                if label not in fe.estimates:
                    lines.append(("{}:{:9s} not seen yet".format(k, label[:9]), BAD))
            ordered = sorted(fe.estimates, key=lambda l: (l not in self.search_list, l))
            for label in ordered:
                pos = fe.estimates[label]
                ok = mapper.well_mapped(label)
                provisional = getattr(mapper, 'is_provisional', lambda l: False)(label)
                colour = BAD if provisional else (GOOD if ok else WARN)
                idx = "{}:".format(self.search_list.index(label) + 1) if label in self.search_list else "  "
                lines.append(("{}{:9s} {:+.2f} {:+.2f}".format(idx, label[:9], pos[0, 0], pos[1, 0]), colour))
                lines.append(("   sd {:.0f}cm  {}v {:.0f}deg  {}".format(
                    mapper.sigma(label) * 100, fe.n_views(label), fe.bearing_spread(label),
                    "ROUGH" if provisional else ("ok" if ok else "weak")), colour))

        for k, (text, colour) in enumerate(lines[:13]):
            self.canvas.blit(self.panel_font.render(text, False, colour), (x + 8, y + 6 + 23 * k))

    def _draw_info(self):
        x, y = self.INFO_POS
        w, h = self.INFO_SIZE
        panel = pygame.Surface((w, h))
        panel.fill((25, 25, 35))
        self.canvas.blit(panel, (x, y))
        pygame.draw.rect(self.canvas, (90, 90, 110), (x, y, w, h), 1)

        lines = []
        n = len(self.visible_tags)
        tags = ", ".join(str(t) for t in self.visible_tags) if self.visible_tags else "none"
        lines.append(("MARKERS {}  ({})".format(n, tags), GOOD if n >= 2 else (WARN if n == 1 else BAD)))

        s = self.ekf.robot.state
        locked = self.nav._localised or self.phase != 'setup'
        lines.append(("{} {:+.2f} {:+.2f} {:+.0f}deg".format(
            "POSE" if locked else "ASSUMED", s[0, 0], s[1, 0],
            math.degrees(math.atan2(math.sin(s[2, 0]), math.cos(s[2, 0])))), TEXT if locked else DIM))
        P = self.ekf.P
        xy_sd = math.sqrt(max(P[0, 0], P[1, 1], 0.0))
        th_sd = math.degrees(math.sqrt(max(P[2, 2], 0.0)))
        sd_colour = GOOD if (xy_sd < 0.05 and th_sd < 10) else (WARN if (xy_sd < 0.10 and th_sd < 20) else BAD)
        lines.append(("SD  {:.1f} cm   {:.1f} deg".format(100 * xy_sd, th_sd), sd_colour))
        if self.ghost is not None:
            gx, gy, gth, resid, gn = self.ghost
            hdg_off = abs(math.degrees(math.atan2(math.sin(gth - s[2, 0]), math.cos(gth - s[2, 0]))))
            pos_off = math.hypot(gx - s[0, 0], gy - s[1, 0])
            if locked and (hdg_off > 10 or pos_off > 0.15):
                lines.append(("FIT  {:+.2f} {:+.2f} {:+.0f}deg  OFF {:.0f}deg {:.0f}cm".format(
                    gx, gy, math.degrees(gth), hdg_off, pos_off * 100), BAD if hdg_off > 20 else WARN))
            else:
                lines.append(("FIT  {:+.2f} {:+.2f} {:+.0f}deg".format(gx, gy, math.degrees(gth)), (110, 210, 210)))
        else:
            lines.append(("", TEXT))

        for k, name in enumerate(self.search_list, start=1):
            status = self.target_status.get(name)
            if status is not None:
                dist, ok = status
                if dist is None:
                    lines.append(("{} {}  skipped".format(k, name), BAD))
                else:
                    lines.append(("{} {}  {:.2f} m {}".format(k, name, dist, "OK" if ok else "FAR"),
                                  GOOD if ok else BAD))
            elif name == self.current_target and self.phase in ('run', 'wait'):
                lines.append(("{} {}  <- now".format(k, name), WARN))
            else:
                lines.append(("{} {}".format(k, name), DIM))

        if self.run_start is not None:
            elapsed = int(time.time() - self.run_start)
            ts = getattr(self.nav, 'turn_scale', None)
            ts0 = getattr(self.nav, 'turn_scale_initial', ts)
            learned = getattr(self.nav, '_turn_scale_samples', 0)
            lines.append(("TIME  {:02d}:{:02d}   TURN x{:.2f}{}".format(
                elapsed // 60, elapsed % 60, ts if ts is not None else 1.0,
                "  (was {:.2f})".format(ts0) if learned and abs(ts - ts0) > 0.01 else ""), DIM))

        for k, (text, colour) in enumerate(lines[:10]):
            self.canvas.blit(self.panel_font.render(text, False, colour), (x + 10, y + 8 + 23 * k))

    def _px(self, x, y):
        """World metres -> map-panel pixels (+x right, +y up, origin at the centre)."""
        return (int(round(self.MAP_RES / 2 + x * self.scale)),
                int(round(self.MAP_RES / 2 - y * self.scale)))

    def _click_to_world(self, pos):
        u = pos[0] - self.MAP_POS[0]
        v = pos[1] - self.MAP_POS[1]
        if not (0 <= u < self.MAP_RES and 0 <= v < self.MAP_RES):
            return None
        x = (u - self.MAP_RES / 2) / self.scale
        y = (self.MAP_RES / 2 - v) / self.scale
        return (round(x, 3), round(y, 3))

    def _draw_map(self):
        res = self.MAP_RES
        s = self.scale
        img = np.empty((res, res, 3), dtype=np.uint8)
        img[:] = BG

        # 0.5 m grid and the axes through the arena centre
        for g in np.arange(-1.5, 1.51, 0.5):
            u, _ = self._px(g, 0)
            _, v = self._px(0, g)
            colour = AXIS if abs(g) < 1e-9 else GRID
            cv2.line(img, (u, 0), (u, res - 1), colour, 1)
            cv2.line(img, (0, v), (res - 1, v), colour, 1)
        cv2.putText(img, "+x", (res - 26, self._px(0, 0)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.putText(img, "+y", (self._px(0, 0)[0] + 6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)

        # boundary tape (inner edge, 2.5 m square) and the legal limit for the robot's centre
        inner = path_planner.ARENA_INNER_SIZE / 2
        cv2.rectangle(img, self._px(-inner, inner), self._px(inner, -inner), TAPE, 2)
        (xmin, xmax), (ymin, ymax) = path_planner.ARENA_BOUNDS
        cv2.rectangle(img, self._px(xmin, ymax), self._px(xmax, ymin), LIMIT, 1)

        # safety circles the planner is avoiding on this leg
        if self.obstacles is not None:
            for ox, oy, r in self.obstacles:
                cv2.circle(img, self._px(ox, oy), int(r * s), SAFETY, 1, cv2.LINE_AA)

        # fruits
        for name, (ox, oy) in self.object_positions.items():
            centre = self._px(ox, oy)
            r_px = max(4, int(self.object_radii.get(name, 0.04) * s))
            colour = FRUIT_RGB.get(name, DEFAULT_FRUIT_RGB)
            cv2.circle(img, centre, r_px, colour, -1, cv2.LINE_AA)
            cv2.circle(img, centre, r_px, (40, 40, 40), 1, cv2.LINE_AA)
            label = name[:3]
            if name in self.search_list:
                label = "{}:{}".format(self.search_list.index(name) + 1, label)
            label_colour = (0, 0, 0) if name in self.search_list else (95, 95, 95)
            cv2.putText(img, label, (centre[0] + r_px + 2, centre[1] - r_px),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, label_colour, 1, cv2.LINE_AA)
            status = self.target_status.get(name)
            if status is not None and status[1]:
                cv2.circle(img, centre, r_px + 4, SEEN, 2, cv2.LINE_AA)

        # Level 3: every FruitEKF estimate with its uncertainty ellipse -- the
        # same diamond + ellipse operate.py's draw_slam_state() uses, so it
        # reads like the M2 map. The ellipse shrinks as sightings from new
        # bearings pin the fruit down; a green ring means "well mapped".
        mapper = getattr(self.nav, 'fruit_mapper', None)
        state = self.ekf.robot.state
        r_px_robot = self._px(float(state[0, 0]), float(state[1, 0]))
        if mapper is not None:
            fe = mapper.fruit_ekf
            recent = time.time() - getattr(self.nav, '_last_detect_time', 0.0) < 2.0
            just_seen = {label for label, _ in getattr(self.nav, '_last_boxes', [])} if recent else set()
            for label, pos in fe.estimates.items():
                fx, fy = float(pos[0, 0]), float(pos[1, 0])
                centre = self._px(fx, fy)
                colour = FRUIT_RGB.get(label, DEFAULT_FRUIT_RGB)
                try:
                    axes_len, angle = self.ekf.make_ellipse(fe.P[label])
                    axes = (max(2, int(axes_len[0] * s)), max(2, int(axes_len[1] * s)))
                    cv2.ellipse(img, centre, axes, angle, 0, 360, colour, 1, cv2.LINE_AA)
                except Exception:
                    pass
                if label in just_seen:
                    cv2.line(img, r_px_robot, centre, colour, 1, cv2.LINE_AA)
                cv2.drawMarker(img, centre, colour, cv2.MARKER_DIAMOND, 10, 2)
                if mapper.well_mapped(label):
                    cv2.circle(img, centre, 9, SEEN, 1, cv2.LINE_AA)
                tag = label[:3]
                if label in self.search_list:
                    tag = "{}:{}".format(self.search_list.index(label) + 1, tag)
                cv2.putText(img, "{} {:.0f}cm".format(tag, mapper.sigma(label) * 100),
                            (centre[0] + 8, centre[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (30, 30, 30), 1, cv2.LINE_AA)

        # the current target's 0.4 m success zone
        if self.target_xy is not None and self.phase in ('run', 'wait'):
            cv2.circle(img, self._px(*self.target_xy), int(0.4 * s), ZONE, 1, cv2.LINE_AA)

        # markers: the physical 6 cm block, ringed green when the camera can see it
        for i, (mx, my) in enumerate(self.aruco_true_pos):
            centre = self._px(mx, my)
            half = max(3, int(0.03 * s))
            cv2.rectangle(img, (centre[0] - half, centre[1] - half), (centre[0] + half, centre[1] + half), (20, 20, 20), -1)
            if (i + 1) in self.visible_tags:
                cv2.circle(img, centre, half + 7, SEEN, 2, cv2.LINE_AA)

        # route
        state = self.ekf.robot.state
        rx, ry, rth = float(state[0, 0]), float(state[1, 0]), float(state[2, 0])
        if self.route:
            pts = [self._px(rx, ry)] + [self._px(*w) for w in self.route]
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, ROUTE, 2, cv2.LINE_AA)
            for w in self.route:
                cv2.circle(img, self._px(*w), 3, ROUTE, -1, cv2.LINE_AA)
        if self.active_wp is not None:
            cv2.circle(img, self._px(*self.active_wp), 7, ROUTE, 2, cv2.LINE_AA)
        if self.goal is not None:
            cv2.drawMarker(img, self._px(*self.goal), ROUTE, cv2.MARKER_TILTED_CROSS, 12, 2)
        if self.click_point is not None and self.phase == 'manual':
            cv2.drawMarker(img, self._px(*self.click_point), ROUTE, cv2.MARKER_TILTED_CROSS, 12, 2)

        # pose the visible markers imply, when it isn't locked in yet (or to compare)
        if self.ghost is not None:
            gx, gy, gth = self.ghost[:3]
            g = self._px(gx, gy)
            cv2.circle(img, g, int(path_planner.ROBOT_RADIUS * s), GHOST, 2, cv2.LINE_AA)
            tip = self._px(gx + 0.16 * math.cos(gth), gy + 0.16 * math.sin(gth))
            cv2.line(img, g, tip, GHOST, 2, cv2.LINE_AA)
            label = "marker fit" if self.ghost[4] >= 2 else "1-marker heading"
            cv2.putText(img, label, (g[0] + 18, g[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GHOST, 1, cv2.LINE_AA)

        # camera field of view, robot footprint and pose uncertainty. Before the
        # pose is locked in setup, the robot's heading is only assumed (centre,
        # facing +x, plus any turns made), so the view is drawn from the marker
        # fit when there is one.
        r = self._px(rx, ry)
        locked = self.nav._localised or self.phase != 'setup'
        # The camera wedge ALWAYS comes from the EKF heading -- the same theta the
        # robot sprite is rotated by -- so the two can never disagree. Before the
        # pose is locked in setup that heading is only assumed (centre, facing
        # +x, plus any turns made), so the wedge is drawn dimmer and labelled.
        fov_colour = FOV if locked else (225, 205, 140)
        for side in (-1, 1):
            a = rth + side * self.fov_half
            cv2.line(img, r, self._px(rx + 1.6 * math.cos(a), ry + 1.6 * math.sin(a)), fov_colour, 1, cv2.LINE_AA)
        if not locked:
            cv2.putText(img, "assumed", (r[0] - 30, r[1] - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
            # ...and the pose the visible markers imply gets its own wedge, in
            # the ghost colour, so it is obvious when the two do not line up
            # (that is what ENTER fixes). Once locked they are one and the same.
            if self.ghost is not None:
                gx, gy, gth = self.ghost[:3]
                for side in (-1, 1):
                    a = gth + side * self.fov_half
                    cv2.line(img, self._px(gx, gy), self._px(gx + 1.2 * math.cos(a), gy + 1.2 * math.sin(a)),
                             GHOST, 1, cv2.LINE_AA)
        cv2.circle(img, r, int(path_planner.ROBOT_RADIUS * s), ROBOT, 1, cv2.LINE_AA)
        try:
            axes_len, angle = self.ekf.make_ellipse(self.ekf.P[0:2, 0:2])
            axes = (max(1, int(axes_len[0] * s)), max(1, int(axes_len[1] * s)))
            cv2.ellipse(img, r, axes, angle, 0, 360, POSE_COV, 1)
        except Exception:
            pass

        img = np.ascontiguousarray(img)
        surface = pygame.image.frombuffer(img.tobytes(), (res, res), 'RGB').copy()

        # marker id icons and the robot sprite, same artwork as operate.py
        for i, (mx, my) in enumerate(self.aruco_true_pos):
            u, v = self._px(mx, my)
            pics = self.ekf.lm_pics
            pic = pics[i] if i < len(pics) - 1 else pics[-1]
            surface.blit(pic, (u - 5, v - 5))
        sprite = self.ekf.rot_center(self.ekf.pibot_pic, math.degrees(rth) + 180.0)
        surface.blit(sprite, (r[0] - 16, r[1] - 16))
        return surface
