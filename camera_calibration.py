#!/usr/bin/env python3
"""
calibrate.py - one-file camera calibration toolchain for the PenguinPi (ECE4078 M1)

Replaces  take_pic.py  +  camera_calibration.py  with a single script that also
tells you whether the result is any good.

    python calibrate.py board                       # make a board to print
    python calibrate.py capture --ip <ROBOT_IP>     # guided live capture
    python calibrate.py calib                       # fit the intrinsics
    python calibrate.py verify --ip <ROBOT_IP>      # check real ArUco distances
    python calibrate.py all    --ip <ROBOT_IP>      # capture then calib

WHY NOT JUST USE camera_calibration.py
    The stock script calls calibrateCamera on whatever images it finds and prints
    one RMS number.  That number is the error on the SAME images used to fit the
    model, so it always looks good and says nothing about whether the model
    generalises.  It is how you end up with a k3 of 10 - a distortion coefficient
    large enough to dominate the correction at the frame corners, fitted to noise,
    with a flattering RMS to go with it.

    What matters for SLAM is not reprojection error.  It is whether the camera
    reports the correct DISTANCE and BEARING to an ArUco marker.  Distance scales
    directly with the focal length; bearing is set by the principal point.  This
    script measures both.

WHAT IT DOES DIFFERENTLY
    1. Rejects blurry frames before they poison the fit (Laplacian variance).
    2. Supports a ChArUco board, so PARTIAL views count.  That is the only way to
       get corners into the frame corners, which is where distortion is otherwise
       unconstrained and free to invent itself.
    3. Shows corner coverage and measured board tilt LIVE, while you can still fix
       them, instead of after you have packed the robot away.
    4. Chooses the distortion model by K-FOLD CROSS-VALIDATION - fitted on some
       images, scored on images the fit never saw.
    5. Bootstraps the focal length so you get an honest uncertainty on every
       distance the robot will ever measure.
    6. Verifies against a tape measure, which is the only test that actually
       answers the question.

Layout: keep this file in  calibration/ , one level below botconnect.py, exactly
where take_pic.py lives now.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.abspath(os.path.join(_HERE, "..")), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PARAM_DIR = os.path.join(_HERE, "param")
PIC_DIR = os.path.join(_HERE, "calib_pics")
WIN = "calibrate"


# =============================================================================
#  Board: a ChArUco board or a plain checkerboard behind one interface
# =============================================================================
class Board:
    """
    ChArUco   : --cols/--rows are SQUARES.  Partial views are fine.
    Checkboard: --cols/--rows are INNER CORNERS.  The whole board must be visible.
    """

    def __init__(self, charuco, cols, rows, square, marker, min_corners=8):
        self.charuco = charuco
        self.cols, self.rows = cols, rows
        self.square, self.marker = square, marker
        self.min_corners = max(6, min_corners)

        if charuco:
            # DICT_5X5_250 is deliberately NOT the arena's DICT_4X4_100, so a board
            # marker can never be mistaken for a landmark, or vice versa.
            self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
            try:
                self.board = cv2.aruco.CharucoBoard((cols, rows), square, marker, self.dict)
            except (AttributeError, TypeError):
                self.board = cv2.aruco.CharucoBoard_create(cols, rows, square, marker, self.dict)
            try:
                self.obj_all = np.array(self.board.getChessboardCorners(), dtype=np.float32)
            except AttributeError:
                self.obj_all = np.array(self.board.chessboardCorners, dtype=np.float32)
            self._detector = self._make_detector()
            self.n_full = (cols - 1) * (rows - 1)
        else:
            objp = np.zeros((cols * rows, 3), np.float32)
            objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            self.obj_all = objp * square
            self.n_full = cols * rows

    # -- construction helpers -------------------------------------------------
    @staticmethod
    def add_args(ap):
        ap.add_argument("--checkerboard", action="store_true",
                        help="use a plain checkerboard instead of ChArUco "
                             "(then --cols/--rows are INNER CORNERS)")
        ap.add_argument("--cols", type=int, default=None, help="squares across (ChArUco) / inner corners across")
        ap.add_argument("--rows", type=int, default=None, help="squares down (ChArUco) / inner corners down")
        ap.add_argument("--square", type=float, default=None, help="MEASURED square side, metres")
        ap.add_argument("--marker", type=float, default=None, help="MEASURED marker side, metres (ChArUco)")
        ap.add_argument("--min-corners", type=int, default=8,
                        help="ChArUco: fewest interpolated corners that still counts as a view")

    @classmethod
    def from_args(cls, a):
        charuco = not getattr(a, "checkerboard", False)
        if charuco:
            cols = a.cols if a.cols else 7
            rows = a.rows if a.rows else 10
            square = a.square if a.square else 0.025
            marker = a.marker if a.marker else round(square * 0.76, 6)
        else:
            cols = a.cols if a.cols else 9
            rows = a.rows if a.rows else 6
            square = a.square if a.square else 0.023
            marker = 0.0
        return cls(charuco, cols, rows, square, marker, getattr(a, "min_corners", 8))

    def describe(self):
        if self.charuco:
            return ("ChArUco {}x{} squares, square {:.2f} mm, marker {:.2f} mm "
                    "(DICT_5X5_250)".format(self.cols, self.rows,
                                            self.square * 1000, self.marker * 1000))
        return "checkerboard {}x{} inner corners, square {:.2f} mm".format(
            self.cols, self.rows, self.square * 1000)

    def _make_detector(self):
        if hasattr(cv2.aruco, "CharucoDetector"):
            return ("class", cv2.aruco.CharucoDetector(self.board))
        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
            return ("legacy", None)
        raise RuntimeError("This OpenCV build has neither cv2.aruco.CharucoDetector nor "
                           "interpolateCornersCharuco.  pip install opencv-contrib-python")

    # -- detection ------------------------------------------------------------
    def detect(self, gray):
        """Returns (objp Nx3 float32, imgp Nx1x2 float32) or (None, None)."""
        if self.charuco:
            return self._detect_charuco(gray)
        return self._detect_chessboard(gray)

    def _detect_charuco(self, gray):
        kind, obj = self._detector
        if kind == "class":
            ch_c, ch_i, _, _ = obj.detectBoard(gray)
        else:
            if hasattr(cv2.aruco, "ArucoDetector"):
                ad = cv2.aruco.ArucoDetector(self.dict, cv2.aruco.DetectorParameters())
                corners, ids, _ = ad.detectMarkers(gray)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dict)
            if ids is None or len(ids) == 0:
                return None, None
            _, ch_c, ch_i = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
        if ch_i is None or len(ch_i) < self.min_corners:
            return None, None
        objp = np.array([self.obj_all[int(i)] for i in ch_i.ravel()], dtype=np.float32)
        return objp, np.asarray(ch_c, dtype=np.float32).reshape(-1, 1, 2)

    def _detect_chessboard(self, gray):
        pattern = (self.cols, self.rows)
        corners = None
        if hasattr(cv2, "findChessboardCornersSB"):
            # markedly more accurate than the classic detector, and far more robust
            # to uneven lighting - but it only exists in newer OpenCV
            ok, c = cv2.findChessboardCornersSB(
                gray, pattern, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok:
                corners = c
        if corners is None:
            ok, c = cv2.findChessboardCorners(
                gray, pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                return None, None
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            corners = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
        return self.obj_all.copy(), np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)


# =============================================================================
#  Small shared helpers
# =============================================================================
def sharpness(gray):
    """Variance of the Laplacian.  Low means blurred."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def guess_K(w, h, param_dir=PARAM_DIR):
    """A rough camera matrix, only ever used for the live tilt readout."""
    p = os.path.join(param_dir, "intrinsic.txt")
    if os.path.exists(p):
        try:
            K = np.loadtxt(p, delimiter=',')
            if K.shape == (3, 3) and K[0, 0] > 1:
                return K
        except Exception:
            pass
    f = 1.69 * w          # this camera runs a cropped sensor mode: ~33 deg HFOV
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def board_pose(objp, imgp, K):
    """(tilt_deg, distance_m) of the board, or None.  Distortion ignored - this is
    only for on-screen guidance, not for the fit."""
    if objp is None or len(objp) < 6:
        return None
    for flag in (cv2.SOLVEPNP_IPPE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, None, flags=flag)
        except cv2.error:
            continue
        if ok:
            R, _ = cv2.Rodrigues(rvec)
            nz = abs(float(R[2, 2]))
            tilt = float(np.degrees(np.arccos(min(1.0, nz))))
            return tilt, float(np.linalg.norm(tvec))
    return None


class Coverage:
    """Where in the frame have we actually SEEN corners?  Distortion is only
    constrained where corners land; everywhere else it extrapolates."""

    def __init__(self, w, h, grid=6):
        self.w, self.h, self.grid = w, h, grid
        self.counts = np.zeros((grid, grid), dtype=int)

    def _cell(self, pt):
        gx = min(self.grid - 1, max(0, int(pt[0] / self.w * self.grid)))
        gy = min(self.grid - 1, max(0, int(pt[1] / self.h * self.grid)))
        return gy, gx

    def add(self, imgp, sign=1):
        for pt in np.asarray(imgp).reshape(-1, 2):
            gy, gx = self._cell(pt)
            self.counts[gy, gx] += sign

    def draw(self, vis):
        cw, ch = self.w / self.grid, self.h / self.grid
        for gy in range(self.grid):
            for gx in range(self.grid):
                n = self.counts[gy, gx]
                col = (60, 60, 200) if n == 0 else ((0, 170, 220) if n < 15 else (60, 190, 60))
                x0, y0 = int(gx * cw), int(gy * ch)
                cv2.rectangle(vis, (x0, y0), (int(x0 + cw) - 1, int(y0 + ch) - 1), col, 1)
                cv2.putText(vis, str(int(n)), (x0 + 4, y0 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        return vis

    def report(self):
        print("\nCorner coverage across the frame ({0}x{0} cells, '.' = never seen):"
              .format(self.grid))
        for row in self.counts:
            print("   " + "".join("  . " if v == 0 else "{:3d} ".format(v) for v in row))
        empty = int((self.counts == 0).sum())
        total = self.counts.sum()
        corner_cells = [self.counts[0, 0], self.counts[0, -1],
                        self.counts[-1, 0], self.counts[-1, -1]]
        print("   {} of {} cells never saw a corner.".format(empty, self.grid ** 2))
        if empty > 0 or (total and min(corner_cells) < total * 0.005):
            print("   WARNING: frame edges/corners are poorly covered.  Distortion")
            print("   coefficients fitted there are extrapolation, not measurement.")
            print("   Re-shoot pushing the board INTO each corner of the image.")
        else:
            print("   Coverage looks good.")
        return empty


def text_block(vis, lines, org=(8, 20), scale=0.5):
    """Draw text with a dark backing box so it stays readable over the image."""
    x, y = org
    for txt, col in lines:
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(vis, (x - 4, y - th - 4), (x + tw + 4, y + 5), (0, 0, 0), -1)
        cv2.putText(vis, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += int(th + 12)
    return vis


# =============================================================================
#  Frame source: the robot, a webcam, or a folder (for testing the UI offline)
# =============================================================================
class FrameSource:
    def __init__(self, source, ip=None):
        self.kind = "robot"
        self.bot = None
        self.cap = None
        self.files, self.i = [], 0

        if source in (None, "robot"):
            from botconnect import BotConnect          # noqa: import kept local
            print("Connecting to robot at {} ...".format(ip))
            self.bot = BotConnect(ip)
            t0 = time.time()
            while time.time() - t0 < 15:
                f = self._robot_frame()
                if f is not None:
                    print("Live frame: {}x{}".format(f.shape[1], f.shape[0]))
                    if f.shape[1] < 600:
                        print("  NOTE: that is smaller than the usual 640x480.  If it")
                        print("  says 480x360 the camera thread had not delivered a real")
                        print("  frame yet - quit and rerun.")
                    return
                time.sleep(0.2)
            raise RuntimeError("No frames from the robot after 15 s.  Check the IP.")
        elif source.startswith("webcam"):
            idx = int(source.split(":")[1]) if ":" in source else 0
            self.kind = "webcam"
            self.cap = cv2.VideoCapture(idx)
            if not self.cap.isOpened():
                raise RuntimeError("Cannot open webcam {}".format(idx))
        else:
            self.kind = "folder"
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                self.files += glob.glob(os.path.join(source, ext))
            self.files.sort()
            if not self.files:
                raise RuntimeError("No images in {}".format(source))

    def _robot_frame(self):
        img = self.bot.get_image()          # BotConnect hands back RGB
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def read(self):
        """Always returns BGR, ready for imshow/imwrite."""
        if self.kind == "robot":
            for _ in range(50):
                f = self._robot_frame()
                if f is not None:
                    return f
                time.sleep(0.02)
            return None
        if self.kind == "webcam":
            ok, f = self.cap.read()
            return f if ok else None
        f = cv2.imread(self.files[self.i % len(self.files)])
        self.i += 1
        return f

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.bot is not None:
            try:
                self.bot.running = False
            except Exception:
                pass


# =============================================================================
#  board:  generate something to print
# =============================================================================
def cmd_board(a):
    board = Board.from_args(a)
    if not board.charuco:
        print("The board generator only makes ChArUco boards.  For a checkerboard, use")
        print("the sheet the lab gave you, then pass --checkerboard to capture/calib.")
        return 1

    # generateImage insists on a whole number of pixels per square, so round the
    # per-square size and report the dpi that actually results
    sq_mm = board.square * 1000
    pps = max(8, int(round(sq_mm * a.dpi / 25.4)))
    px = (board.cols * pps, board.rows * pps)
    dpi = pps / sq_mm * 25.4
    w_mm, h_mm = board.cols * sq_mm, board.rows * sq_mm
    try:
        img = board.board.generateImage(px, marginSize=0, borderBits=1)
    except AttributeError:
        img = board.board.draw(px, marginSize=0, borderBits=1)

    png = os.path.join(_HERE, "charuco_board.png")
    cv2.imwrite(png, img)
    print("Wrote {}  ({}x{} px = {:.1f} x {:.1f} mm, so print it at {:.1f} dpi)".format(
        png, px[0], px[1], w_mm, h_mm, dpi))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib not installed, so no PDF.  Print the PNG at exactly {} dpi."
              .format(dpi))
        return 0

    PW, PH = 210.0, 297.0                             # A4 in mm
    if w_mm > PW - 20 or h_mm > PH - 30:
        print("WARNING: board is {:.0f}x{:.0f} mm, which will not fit A4 with margins."
              .format(w_mm, h_mm))
    fig = plt.figure(figsize=(PW / 25.4, PH / 25.4))
    ax = fig.add_axes([(PW - w_mm) / 2 / PW, (PH - h_mm) / 2 / PH + 0.03, w_mm / PW, h_mm / PH])
    ax.imshow(img, cmap="gray", interpolation="none", aspect="auto")
    ax.axis("off")

    # A printed ruler, so the first thing you do is check the scale.
    y = 0.045
    x0 = (PW - 100.0) / 2 / PW
    fig.add_artist(Line2D([x0, x0 + 100.0 / PW], [y, y], color="black", lw=1.2))
    for t in range(0, 101, 10):
        xx = x0 + t / PW
        fig.add_artist(Line2D([xx, xx], [y, y + (0.008 if t % 50 else 0.014)], color="black", lw=1.2))
    fig.text(0.5, y - 0.018, "this line is exactly 100 mm - measure it before you use the board",
             ha="center", fontsize=8)
    fig.text(0.5, 1 - 0.035,
             "ChArUco {}x{} squares  |  nominal square {:.2f} mm, marker {:.2f} mm  |  "
             "print at 100% / Actual size".format(board.cols, board.rows,
                                                  board.square * 1000, board.marker * 1000),
             ha="center", fontsize=8)
    pdf = os.path.join(_HERE, "charuco_board.pdf")
    fig.savefig(pdf)
    plt.close(fig)
    print("Wrote {}".format(pdf))
    print("""
NEXT
  1. Print at 100%% / "Actual size".  NOT "Fit to page".
  2. Measure the 100 mm ruler on the print.  Printers scale even when told not to.
  3. Measure across all %d squares one way and all %d the other, divide, and check
     the two answers agree within about 1%%.  If they do not, the print is stretched
     and will corrupt the focal length - print it somewhere else.
  4. Glue it to card or a clipboard.  A curled page bends the exact geometry you
     are trying to solve for.
  5. Pass the MEASURED numbers to capture and calib:
        --square <measured m>  --marker <measured * 0.76>
""" % (board.cols, board.rows))
    return 0


# =============================================================================
#  capture:  guided live capture
# =============================================================================
TARGETS = {
    "CENTRE":       (0.50, 0.50), "LEFT": (0.15, 0.50), "RIGHT": (0.85, 0.50),
    "TOP":          (0.50, 0.15), "BOTTOM": (0.50, 0.85),
    "TOP-LEFT":     (0.15, 0.15), "TOP-RIGHT": (0.85, 0.15),
    "BOTTOM-LEFT":  (0.15, 0.85), "BOTTOM-RIGHT": (0.85, 0.85),
}


def build_plan(charuco):
    """27 shots in 4 distance stations: set up at one distance, do every position and
    tilt there, then move.  `clip` shots deliberately let the frame cut the board -
    only possible with ChArUco, which is the whole reason for using one."""
    P = []

    def add(d, where, tilt, axis, clip=False):
        P.append(dict(dist=d, where=where, tilt=tilt, axis=axis, clip=clip and charuco))

    for d, shots in [
        (0.35, [("CENTRE", 0, "flat"), ("CENTRE", 30, "yaw"), ("CENTRE", 30, "pitch"),
                ("TOP-LEFT", 15, "yaw", True), ("BOTTOM-RIGHT", 15, "pitch", True)]),
        (0.55, [("CENTRE", 0, "flat"), ("LEFT", 25, "yaw"), ("RIGHT", 25, "yaw"),
                ("TOP", 25, "pitch"), ("BOTTOM", 25, "pitch"),
                ("TOP-RIGHT", 20, "yaw", True), ("BOTTOM-LEFT", 20, "pitch", True)]),
        (0.80, [("CENTRE", 0, "flat"), ("CENTRE", 40, "yaw"), ("CENTRE", 40, "pitch"),
                ("LEFT", 30, "yaw"), ("RIGHT", 30, "yaw"),
                ("TOP-LEFT", 20, "yaw", True), ("TOP-RIGHT", 20, "yaw", True),
                ("BOTTOM-LEFT", 20, "pitch", True), ("BOTTOM-RIGHT", 20, "pitch", True)]),
        (1.10, [("CENTRE", 0, "flat"), ("CENTRE", 30, "yaw"), ("CENTRE", 30, "pitch"),
                ("CENTRE", 45, "roll"), ("TOP-LEFT", 20, "yaw", True),
                ("BOTTOM-RIGHT", 20, "pitch", True)]),
    ]:
        for s in shots:
            add(d, s[0], s[1], s[2], len(s) > 3)
    return P


AXIS_HELP = {
    "flat":  "square on to the camera",
    "yaw":   "rotate about the VERTICAL axis - one side edge nearer than the other",
    "pitch": "rotate about the HORIZONTAL axis - top nearer than bottom",
    "roll":  "spin the board in its own plane, stays face-on",
}


def cmd_capture(a):
    board = Board.from_args(a)
    out = a.dir or PIC_DIR
    if a.fresh and os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    print(board.describe())
    if board.charuco and abs(board.marker / board.square - 0.76) > 0.05:
        print("NOTE: marker/square is normally 0.76.  Yours is {:.3f} - check you measured"
              .format(board.marker / board.square))
        print("      both numbers off the same print.")
    print("""
CONTROLS   SPACE capture   s skip this shot   d delete last & redo   f force   q quit
IMPORTANT  click the image window before pressing keys - OpenCV only receives
           keystrokes when its own window has focus.
SETUP      raise the robot ~30 cm on a box.  The camera sits ~6 cm off the ground,
           so on the floor you physically cannot put the board below the camera
           axis, and the bottom of the frame never gets any corners.
""")
    src = FrameSource(a.source, a.ip)
    plan = build_plan(board.charuco)
    saved = []                        # [(path, imgp)]
    cov = None
    idx = 0
    msg = ""
    cv2.namedWindow(WIN)

    while idx < len(plan):
        frame = src.read()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        if cov is None:
            cov = Coverage(w, h)
            K = guess_K(w, h, a.out)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objp, imgp = board.detect(gray)
        sharp = sharpness(gray)
        shot = plan[idx]

        vis = frame.copy()
        cov.draw(vis)

        # where the board should be
        tx, ty = TARGETS[shot["where"]]
        cv2.drawMarker(vis, (int(tx * w), int(ty * h)), (0, 255, 255),
                       cv2.MARKER_CROSS, 26, 2)
        cv2.circle(vis, (int(tx * w), int(ty * h)), 34, (0, 255, 255), 1)

        ok_detect = imgp is not None
        n = len(imgp) if ok_detect else 0
        tilt = dist = None
        if ok_detect:
            for pt in imgp.reshape(-1, 2):
                cv2.circle(vis, (int(pt[0]), int(pt[1])), 2, (0, 220, 0), -1)
            c = imgp.reshape(-1, 2).mean(axis=0)
            cv2.circle(vis, (int(c[0]), int(c[1])), 6, (255, 120, 0), 2)
            cv2.line(vis, (int(c[0]), int(c[1])), (int(tx * w), int(ty * h)), (255, 120, 0), 1)
            pose = board_pose(objp, imgp, K)
            if pose:
                tilt, dist = pose

        if not ok_detect:
            status, scol = "board NOT detected - too far out of frame, or too oblique", (60, 60, 255)
        elif sharp < a.blur:
            status, scol = "TOO BLURRY ({:.0f}) - brace your elbows".format(sharp), (0, 165, 255)
        else:
            status, scol = "ready - press SPACE", (60, 220, 60)

        clip_note = "  [LET THE FRAME CUT IT - aim for 20-35 corners]" if shot["clip"] else ""
        lines = [
            ("shot {}/{}   station {:.2f} m".format(idx + 1, len(plan), shot["dist"]), (255, 255, 255)),
            ("put the board {}{}".format(shot["where"], clip_note), (0, 255, 255)),
            ("tilt ~{} deg {}: {}".format(shot["tilt"], shot["axis"], AXIS_HELP[shot["axis"]]),
             (0, 255, 255)),
            ("measured tilt {}   dist {}   corners {}/{}".format(
                "--" if tilt is None else "{:.0f} deg".format(tilt),
                "--" if dist is None else "{:.2f} m".format(dist), n, board.n_full),
             (255, 255, 255)),
            (status, scol),
        ]
        if msg:
            lines.append((msg, (255, 200, 0)))
        text_block(vis, lines)
        cv2.imshow(WIN, vis)

        k = cv2.waitKey(30) & 0xFF
        msg = ""
        if k == ord('q'):
            break
        elif k == ord('s'):
            msg = "skipped"
            idx += 1
        elif k == ord('d'):
            if saved:
                p, ip = saved.pop()
                cov.add(ip, sign=-1)
                os.remove(p)
                idx = max(0, idx - 1)
                msg = "deleted {}".format(os.path.basename(p))
            else:
                msg = "nothing to delete"
        elif k in (ord(' '), ord('f')):
            if not ok_detect:
                msg = "cannot save: no board detected"
            elif sharp < a.blur and k != ord('f'):
                msg = "too blurry - hold still, or press f to force"
            else:
                name = "calib_{:02d}_{:.0f}cm_{}.png".format(
                    len(saved), shot["dist"] * 100, shot["where"].lower().replace("-", ""))
                path = os.path.join(out, name)
                cv2.imwrite(path, frame)
                saved.append((path, imgp))
                cov.add(imgp)
                msg = "saved {} ({} corners)".format(name, n)
                print("  {}  {} corners, tilt {}, sharpness {:.0f}".format(
                    name, n, "--" if tilt is None else "{:.0f}".format(tilt), sharp))
                idx += 1

    cv2.destroyAllWindows()
    src.close()
    print("\nSaved {} images to {}".format(len(saved), out))
    if cov is not None:
        cov.report()
    if len(saved) < 20:
        print("\nFewer than 20 images.  Rerun WITHOUT --fresh to add more without")
        print("losing these.  More VARIETY (distance, tilt) helps; more of the same")
        print("geometry does not.")
    return 0


# =============================================================================
#  calib:  fit, with the model chosen by cross-validation
# =============================================================================
MODELS = [
    # fx == fy enforced.  The sensor has square pixels, so they SHOULD be equal;
    # letting them float independently spends a parameter on noise.
    ("fixed aspect, k1,k2", cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST),
    ("k1,k2 only (no k3)", cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST),
    ("k1,k2 + tangential", cv2.CALIB_FIX_K3),
    ("k1,k2,k3 + tangential", 0),
    ("rational (k1..k6)", cv2.CALIB_RATIONAL_MODEL),
]


def collect(image_dir, board, blur_threshold):
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    paths = sorted(set(paths))
    if not paths:
        raise RuntimeError("No images found in {}".format(image_dir))

    objpoints, imgpoints, used, shape = [], [], [], None
    print("Screening {} images...".format(len(paths)))
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print("  {:30s} unreadable, skipped".format(os.path.basename(p)))
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if shape is None:
            shape = gray.shape[::-1]
        elif gray.shape[::-1] != shape:
            # mixing resolutions silently produces a meaningless calibration
            print("  {:30s} WRONG SIZE {}, skipped".format(os.path.basename(p), gray.shape[::-1]))
            continue
        s = sharpness(gray)
        if s < blur_threshold:
            print("  {:30s} too blurry ({:.0f}), skipped".format(os.path.basename(p), s))
            continue
        objp, imgp = board.detect(gray)
        if objp is None:
            print("  {:30s} board not found / too few corners, skipped".format(os.path.basename(p)))
            continue
        objpoints.append(objp)
        imgpoints.append(imgp)
        used.append(p)
        print("  {:30s} OK  {:3d} corners, sharpness {:.0f}".format(
            os.path.basename(p), len(objp), s))
    return objpoints, imgpoints, used, shape


def _fit(objp, imgp, shape, flags):
    return cv2.calibrateCamera(objp, imgp, shape, None, None, flags=flags)


def holdout_error(objp_test, imgp_test, K, D):
    """Reprojection error on images the fit never saw.  Pose is solved per image with
    the intrinsics held fixed, so this measures generalisation, not memorisation."""
    errs = []
    for op, ip in zip(objp_test, imgp_test):
        try:
            ok, rvec, tvec = cv2.solvePnP(op, ip, K, D)
        except cv2.error:
            continue
        if not ok:
            continue
        proj, _ = cv2.projectPoints(op, rvec, tvec, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - ip.reshape(-1, 2)) ** 2, axis=1)))))
    return float(np.mean(errs)) if errs else float("inf")


def cross_validate(objp, imgp, shape, flags, folds=4):
    n = len(objp)
    if n < folds * 2:
        return float("nan")
    idx = np.arange(n)
    np.random.default_rng(0).shuffle(idx)
    scores = []
    for f in range(folds):
        test = set(idx[f::folds].tolist())
        train = [i for i in idx if i not in test]
        if len(train) < 6:
            continue
        try:
            _, K, D, _, _ = _fit([objp[i] for i in train], [imgp[i] for i in train], shape, flags)
        except cv2.error:
            return float("inf")
        scores.append(holdout_error([objp[i] for i in sorted(test)],
                                    [imgp[i] for i in sorted(test)], K, D))
    return float(np.mean(scores)) if scores else float("nan")


def per_image_errors(objp, imgp, K, D, rvecs, tvecs):
    errs = []
    for i in range(len(objp)):
        proj, _ = cv2.projectPoints(objp[i], rvecs[i], tvecs[i], K, D)
        errs.append(float(np.sqrt(np.mean(np.sum(
            (proj.reshape(-1, 2) - imgp[i].reshape(-1, 2)) ** 2, axis=1)))))
    return np.array(errs)


def sanity_check(K, D, shape, log):
    w, h = shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    log("\nSanity checks:")
    off = float(np.hypot(cx - w / 2.0, cy - h / 2.0))
    log("  principal point   ({:.1f}, {:.1f}), {:.1f} px from centre  [{}]".format(
        cx, cy, off, "ok" if off < 0.05 * w else "SUSPECT - should sit near the centre"))
    log("    -> constant bearing bias {:+.2f} deg yaw, {:+.2f} deg pitch".format(
        np.degrees(np.arctan((cx - w / 2) / fx)), np.degrees(np.arctan((cy - h / 2) / fy))))
    log("    -> lateral error on a marker 2 m away: {:+.0f} mm".format(
        2000 * (cx - w / 2) / fx))
    r = fx / fy
    log("  fx/fy ratio       {:.4f}  [{}]".format(
        r, "ok" if 0.98 < r < 1.02 else "SUSPECT - pixels should be near square"))
    log("  horizontal FOV    {:.1f} deg  (narrow is expected: cropped sensor mode)".format(
        2 * np.degrees(np.arctan(w / (2 * fx)))))
    d = np.asarray(D).ravel()
    log("  distortion        " + ", ".join("{:.4f}".format(v) for v in d[:5]))
    if len(d) > 4 and abs(d[4]) > 2.0:
        log("    WARNING: |k3| = {:.2f} is large.  Check what fraction of the".format(abs(d[4])))
        log("    correction it carries at the frame corner - if it dominates there,")
        log("    it is fitted to noise, not to the lens.")
    # how much of the correction does k3 carry where you have least data?
    if len(d) >= 5:
        rmax = float(np.hypot(max(cx, w - cx), max(cy, h - cy)) / fx)
        k1, k2, k3 = d[0], d[1], d[4]
        tot = k1 * rmax ** 2 + k2 * rmax ** 4 + k3 * rmax ** 6
        if abs(tot) > 1e-9:
            frac = abs(k3 * rmax ** 6) / abs(tot) * 100
            log("  k3 share of the radial correction at the frame corner: {:.0f}%".format(frac))
            if frac > 60:
                log("    WARNING: over 60% - the highest-order term is doing most of the")
                log("    work in the region you have the fewest corners.  Classic overfit.")


def cmd_calib(a):
    board = Board.from_args(a)
    print(board.describe())
    objp, imgp, used, shape = collect(a.dir or PIC_DIR, board, a.blur)

    print("\n{} usable images at {}x{}".format(len(used), shape[0], shape[1]))
    if len(used) < 10:
        raise RuntimeError("Need at least 10 usable images; 25-40 is much better.")
    if len(used) < 20:
        print("WARNING: fewer than 20 images.  The fit will be noticeably less stable.")

    cov = Coverage(shape[0], shape[1])
    for ip in imgp:
        cov.add(ip)
    cov.report()

    report = []

    def log(s=""):
        print(s)
        report.append(s)

    log("\nComparing distortion models by {}-fold cross-validation.".format(a.folds))
    log("In-sample RMS always improves with more parameters, which is exactly why it")
    log("is the wrong thing to choose on.  Trust the held-out column.\n")
    log("  {:26s} {:>12s} {:>14s}".format("model", "in-sample", "held-out"))
    results = []
    for name, flags in MODELS:
        try:
            rms, K, D, _, _ = _fit(objp, imgp, shape, flags)
        except cv2.error as e:
            log("  {:26s} failed ({})".format(name, str(e).split('\n')[0][:40]))
            continue
        cv_err = cross_validate(objp, imgp, shape, flags, a.folds)
        results.append((cv_err, name, flags))
        log("  {:26s} {:12.4f} {:14.4f}".format(name, rms, cv_err))
    if not results:
        raise RuntimeError("Every model failed to fit.")
    results.sort(key=lambda r: (np.inf if np.isnan(r[0]) else r[0]))
    cv_err, best_name, best_flags = results[0]
    log("\nSelected: {}  (held-out error {:.4f} px)".format(best_name, cv_err))

    rms, K, D, rvecs, tvecs = _fit(objp, imgp, shape, best_flags)
    errs = per_image_errors(objp, imgp, K, D, rvecs, tvecs)
    thresh = errs.mean() + a.outlier * errs.std()
    keep = [i for i in range(len(errs)) if errs[i] <= thresh]
    if len(keep) < len(errs):
        log("\nDropping {} outlier image(s) above {:.3f} px:".format(len(errs) - len(keep), thresh))
        for i in range(len(errs)):
            if i not in keep:
                log("   {:30s} {:.3f} px".format(os.path.basename(used[i]), errs[i]))
        rms, K, D, rvecs, tvecs = _fit([objp[i] for i in keep], [imgp[i] for i in keep],
                                       shape, best_flags)
        errs = per_image_errors([objp[i] for i in keep], [imgp[i] for i in keep],
                                K, D, rvecs, tvecs)
        log("Refitted on {} images.".format(len(keep)))

    log("\nFinal RMS reprojection error: {:.4f} px".format(rms))
    log("  per-image: min {:.3f}, median {:.3f}, max {:.3f}".format(
        errs.min(), float(np.median(errs)), errs.max()))
    log("\nCamera matrix:")
    log("  fx {:.3f}   fy {:.3f}".format(K[0, 0], K[1, 1]))
    log("  cx {:.3f}   cy {:.3f}".format(K[0, 2], K[1, 2]))
    sanity_check(K, D, shape, log)

    # --- how repeatable is fx? ------------------------------------------------
    # Reprojection error says nothing about whether the FOCAL LENGTH is stable, and
    # focal length is what sets ArUco range: 1% error in fx is 1% error in every
    # distance the robot measures, forever.
    log("\nFocal-length stability (bootstrap over {} resamples):".format(a.boot))
    rng = np.random.default_rng(0)
    n = len(objp)
    fxs = []
    for _ in range(a.boot):
        pick = rng.choice(n, size=max(8, int(n * 0.7)), replace=False)
        try:
            _, Kb, _, _, _ = _fit([objp[i] for i in pick], [imgp[i] for i in pick],
                                  shape, best_flags)
            fxs.append(float(Kb[0, 0]))
        except cv2.error:
            continue
    if len(fxs) >= 4:
        fxs = np.array(fxs)
        spread = fxs.std() / fxs.mean() * 100
        log("  fx = {:.1f} +/- {:.1f}  ({:.2f}% spread)".format(fxs.mean(), fxs.std(), spread))
        log("  -> every ArUco distance inherits about this much uncertainty.")
        if spread > 1.0:
            log("  WARNING: over 1%.  The image set is not constraining fx well.")
            log("  Add frames at MORE VARIED DISTANCES AND TILTS - not just more frames.")
            log("  Rerun capture WITHOUT --fresh to add to what you already have.")
        else:
            log("  Good - fx is well determined by this image set.")
    else:
        log("  too few images to bootstrap")

    # --- back up whatever is there now, then compare --------------------------
    os.makedirs(a.out, exist_ok=True)
    cur_K = os.path.join(a.out, "intrinsic.txt")
    cur_D = os.path.join(a.out, "distCoeffs.txt")
    bak_K = os.path.join(a.out, "intrinsic_WORKING.txt")
    bak_D = os.path.join(a.out, "distCoeffs_WORKING.txt")
    if os.path.exists(cur_K) and not os.path.exists(bak_K):
        shutil.copyfile(cur_K, bak_K)
        if os.path.exists(cur_D):
            shutil.copyfile(cur_D, bak_D)
        print("\nBacked up your existing calibration to intrinsic_WORKING.txt /")
        print("distCoeffs_WORKING.txt.  If this all goes wrong, copy those back.")
    if os.path.exists(bak_K):
        try:
            old = np.loadtxt(bak_K, delimiter=',')
            change = (K[0, 0] / old[0][0] - 1) * 100
            log("\nCompared with intrinsic_WORKING.txt:")
            log("  fx {:.1f} -> {:.1f}  ({:+.2f}%)".format(old[0][0], K[0, 0], change))
            log("  cx {:.1f} -> {:.1f},  cy {:.1f} -> {:.1f}".format(
                old[0][2], K[0, 2], old[1][2], K[1, 2]))
            log("  ArUco ranges scale with fx, so every distance the robot measures")
            log("  should move by about {:+.2f}%.  If your bench test said ranges were".format(change))
            log("  too LONG, you want this number NEGATIVE.")
        except Exception:
            pass

    if a.dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    np.savetxt(cur_K, K, delimiter=',')
    # The rational model returns 8 coefficients.  solvePnP accepts them, but keep the
    # file shape predictable for whatever else in the repo reads it.
    np.savetxt(cur_D, np.asarray(D).reshape(1, -1), delimiter=',')
    with open(os.path.join(a.out, "calibration_report.txt"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("\nWrote {}/intrinsic.txt, {}/distCoeffs.txt and calibration_report.txt".format(
        a.out, a.out))
    print("\nNEXT: reprojection error does NOT tell you whether ranges are right.")
    print("      python calibrate.py verify --ip <IP>")
    return 0


# =============================================================================
#  verify:  do the measured distances actually match a tape measure?
# =============================================================================
def load_params(param_dir):
    K = np.loadtxt(os.path.join(param_dir, "intrinsic.txt"), delimiter=',')
    D = np.loadtxt(os.path.join(param_dir, "distCoeffs.txt"), delimiter=',').reshape(1, -1)
    return K, D


def measure_markers(img, K, D, detector, marker_length):
    """Same convention as slam/aruco_sensor.py: forward = +z, lateral = -x."""
    corners, ids, _ = detector.detectMarkers(img)
    out = []
    if ids is None or len(corners) == 0:
        return out
    half = marker_length / 2.0
    obj = np.array([[-half, half, 0], [half, half, 0],
                    [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
    for c, i in zip(corners, ids.ravel()):
        ok, rvec, tvec = cv2.solvePnP(obj, c[0], K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        t = tvec.ravel()
        fwd, lat = float(t[2]), float(-t[0])
        out.append(dict(tag=int(i), forward=fwd, lateral=lat,
                        rng=float(np.hypot(fwd, lat)),
                        bearing=float(np.degrees(np.arctan2(lat, fwd))),
                        corners=c))
    return out


def analyse(rows, label=""):
    if not rows:
        print("No readings to analyse.")
        return None
    print("\n=== {} {} reading(s) ===".format(label or "results", len(rows)))
    print("  {:>6s} {:>7s} | {:>8s} {:>8s} | {:>7s} {:>8s}".format(
        "true_f", "true_l", "meas_f", "meas_l", "ratio", "d_brg"))
    ratios, dbrg = [], []
    for r in rows:
        tf, tl = r["true_forward"], r["true_lateral"]
        tr = float(np.hypot(tf, tl))
        if tr <= 0:
            continue
        ratio = r["meas_range"] / tr
        db = r["meas_bearing"] - float(np.degrees(np.arctan2(tl, tf)))
        ratios.append(ratio)
        dbrg.append(db)
        print("  {:6.2f} {:7.3f} | {:8.3f} {:8.3f} | {:7.4f} {:+8.2f}".format(
            tf, tl, r["meas_forward"], r["meas_lateral"], ratio, db))
    if not ratios:
        return None
    ratios, dbrg = np.array(ratios), np.array(dbrg)
    print("\n  measured range / true range: mean {:.4f}  (i.e. {:+.2f}%), spread +/-{:.2f}%".format(
        ratios.mean(), (ratios.mean() - 1) * 100, ratios.std() * 100))
    m = abs(ratios.mean() - 1) * 100
    if m < 2:
        print("  -> within 2%.  Calibration is good, keep it.")
    elif m < 5:
        print("  -> {:.1f}% off.  Better than nothing but there is more to get.".format(m))
    else:
        print("  -> {:.1f}% off.  This did NOT fix the problem.  Restore".format(m))
        print("     intrinsic_WORKING.txt / distCoeffs_WORKING.txt before your next run.")
    print("\n  bearing offset: mean {:+.2f} deg (spread {:.2f})".format(dbrg.mean(), dbrg.std()))
    print("  A constant offset means the camera is mounted slightly rotated on the")
    print("  chassis, or the principal point is off.  It shows up as lateral error")
    print("  growing with distance: {:+.0f} mm at 2 m.".format(
        2000 * np.tan(np.radians(dbrg.mean()))))
    if abs(dbrg.mean()) < 1.0:
        print("  Under 1 deg - not worth chasing.")
    return ratios.mean()


def read_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            rows.append({k: (float(v) if k != "tag" else int(float(v))) for k, v in d.items()})
    return rows


def cmd_verify(a):
    K, D = load_params(a.out)
    print("Using fx {:.1f}, cx {:.1f}, cy {:.1f}".format(K[0, 0], K[0, 2], K[1, 2]))
    csv_path = a.csv or os.path.join(_HERE, "marker_accuracy_data.csv")
    if a.analyse_only:
        return 0 if analyse(read_csv(csv_path), "logged") is not None else 1
    if a.fresh and os.path.exists(csv_path):
        os.rename(csv_path, csv_path.replace(".csv", "_OLD.csv"))
        print("Moved the old CSV aside - the analysis pools everything in the file.")

    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100),
        cv2.aruco.DetectorParameters())
    print("""
SETUP  Robot stationary.  Tape measure on the floor running straight out from the
       lens, zero AT the lens.  Pick one reference point on the robot and use it
       every single time - consistency matters more than being exactly right.
       Stand one arena ArUco marker up facing the camera.

       For each reading: position the marker, press SPACE, then switch to the
       TERMINAL and type the true position.  Suggested set:
           forward 0.4 / 0.8 / 1.2 / 2.0 m, lateral 0, facing the camera.
       Then optionally repeat at 0.8 and 1.2 m with the marker yawed 20/40/60 deg.
       q when done.
""")
    src = FrameSource(a.source, a.ip)
    rows = read_csv(csv_path)
    new_file = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    wr = csv.writer(fh)
    if new_file:
        wr.writerow(["tag", "true_forward", "true_lateral", "true_yaw",
                     "meas_forward", "meas_lateral", "meas_range", "meas_bearing"])
    cv2.namedWindow(WIN)
    while True:
        frame = src.read()
        if frame is None:
            continue
        dets = measure_markers(frame, K, D, detector, a.marker_length)
        vis = frame.copy()
        lines = [("readings logged: {}   SPACE record   q finish".format(len(rows)), (255, 255, 255))]
        for d in dets:
            cv2.aruco.drawDetectedMarkers(vis, [d["corners"]], np.array([[d["tag"]]]))
            lines.append(("id {:2d}  forward {:.3f} m  lateral {:+.3f} m  bearing {:+.1f} deg"
                          .format(d["tag"], d["forward"], d["lateral"], d["bearing"]),
                          (60, 220, 60)))
        if not dets:
            lines.append(("no marker detected", (60, 60, 255)))
        text_block(vis, lines)
        cv2.imshow(WIN, vis)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q'):
            break
        if k == ord(' '):
            if not dets:
                print("  no marker in view")
                continue
            d = dets[0]
            print("\n  measured: forward {:.3f}  lateral {:+.3f}  range {:.3f}  bearing {:+.1f} deg"
                  .format(d["forward"], d["lateral"], d["rng"], d["bearing"]))
            try:
                tf = float(input("  TRUE forward (m): ").strip())
                sl = input("  TRUE lateral (m, + = left, blank = 0): ").strip()
                sy = input("  TRUE marker yaw (deg, blank = 0): ").strip()
            except (ValueError, EOFError):
                print("  discarded")
                continue
            tl = float(sl) if sl else 0.0
            ty = float(sy) if sy else 0.0
            row = dict(tag=d["tag"], true_forward=tf, true_lateral=tl, true_yaw=ty,
                       meas_forward=d["forward"], meas_lateral=d["lateral"],
                       meas_range=d["rng"], meas_bearing=d["bearing"])
            rows.append(row)
            wr.writerow([row[c] for c in ["tag", "true_forward", "true_lateral", "true_yaw",
                                          "meas_forward", "meas_lateral", "meas_range",
                                          "meas_bearing"]])
            fh.flush()
            print("  logged.  error {:+.1f}% on range.".format(
                (d["rng"] / max(1e-9, np.hypot(tf, tl)) - 1) * 100))
    fh.close()
    cv2.destroyAllWindows()
    src.close()
    analyse(rows, "this session")
    if a.compare:
        analyse(read_csv(a.compare), "baseline " + os.path.basename(a.compare))
    print("\nReadings appended to {}".format(csv_path))
    return 0


# =============================================================================
#  CLI
# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="One-file camera calibration for the PenguinPi (ECE4078).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Typical run:\n"
               "  python calibrate.py board\n"
               "  python calibrate.py capture --ip 192.168.137.226 --square 0.02335 --marker 0.01775\n"
               "  python calibrate.py calib             --square 0.02335 --marker 0.01775\n"
               "  python calibrate.py verify  --ip 192.168.137.226\n")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, robot=False):
        Board.add_args(p)
        p.add_argument("--out", default=PARAM_DIR, help="folder holding intrinsic.txt etc.")
        if robot:
            p.add_argument("--ip", default="localhost", help="robot IP")
            p.add_argument("--source", default="robot",
                           help="'robot' (default), 'webcam[:N]', or a folder of images "
                                "to replay - handy for trying the UI without the robot")

    p = sub.add_parser("board", help="generate a ChArUco board to print")
    common(p)
    p.add_argument("--dpi", type=float, default=300.0)
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("capture", help="guided live capture from the robot camera")
    common(p, robot=True)
    p.add_argument("--dir", default=PIC_DIR, help="where to save images")
    p.add_argument("--fresh", action="store_true", help="empty the folder first")
    p.add_argument("--blur", type=float, default=60.0, help="minimum Laplacian variance")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("calib", help="fit the intrinsics")
    common(p)
    p.add_argument("--dir", default=PIC_DIR, help="folder of calibration images")
    p.add_argument("--blur", type=float, default=60.0)
    p.add_argument("--outlier", type=float, default=2.5,
                   help="drop images beyond this many std devs of reprojection error")
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--boot", type=int, default=12, help="bootstrap resamples for fx")
    p.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    p.set_defaults(func=cmd_calib)

    p = sub.add_parser("verify", help="check measured ArUco distances against a tape measure")
    common(p, robot=True)
    p.add_argument("--marker-length", type=float, default=0.06, help="arena marker side, m")
    p.add_argument("--csv", default=None)
    p.add_argument("--fresh", action="store_true", help="move any existing CSV aside")
    p.add_argument("--compare", default=None, help="another CSV to print alongside")
    p.add_argument("--analyse-only", action="store_true", help="just re-print the analysis")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("all", help="capture then calib")
    common(p, robot=True)
    p.add_argument("--dir", default=PIC_DIR)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--blur", type=float, default=60.0)
    p.add_argument("--outlier", type=float, default=2.5)
    p.add_argument("--folds", type=int, default=4)
    p.add_argument("--boot", type=int, default=12)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=lambda a: cmd_capture(a) or cmd_calib(a))

    a = ap.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())