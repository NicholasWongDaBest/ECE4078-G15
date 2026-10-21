import socket
import struct
import io
import threading
import argparse
import time
from time import monotonic
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# Network Configuration
HOST = '0.0.0.0'
WHEEL_PORT = 8000
CAMERA_PORT = 8001
PID_CONFIG_PORT = 8002

# Pins
RIGHT_MOTOR_ENA = 18
RIGHT_MOTOR_IN1 = 17
RIGHT_MOTOR_IN2 = 27
LEFT_MOTOR_ENB = 25
LEFT_MOTOR_IN3 = 23
LEFT_MOTOR_IN4 = 24
LEFT_ENCODER = 26
RIGHT_ENCODER = 16

# PID Constants (default values, will be overridden by client)
use_PID = 1
KP, KI, KD = 0, 0, 0
MAX_CORRECTION = 30  # Maximum PWM correction value

# Global variables
running = True
left_pwm, right_pwm = 0, 0
left_count, right_count = 0, 0
prev_left_state, prev_right_state = None, None
MIN_PWM_THRESHOLD = 20

# Encoder noise filter. Electrical noise from the motors (worse at a higher
# motor voltage) shows up on the encoder lines as bursts of extra edges. Each
# one counts as a tick, so a mode-2 move reaches its tick target early and
# stops short -- while the laptop is told the full move happened, which is how
# the robot loses its position. An edge only counts if
#   1. the pin reads the new level ENCODER_STABLE_READS times in a row (a
#      noise spike is microseconds long and gone by the second read; a real
#      slot edge stays), and
#   2. it comes at least MIN_EDGE_INTERVAL after the last counted edge on that
#      wheel. One tick is ~5 mm of wheel travel, so even 1 m/s is ~5 ms
#      between edges; 2 ms leaves plenty of margin.
# A rejected edge does not change the remembered level, so the next real edge
# is still counted. --min-edge-ms 0 --stable-reads 1 turns the filter off.
MIN_EDGE_INTERVAL = 0.002     # s
ENCODER_STABLE_READS = 3
left_last_edge, right_last_edge = 0.0, 0.0
left_rejected, right_rejected = 0, 0
current_movement, prev_movement = 'stop', 'stop'
CAMERA_MODE = 'old'            # --camera-mode (see camera_stream_server, FrameGrabber)
motion_wake_event = threading.Event()  # wakes pid_control() when a new drive command arrives

# Protects left_pwm/right_pwm: wheel_server() (writer, socket thread) and
# pid_control() (reader, its own thread) touch these globals concurrently.
# BOTH sides must take this lock -- locking only the writer isn't enough,
# since an unsynchronized reader can still observe a "mixed" state where
# left_pwm reflects a new command but right_pwm still reflects the old one
# (or vice versa). That mixed read gets misclassified as a turn even when
# the client only ever sent straight forward/backward -- this was the root
# cause of the random swerve when rapidly alternating direction.
motor_pwm_lock = threading.Lock()


def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    # Motor
    GPIO.setup(RIGHT_MOTOR_ENA, GPIO.OUT)
    GPIO.setup(RIGHT_MOTOR_IN1, GPIO.OUT)
    GPIO.setup(RIGHT_MOTOR_IN2, GPIO.OUT)
    GPIO.setup(LEFT_MOTOR_ENB, GPIO.OUT)
    GPIO.setup(LEFT_MOTOR_IN3, GPIO.OUT)
    GPIO.setup(LEFT_MOTOR_IN4, GPIO.OUT)

    # This prevents slight motor jerk when connection is established
    GPIO.output(RIGHT_MOTOR_ENA, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_ENB, GPIO.LOW)

    # Encoder setup and interrupt (both activated and deactivated)
    GPIO.setup(LEFT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(RIGHT_ENCODER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.add_event_detect(LEFT_ENCODER, GPIO.BOTH, callback=left_encoder_callback)
    GPIO.add_event_detect(RIGHT_ENCODER, GPIO.BOTH, callback=right_encoder_callback)

    # Initialize PWM (frequency: 5000Hz)
    global left_motor_pwm, right_motor_pwm
    left_motor_pwm = GPIO.PWM(LEFT_MOTOR_ENB, 1100)
    right_motor_pwm = GPIO.PWM(RIGHT_MOTOR_ENA, 1100)
    left_motor_pwm.start(0)
    right_motor_pwm.start(0)


def _stable_level(pin, level):
    """True if `pin` still reads `level` on ENCODER_STABLE_READS - 1 more reads."""
    for _ in range(ENCODER_STABLE_READS - 1):
        if GPIO.input(pin) != level:
            return False
    return True


def left_encoder_callback(channel):
    global left_count, prev_left_state, left_last_edge, left_rejected
    now = monotonic()
    current_state = GPIO.input(LEFT_ENCODER)
    if prev_left_state is None:
        prev_left_state = current_state  # First reading
        left_last_edge = now
        return
    if current_state == prev_left_state:
        return   # no real change (the level is what it was): electrical noise
    if not _stable_level(LEFT_ENCODER, current_state) or now - left_last_edge < MIN_EDGE_INTERVAL:
        left_rejected += 1   # a spike, or faster than the wheel can turn: not a tick
        return
    left_count += 1
    prev_left_state = current_state
    left_last_edge = now


def right_encoder_callback(channel):
    global right_count, prev_right_state, right_last_edge, right_rejected
    now = monotonic()
    current_state = GPIO.input(RIGHT_ENCODER)
    if prev_right_state is None:
        prev_right_state = current_state
        right_last_edge = now
        return
    if current_state == prev_right_state:
        return
    if not _stable_level(RIGHT_ENCODER, current_state) or now - right_last_edge < MIN_EDGE_INTERVAL:
        right_rejected += 1
        return
    right_count += 1
    prev_right_state = current_state
    right_last_edge = now


def read_counts():
    # Encoder totals since this server started. They are NEVER reset: each
    # movement measures its own ticks as the difference from the totals when
    # it started, so ticks from the robot rolling on after the brake (or
    # rocking as it settles) are still recorded and the laptop can see them.
    # The encoders are single-channel (every edge counts, whatever the
    # direction), so the totals only ever go up.
    return left_count, right_count


def set_motors(left, right):
    global prev_movement, current_movement

    # Pre-Start Kick (Motor Priming), to reduce initial jerk and slight orientation change when movement starts
    if prev_movement == 'stop':
        if current_movement == 'forward':
            GPIO.output(RIGHT_MOTOR_IN1, GPIO.HIGH)
            GPIO.output(RIGHT_MOTOR_IN2, GPIO.LOW)
            GPIO.output(LEFT_MOTOR_IN3, GPIO.HIGH)
            GPIO.output(LEFT_MOTOR_IN4, GPIO.LOW)
        elif current_movement == 'backward':
            GPIO.output(RIGHT_MOTOR_IN1, GPIO.LOW)
            GPIO.output(RIGHT_MOTOR_IN2, GPIO.HIGH)
            GPIO.output(LEFT_MOTOR_IN3, GPIO.LOW)
            GPIO.output(LEFT_MOTOR_IN4, GPIO.HIGH)
        elif current_movement == 'clockwise':
            GPIO.output(RIGHT_MOTOR_IN1, GPIO.LOW)
            GPIO.output(RIGHT_MOTOR_IN2, GPIO.HIGH)
            GPIO.output(LEFT_MOTOR_IN3, GPIO.HIGH)
            GPIO.output(LEFT_MOTOR_IN4, GPIO.LOW)
        elif current_movement == 'anticlockwise':
            GPIO.output(RIGHT_MOTOR_IN1, GPIO.HIGH)
            GPIO.output(RIGHT_MOTOR_IN2, GPIO.LOW)
            GPIO.output(LEFT_MOTOR_IN3, GPIO.LOW)
            GPIO.output(LEFT_MOTOR_IN4, GPIO.HIGH)

        left_motor_pwm.ChangeDutyCycle(75)
        right_motor_pwm.ChangeDutyCycle(75)
        time.sleep(0.025)

    # Set the desired PWM
    if right > 0:
        GPIO.output(RIGHT_MOTOR_IN1, GPIO.HIGH)
        GPIO.output(RIGHT_MOTOR_IN2, GPIO.LOW)
        right_motor_pwm.ChangeDutyCycle(min(right, 100))
    elif right < 0:
        GPIO.output(RIGHT_MOTOR_IN1, GPIO.LOW)
        GPIO.output(RIGHT_MOTOR_IN2, GPIO.HIGH)
        right_motor_pwm.ChangeDutyCycle(min(abs(right), 100))
    else:
        # when pwm = 0, implement Active Braking mode, better than putting duty cycle to 0 which may cause uneven stopping
        GPIO.output(RIGHT_MOTOR_IN1, GPIO.HIGH)
        GPIO.output(RIGHT_MOTOR_IN2, GPIO.HIGH)
        right_motor_pwm.ChangeDutyCycle(100)

    if left > 0:
        GPIO.output(LEFT_MOTOR_IN3, GPIO.HIGH)
        GPIO.output(LEFT_MOTOR_IN4, GPIO.LOW)
        left_motor_pwm.ChangeDutyCycle(min(left, 100))
    elif left < 0:
        GPIO.output(LEFT_MOTOR_IN3, GPIO.LOW)
        GPIO.output(LEFT_MOTOR_IN4, GPIO.HIGH)
        left_motor_pwm.ChangeDutyCycle(min(abs(left), 100))
    else:
        GPIO.output(LEFT_MOTOR_IN3, GPIO.HIGH)
        GPIO.output(LEFT_MOTOR_IN4, GPIO.HIGH)
        left_motor_pwm.ChangeDutyCycle(100)


def apply_min_threshold(pwm_value, min_threshold):
    if pwm_value == 0:
        return 0  # Zero means stop
    elif abs(pwm_value) < min_threshold:
        return min_threshold if pwm_value > 0 else -min_threshold  # Boost small values to minimum threshold, preserving direction
    else:
        return pwm_value


def pid_control():
    '''This function sets the motor pwm using pid control'''
    global prev_movement, current_movement

    integral = 0
    last_error = 0
    base_left, base_right = read_counts()  # totals when the current movement started
    last_time = monotonic()
    while running:
        current_time = monotonic()
        dt = current_time - last_time
        last_time = current_time

        # Snapshot left_pwm/right_pwm ONCE per iteration, under the lock, and use
        # only this snapshot for the rest of the iteration -- this is the other
        # half of the fix (see motor_pwm_lock comment above).
        with motor_pwm_lock:
            l_pwm, r_pwm = left_pwm, right_pwm

        prev_movement = current_movement
        if l_pwm > 0 and r_pwm > 0:
            current_movement = 'forward'
        elif l_pwm < 0 and r_pwm < 0:
            current_movement = 'backward'
        elif l_pwm == 0 and r_pwm == 0:
            current_movement = 'stop'
        elif l_pwm > 0 and r_pwm < 0:
            current_movement = 'clockwise'
        else:
            current_movement = 'anticlockwise'

        # Detect an abrupt reversal that skipped 'stop' (e.g. pygame coalesced
        # KEYUP+KEYDOWN into one frame so the PC never sent a zero-speed packet).
        # Without this, PID integral/last_error and the left/right difference
        # carry over from the old direction and cause a spurious correction spike.
        reversed_direction = (
            (prev_movement == 'forward' and current_movement == 'backward') or
            (prev_movement == 'backward' and current_movement == 'forward') or
            (prev_movement == 'clockwise' and current_movement == 'anticlockwise') or
            (prev_movement == 'anticlockwise' and current_movement == 'clockwise')
        )
        if reversed_direction:
            integral = 0
            last_error = 0

        # A new movement (from a stop, or a reversal): the PID compares the
        # wheels' ticks from here on. This replaces zeroing the counters.
        if current_movement != 'stop' and (prev_movement == 'stop' or reversed_direction):
            base_left, base_right = read_counts()

        if not use_PID:
            target_left_pwm = l_pwm
            target_right_pwm = r_pwm
        else:
            if current_movement != 'stop':
                error = (left_count - base_left) - (right_count - base_right)
                proportional = KP * error
                integral += KI * error * dt
                integral = max(-MAX_CORRECTION, min(integral, MAX_CORRECTION))  # Anti-windup
                derivative = KD * (error - last_error) / dt if dt > 0 else 0
                correction = proportional + integral + derivative
                correction = max(-MAX_CORRECTION, min(correction, MAX_CORRECTION))
                last_error = error

                if current_movement == 'forward':
                    target_left_pwm = l_pwm - correction
                    target_right_pwm = r_pwm + correction
                elif current_movement == 'backward':
                    target_left_pwm = l_pwm + correction
                    target_right_pwm = r_pwm - correction
                elif current_movement == 'clockwise':
                    target_left_pwm = l_pwm - correction
                    target_right_pwm = r_pwm - correction
                else:  # anticlockwise
                    target_left_pwm = l_pwm + correction
                    target_right_pwm = r_pwm + correction
            else:
                # Reset the PID when stopped. The encoder totals are left alone,
                # so ticks from the robot rolling on after the brake still count.
                integral = 0
                last_error = 0
                set_motors(0, 0)

                # Block until a new drive command wakes us, instead of polling every 20ms.
                # Clear first so we don't immediately re-trigger on a stale set() from
                # the command that just brought us to a stop. A timeout is kept as a
                # safety net in case a wake-up is missed in the brief window between
                # the movement check above and clear() below -- worst case we just
                # fall back to the old 0.02s-ish polling cadence rather than hanging.
                motion_wake_event.clear()
                motion_wake_event.wait(timeout=0.5)

                # Avoid a large dt spike (and matching integral/derivative spike)
                # on the first PID iteration after waking from a long idle wait.
                last_time = monotonic()
                continue

        final_left_pwm = apply_min_threshold(target_left_pwm, MIN_PWM_THRESHOLD)
        final_right_pwm = apply_min_threshold(target_right_pwm, MIN_PWM_THRESHOLD)
        set_motors(final_left_pwm, final_right_pwm)

        if target_left_pwm != 0 and args.verbose:  # print pwm values for debugging purpose
            print(f"L/R PWM: ({target_left_pwm:.2f},{target_right_pwm:.2f}), "
                  f"L/R Enc this movement: ({left_count - base_left}, {right_count - base_right})")

        time.sleep(0.01)


JPEG_QUALITY = 90          # what picam2.capture_file() uses by default: the same pictures as before
FRESH_WAIT_MAX = 1.0       # s: longest wait for a frame exposed after the request (then: the newest one)


class FrameGrabber:
    """
    Captures non-stop in a thread of its own and keeps only the NEWEST frame:
    its pixels (a copy), when its exposure started (Pi monotonic clock, s),
    when it was ready to send, a sequence number and -- with pre_encode --
    its JPEG.

    The old server called capture_file() only once a request came in, so
    every frame paid for the wait for the next sensor frame AND the JPEG
    encode before it could be sent.

    pre_encode (--camera-mode continuous): every frame is encoded as it
    arrives (a thread of its own: capture never misses a frame), and a
    request is answered with the first frame READY after the request arrived
    and newer than the last one sent -- the encode no longer adds to the
    wait. Against a fake 30 fps camera with a 25 ms encode: wheels still ->
    the laptop's 2nd new frame 54 ms, against 80 ms with the old server. But
    a frame's exposure can then start up to ~90 ms before its request (old:
    ~30 ms, one exposure + readout), so the laptop's 2nd frame after 0.10 s
    of still wheels starts only ~10 ms after the last tick (old: ~70 ms).
    Hence not the default.

    not pre_encode (--camera-mode strict): a request is answered with the
    first frame whose exposure STARTED after the request -- the strictest
    freshness, but it waits for a whole new exposure (the same test: 118 ms,
    slower than the old server). use_timestamps False (the clock check
    failed): "the second frame completed after the request" instead.
    """

    def __init__(self, picam2, use_timestamps=True, pre_encode=True):
        self.picam2 = picam2
        self.use_timestamps = use_timestamps
        self.pre_encode = pre_encode
        self.cond = threading.Condition()
        self.arr = None
        self.jpeg = None
        self.start = None      # exposure start of the newest frame (s, time.monotonic clock), or None
        self.ready = 0.0       # when the newest frame was ready to send (time.monotonic)
        self.seq = 0
        self.errors = 0
        # Capture and encoding run in separate threads: capture_request() waits
        # for the next frame to finish AFTER it is called, so a thread that
        # also encoded would miss every frame finishing during an encode and
        # run at half the camera's rate.
        self.raw = (None, None, 0)   # newest captured frame: (pixels, exposure start, its number)
        self.raw_cond = threading.Condition()
        threading.Thread(target=self._capture, daemon=True).start()
        if pre_encode:
            threading.Thread(target=self._encode, daemon=True).start()

    def _capture(self):
        n = 0
        while running:
            try:
                arr, start = grab(self.picam2)
            except Exception as e:
                self.errors += 1
                if self.errors <= 3:
                    print(f"Camera capture error: {e}")
                time.sleep(0.05)
                continue
            n += 1
            if self.pre_encode:
                with self.raw_cond:
                    self.raw = (arr, start, n)
                    self.raw_cond.notify_all()
            else:
                self._publish(arr, None, start)

    def _encode(self):
        """Encode the newest captured frame, over and over (one skipped while
        it was busy is simply never sent -- a newer one is)."""
        done = 0
        while running:
            with self.raw_cond:
                while running and self.raw[2] == done:
                    self.raw_cond.wait(0.5)
                arr, start, done = self.raw
            if arr is None:
                continue
            try:
                jpeg = encode(arr)
            except Exception as e:
                self.errors += 1
                if self.errors <= 3:
                    print(f"Camera encode error: {e}")
                continue
            self._publish(arr, jpeg, start)

    def _publish(self, arr, jpeg, start):
        with self.cond:
            self.arr, self.jpeg, self.start = arr, jpeg, start
            self.ready = time.monotonic()
            self.seq += 1
            self.cond.notify_all()

    def wait_fresh(self, t_request, last_seq=0, timeout=FRESH_WAIT_MAX):
        """The frame to answer a request that arrived at t_request
        (time.monotonic()) with: pre_encode -- the first one ready after it
        and newer than last_seq; else the first one exposed after it. The
        newest one after `timeout` s.
        @return: (pixels, jpeg or None, True if fresh, exposure start, seq)"""
        deadline = time.monotonic() + timeout
        with self.cond:
            seq0 = self.seq
            while True:
                if self.arr is not None:
                    if self.pre_encode:
                        fresh = self.ready >= t_request and self.seq > last_seq
                    elif self.use_timestamps and self.start is not None:
                        fresh = self.start >= t_request
                    else:
                        fresh = self.seq >= seq0 + 2
                    if fresh:
                        return self.arr, self.jpeg, True, self.start, self.seq
                left = deadline - time.monotonic()
                if left <= 0:
                    return self.arr, self.jpeg, False, self.start, self.seq
                self.cond.wait(left)


def grab(picam2):
    """One frame from the main stream: (pixels, exposure start in s on the
    time.monotonic() clock, or None if the frame has no timestamp)."""
    request = picam2.capture_request()
    try:
        arr = request.make_array('main')
        md = request.get_metadata()
    finally:
        request.release()
    ts, exposure = md.get('SensorTimestamp'), md.get('ExposureTime')
    # SensorTimestamp (ns) is taken at the start of the frame; counting the
    # whole exposure before it makes "exposed after the request" err strict.
    start = ts / 1e9 - (exposure or 0) / 1e6 if ts is not None else None
    return arr, start


def encode(arr):
    """JPEG bytes of a main-stream frame, as capture_file() makes them. The
    preview configuration's main stream is XBGR8888: the bytes of each pixel
    are R, G, B, 255 -- arr[..., :3] is RGB."""
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(arr[..., :3]).save(stream, format='JPEG', quality=JPEG_QUALITY)
    return stream.getvalue()


def _jpeg_means(jpeg):
    from PIL import Image
    import numpy as np
    img = np.asarray(Image.open(io.BytesIO(jpeg)).convert('RGB'), dtype=float)
    return img.reshape(-1, 3).mean(axis=0)


def colour_check(picam2):
    """Encode one frame the old way (capture_file) and one the new way, and
    compare their colours. @return: (ok, message). A red/blue swap would
    ruin the fruit detector, so anything doubtful counts as a failure."""
    try:
        stream = io.BytesIO()
        picam2.capture_file(stream, format='jpeg')
        old = _jpeg_means(stream.getvalue())
        arr, _ = grab(picam2)
        new = _jpeg_means(encode(arr))
    except Exception as e:
        return False, f"could not compare ({e})"
    same = float(abs(old - new).sum())
    swapped = float(abs(old - new[::-1]).sum())
    msg = "old RGB means {} vs new {}".format([round(v) for v in old], [round(v) for v in new])
    if same > 25 or (swapped < same and abs(old[0] - old[2]) > 8):
        return False, msg
    return True, msg


def clock_check(picam2, n=5):
    """True if the frames' exposure starts are on the time.monotonic() clock:
    each one just before the frame came back (0-0.5 s). @return: (ok, message)"""
    lags = []
    for _ in range(n):
        try:
            _, start = grab(picam2)
        except Exception as e:
            return False, f"capture failed ({e})"
        if start is None:
            return False, "no SensorTimestamp in the frame metadata"
        lags.append(time.monotonic() - start)
    ok = all(0.0 <= lag <= 0.5 for lag in lags)
    return ok, "frame done {:.0f}-{:.0f} ms after its exposure started".format(1000 * min(lags), 1000 * max(lags))


def camera_stream_server():
    picam2 = Picamera2()
    camera_config = picam2.create_preview_configuration(lores={"size": (480, 360)})
    picam2.configure(camera_config)
    picam2.start()

    grabber = None
    if CAMERA_MODE in ('continuous', 'strict'):
        ok_colour, msg_colour = colour_check(picam2)
        if not ok_colour:
            print(f"Camera: continuous capture OFF -- colour check failed ({msg_colour}); using capture_file as before")
        elif CAMERA_MODE == 'continuous':
            grabber = FrameGrabber(picam2, pre_encode=True)
            print("Camera: continuous capture ON, colours match ({}); every frame encoded as it arrives, a request "
                  "gets the first one ready after it".format(msg_colour))
        else:
            ok_clock, msg_clock = clock_check(picam2)
            grabber = FrameGrabber(picam2, use_timestamps=ok_clock, pre_encode=False)
            print("Camera: continuous capture ON (strict), colours match ({}); a request gets {} ({})".format(
                msg_colour, "the first frame exposed after it" if ok_clock else
                "the second frame after it (timestamps not on this clock)", msg_clock))
    else:
        print("Camera: capture_file per request (--camera-mode old)")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, CAMERA_PORT))
    server_socket.listen(1)
    print(f"Camera stream server started on port {CAMERA_PORT}")

    while running:
        try:
            client_socket, _ = server_socket.accept()
            print("Camera stream client connected")
            stale, last_seq = 0, 0
            while running:
                ready = client_socket.recv(1)
                if not ready:
                    break
                t_request = time.monotonic()

                if grabber is not None:
                    arr, jpeg_data, fresh, _, last_seq = grabber.wait_fresh(t_request, last_seq)
                    if not fresh:
                        stale += 1
                        if stale <= 3:
                            print("Camera: no new frame within {:.1f} s -- sending the newest".format(FRESH_WAIT_MAX))
                    if arr is None:
                        break
                    if jpeg_data is None:
                        jpeg_data = encode(arr)
                else:
                    stream = io.BytesIO()
                    picam2.capture_file(stream, format='jpeg')
                    stream.seek(0)
                    jpeg_data = stream.getvalue()
                jpeg_size = len(jpeg_data)
                try:
                    client_socket.sendall(struct.pack("!I", jpeg_size) + jpeg_data)
                except Exception:
                    print("Camera stream client disconnected")
                    break

        except Exception as e:
            print(f"Camera stream server error: {str(e)}")

        if 'client_socket' in locals() and client_socket:
            client_socket.close()

    server_socket.close()
    picam2.stop()


def pid_config_server():
    global use_PID, KP, KI, KD

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, PID_CONFIG_PORT))
    server_socket.listen(1)
    print(f"PID config server started on port {PID_CONFIG_PORT}")

    while running:
        try:
            client_socket, _ = server_socket.accept()
            print("PID config client connected")

            try:
                data = client_socket.recv(16)
                if data and len(data) == 16:
                    use_PID, KP, KI, KD = struct.unpack("!ffff", data)
                    if use_PID:
                        print(f"Updated PID constants: KP={KP}, KI={KI}, KD={KD}")
                    else:
                        print("The robot is not using PID.")
                    response = struct.pack("!i", 1)
                else:
                    response = struct.pack("!i", 0)

                client_socket.sendall(response)

            except Exception as e:
                print(f"PID config socket error: {str(e)}")
                try:
                    response = struct.pack("!i", 0)
                    client_socket.sendall(response)
                except Exception:
                    pass

            client_socket.close()

        except Exception as e:
            print(f"PID config server error: {str(e)}")

    server_socket.close()


def wheel_server():
    global left_pwm, right_pwm, running, left_count, right_count

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST, WHEEL_PORT))
    server_socket.listen(1)
    print(f"Wheel server started on port {WHEEL_PORT}")

    while running:
        try:
            client_socket, _ = server_socket.accept()
            print("Wheel client connected")

            while running:
                try:
                    move_mode = client_socket.recv(1)
                    if not move_mode or len(move_mode) != 1:
                        print("Wheel client sending error")
                        break
                    move_mode = struct.unpack("!B", move_mode)[0]

                    if move_mode == 0:
                        data = client_socket.recv(8)
                        if not data or len(data) != 8:
                            print("Wheel client sending error")
                            break

                        left_speed, right_speed = struct.unpack("!ff", data)
                        with motor_pwm_lock:
                            left_pwm, right_pwm = left_speed * 100, right_speed * 100
                        motion_wake_event.set()

                    elif move_mode == 1:
                        data = client_socket.recv(12)
                        if not data or len(data) != 12:
                            print("Wheel client sending error")
                            break

                        left_speed, right_speed, duration = struct.unpack("!fff", data)
                        print(f"Received Mode 1 with L/R speed: {left_speed:.4f}, {right_speed:.4f}, Duration: {duration:.2f}s")
                        with motor_pwm_lock:
                            left_pwm, right_pwm = left_speed * 100, right_speed * 100
                        motion_wake_event.set()

                        autonomous_start_time = monotonic()
                        while running:
                            elapsed_time = monotonic() - autonomous_start_time
                            if elapsed_time >= duration:
                                with motor_pwm_lock:
                                    left_pwm, right_pwm = 0, 0
                                print(f"Timed movement completed after {elapsed_time:.2f}s")
                                break
                            time.sleep(0.02)

                    elif move_mode == 2:
                        data = client_socket.recv(16)
                        if not data or len(data) != 16:
                            print("Wheel client sending error")
                            break

                        left_speed, right_speed, target_left_enc, target_right_enc = struct.unpack("!ffii", data)
                        print(f"Received Mode 2 with L/R speed: {left_speed:.4f}, {right_speed:.4f}, "
                              f"L/R enc: {target_left_enc}, {target_right_enc}")
                        # Count this move from the totals right now (they are never reset).
                        start_left, start_right = read_counts()
                        rej_left0, rej_right0 = left_rejected, right_rejected
                        with motor_pwm_lock:
                            left_pwm, right_pwm = left_speed * 100, right_speed * 100
                        motion_wake_event.set()

                        # Monitor progress toward the target encoder counts. This loop was
                        # present in the original script but had been lost along the way --
                        # restored here, since without it mode 2 never stops on its own.
                        autonomous_start_time = monotonic()
                        while running:
                            elapsed_time = monotonic() - autonomous_start_time
                            done_left = left_count - start_left
                            done_right = right_count - start_right
                            if done_left >= target_left_enc and done_right >= target_right_enc:
                                with motor_pwm_lock:
                                    left_pwm, right_pwm = 0, 0
                                rej = (left_rejected - rej_left0, right_rejected - rej_right0)
                                print(f"Encoder-based movement completed at L/R enc: {done_left}, {done_right} "
                                      f"(target {target_left_enc}, {target_right_enc})"
                                      + (f"; filtered out {rej[0]} / {rej[1]} noise edge(s)" if any(rej) else ""))
                                break
                            elif elapsed_time >= 8:  # safety timeout
                                with motor_pwm_lock:
                                    left_pwm, right_pwm = 0, 0
                                print("Encoder-based movement failed (safety timeout).")
                                break
                            time.sleep(0.02)

                    # Reply with the running totals (never reset); the laptop works
                    # out a move's ticks as the difference between two replies.
                    response = struct.pack("!ii", left_count, right_count)
                    client_socket.sendall(response)

                except Exception:
                    print("Wheel client disconnected")
                    break

        except Exception as e:
            print(f"Wheel server error: {str(e)}")

        if 'client_socket' in locals() and client_socket:
            client_socket.close()

    server_socket.close()


def main():
    try:
        setup_gpio()

        pid_thread = threading.Thread(target=pid_control)
        pid_thread.daemon = True
        pid_thread.start()

        camera_thread = threading.Thread(target=camera_stream_server)
        camera_thread.daemon = True
        camera_thread.start()

        pid_config_thread = threading.Thread(target=pid_config_server)
        pid_config_thread.daemon = True
        pid_config_thread.start()

        wheel_server()

    except KeyboardInterrupt:
        print("Stopping...")

    finally:
        global running
        running = False
        GPIO.cleanup()
        print("Cleanup complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--min-edge-ms', type=float, default=MIN_EDGE_INTERVAL * 1000,
                        help='ignore an encoder edge closer than this to the last counted one on that wheel, ms '
                             '(0 = off)')
    parser.add_argument('--stable-reads', type=int, default=ENCODER_STABLE_READS,
                        help='times the encoder pin must read the new level before an edge counts (1 = off)')
    parser.add_argument('--camera-mode', choices=['old', 'continuous', 'strict'], default='old',
                        help='old (default): capture_file() per request, as before; continuous: capture and encode '
                             'all the time and answer each request with the first frame ready after it -- faster, '
                             'but a frame can start up to ~90 ms before the request (old: ~30 ms), which eats into '
                             "the laptop's 0.10 s wheels-still margin; strict: the first frame EXPOSED after the "
                             'request -- the safest, but slower than old')
    args = parser.parse_args()
    CAMERA_MODE = args.camera_mode
    MIN_EDGE_INTERVAL = max(0.0, args.min_edge_ms) / 1000.0
    ENCODER_STABLE_READS = max(1, args.stable_reads)
    print(f"Encoder filter: edges at least {MIN_EDGE_INTERVAL * 1000:.1f} ms apart, "
          f"{ENCODER_STABLE_READS} stable read(s)")
    main()
