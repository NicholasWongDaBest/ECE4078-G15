# Encoder count logger for the PiBot: timestamps every command sent to the Pi
# and every encoder count it sends back, so you can see exactly when a move's
# ticks start and stop, and how many more arrive after the Pi says "done"
# (the robot rolling on after the brake).
#
# For the no-reset listen.py: the Pi now reports running TOTALS that are never
# reset, so this script shows each move's ticks as differences. If it ever sees
# the totals go DOWN it says so -- that means the old listen.py (which zeroes
# the counters when the robot stops) is still running on the Pi.
#
# It talks to the Pi's wheel server directly (same packets as botconnect.py)
# instead of going through BotConnect, because BotConnect's thread blocks for
# the whole of a move_auto_encoder() move and hides the timing. Close
# operate.py / auto_fruit_search.py first -- only one program can hold the
# wheel connection.
#
# Usage:
#   python encoder_log.py --ip <robot_ip>
# then type commands at the prompt:
#   f 100     move_auto_encoder forward 100 ticks (same speed as M3: 0.4),
#             then keep polling at zero speed for --after seconds (default 1.0)
#   b 100     backward      l 20 / r 20   turn left / right (speed 0.35)
#   ff 100    two moves back to back with NO zero-speed polls in between
#             (also bb / ll / rr) -- checks each move counts from its own start
#   m f 0.5   manual drive (mode 0, like the keyboard) for 0.5 s, then stop
#   w 2       just watch the counts for 2 s with the robot still
#   after 1.5 change how long to watch after each move
#   q         quit
#
# Every line starts with the time in ms since the script started. Only lines
# where the counts CHANGE are printed while polling (plus the first and last),
# so the output is short enough to paste. --csv <file> also saves every packet.

import argparse
import csv
import datetime
import os
import socket
import struct
import time

MODES = {
    'f': ('Forward', (0.4, 0.4)),
    'b': ('Backward', (-0.4, -0.4)),
    'l': ('Turn left', (-0.35, 0.35)),
    'r': ('Turn right', (0.35, -0.35)),
}
PID_GAINS = (2.0, 0.04, 0.29)   # kp, ki, kd -- same as operate.py / M3


class Log:
    def __init__(self, csv_path=None):
        self.t0 = time.perf_counter()
        self.csv = None
        if csv_path:
            self._csv_file = open(csv_path, 'w', newline='')
            self.csv = csv.writer(self._csv_file)
            self.csv.writerow(['t_ms', 'event', 'cmd_left', 'cmd_right', 'target_left', 'target_right',
                               'left', 'right', 'round_trip_ms', 'note'])

    def ms(self, t=None):
        return ((time.perf_counter() if t is None else t) - self.t0) * 1000.0

    def show(self, t, event, left=None, right=None, note=''):
        lr = '{:>6} {:>6}'.format(left, right) if left is not None else ' ' * 13
        print('{:9.1f}  {:<7} {}   {}'.format(self.ms(t), event, lr, note))

    def row(self, t, event, cmd=(None, None), target=(None, None), counts=(None, None), rtt=None, note=''):
        if self.csv:
            self.csv.writerow(['{:.1f}'.format(self.ms(t)), event, cmd[0], cmd[1], target[0], target[1],
                               counts[0], counts[1], '' if rtt is None else '{:.1f}'.format(rtt), note])

    def close(self):
        if self.csv:
            self._csv_file.close()


class Wheel:
    """Minimal client for the Pi's wheel server, same packets as botconnect.py."""

    def __init__(self, ip, port, move_timeout):
        self.ip, self.port, self.move_timeout = ip, port, move_timeout
        self.sock = None
        self.connect()

    def connect(self):
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(5.0)
        self.sock.connect((self.ip, self.port))

    def _recv_counts(self, timeout):
        self.sock.settimeout(timeout)
        data = b''
        while len(data) < 8:
            chunk = self.sock.recv(8 - len(data))
            if not chunk:
                raise ConnectionError('wheel server closed the connection')
            data += chunk
        return struct.unpack('!ii', data)

    def manual(self, left, right):
        """Mode 0: set wheel speeds; the Pi replies with its counts at once.
        @return: (left, right), t_sent, t_received"""
        t_send = time.perf_counter()
        self.sock.sendall(struct.pack('!Bff', 0, left, right))
        counts = self._recv_counts(5.0)
        return counts, t_send, time.perf_counter()

    def auto_encoder(self, left, right, target_left, target_right):
        """Mode 2: the Pi drives until it has counted the targets, THEN replies.
        @return: (left, right) or None on timeout, t_sent, t_received"""
        t_send = time.perf_counter()
        self.sock.sendall(struct.pack('!Bffii', 2, left, right, int(target_left), int(target_right)))
        try:
            counts = self._recv_counts(self.move_timeout)
        except socket.timeout:
            return None, t_send, time.perf_counter()
        return counts, t_send, time.perf_counter()


def set_pid(ip, port, kp, ki, kd):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((ip, port))
        s.sendall(struct.pack('!ffff', 1.0, kp, ki, kd))
        ok = struct.unpack('!i', s.recv(4))[0] == 1
        s.close()
        return ok
    except Exception as e:
        print('  could not set PID: {}'.format(e))
        return False


class Tester:
    def __init__(self, wheel, log, poll_s, tpm, baseline):
        self.wheel, self.log, self.poll_s = wheel, log, poll_s
        self.tpm, self.baseline = tpm, baseline
        self.after = 1.0

    # --------------------------------------------------------------
    def poll(self, duration, speeds=(0.0, 0.0), label='poll', ref=None, min_gap_ms=0.0, ref_name='done reply'):
        """Send mode-0 packets every poll_s for `duration` s and print each reply
        whose counts changed. `ref` = (t, left, right) totals at the move's
        "done" reply (or the stop), used to show the ticks that arrive after it.
        @return: list of (t_received, left, right), round-trip times (ms), went_down"""
        samples, rtts = [], []
        last, last_print_t = None, -1e9
        went_down = False
        end = time.perf_counter() + duration
        first = True
        while True:
            now = time.perf_counter()
            if now >= end and not first:
                break
            (l, r), t_s, t_r = self.wheel.manual(*speeds)
            rtt = (t_r - t_s) * 1000.0
            samples.append((t_r, l, r))
            rtts.append(rtt)
            self.log.row(t_r, label, cmd=speeds, counts=(l, r), rtt=rtt)
            note = ''
            important = first
            if last is not None and (l < last[0] or r < last[1]):
                note = 'totals went DOWN -- is the OLD listen.py (which resets) still running on the Pi?'
                went_down = important = True
            elif ref is not None:
                note = '{:+d} / {:+d} since the {} ({:.0f} ms)'.format(
                    l - ref[1], r - ref[2], ref_name, (t_r - ref[0]) * 1000.0)
            if (l, r) != last and (important or (t_r - last_print_t) * 1000.0 >= min_gap_ms):
                self.log.show(t_r, label, l, r, note)
                last_print_t = t_r
            last = (l, r)
            first = False
            time.sleep(self.poll_s)
        self.log.show(samples[-1][0], 'end', samples[-1][1], samples[-1][2],
                      '{} polls, round trip {:.1f} ms avg / {:.1f} max'.format(
                          len(samples), sum(rtts) / len(rtts), max(rtts)))
        return samples, rtts, went_down

    # --------------------------------------------------------------
    def _units(self, key, ticks):
        if key in ('l', 'r'):
            if not self.baseline:
                return '{} ticks'.format(ticks)
            deg = ticks * 2.0 / (self.baseline * self.tpm) * 57.2958
            return '~{:.1f} deg'.format(deg)
        return '~{:.1f} cm'.format(ticks / self.tpm * 100.0)

    def _one_move(self, key, ticks):
        name, (sl, sr) = MODES[key]
        (l0, r0), _, t0 = self.wheel.manual(0.0, 0.0)
        self.log.show(t0, 'before', l0, r0, 'totals just before the move command')
        self.log.row(t0, 'before', cmd=(0.0, 0.0), counts=(l0, r0))
        counts, t_s, t_r = self.wheel.auto_encoder(sl, sr, ticks, ticks)
        self.log.show(t_s, 'SEND', note='mode 2 {} speed {:+.2f}/{:+.2f} target {}/{}'.format(
            name, sl, sr, ticks, ticks))
        self.log.row(t_s, 'SEND', cmd=(sl, sr), target=(ticks, ticks))
        if counts is None:
            self.log.show(t_r, 'TIMEOUT', note='no reply in {:.0f} s -- stopping and reconnecting'.format(
                self.wheel.move_timeout))
            self.wheel.connect()
            self.wheel.manual(0.0, 0.0)
            return None
        dl, dr = counts[0] - l0, counts[1] - r0
        note = 'Pi says done after {:.0f} ms: {:+d} / {:+d} this move (target {})'.format(
            (t_r - t_s) * 1000.0, dl, dr, ticks)
        if dl < 0 or dr < 0:
            note += ' -- totals went DOWN: OLD listen.py still running?'
        self.log.show(t_r, 'REPLY', counts[0], counts[1], note)
        self.log.row(t_r, 'REPLY', cmd=(sl, sr), target=(ticks, ticks), counts=counts,
                     rtt=(t_r - t_s) * 1000.0)
        return (l0, r0), counts, t_r

    def move(self, key, ticks, back_to_back=False):
        res = self._one_move(key, ticks)
        if res is None:
            return
        before, reply, t_reply = res
        if back_to_back:
            res2 = self._one_move(key, ticks)
            if res2 is None:
                return
            before, reply, t_reply = res2
        samples, _, went_down = self.poll(self.after, ref=(t_reply, reply[0], reply[1]))
        self.summary(key, ticks, before, reply, t_reply, samples, went_down)

    def summary(self, key, ticks, before, reply, t_reply, samples, went_down, manual=False):
        end_t, end_l, end_r = samples[-1]
        # time of the last tick that arrived after the reply/stop
        last_tick_t, prev = None, (reply[0], reply[1])
        for t, l, r in samples:
            if (l, r) != prev:
                last_tick_t = t
            prev = (l, r)
        to_done = ((reply[0] - before[0]), (reply[1] - before[1]))
        roll = ((end_l - reply[0]), (end_r - reply[1]))
        whole = ((end_l - before[0]), (end_r - before[1]))
        avg = lambda p: round((p[0] + p[1]) / 2.0)
        print('  SUMMARY {} {} | {} {:+d}/{:+d} ({}) | after it {:+d}/{:+d} ({}{}) | whole move {:+d}/{:+d} ({})'.format(
            MODES[key][0], 'manual' if manual else 'target {}'.format(ticks),
            'while driving' if manual else 'counted to done', to_done[0], to_done[1],
            self._units(key, avg(to_done)), roll[0], roll[1], self._units(key, avg(roll)),
            ', last tick at +{:.0f} ms'.format((last_tick_t - t_reply) * 1000.0) if last_tick_t else '',
            whole[0], whole[1], self._units(key, avg(whole))))
        if went_down:
            print('  WARNING: the totals went down during this test -- the Pi is still resetting its '
                  'counters, so the numbers above are not valid. Is the new listen.py running?')
        print()

    def manual(self, key, seconds):
        name, (sl, sr) = MODES[key]
        (l0, r0), _, t0 = self.wheel.manual(0.0, 0.0)
        self.log.show(t0, 'before', l0, r0, 'manual {} {:.2f} s at {:+.2f}/{:+.2f}'.format(
            name, seconds, sl, sr))
        moving, _, down1 = self.poll(seconds, speeds=(sl, sr), label='drive', min_gap_ms=40.0)
        t_stop, l_stop, r_stop = moving[-1]
        self.log.show(t_stop, 'STOP', l_stop, r_stop, 'last count while driving; now sending speed 0')
        after, _, down2 = self.poll(self.after, ref=(t_stop, l_stop, r_stop), ref_name='stop')
        self.summary(key, 0, (l0, r0), (l_stop, r_stop), t_stop, after, down1 or down2, manual=True)


def load_baseline(path):
    try:
        with open(path) as f:
            return abs(float(f.read().strip().split(',')[0]))
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ip', type=str, default='localhost')
    p.add_argument('--port', type=int, default=8000, help='wheel server port (botconnect default 8000)')
    p.add_argument('--pid-port', type=int, default=8002)
    p.add_argument('--no-pid', action='store_true', help="don't send the PID gains first")
    p.add_argument('--poll-ms', type=float, default=10.0, help='gap between zero-speed polls (botconnect: 10)')
    p.add_argument('--after', type=float, default=1.0, help='seconds to keep polling after each move')
    p.add_argument('--tpm', type=float, default=196.1, help='ticks per metre, only for the cm/deg notes')
    p.add_argument('--baseline', type=str, default=os.path.join('calibration', 'param', 'baseline.txt'))
    p.add_argument('--timeout', type=float, default=15.0, help='give up on a move after this many s')
    p.add_argument('--csv', type=str, default=None, help='also save every packet to this CSV file')
    args = p.parse_args()

    print('encoder_log.py  {}  ip={}  poll={:.0f} ms  after={:.1f} s'.format(
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), args.ip, args.poll_ms, args.after))
    if not args.no_pid:
        ok = set_pid(args.ip, args.pid_port, *PID_GAINS)
        print('PID {} kp={} ki={} kd={}'.format('on' if ok else 'NOT set', *PID_GAINS))
    else:
        print('PID not sent (--no-pid)')

    log = Log(args.csv)
    wheel = Wheel(args.ip, args.port, args.timeout)
    tester = Tester(wheel, log, args.poll_ms / 1000.0, args.tpm, load_baseline(args.baseline))
    tester.after = args.after
    print('Connected. Commands: f/b/l/r <ticks>, ff/bb/ll/rr <ticks>, m <f|b|l|r> <seconds>, '
          'w <seconds>, after <seconds>, q')
    print('     t(ms)  event        L      R   note')
    tester.poll(0.2, label='idle')
    print()

    try:
        while True:
            cmd = input('> ').strip().lower()
            t_in = time.perf_counter()
            if not cmd:
                continue
            log.show(t_in, 'INPUT', note=repr(cmd))
            log.row(t_in, 'INPUT', note=cmd)
            parts = cmd.split()
            try:
                if parts[0] == 'q':
                    break
                elif parts[0] == 'after' and len(parts) == 2:
                    tester.after = max(0.05, float(parts[1]))
                    print('  watching {:.2f} s after each move\n'.format(tester.after))
                elif parts[0] == 'w' and len(parts) == 2:
                    tester.poll(float(parts[1]), label='watch')
                    print()
                elif parts[0] == 'm' and len(parts) == 3 and parts[1] in MODES:
                    tester.manual(parts[1], min(5.0, float(parts[2])))
                elif len(parts) == 2 and parts[0] in MODES:
                    tester.move(parts[0], int(round(float(parts[1]))))
                elif len(parts) == 2 and len(parts[0]) == 2 and parts[0][0] == parts[0][1] \
                        and parts[0][0] in MODES:
                    tester.move(parts[0][0], int(round(float(parts[1]))), back_to_back=True)
                else:
                    print('  ? try: f 100 | l 20 | ff 100 | m f 0.5 | w 2 | after 1.5 | q\n')
            except ValueError:
                print('  not a number\n')
    except KeyboardInterrupt:
        print()
    finally:
        try:
            wheel.manual(0.0, 0.0)
        except Exception:
            pass
        log.close()
        print('stopped.')


if __name__ == '__main__':
    main()
