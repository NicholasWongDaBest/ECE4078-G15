"""
Summarise a final_demo_l3.py run log (run_logs/<date-time>/, see RunLog).

    python analyze_run.py                      # newest run in run_logs/
    python analyze_run.py run_logs/20261007-141500
    python analyze_run.py --truemap truemap.txt   # + measurement errors against a known map

Sections:
  phases        time spent in each phase
  moves         turns / drives commanded, timeouts
  EKF fit       NIS (normalised innovation squared) by range. With 2-D
                readings a well-tuned noise model averages ~2.0: well above
                -> the model claims more accuracy than the readings have
                (raise the noise), well below -> it is too cautious.
                Also how often readings were rejected by the gate.
  pose jumps    how far each frame moved the pose estimate
  markers       sd at the lock, late markers, final positions
  fruits        sightings handed to the mapper, accepted, per fruit
  targets       'found' events and their distances
  vs truth      (--truemap, or the run's own --compare-map) marker errors
                after alignment, and every marker READING compared with
                where the true marker should have appeared from the
                logged pose: bias and sd in depth and sideways by range,
                corrected vs raw -- the numbers to set the noise model from.
"""
import os
import sys
import json
import math
import glob
import argparse
import numpy as np

BINS = [(0.0, 0.6), (0.6, 0.9), (0.9, 1.2), (1.2, 1.6), (1.6, 2.0), (2.0, 9.0)]


def latest_run(root='run_logs'):
    runs = sorted(d for d in glob.glob(os.path.join(root, '*')) if os.path.isdir(d))
    return runs[-1] if runs else None


def load_run(run_dir):
    def read(name, default=None):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            return default
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    events = []
    path = os.path.join(run_dir, 'events.jsonl')
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except ValueError:
                        pass   # a half-written last line from a crash
    return read('meta.json', {}), events, read('final.json', {})


def bin_label(lo, hi):
    return "{:.1f}-{:.1f} m".format(lo, hi) if hi < 9 else ">{:.1f} m  ".format(lo)


def header(title):
    print("\n" + title)
    print("-" * len(title))


def fit_rigid_2d(est, true):
    mu_e = est.mean(axis=1, keepdims=True)
    mu_t = true.mean(axis=1, keepdims=True)
    a, b = est - mu_e, true - mu_t
    th = math.atan2(float(np.sum(a[0] * b[1] - a[1] * b[0])), float(np.sum(a[0] * b[0] + a[1] * b[1])))
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    return R, mu_t - R @ mu_e


def section_phases(events, final):
    header("Phases")
    times = final.get('phase_times')
    if not times:
        times, last = {}, None
        for e in events:
            if e['kind'] == 'phase':
                if last is not None:
                    times[last][1] = e['t']
                times[e['name']] = [e['t'], None]
                last = e['name']
        if last is not None and events:
            times[last][1] = events[-1]['t']
    for name, (a, b) in times.items():
        print("  {:8s} {:7.1f} s".format(name, (b or a) - a))
    end = [e for e in events if e['kind'] in ('stopped_by_user', 'crash')]
    for e in end:
        print("  ended by: {} {}".format(e['kind'], e.get('error', '')))


def section_moves(events):
    header("Moves")
    moves = [e for e in events if e['kind'] == 'move']
    turns = [m for m in moves if abs(m['dtheta']) > 1e-9 and abs(m['distance']) < 1e-9]
    drives = [m for m in moves if abs(m['distance']) > 1e-9]
    fails = [m for m in moves if not m['completed']]
    print("  {} turns ({:.0f} deg in all), {} drives ({:.2f} m in all), {} timed out".format(
        len(turns), math.degrees(sum(abs(m['dtheta']) for m in turns)), len(drives),
        sum(abs(m['distance']) for m in drives), len(fails)))


def section_nis(events):
    header("EKF fit: NIS by range (target ~2.0)")
    for phase in ('phase0', 'level3'):
        # (single-marker heading fixes write a smaller record with no NIS -- not counted here)
        diags = [d for e in events if e['kind'] == 'frame' and e['phase'] == phase for d in e.get('diag', [])
                 if d.get('tag', 999) < 100 and 'nis' in d]
        if not diags:
            continue
        print("  {}: {} marker updates".format(phase, len(diags)))
        print("    range        n   mean NIS  median   gated   mean |innovation|")
        for lo, hi in BINS:
            sel = [d for d in diags if lo <= d['distance'] < hi]
            if not sel:
                continue
            nis = np.array([d['nis'] for d in sel])
            inn = np.array([math.hypot(d['innov_x'], d['innov_y']) for d in sel])
            gated = np.mean([d['gated'] for d in sel])
            verdict = "  <- noise model too optimistic" if np.median(nis) > 4 else (
                "  <- too cautious" if np.median(nis) < 0.5 else "")
            print("    {}  {:4d}   {:6.2f}   {:6.2f}   {:4.0f}%     {:5.1f} cm{}".format(
                bin_label(lo, hi), len(sel), nis.mean(), np.median(nis), 100 * gated, 100 * inn.mean(), verdict))


def section_pose_jumps(events):
    header("Pose jumps per frame (how far one frame moved the estimate)")
    for phase in ('phase0', 'level3'):
        fr = [e for e in events if e['kind'] == 'frame' and e['phase'] == phase and e['mode'] != 'none']
        if not fr:
            continue
        dxy = np.array([math.hypot(e['pose_after'][0] - e['pose_before'][0],
                                   e['pose_after'][1] - e['pose_before'][1]) for e in fr])
        dth = np.array([abs(math.degrees(math.atan2(math.sin(e['pose_after'][2] - e['pose_before'][2]),
                                                    math.cos(e['pose_after'][2] - e['pose_before'][2]))))
                        for e in fr])
        modes = {}
        for e in fr:
            modes[e['mode']] = modes.get(e['mode'], 0) + 1
        print("  {}: {} frames ({})".format(phase, len(fr), ", ".join("{} {}".format(k, v) for k, v in modes.items())))
        print("    position: median {:.1f} cm, 90% {:.1f} cm, max {:.1f} cm, {} over 10 cm".format(
            100 * np.median(dxy), 100 * np.percentile(dxy, 90), 100 * dxy.max(), int(np.sum(dxy > 0.10))))
        print("    heading : median {:.1f} deg, 90% {:.1f} deg, max {:.1f} deg, {} over 10 deg".format(
            np.median(dth), np.percentile(dth, 90), dth.max(), int(np.sum(dth > 10))))


def section_markers(events, final):
    header("Markers")
    status = [e for e in events if e['kind'] == 'marker_status']
    for e in status:
        sds = [v[2] for v in e['markers'].values()]
        print("  after {}: {} placed, sd median {:.1f} cm, worst {:.1f} cm".format(
            e['where'], len(sds), 100 * np.median(sds) if sds else float('nan'), 100 * max(sds) if sds else float('nan')))
    lock = next((e for e in events if e['kind'] == 'lock'), None)
    if lock:
        print("  locked at t={:.0f}s: ".format(lock['t']) + ", ".join(
            "{}({:.0f}cm)".format(t, 100 * v[2]) for t, v in sorted(lock['markers'].items(), key=lambda kv: int(kv[0]))))
    for e in events:
        if e['kind'] == 'late_marker':
            print("  late marker {} at t={:.0f}s, [{:+.2f}, {:+.2f}]".format(e['tag'], e['t'], *e['xy']))
    readings = {}
    for e in events:
        if e['kind'] == 'frame':
            for m in e['markers']:
                readings[m['tag']] = readings.get(m['tag'], 0) + 1
    if readings:
        print("  readings per tag: " + ", ".join("{}:{}".format(t, n) for t, n in sorted(readings.items())))


def section_fruits(events, final):
    header("Fruits")
    obs = [e for e in events if e['kind'] == 'fruit_obs']
    per = {}
    for e in obs:
        for label, _ in e['boxes']:
            per[label] = per.get(label, 0) + 1
    acc = sum(e['accepted'] for e in obs)
    print("  {} batches handed to the mapper, {} sightings accepted in all (observe() returns a count, "
          "not which fruits)".format(len(obs), acc))
    if per:
        print("  sightings handed over per fruit: " + ", ".join("{}:{}".format(l, n) for l, n in sorted(per.items())))
    for l, f in sorted(final.get('fruits', {}).items()):
        print("  {:10s} [{:+.2f}, {:+.2f}] sd {:4.1f} cm, {} views over {:.0f} deg{}".format(
            l, f['xy'][0], f['xy'][1], 100 * f['sigma'], f['views'], f['spread_deg'],
            "" if f['well_mapped'] else "  (weak)"))


def section_targets(events):
    header("Targets")
    any_ = False
    for e in events:
        if e['kind'] == 'ui_target_done':
            any_ = True
            name, dist, ok = (e['args'] + [None, None, None])[:3]
            print("  t={:6.1f}s {:10s} {}".format(e['t'], str(name), "skipped" if dist is None else
                                                  "{:.2f} m {}".format(dist, "OK" if ok else "TOO FAR")))
    if not any_:
        print("  no targets reached")


def section_truth(events, final, truemap):
    header("Against the true map ({})".format(truemap))
    with open(truemap) as f:
        gt = json.load(f)
    true_m = {int(k[5:].split('_')[0]): np.array([v['x'], v['y']], dtype=float)
              for k, v in gt.items() if k.startswith('aruco')}
    est_m = {int(t): np.array(xy, dtype=float) for t, xy in final.get('markers', {}).items()}
    if not est_m:
        lock = next((e for e in events if e['kind'] == 'lock'), None)
        est_m = {int(t): np.array(v[:2]) for t, v in (lock or {}).get('markers', {}).items()}
    tags = sorted(set(est_m) & set(true_m))
    if len(tags) < 2:
        print("  fewer than 2 markers in common -- nothing to align")
        return
    E = np.array([est_m[t] for t in tags]).T
    T = np.array([true_m[t] for t in tags]).T
    R, t = fit_rigid_2d(E, T)
    al = R @ E + t
    err = np.hypot(*(al - T))
    print("  final markers: {} of {}, aligned RMSE {:.3f} m (rotate {:+.1f} deg, shift {:+.2f}, {:+.2f})".format(
        len(tags), len(true_m), float(np.sqrt(np.mean(err ** 2))), math.degrees(math.atan2(R[1, 0], R[0, 0])),
        t[0, 0], t[1, 0]))
    print("  per marker (cm): " + ", ".join("{}:{:.1f}".format(tg, 100 * e) for tg, e in zip(tags, err)))
    sd = final.get('locked_marker_sd', {})
    if sd:
        print("  sd claimed at lock vs actual error: " + ", ".join(
            "{}:{:.0f}/{:.0f}".format(tg, 100 * sd[str(tg)], 100 * e) for tg, e in zip(tags, err) if str(tg) in sd))

    # every reading vs where the true marker should have appeared from the logged pose
    true_in_est = {tg: R.T @ (true_m[tg].reshape(2, 1) - t) for tg in true_m}
    rows = {'z': [], 'raw': []}
    for e in events:
        if e['kind'] != 'frame':
            continue
        x, y, th = e['pose_after']
        c, s = math.cos(th), math.sin(th)
        for m in e['markers']:
            tg = m['tag']
            if tg not in true_in_est:
                continue
            dx, dy = float(true_in_est[tg][0, 0]) - x, float(true_in_est[tg][1, 0]) - y
            pred = (c * dx + s * dy, -s * dx + c * dy)        # (ahead, left) the reading should be
            d = math.hypot(*pred)
            for key in ('z', 'raw'):
                if key in m:
                    rows[key].append((e['phase'], d, m[key][0] - pred[0], m[key][1] - pred[1]))
    for key, title in (('z', 'readings as the EKF used them (corrected)'), ('raw', 'raw readings (no correction)')):
        r = rows[key]
        if not r:
            continue
        print("\n  {}: {} readings, residual vs the true marker from the logged pose".format(title, len(r)))
        print("    range        n   depth bias  depth sd   side bias  side sd")
        fit = []
        for lo, hi in BINS:
            sel = np.array([(dd, a, b) for _, dd, a, b in r if lo <= dd < hi])
            if len(sel) < 3:
                continue
            db, ds = sel[:, 1].mean(), sel[:, 1].std()
            lb, ls = sel[:, 2].mean(), sel[:, 2].std()
            fit.append(((lo + min(hi, 2.6)) / 2, ds, ls, len(sel)))
            print("    {}  {:4d}   {:+6.1f} cm   {:5.1f} cm   {:+6.1f} cm  {:5.1f} cm".format(
                bin_label(lo, hi), len(sel), 100 * db, 100 * ds, 100 * lb, 100 * ls))
        if key == 'z' and len(fit) >= 2:
            f = np.array(fit)
            w = np.sqrt(f[:, 3])
            A = np.stack([np.ones(len(f)), f[:, 0]], 1) * w[:, None]
            a_d = np.linalg.lstsq(A, f[:, 1] * w, rcond=None)[0]
            a_l = np.linalg.lstsq(A, f[:, 2] * w, rcond=None)[0]
            print("    -> depth sd ~ {:.3f} + {:.3f}*d m, side sd ~ {:.3f} + {:.3f}*d m "
                  "(includes pose error; the model in use is in meta.json 'noise')".format(
                      a_d[0], a_d[1], a_l[0], a_l[1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?', default=None, help="run folder (default: newest in run_logs/)")
    ap.add_argument('--truemap', default=None, help="known map to compare with (default: the run's --compare-map)")
    args = ap.parse_args()
    run_dir = args.run or latest_run()
    if not run_dir or not os.path.isdir(run_dir):
        sys.exit("No run folder found (run_logs/ is empty?)")
    meta, events, final = load_run(run_dir)
    print("Run {}  ({}, git {})".format(run_dir, meta.get('started', '?'), meta.get('git', '?')))
    a = meta.get('args', {})
    dc = meta.get('dist_correction')
    print("search list: {}   distance correction: {}   noise: {}".format(
        ", ".join(meta.get('search_list', [])), "on" if dc else "OFF",
        {k: v for k, v in meta.get('noise', {}).items() if k in ('marker_sd_base', 'marker_sd_per_m',
                                                                   'innovation_gate')}))
    print("{} events".format(len(events)))
    section_phases(events, final)
    section_moves(events)
    section_nis(events)
    section_pose_jumps(events)
    section_markers(events, final)
    section_fruits(events, final)
    section_targets(events)
    truemap = args.truemap or a.get('compare_map')
    if truemap and os.path.exists(truemap):
        section_truth(events, final, truemap)
    elif truemap:
        print("\n(true map {} not found -- comparison skipped)".format(truemap))


if __name__ == '__main__':
    main()
