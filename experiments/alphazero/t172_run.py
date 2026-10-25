#!/usr/bin/env python3
"""Task 172 controlled scratch training, final-only light finish, and paired frame.

--pilot N loads the FULL R manifest and runs N CUDA updates in an isolated stage,
reporting allocator peaks plus nvidia-smi process usage; no export or games.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parent)]
import std_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402
from t172_build import build, dump, env_paths, sources  # noqa: E402
from trainer.multi_cache import Manifest, file_stamp  # noqa: E402

STEPS = 749_424
ADOPTED = 'task154_R_result30'
N_PUBLIC = 7_240_673


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def code_stamps():
    return {str(p.relative_to(REPO)): file_stamp(p)['sha256'] for p in
            [REPO / 'trainer/train_nnue.py', REPO / 'trainer/multi_cache.py',
             REPO / 'trainer/nnue.py', Path(__file__).resolve()]}


def verify_control(model, manifest):
    """Fail closed: old history alone cannot prove the historical seed/input identity.

    An existing provenance record must have been made during the original run;
    do not synthesize one from today's inputs. Legacy Task135 lacks this record,
    so it is automatically retrained. No manual reuse override.
    """
    evidence = model / 'task172_compatible_provenance.json'
    if not evidence.exists():
        return False, 'historical seed/input stamps not recorded; retrain C'
    try:
        rec = json.loads(evidence.read_text())
        hist = json.loads((model / 'history.json').read_text())
        cfg = json.loads((model / 'model_config.json').read_text())
        expected = dict(seed=20260621, steps=STEPS, fast=True, init_weights=None,
                        cp_result_lambda=1, lr=6e-5, batch=1024, epochs=12)
        good = all(rec.get(k) == v for k, v in expected.items())
        good &= rec['source_stamp'] == manifest.parts[0]['source_stamp']
        good &= rec['labels_stamp'] == manifest.parts[0]['labels_stamp']
        good &= rec['final_stamp'] == file_stamp(model / 'nnue_final.pt')
        good &= hist['n_train'] == 63_950_000 and hist['epochs'] == 12 and len(hist['history']) == 12
        good &= all(cfg.get(k) == v for k, v in dict(qat=True, qat_tie=True, qat_fake_quant=True,
                       factorize=True, augment_mirror=True, feature_dim=256, hidden_dim=32,
                       objective='cp', scale_cp=600, clamp_cp=2000, huber_delta=600, loss='huber',
                       dropout=0, wdl_weight=0, vm=False).items())
        good &= cfg.get('hidden_dim2', 32) == 32
        good &= bool(manifest.vector('mask').all())
        return bool(good), 'verified original provenance' if good else 'provenance mismatch'
    except (KeyError, ValueError, OSError):
        return False, 'incomplete provenance; retrain C'


def train_stage(name, cmd, out, logs, env, expected_steps=None):
    """A history file is not a completion marker. Bind resume/skip to all inputs."""
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(command=[str(x) for x in cmd], code=code_stamps())
    for flag in ('--manifest', '--init-weights'):
        if flag in cmd:
            identity[flag] = file_stamp(cmd[cmd.index(flag) + 1])
    if '--eval-npy' in cmd:
        directory = Path(cmd[cmd.index('--eval-npy') + 1])
        identity['labels'] = [file_stamp(directory / f'{s}_dlshogi_evals.npy') for s in ('train', 'val')]
    if '--compact-cache' in cmd:
        comp = Path(cmd[cmd.index('--compact-cache') + 1])
        identity['compact'] = file_stamp(comp / 'compact.json')
        # CompactCache verifies its source stamp on load. Bind the stage skip too.
        from trainer.compact_cache import source_stamp
        identity['source'] = [source_stamp(cmd[4], s) for s in ('train', 'val')]
    record = out / 'stage.json'
    done = out / 'done.json'
    if record.exists():
        if json.loads(record.read_text()) != identity:
            raise ValueError(f'{name}: stage inputs/code changed; use a fresh output directory')
    else:
        if (out / 'train_state.pt').exists() or (out / 'history.json').exists():
            raise ValueError(f'{name}: untracked previous training output')
        dump(record, identity)
    if done.exists():
        saved = json.loads(done.read_text())
        if saved['final'] != file_stamp(out / 'nnue_final.pt') or saved['history'] != file_stamp(out / 'history.json'):
            raise ValueError(f'{name}: completed artifacts changed')
        print(f'T172_SKIP {name}', flush=True)
        return saved
    log = logs / f'{name}.log'
    print(f'T172_TRAIN {name} log={log}', flush=True)
    # Poll the child without launching any additional work; collect external VRAM
    # usage as well as torch's allocated/reserved peaks printed by the trainer.
    peak_mib = 0
    with log.open('a', encoding='utf-8') as stream:
        stream.write('\nT172_COMMAND ' + json.dumps(identity['command']) + '\n')
        stream.flush()
        proc = subprocess.Popen([str(x) for x in cmd], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            try:
                stat = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                       '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=2)
                for line in stat.stdout.splitlines():
                    pid, memory = line.split(',')
                    if int(pid) == proc.pid:
                        peak_mib = max(peak_mib, int(memory))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                pass
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                continue
    if proc.returncode:
        atomic_json(out / 'last_failure.json', dict(returncode=proc.returncode, peak_nvidia_smi_mib=peak_mib, log=str(log)))
        raise RuntimeError(f'{name}: exit={proc.returncode}; see {log}; rerun same command to resume')
    hist = json.loads((out / 'history.json').read_text())
    if expected_steps is not None and hist.get('updates') != expected_steps:
        raise ValueError(f'{name}: expected {expected_steps} updates, got {hist.get("updates")}')
    peak = re.findall(r'CUDA_PEAK allocated=(\d+) reserved=(\d+)', log.read_text())
    result = dict(final=file_stamp(out / 'nnue_final.pt'), history=file_stamp(out / 'history.json'),
                  steps=hist.get('updates'), peak_nvidia_smi_mib=peak_mib,
                  torch_peak=dict(zip(('allocated', 'reserved'), map(int, peak[-1]))) if peak else None)
    atomic_json(done, result)
    print(f'T172_TRAIN_DONE {name} memory={result["torch_peak"]} smi_mib={peak_mib}', flush=True)
    return result


def export_stage(name, out, data, logs, py, env):
    identity = dict(weights=file_stamp(out / 'nnue_final.pt'), config=file_stamp(out / 'model_config.json'),
                    engines=[file_stamp(data / 'engines/yaneuraou/source/YaneuraOu-by-gcc'),
                             file_stamp(Path(env['BINBO'])), file_stamp(data / 'engines/yaneuraou/eval/nn.bin')],
                    code={str(p.relative_to(REPO)): file_stamp(p)['sha256'] for p in
                          [REPO / 'trainer/export_nnue.py', REPO / 'trainer/binbo_to_yane.py',
                           REPO / 'scripts/binbo_to_yane.sh',
                           REPO / 'experiments/alphazero/t147_self_relabel_train.py']})
    record, done = out / 'export_stage.json', out / 'export_done.json'
    eval_path = data / 'eval_dirs' / name / 'nn.bin'
    bnnq, gate = out / 'nnue_final.bnnq', logs / f'{name}_0cp.log'
    if record.exists():
        if json.loads(record.read_text()) != identity:
            raise ValueError(f'{name}: export input changed')
    else:
        if eval_path.exists() or bnnq.exists() or gate.exists():
            raise ValueError(f'{name}: existing export without provenance')
        dump(record, identity)
    if done.exists():
        expected = json.loads(done.read_text())
        if expected != [file_stamp(p) for p in (eval_path, bnnq, gate)]:
            raise ValueError(f'{name}: exported artifacts changed')
        return []
    # Without a completion marker, nn.bin or the gate log may be truncated.
    # These files belong to this recorded stage; regenerate both deterministically.
    eval_path.unlink(missing_ok=True)
    gate.unlink(missing_ok=True)
    bad = export(name, out, 'nnue_final.pt', data, logs, py, REPO, env)
    if not bad:
        atomic_json(done, [file_stamp(p) for p in (eval_path, bnnq, gate)])
    return bad


def common_args(mirror):
    return ['--qat', '--qat-tie', '--feature-dim', '256', '--hidden-dim', '32',
            '--batch-size', '1024', '--seed', '20260621', '--objective', 'cp', '--loss', 'huber',
            '--huber-delta', '600', '--clamp-cp', '2000', '--scale-cp', '600',
            '--cp-result-lambda', '1', '--factorize', '--augment-mirror', '--mirror-cache', mirror,
            '--fast', '--checkpoint-minutes', '10', '--resume', '--device', 'cuda']


def measure_all(names, env):
    failures, frames, results = [], {}, {}
    for name in [*names.values(), ADOPTED]:
        frames[name], bad = std_frame.measure_sets(name, env, sets=std_frame.SEED_SETS)
        failures += bad
    comparisons = [(names['R1'], names['C1']), (names['W1'], names['C1'])]
    comparisons += [(name, ADOPTED) for name in names.values()]
    for a, b in comparisons:
        total, per, ps, bad = std_frame.delta_sets(frames[a], frames[b], f'{a} - {b}')
        failures += bad
        results[f'{a} - {b}'] = dict(pooled=total, per_anchor=per, per_set=ps)
        print(f"T172_DELTA {a}-{b} elo={total['delta_elo']:+.2f} ci={total['ci95']}", flush=True)
    best = max(names.values(), key=lambda n: results[f'{n} - {ADOPTED}']['pooled']['delta_elo'])
    verdict = 'INVALID' if failures else 'NO_BETTER'
    if not failures and results[f'{best} - {ADOPTED}']['pooled']['ci95'][0] > 0:
        conf = {}
        cs = {'conf172': std_frame.spaced_set(20295000)}
        for label, which in [('routine', std_frame.ANCHORS), ('micro', std_frame.CONFIRM_ANCHORS)]:
            base, b1 = std_frame.measure_sets(ADOPTED, env, sets=cs, which=which)
            cand, b2 = std_frame.measure_sets(best, env, sets=cs, which=which)
            total, per, ps, b3 = std_frame.delta_sets(cand, base, f'confirm {label} {best}')
            failures += b1 + b2 + b3
            conf[label] = dict(pooled=total, per_anchor=per, per_set=ps)
            print(f"T172_CONFIRM {label} {best} elo={total['delta_elo']:+.2f} ci={total['ci95']}", flush=True)
        results['confirmation'] = conf
        good = all(d['delta_elo'] > 0 for c in conf.values() for d in c['per_anchor'].values())
        verdict = 'INVALID' if failures else 'ADOPT_CANDIDATE' if good else 'NOT_CONFIRMED'
    return dict(task=172, results=results, best=best, failures=failures, verdict=verdict,
                note='Paired game CI does not include training-seed variance; best of three is exploratory.')


@contextmanager
def lock(work):
    import fcntl
    with (work / '.run.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--pilot', type=int, metavar='STEPS')
    parser.add_argument('--models-root', type=Path)
    parser.add_argument('--logs-dir', type=Path)
    args = parser.parse_args()
    if args.pilot is not None and args.pilot < 1:
        parser.error('--pilot must be positive')
    env = env_paths()
    env.update(PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', FRAME_PARALLEL='4')
    data = Path(env['DATA_ROOT'])
    work = (args.work or data / 'work/task172').resolve()
    work.mkdir(parents=True, exist_ok=True)
    logs = args.logs_dir or Path(env['LOGS_DIR'])
    models = args.models_root or Path(env['MODELS_DIR'])
    logs.mkdir(parents=True, exist_ok=True)
    # std_frame reads os.environ itself; export both canonical paths and limits.
    os.environ.update(env)
    os.environ['LOGS_DIR'] = str(logs)
    std_frame.t106.PARALLEL = 4
    with lock(work):
        parts = sources(data)
        val = dict(cache=parts[0]['cache'], compact=parts[0]['compact'],
                   labels=str(data / 'work/task129/labels/val_dlshogi_evals.npy'))
        build(parts, val, work)
        manifests = {a: Manifest(work / f'{a}.json') for a in ('C', 'R', 'W')}
        mirror = work / 'mirror.npy'  # trainer generates mirror_shared.npy within the writable work dir
        common = common_args(mirror)
        if args.pilot:
            name = f'task172_pilot_R_{args.pilot}'
            out = work / name
            n = int(manifests['R'].vector('mask').sum())
            cmd = [env['VENV_PY'], '-B', '-m', 'trainer.train_nnue', val['cache'], '--out', out,
                   '--manifest', work / 'R.json', *common, '--lr', '6e-5',
                   '--epochs', str(math.ceil(args.pilot / math.ceil(n / 1024))), '--max-steps', str(args.pilot)]
            result = train_stage(name, cmd, out, logs, env, args.pilot)
            atomic_json(work / f'pilot_{args.pilot}.json', result)
            print(f'T172_PILOT_DONE steps={args.pilot} memory={result}', flush=True)
            return 0
        starts, training = {}, {}
        reuse, reason = verify_control(Path(env['MODELS_DIR']) / 'task135_student_fast', manifests['C'])
        print(f'T172_CONTROL_REUSE ok={reuse} reason={reason}', flush=True)
        for arm in ('C', 'R', 'W'):
            if arm == 'C' and reuse:
                starts[arm] = Path(env['MODELS_DIR']) / 'task135_student_fast'
                continue
            name = f'task172_{arm}'
            out = models / name
            n = int(manifests[arm].vector('mask').sum())
            cmd = [env['VENV_PY'], '-B', '-m', 'trainer.train_nnue', val['cache'], '--out', out,
                   '--manifest', work / f'{arm}.json', *common, '--lr', '6e-5',
                   '--epochs', str(math.ceil(STEPS / math.ceil(n / 1024))), '--max-steps', str(STEPS)]
            training[arm] = train_stage(name, cmd, out, logs, env, STEPS)
            starts[arm] = out
        pub = data / 'caches/public_suisho5_val7m'
        names, failures = {}, []
        for arm in ('C', 'R', 'W'):
            key, name = arm + '1', f'task172_{arm}1'
            names[key] = name
            out = models / name
            cmd = [env['VENV_PY'], '-B', '-m', 'trainer.train_nnue', pub, '--out', out,
                   *common, '--lr', '1e-5', '--epochs', '2', '--init-weights', starts[arm] / 'nnue_final.pt',
                   '--eval-npy', data / 'work/task142/labels_public', '--limit-train', str(N_PUBLIC),
                   '--compact-cache', Path(str(pub) + '_compact')]
            training[key] = train_stage(name, cmd, out, logs, env)
            bad = export_stage(name, out, data, logs, Path(env['VENV_PY']), env)
            failures += bad
        if failures:
            result = dict(task=172, failures=failures, verdict='INVALID')
        else:
            result = measure_all(names, env)
        result.update(training=training, audit=file_stamp(work / 'audit.json'), control_reused=reuse,
                      control_reuse_reason=reason)
        atomic_json(logs / 'task172_results.json', result)
        print(f"T172_VERDICT verdict={result['verdict']} results={logs / 'task172_results.json'}", flush=True)
        return int(bool(result['failures']))


if __name__ == '__main__':
    sys.exit(main())
