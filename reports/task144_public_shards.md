# Task 144: public PSV shard conversion

The new tool is `experiments/alphazero/t144_public_shards.py`; the reproducible CPU integration test is `experiments/alphazero/t144_public_shards_test.py`. Existing tracked files remain unchanged, including `STATUS.md`, as requested. No commits, downloads, GPU work or training were performed. Test outputs are confined to `~/binbo-shogi-data/work/task144_dryrun/`.

Restart review, 2026-10-19: reviewed the partial implementation against `qs_replace.py`, the installed YaneuraOu `qsearch_psv_record` implementation, Task 142's cache builder, `trainer/kp.py`, `trainer/mirror.py` and `trainer/compact_cache.py`. Added parent-directory sync after atomic JSON publication, embedded each conversion recipe (including engine/eval/code hashes) in final cache provenance, and rejected repeated shard paths in an HF listing. Extended the reproducible test to cover the previously separately reported recovery/CPU-loader checks and additional failure paths. The measurements below come from the fresh restart run; they do not rely on `/tmp` or previous outputs.

## Label and identity rules

The public score and signed game result describe the **original side to move**. The tool gives qsearch a disposable copy with score `+1`, uses `qs_replace.QSearchEngine` and `leaf_rows`, and applies the returned parity to the original public score **and** result. It does not use the marker score as a label or negate an already negated engine score. The loaded eval defaults to `eval_dirs/task129_student_64m_qsrel_e12`, as in Tasks 136b/139. Threads are restricted to 1 or 2, and NumPy/OpenMP thread limits are 2.

`--changed drop` is the default: discard every row whose packed position changed. Thus a retained label still describes exactly its original position. `--changed inherit` implements Task 119's original label with a parity adjustment, explicitly **without** relabelling. Task 128's results are the reason this is opt-in. There is no claim that a changed leaf has the original position's value. Leaves in check are counted and dropped in either mode. Engine decode errors or illegal PVs fail the part; no successful receipt is written. A sign-adjusted score outside int16 (only possible for negating -32768) fails explicitly instead of silently wrapping.

A position's hash is BLAKE2b-128 over the little-endian int32 canonical shared row, with a format-specific personalization string. These 77 features reversibly encode board, hands, both kings and absolute turn. Move counters, labels and source record metadata are excluded. Each half is canonically sorted; validation mirrors use `trainer.mirror.build_mirror_perm_shared` followed by sorting each half again. On every hash hit, all 77 features are compared, so a collision cannot discard a different position. New train rows are deduplicated exactly; only validation also excludes its left-right mirror.

## Operation

Run from this worktree, sourcing the canonical environment. For this sandbox, keep all outputs under the allowed dryrun directory. Outside the sandbox, choose a persistent output location with sufficient free disk.

```bash
source scripts/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
TOOL=experiments/alphazero/t144_public_shards.py
WORK="$DATA_ROOT/work/task144_dryrun/hao_run"

# Prefer a complete, locally supplied HF tree listing, including all API pages.
"$VENV_PY" "$TOOL" plan --dataset hao --target-rows 20000000 \
  --listing /path/to/hao_tree.json > "$DATA_ROOT/work/task144_dryrun/hao_plan.json"

# After each download, append its bounded parts to the same processing directory.
"$VENV_PY" "$TOOL" process /path/to/downloaded_shard.bin \
  --out "$WORK/processed" --threads 2
# Delete that download yourself ONLY after source_complete is printed.
# Repeat process for subsequent shards; the tool never deletes a source file.

# Build one snapshot after the desired shards have been processed.
"$VENV_PY" "$TOOL" dedup "$WORK/processed" --out "$WORK/dedup"
"$VENV_PY" "$TOOL" assemble "$WORK/dedup" --out "$WORK/cache" \
  --labels "$WORK/labels"
# assemble prints the exact Python interpreter, worktree and compact_cache command.
"$VENV_PY" -m trainer.compact_cache "$WORK/cache" "$WORK/cache_compact"
```

Without `--listing`, `plan` uses estimated 290,000,000-byte shards, 998 indices starting at 000, and the dataset-specific start time from the task. It marks the file names and sizes as estimates. This fallback is not a verified HF file inventory. The default changed drop of 50% and subsequent dedup drop of 5% are planning assumptions, not measurements of Hao/tanuki. Override `--changed-drop-rate` and `--dedup-drop-rate` after a representative downloaded shard. JSON listings accept the HF tree `path`/`size` fields or `siblings`/`rfilename` entries with sizes; duplicate shard paths are rejected and an exhausted listing fails rather than reporting an achieved target. The tool has no network path.

Processing uses default 200,000-record parts, never loads an entire multi-million-row source, and saves source path/size/SHA256, engine/eval/tool/feature-code identity, ranges, labels, audit arrays, timing and error statistics. A completed part is checksummed, flushed and renamed before the atomic manifest update. A rerun recovers published parts absent from the manifest and checks checksums. Incomplete parts are restarted. Changing the recipe requires a new output directory. `--keep-psv` is intended for audits; omit it for bulk processing.

Dedup builds a disk-backed SQLite index for the existing public train set, our 50,000 val positions and their mirrors, then visits new parts in explicit argument order (within process directories: source SHA256, then record offset). The first new surviving row wins, preserving its labels. Reference matches take priority over new-part matches. SQLite uses one thread and a 32 MiB page cache. Each keep mask is durable before its database transaction commits. Reference loading resumes by chunk; candidate work resumes by part. An interrupted mask publication is safely repeated. Inputs and masks are checksummed. A dedup directory represents a fixed snapshot: adding processed parts requires a new dedup output directory.

Assembly streams selected rows with memmaps into the normal int32 `(N,77)` train indices, int16 train evals and int8 signed results. `val_*` symlinks point to `caches/shared_64m`; validation labels are copied from `work/task129/labels`; train labels equal the public scores. It checks that the installed validation features match the dedup reference. Metadata preserves trainer-required feature settings and records source SHA256s, ranges, label policy and dedup counts. Cache and labels directories are staged and renamed individually. A rerun recovers a crash between the two publications. Assembly requires new final output directories. Training minibatch shuffling remains the trainer's job; the tool does not claim to globally shuffle the source shards.

Keep disk space for all derived data, not only one downloaded shard. Each retained processed row takes about **337 B**, final cache plus separate labels **313 B**, compact cache about **156 B**, and the dedup index about **30–50 B** per indexed position (plus reference positions and transaction space). Allow roughly **0.85 GB per million output rows**, more when dedup drops many processed rows, plus the ~0.21 GB reference index observed here, a download and the current working part. A 20M-row run is roughly 17–20 GB; 50M rows can exceed 45 GB free when staging/duplicates are included. Available disk at restart review was **32 GiB**, below the background estimate of 45 GB; use the plan's total-disk estimate before choosing the target. Do not remove processed parts/masks until assembly and compact round-trip verification succeed.

## Verification

Reproduce the complete test in a fresh directory:

```bash
source scripts/env.sh
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
"$VENV_PY" experiments/alphazero/t144_public_shards_test.py \
  --out "$DATA_ROOT/work/task144_dryrun/retest"
```

The test copies exactly the first 20,000 records of `public/nodchip_suisho5_depth9/shuffled.bin`; it checks both label policies, parity against decoded absolute turns, 500 random feature-to-board round trips, material/score correlations separately for both colours, real full-reference dedup, duplicate input parts, independent board-level mirrors, deliberately forced hash collisions, resume recovery, label arrays and the unchanged trainer's full compact-cache round trip. A quiet Suisho5 leaf may still change under our different qsearch eval; this fixture does not measure the expected Hao/tanuki drop rate.

Completed restart run: `~/binbo-shogi-data/work/task144_dryrun/restart_verified/results.json`,
with `command_*.log` and `compact.log` in the same directory, plus the complete
`~/binbo-shogi-data/work/task144_dryrun/restart_verified.log`. Exit status **0**.
Total **189.15 s**; real-reference dedup **152.40 s**. The 20,000-record fixture's
SHA256 is `215a0dad003abe609f88aba4259d83e5affb77fb590a2de00bc0f63a0d0b3e8d`.

| Check | Result |
|---|---:|
| PSV input / qsearch output | 20,000 / 20,000 |
| Changed / odd PV parity | 4,314 / 3,039 |
| Default retained before dedup | 15,686 |
| Decode errors / illegal PV / leaves in check | 0 / 0 / 0 |
| Independent sampled feature-to-leaf round trips | 500 / 500 |
| Original material vs score, black / white | +0.625743 / +0.642170 |
| Leaf material vs adjusted score, black / white | +0.636476 / +0.643146 |
| Flipped rows only, adjusted, black / white | +0.670122 / +0.651219 |
| Flipped rows only, omitting sign adjustment | −0.670122 / −0.651219 |
| Default part against existing public set | 0 new rows (expected: source is that same set) |
| Inherited part after full-reference and internal dedup | 3,937 new rows |
| Second identical inherited part | 0 new rows; 3,956 new-part matches including 19 internal repeats |
| Real hash collision comparisons | 0 |
| Forced-hash fixture | 13 unequal full-row comparisons; both distinct rows retained |
| Fixture existing / val / mirrored val / internal duplicate | 1 / 1 / 1 / 1 correctly excluded |
| Compact train / val round trip | 3,937 / 50,000 rows, 0 mismatches, 0 escapes |

The complete dedup test feeds the default part twice and the inherited part twice:
71,372 candidate rows, 63,460 existing-public matches, 3,975 new-part duplicate matches,
3,937 kept. The real fixture has no val overlap; explicit val and mirrored-val exclusions
are exercised by separate board-derived fixtures. Completed process resume, orphan-part
manifest recovery, and a mask written before its database commit also pass.
Additional checks cover four part boundaries in a 128-record file, recovery between
cache/labels publication, and loading the final compact cache on CPU and comparing
500 decoded rows to the source. An owned interrupted part is regenerated, an all-changed
input publishes a valid zero-row part, and a changed resume recipe is rejected.
Injected engine decode/illegal-PV failures and int16 overflow publish no completed part;
a real board in check returned by the fixture engine is counted and excluded.
Planning also rejects an exhausted listing and duplicate shard paths.

Measured qsearch time is **0.30157 s** for 20,000 records at 2 threads:
**66,319 rows/s total, 33,159 rows/s per configured thread**. Feature conversion,
hashing, filtering, audit round trips and bounded-stream writing take **1.31977 s**
for the inherited 20,000 rows: **15,154 rows/s on one CPU thread**. These phase
times exclude engine startup and final `.npy` publication/checksumming; the default
20k part takes **1.63 s** end to end (**2.14 s** with label inheritance). At these
measured rates, 7.5M rows would take about **113 s** of qsearch plus **495 s** of
feature conversion if all rows were retained. Dedup's timed phase visits 7,340,673
reference rows and 71,372 candidate rows in 152.40 s, about **48,637 rows/s on one
CPU thread**, dominated by reference-index construction. Its 212,291,584-byte
SQLite index can be resumed/reused within that fixed snapshot. These rates do not
include source fingerprinting, mirror permutation construction, assembly or compaction.
This is only an extrapolation from already Suisho5-qsearch-replaced positions;
raw Hao/tanuki positions may make qsearch slower and have different drop rates.
The machine was also running another CPU-heavy evaluation, so timing is load-dependent.
Use the first full downloaded shard's per-part timings to estimate the bulk run.
