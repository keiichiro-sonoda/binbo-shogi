"""CPU integration/regression checks using read-only real Task 153 games.

PYTHONDONTWRITEBYTECODE=1 <pgx-python> tests/test_t173_wst2.py
Outputs are isolated in a fresh directory below TASK173_TEST_WORK (persistent).
"""
import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import statistics
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
os.environ["JAX_PLATFORMS"] = "cpu"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
SCRIPT = Path(__file__).resolve().parents[1] / "experiments/alphazero/t173_wst2.py"
spec = importlib.util.spec_from_file_location("t173", SCRIPT)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
REPO = Path(os.environ.get("BINBO_REPO", Path.cwd())).resolve()
sys.path.insert(0, str(REPO / "experiments/alphazero"))
import numpy as np
import jax
import jax.numpy as jnp
from pgx.experimental.shogi import from_sfen, _from_board
from pgx._src.games.shogi import _observe
import m2_moves


class ToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent = Path(os.environ.get("TASK173_TEST_WORK", t.WORK / "tests_output"))
        parent.mkdir(parents=True, exist_ok=True)
        cls.work = Path(tempfile.mkdtemp(prefix="cpu_", dir=parent))
        print("TEST_WORK=" + str(cls.work), flush=True)
        cls.probes = [t.DATA_ROOT / "work/strategy/gate0", t.DATA_ROOT / "work/strategy/probe13/main"]
        paths, keys, _ = t.exclusions(cls.probes)
        cls.excluded_keys = keys
        cls.folders = []
        # Both rounds contain game_id=0: their keys must stay distinct.
        for idx, task in enumerate(("task154", "task159")):
            folder = cls.work / f"fixtures/{task}/games"
            folder.mkdir(parents=True)
            cls.folders.append(folder)
            original = t.DATA_ROOT / "work" / task / "games"
            for gid in range(12):
                path = original / f"game_{gid:012d}.json"
                shutil.copyfile(path, folder / path.name)
            # Copy a known probe game under a different path. Path-only filters
            # would leak this game; recipe+id filtering must exclude it.
            root = t.read(cls.probes[idx] / "roots.json")["roots"][0]
            shutil.copyfile(root["source"], folder / "game_probe_alias.json")
        cls.dataset = cls.work / "dataset"
        cls.build_to(cls.dataset)
        cls.data = t.Dataset(cls.dataset)
        cls.encoder = t.Encoder(jax)

    @classmethod
    def args(cls, command, *extras):
        return t.parser().parse_args(["--repo", str(REPO), "--device", "cpu", command, *map(str, extras)])

    @classmethod
    def build_to(cls, out, reverse=False):
        folders = cls.folders[::-1] if reverse else cls.folders
        args = cls.args("build", "--games", *folders, "--out", out, "--max-positions-per-game", 12)
        t.build(args)

    def test_01_exclusion_split_determinism(self):
        second = self.work / "dataset_repeat"
        self.build_to(second, reverse=True)
        self.assertEqual(t.read(second / "meta.json"), self.data.meta)
        for name in ("train.bin", "val.bin", "games.jsonl"):
            self.assertEqual((second / name).read_bytes(), (self.dataset / name).read_bytes())
        games = self.data.games
        self.assertTrue(not ({g["key"] for g in games} & self.excluded_keys))
        self.assertEqual(self.data.meta["counts"]["excluded_aliases"], 2)
        self.assertEqual(len(games), 24)
        self.assertEqual(sum(g["split"] == "val" for g in games), 1)
        self.assertEqual(len({g["key"] for g in games if g["game_id"] == "0"}), 2)
        self.assertEqual(sum(len(r) for r in self.data.rows.values()), 288)
        large = [dict(key=str(i)) for i in range(1000)]
        split = t.split_games(large)
        self.assertEqual(list(split.values()).count("val"), 20)
        # Tamper the audit, recompute checksum: semantic exclusion still fails.
        bad = self.work / "dataset_bad_exclusion"
        shutil.copytree(self.dataset, bad)
        meta = t.read(bad / "meta.json")
        meta["excluded_keys"].append(games[0]["key"])
        meta["dataset_id"] = t.identity({k: v for k, v in meta.items() if k != "dataset_id"})
        t.json_write(bad / "meta.json", meta)
        with self.assertRaisesRegex(ValueError, "excluded"):
            t.Dataset(bad)

    def test_02_targets(self):
        self.assertAlmostEqual(t.value_target(0, 1), .3)
        self.assertAlmostEqual(t.value_target(0, -1), -.3)
        self.assertEqual(t.value_target(32000, 1, True), 1)
        self.assertEqual(t.value_target(-32000, -1, True), -1)
        self.assertAlmostEqual(t.value_target(32000, -1, True), .4)
        self.assertAlmostEqual(t.value_target(-32000, 1, True), -.4)
        self.assertEqual(t.value_target(32000, 0, True), 1)
        self.assertEqual(t.value_target(-32000, 0, True), -1)
        for result in (-1, 0, 1):
            self.assertEqual(t.value_target(3000, result), t.value_target(9000, result))
            self.assertEqual(t.value_target(-3000, result), t.value_target(-9000, result))
            self.assertAlmostEqual(t.value_target(700, result), -t.value_target(-700, -result))
        self.assertAlmostEqual(t.value_target(600, 0), 2 / (1 + math.exp(-1)) - 1)
        for game in self.data.games:
            raw = t.load_game(game["source"])
            targets = {r["ply"]: r for r in raw["rows"]}
            rows = self.data.rows[game["split"]]
            game_index = self.data.games.index(game)
            for row in rows[rows["game"] == game_index]:
                original = targets[int(row["ply"])]
                self.assertGreaterEqual(row["ply"], 16)
                self.assertNotIn(int(row["ply"]), raw["random_plies"])
                self.assertEqual(row["action"], m2_moves.usi_to_action(original["move"], original["turn"]))
                self.assertAlmostEqual(float(row["value"]), t.value_target(original["score"], original["result"], original["mate"]), places=6)

    def test_03_encode_oracle_and_mask(self):
        sfens = [s.decode() for rows in self.data.rows.values() for s in rows["sfen"]]
        # Material-conserving positions exercise the known pgx 18-pawn bug
        # for both colors; ordinary sampled roots need not have >=10 in hand.
        extra = {"b": "4k4/9/9/9/9/9/9/9/4K4 b 2R2B4G4S4N4L18P 1",
                 "w": "4k4/9/9/9/9/9/9/9/4K4 w 2r2b4g4s4n4l18p 1"}
        sfens.extend(extra.values())
        for start in range(0, len(sfens), 32):
            chunk = sfens[start:start + 32]
            self.encoder.batch(chunk, size=32, check_legal=True)
        observe = jax.jit(lambda x: _observe(x, flip=jnp.bool_(True)))
        for sfen in sfens[:4] + list(extra.values()):
            reference = from_sfen(sfen)
            if t.re.search(r"\d{2}", sfen.split()[2]):
                absolute = from_sfen(sfen.split()[0] + " b - 1")
                hand = np.zeros((2, 7), np.int32)
                for n, p in t.re.findall(r"(\d*)([A-Za-z])", sfen.split()[2]):
                    hand[int(p.islower()), "PLNSBRG".index(p.upper())] = int(n or 1)
                reference = _from_board(int(sfen.split()[1] == "w"), absolute._x.board, hand)
            obs, mask = self.encoder.batch([sfen], size=32)
            np.testing.assert_array_equal(obs[0], np.asarray(observe(reference._x)))
            np.testing.assert_array_equal(mask[0], np.asarray(reference.legal_action_mask))
        # Huge illegal logits must not displace a legal top-1/top-2.
        _, legal = self.encoder.batch(sfens[:4])

        def fake_predict(model, obs):
            del model, obs
            return jnp.asarray(np.where(legal, np.arange(2187)[None, :], 1e9)), jnp.zeros(4)

        preds = list(t.policy_predictions(sfens[:4], fake_predict, None, self.encoder, jax, jax.devices()[0], 4))
        for mask, pred in zip(legal, preds):
            self.assertTrue(all(mask[a] for a in pred["top_actions"]))
            self.assertEqual(pred["top_actions"][0], np.flatnonzero(mask)[-1])
        # Training CE must ignore illegal logits too, including their gradients.
        class Forward:
            def apply(self, params, state, obs, is_eval=False):
                return (jnp.tile(params["logits"], (len(obs), 1)), jnp.zeros(len(obs))), state

        optimizer, step = t.train_step_fn(Forward(), jax)
        logits = jnp.arange(2187, dtype=jnp.float32).at[100].set(1e9)
        model = ({"logits": logits}, {})
        legal = np.zeros((2, 2187), bool)
        legal[:, [7, 9]] = True
        updated, _, losses = step(model, optimizer.init(model[0]), jnp.zeros((2, 1)),
                                  jnp.array(legal), jnp.array([7, 7]), jnp.zeros(2), 1e-3)
        self.assertTrue(np.isfinite(np.asarray(losses)).all())
        self.assertLess(float(losses[1]), 3)
        self.assertEqual(float(updated[0]["logits"][100]), 1e9)
        self.assertNotEqual(float(updated[0]["logits"][7]), 7.)

    def test_03b_sampler_and_schedule(self):
        sampler = t.Sampler(5, 3, t.SEED)
        all_indices = np.concatenate([sampler.indices(i) for i in range(5)])
        np.testing.assert_array_equal(np.sort(all_indices), np.repeat(np.arange(5), 3))
        for i in range(10):
            np.testing.assert_array_equal(sampler.indices(i), t.Sampler(5, 3, t.SEED).indices(i))
        settings = dict(warmup=10, total_steps=100, lr=1e-5)
        self.assertAlmostEqual(t.learning_rate(0, settings), 1e-6)
        self.assertAlmostEqual(t.learning_rate(9, settings), 1e-5)
        self.assertAlmostEqual(t.learning_rate(99, settings), 1e-6)

    def test_04_gate_rule_exact_oracle(self):
        source = ast.parse((REPO / "experiments/alphazero/t167_posthoc_agree_alloc.py").read_text())
        definition = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "rule")
        scope = {"st": statistics}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), "posthoc-rule", "exec"), scope)
        records, _ = t.probe_records(self.probes)
        self.assertEqual(len(records), 1184)
        rows = []
        policies = {str(folder.resolve()): t.read(folder / "policy.json")["roots"] for folder in self.probes}
        for record in records:
            row = record["report"]
            policy = policies[record["folder"]][row["root_id"]]["top_moves"][0]
            agree = policy["move"] == record["short_move"]
            rows.append(dict(ln=row["loss_n"], lu=row["loss_uniform"], l4=row["loss_4n"], ll=row["loss_live_4n"],
                             agree=agree, agree3=agree and policy["probability"] >= .3))
        for key in ("agree", "agree3"):
            for big in ("l4", "ll"):
                np.testing.assert_allclose(t.gate_rule(rows, key, big), scope["rule"](rows, key, big), rtol=0, atol=1e-12)
        print("GATE_ORACLE=" + t.canonical(t.gate_summary(rows, 10)), flush=True)

    def test_05_train_resume_and_eval(self):
        # Full-sized WS-T, 4 tiny CPU updates; resume after update 2.
        complete, resumed = self.work / "train_complete", self.work / "train_resume"
        common = ["--data", self.dataset, "--batch", "4", "--steps", "4", "--bench-steps", "1",
                  "--checkpoint-every", "2", "--log-every", "1"]
        t.train(self.args("train", *common, "--out", complete))
        t.train(self.args("train", *common, "--out", resumed, "--stop-after", 2))
        t.train(self.args("train", "--data", self.dataset, "--out", resumed,
                          "--resume", resumed / "step_000000002.ckpt", "--log-every", 1))
        with (complete / "step_000000004.ckpt").open("rb") as f:
            a = pickle.load(f)
        with (resumed / "step_000000004.ckpt").open("rb") as f:
            b = pickle.load(f)
        for x, y in zip(jax.tree_util.tree_leaves((a["model"], a["opt_state"])),
                        jax.tree_util.tree_leaves((b["model"], b["opt_state"]))):
            np.testing.assert_array_equal(x, y)
        self.assertEqual(a["train_settings"], b["train_settings"])
        checkpoint = complete / "step_000000004.ckpt"
        result_dir = self.work / "eval"
        t.evaluate(self.args("eval", "--data", self.dataset, "--checkpoint", checkpoint,
                             "--out", result_dir, "--batch", 4, "--bootstrap", 10,
                             "--probe-limit", 8, "--allow-subset", "--check-val-legal"))
        result = t.read(result_dir / "eval.json")
        self.assertTrue(result["subset"])
        for label in ("ws2", "wst2"):
            self.assertEqual(result["results"][label]["probe"]["n"], 8)
            self.assertEqual(result["results"][label]["heldout"]["positions"], 12)
            self.assertTrue(0 <= result["results"][label]["heldout"]["value_brier"] <= 1)
        with self.assertRaisesRegex(ValueError, "subset"):
            t.evaluate(self.args("eval", "--data", self.dataset, "--checkpoint", checkpoint))
        # Existing m2 USI loader must understand the fine-tuned checkpoint.
        from m2_usi import load_model
        _, _, info = load_model(str(checkpoint))
        self.assertEqual(info["arch"], "transformer")
        t.json_write(self.work / "test_summary.json", dict(positions=288, probe_legal_extra=2,
                     stable_nonmate_roots=1184, resume="model + Adam bitwise equal", eval=result["delta"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
