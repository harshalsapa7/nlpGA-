"""Offline checks for the safe Kaggle notebook; no GPU or downloads needed."""

import ast
import contextlib
import datetime
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "04_kaggle_safe_run.ipynb"


def git(*args):
    return subprocess.check_output(
        ["git", *(str(arg) for arg in args)], text=True, stderr=subprocess.STDOUT,
    ).strip()


class KaggleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = {
            cell["id"]: "".join(cell["source"])
            for cell in cls.notebook["cells"]
        }

    def test_valid_unexecuted_python_cells(self):
        self.assertEqual(self.notebook["nbformat"], 4)
        ids = [cell["id"] for cell in self.notebook["cells"]]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                compile("".join(cell["source"]), f"{NOTEBOOK.name}:{cell['id']}", "exec")

    def test_long_runs_default_to_disabled(self):
        namespace = {}
        exec(self.cells["configuration"], namespace)
        for flag in ("RUN_FULL", "RUN_PROPORTIONAL", "RUN_LAYER_BUDGET", "ALLOW_LOW_DISK"):
            self.assertIs(namespace[flag], False)
        self.assertEqual(namespace["REPO_URL"], "https://github.com/harshalsapa7/nlpGA-.git")
        self.assertEqual(namespace["REPO_BRANCH"], "master")

    @contextlib.contextmanager
    def checkout_fixture(self):
        """A real local Git remote: clone/update tests never contact GitHub."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            remote, seed, checkout = root / "remote.git", root / "seed", root / "checkout"
            git("init", "--bare", "--initial-branch=master", remote)
            git("init", "--initial-branch=master", seed)
            git("-C", seed, "config", "user.email", "test@example.invalid")
            git("-C", seed, "config", "user.name", "Notebook test")
            (seed / "routerguard").mkdir()
            (seed / "routerguard/cli.py").write_text("# tiny test checkout\n")
            git("-C", seed, "add", ".")
            git("-C", seed, "commit", "-m", "initial")
            git("-C", seed, "remote", "add", "origin", remote)
            git("-C", seed, "push", "origin", "master")

            def run_checked(command, label):
                subprocess.run(
                    [str(arg) for arg in command], check=True, capture_output=True, text=True,
                )

            namespace = {
                "REPO_URL": str(remote), "REPO_BRANCH": "master", "REPO_DIR": checkout,
                "subprocess": subprocess, "run_checked": run_checked,
                "PREFLIGHT_OK": True, "SMOKE_OK": True, "FULL_OK": True,
            }
            yield namespace, seed

    def test_clone_and_clean_checkout_fetch_latest_code(self):
        with self.checkout_fixture() as (namespace, seed):
            exec(self.cells["clone"], namespace)
            checkout = namespace["REPO_DIR"]
            old_sha = namespace["CODE_SHA"]
            self.assertEqual(old_sha, git("-C", seed, "rev-parse", "HEAD"))
            (seed / "routerguard/cli.py").write_text("# updated code\n")
            git("-C", seed, "commit", "-am", "fix")
            git("-C", seed, "push", "origin", "master")
            namespace.update(PREFLIGHT_OK=True, SMOKE_OK=True, FULL_OK=True)
            exec(self.cells["clone"], namespace)
            self.assertNotEqual(namespace["CODE_SHA"], old_sha)
            self.assertEqual(namespace["CODE_SHA"], git("-C", seed, "rev-parse", "HEAD"))
            self.assertEqual((checkout / "routerguard/cli.py").read_text(), "# updated code\n")
            for flag in ("PREFLIGHT_OK", "SMOKE_OK", "FULL_OK"):
                self.assertFalse(namespace[flag])
            # Already up to date is also safe.
            exec(self.cells["clone"], namespace)

    def test_checkout_update_preserves_local_edits(self):
        for untracked in (False, True):
            with self.subTest(untracked=untracked), self.checkout_fixture() as (namespace, seed):
                exec(self.cells["clone"], namespace)
                checkout = namespace["REPO_DIR"]
                path = checkout / ("notes.txt" if untracked else "routerguard/cli.py")
                path.write_text("keep my work\n")
                with self.assertRaisesRegex(RuntimeError, "local changes"):
                    exec(self.cells["clone"], namespace)
                self.assertEqual(path.read_text(), "keep my work\n")

    def test_checkout_update_rejects_other_branch_or_origin(self):
        with self.checkout_fixture() as (namespace, seed):
            exec(self.cells["clone"], namespace)
            checkout = namespace["REPO_DIR"]
            git("-C", checkout, "switch", "-c", "my-work")
            with self.assertRaisesRegex(RuntimeError, "Expected branch"):
                exec(self.cells["clone"], namespace)
            git("-C", checkout, "remote", "set-url", "origin", "https://example.invalid/other.git")
            with self.assertRaisesRegex(RuntimeError, "another origin"):
                exec(self.cells["clone"], namespace)

    def test_checkout_update_never_resets_divergent_commits(self):
        with self.checkout_fixture() as (namespace, seed):
            exec(self.cells["clone"], namespace)
            checkout = namespace["REPO_DIR"]
            git("-C", checkout, "config", "user.email", "test@example.invalid")
            git("-C", checkout, "config", "user.name", "Notebook test")
            (checkout / "local.txt").write_text("local work\n")
            git("-C", checkout, "add", "local.txt")
            git("-C", checkout, "commit", "-m", "local work")
            local_sha = git("-C", checkout, "rev-parse", "HEAD")
            (seed / "remote.txt").write_text("remote work\n")
            git("-C", seed, "add", "remote.txt")
            git("-C", seed, "commit", "-m", "remote work")
            git("-C", seed, "push", "origin", "master")
            with self.assertRaises(subprocess.CalledProcessError):
                exec(self.cells["clone"], namespace)
            self.assertEqual(git("-C", checkout, "rev-parse", "HEAD"), local_sha)
            self.assertEqual((checkout / "local.txt").read_text(), "local work\n")

    def test_smoke_explicitly_caps_dataset_and_memory(self):
        namespace = {"RESULTS_DIR": Path("/unused")}
        exec(self.cells["experiment-helper"], namespace)
        base, smoke = namespace["BASE_SETTINGS"], namespace["SMOKE_SETTINGS"]
        for key in ("stats_batch_size", "ppl_batch_size", "mmlu_batch_size", "similarity_batch_size"):
            self.assertEqual(base[key], 1)
        self.assertEqual(base["precision"], "8bit")
        self.assertFalse(base["with_merge"])
        for key in ("do_mmlu", "do_throughput", "with_refit"):
            self.assertFalse(smoke[key])
        self.assertEqual(smoke["overrides"]["calibration"], {"n_seqs": 4, "seq_len": 256})
        self.assertEqual(smoke["overrides"]["eval"], {"n_seqs": 2, "seq_len": 256})
        self.assertNotIn("overrides", base)
        self.assertNotIn("ratios", base)
        self.assertNotIn("seeds", base)

    def test_embedded_child_programs_compile(self):
        for cell_id in ("gpu-check", "preflight", "experiment-helper"):
            tree = ast.parse(self.cells[cell_id])
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    if any(
                        isinstance(target, ast.Name) and target.id in ("gpu_check", "preflight", "program")
                        for target in node.targets
                    ):
                        compile(node.value.value, f"child:{cell_id}", "exec")

    def test_preflight_exercises_cuda_calibration_before_download(self):
        self.assertIn("OlmoeForCausalLM(config)", self.cells["preflight"])
        self.assertIn("biases, fit = fit_biases(", self.cells["preflight"])
        self.assertIn("device='cuda:0'", self.cells["preflight"])
        self.assertIn("state = run_pipeline(PipelineSpec(", self.cells["preflight"])
        self.assertIn("TokenSplit('synthetic_calib'", self.cells["preflight"])
        self.assertNotIn("from_pretrained", self.cells["preflight"])

    def test_complete_tiny_preflight_program_offline_on_cpu(self):
        """Execute the notebook's actual synthetic pipeline, not a parallel copy."""
        import torch
        from transformers import OlmoeConfig, OlmoeForCausalLM
        from routerguard.balance import fit_biases
        from routerguard.data import TokenSplit
        from routerguard.pipeline import PipelineSpec, run_pipeline

        tree = ast.parse(self.cells["preflight"])
        program = next(
            node.value.value for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "preflight" for t in node.targets)
        )
        # Only replace hardware/precision, not the pipeline or its data/settings.
        program = program[program.index("# Exercise the complete tiny pipeline"):]
        program = program.replace("cuda:0", "cpu").replace("float16", "float32").replace("'fp16'", "'fp32'")
        namespace = {
            "torch": torch, "tempfile": tempfile, "SimpleNamespace": SimpleNamespace,
            "OlmoeConfig": OlmoeConfig, "OlmoeForCausalLM": OlmoeForCausalLM,
            "fit_biases": fit_biases, "TokenSplit": TokenSplit,
            "PipelineSpec": PipelineSpec, "run_pipeline": run_pipeline,
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(program, namespace)
        self.assertGreater(namespace["state"]["checkpoint_bytes_saved"], 0)
        self.assertEqual(namespace["fit"]["updates"], 2)

    def test_experiment_gates(self):
        # Later cells remain safe if a user manually skips over a failed cell.
        namespace = {
            "RUN_FULL": False, "RUN_PROPORTIONAL": False, "RUN_LAYER_BUDGET": False,
            "run_experiment": lambda *args: self.fail("Disabled cell launched a run"),
        }
        for name in ("full", "proportional", "layer-budget"):
            exec(self.cells[name], namespace)
        namespace.update(RUN_FULL=True, SMOKE_OK=False)
        with self.assertRaises(AssertionError):
            exec(self.cells["full"], namespace)
        namespace.update(RUN_PROPORTIONAL=True, FULL_OK=False)
        with self.assertRaises(AssertionError):
            exec(self.cells["proportional"], namespace)
        namespace.update(RUN_LAYER_BUDGET=True)
        with self.assertRaises(AssertionError):
            exec(self.cells["layer-budget"], namespace)

    def test_summary_path_and_archive_exclude_weights(self):
        self.assertIn("*/metrics/summary.json", self.cells["export"])
        self.assertIn("for root in (RESULTS_DIR, LOGS_DIR)", self.cells["export"])
        self.assertNotIn("CACHE_DIR", self.cells["export"])
        self.assertNotIn("ENV_DIR", self.cells["export"])

    def test_child_failure_is_logged_and_raised(self):
        tree = ast.parse(self.cells["helpers"])
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_checked"
        )
        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "datetime": datetime, "LOGS_DIR": Path(directory),
                "subprocess": subprocess, "CHILD_ENV": {},
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "helper", "exec"), namespace)
            with contextlib.redirect_stdout(io.StringIO()):
                namespace["run_checked"](
                    [sys.executable, "-c", "print('success-marker')"], "success",
                )
                with self.assertRaisesRegex(RuntimeError, "failed.*exit 3"):
                    namespace["run_checked"](
                        [sys.executable, "-c", "print('failure-marker'); raise SystemExit(3)"],
                        "failure",
                    )
            self.assertIn("success-marker", next(Path(directory).glob("*_success.log")).read_text())
            self.assertIn("failure-marker", next(Path(directory).glob("*_failure.log")).read_text())

    def test_install_repairs_partial_environment_without_ensurepip(self):
        with tempfile.TemporaryDirectory() as directory:
            env_dir = Path(directory) / "env"
            python = env_dir / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()  # A failed venv attempt already left an interpreter behind.
            commands = []
            namespace = {
                "sys": sys, "ENV_DIR": env_dir, "PYTHON": python, "CHILD_ENV": {},
                "run_checked": lambda command, label: commands.append((command, label)),
                "subprocess": SimpleNamespace(check_output=lambda *args, **kwargs: "2.11.0+cu130\n"),
            }
            exec(self.cells["install"], namespace)
            create, label = commands[0]
            self.assertEqual(label, "create-environment")
            self.assertIn("--without-pip", create)
            self.assertIn("--system-site-packages", create)
            self.assertNotIn("--clear", create)
            self.assertEqual(namespace["PIP_COMMAND"], [
                sys.executable, "-m", "pip", "--python", str(python),
            ])
            self.assertEqual(commands[-1][0][:5], namespace["PIP_COMMAND"])
            self.assertTrue(namespace["SETUP_OK"])
            for command, label in commands:
                if label == "verify-environment":
                    compile(command[2], "verify-environment", "exec")
            self.assertEqual(
                (env_dir / "kaggle-torch-constraint.txt").read_text(),
                "torch==2.11.0+cu130\n",
            )

    def test_pip_targets_environment_without_ensurepip_or_network(self):
        """Install a tiny local wheel into a pip-less venv, never into the host."""
        import venv

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_dir = root / "env"
            # Fail the test if the bootstrap accidentally requests ensurepip.
            with mock.patch.object(venv.EnvBuilder, "_setup_pip", side_effect=AssertionError("ensurepip called")):
                venv.EnvBuilder(system_site_packages=True, with_pip=False).create(env_dir)
            python = env_dir / "bin/python"
            pip_command = [sys.executable, "-m", "pip", "--python", str(python)]
            wheel = root / "rg_bootstrap_probe-0.0.1-py3-none-any.whl"
            dist_info = "rg_bootstrap_probe-0.0.1.dist-info"
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("rg_bootstrap_probe/__init__.py", "MARKER = 'venv-only'\n")
                archive.writestr(
                    f"{dist_info}/METADATA",
                    "Metadata-Version: 2.1\nName: rg-bootstrap-probe\nVersion: 0.0.1\n",
                )
                archive.writestr(
                    f"{dist_info}/WHEEL",
                    "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
                )
                archive.writestr(f"{dist_info}/RECORD", "")
            subprocess.run(
                [*pip_command, "install", "--no-deps", "--no-index", str(wheel)],
                check=True, capture_output=True, text=True,
            )
            verify = (
                "import pathlib, sys, rg_bootstrap_probe; "
                "assert rg_bootstrap_probe.MARKER == 'venv-only'; "
                "assert pathlib.Path(rg_bootstrap_probe.__file__).is_relative_to(pathlib.Path(sys.prefix))"
            )
            subprocess.run([str(python), "-c", verify], check=True, capture_output=True, text=True)
            subprocess.run(
                [sys.executable, "-c",
                 "import importlib.util; assert importlib.util.find_spec('rg_bootstrap_probe') is None"],
                check=True, capture_output=True, text=True,
            )


if __name__ == "__main__":
    unittest.main()
