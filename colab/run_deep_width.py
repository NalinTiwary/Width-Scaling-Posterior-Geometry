"""Run the deep-width sweep on the CUDA GPU in this Colab runtime and publish the small files.

One runtime has one GPU, so the tasks in this process run one after another. Open more
notebooks, each with its own GPU runtime, to run other task ids at the same time.
The queued A10 job writes ``results/deep_width``. This writer uses
``results/deep_width_colab`` and records the GPU name Colab assigned.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(ROOT), str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])

KEEP = {".csv", ".json", ".pdf", ".png", ".txt", ".yaml", ".yml", ".npz"}
MAX_BYTES = 50 * 1024 * 1024
PUBLIC_REMOTE = "https://github.com/NalinTiwary/Width-Scaling-Posterior-Geometry.git"


def run_argv(task, out, step, level):
    return [sys.executable, "-m", "deep_width.run", "run",
            "--task", str(int(task)), "--device", "cuda", "--allow-any-device",
            "--kernel", "pcnl", "--step", str(step), "--level", str(int(level)),
            "--pilot-steps", "128", "--config", "configs/revision.yaml",
            "--out", str(out)]


def files_to_publish(root):
    chosen = []
    root = Path(root)
    if not root.exists():
        return chosen
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in KEEP and path.stat().st_size <= MAX_BYTES:
            chosen.append(path)
    return chosen


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is visible. In Colab choose Runtime, Change runtime type, "
                         "select a GPU, then Runtime, Restart and run all.")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory
    value = torch.zeros(4, dtype=torch.float64, device="cuda")
    value = value + torch.eye(4, dtype=torch.float64, device="cuda")
    if value.dtype != torch.float64 or not torch.isfinite(value).all():
        raise SystemExit(f"{name} did not complete a float64 CUDA operation")
    print(f"GPU {name}, {total / 1024**3:.1f} GiB, float64 CUDA ok")
    return name


def _git(repo, args, token=None):
    """Run git. A token is used only for push, through an askpass script."""
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    command = ["git"]
    askpass_path = None
    if token:
        handle, name = tempfile.mkstemp(prefix="bnn-askpass-", suffix=".sh")
        os.close(handle)
        askpass_path = Path(name)
        askpass_path.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "*[Uu]sername*) printf '%s\\n' 'x-access-token' ;;\n"
            "*) printf '%s\\n' \"$GITHUB_TOKEN\" ;;\n"
            "esac\n"
        )
        askpass_path.chmod(askpass_path.stat().st_mode | stat.S_IEXEC)
        env["GITHUB_TOKEN"] = token
        env["GIT_ASKPASS"] = str(askpass_path)
        env["SSH_ASKPASS"] = str(askpass_path)
        # Drop helpers that try to open a terminal. The empty value clears the list.
        command += ["-c", "credential.helper="]
    command += list(args)
    try:
        result = subprocess.run(command, cwd=repo, env=env, text=True, capture_output=True)
    finally:
        if askpass_path is not None:
            askpass_path.unlink(missing_ok=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if token:
            detail = detail.replace(token, "TOKEN")
        print(detail, flush=True)
        shown = " ".join(args)
        print(f"GIT_FAILED: git {shown} status {result.returncode}", flush=True)
        raise SystemExit(f"GIT_FAILED: git {shown} status {result.returncode}")
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)


def _ensure_identity(repo):
    for key, value in (("user.email", "nalint2@illinois.edu"), ("user.name", "NalinTiwaryUIUC")):
        current = subprocess.run(["git", "config", "--get", key], cwd=repo,
                                 text=True, capture_output=True)
        if current.returncode != 0:
            subprocess.check_call(["git", "config", key, value], cwd=repo)


def trackable(repo, relative):
    """Drop paths Git is configured to ignore. Adding those makes git add exit 1."""
    if not relative:
        return []
    result = subprocess.run(["git", "check-ignore", "--stdin"], cwd=repo,
                            input="\n".join(relative) + "\n", text=True, capture_output=True)
    ignored = set(result.stdout.splitlines())
    kept = [path for path in relative if path not in ignored]
    if ignored:
        print("Skipping ignored files: " + ", ".join(sorted(ignored)), flush=True)
    return kept


def publish(repo, root, message, token=None):
    """Commit the small result files. Push when a GitHub token is supplied."""
    repo = Path(repo)
    _ensure_identity(repo)
    roots = [root] if isinstance(root, (str, Path)) else list(root)
    files = []
    for item in roots:
        files.extend(files_to_publish(item))
    if not files:
        print(f"No small result files under {root}")
        return []
    relative = trackable(repo, [str(path.relative_to(repo)) for path in files])
    if not relative:
        print(f"No trackable result files under {root}")
        return []
    _git(repo, ["add", "--", *relative])
    cached = subprocess.call(["git", "diff", "--cached", "--quiet"], cwd=repo)
    if cached == 0:
        print("Result files are already committed")
        return []
    _git(repo, ["commit", "-m", message])
    print(f"Committed {len(relative)} files")
    if token:
        _git(repo, ["pull", "--ff-only"])
        _git(repo, ["push", "origin", "HEAD:main"], token)
        print("Pushed to main")
    return relative


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="9,10,11,21,22,23",
                        help="Comma-separated task ids. 9-11 are width 256. 21-23 are width 4096.")
    parser.add_argument("--out", default="results/deep_width_colab")
    parser.add_argument("--step", type=float, default=0.16)
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--report", action="store_true",
                        help="Build the figure from whatever cells are already in --out")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    args = parser.parse_args()
    os.chdir(ROOT)
    repo = ROOT
    token = (os.environ.get(args.token_env) or "").strip() or None
    out = Path(args.out)
    if not out.is_absolute():
        out = repo / out
    if not args.skip_run:
        require_cuda()
        subprocess.check_call([sys.executable, "-m", "deep_width.run", "inputs",
                               "--config", "configs/revision.yaml", "--out", str(out)])
        from deep_width.core import task_matrix
        tasks = [int(piece) for piece in args.tasks.split(",") if piece.strip()]
        failed = []
        for task in tasks:
            print(f"START task {task}", flush=True)
            status = subprocess.call(run_argv(task, out, args.step, args.level))
            cell = task_matrix()[task]["id"]
            arm = f"pcnl_h_{args.step:g}"
            _git(repo, ["pull", "--ff-only"])
            publish(repo, [out / "inputs", out / "run" / arm / cell, out / "pilot" / arm / cell],
                    f"Add Colab deep-width task {task}.", token)
            if status != 0:
                failed.append((task, status))
                print(f"TASK_FAILED {task} status {status}. The other tasks in this runtime still run.",
                      flush=True)
        if failed:
            detail = ", ".join(f"{task}={status}" for task, status in failed)
            raise SystemExit(f"Some tasks failed: {detail}")
    if args.report:
        _git(repo, ["pull", "--ff-only"])
        subprocess.check_call([sys.executable, "-m", "deep_width.report",
                               "--out", str(out), "--step", str(args.step), "--steps", "8192"])
        publish(repo, out, "Add the Colab deep-width report.", token)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            print(f"RUNNER_ERROR: {exc}", flush=True)
        raise
    except Exception:
        traceback.print_exc()
        print("RUNNER_ERROR: the traceback above is the failure", flush=True)
        raise SystemExit(1)
