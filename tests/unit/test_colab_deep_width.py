"""Colab publishes the small result files and keeps each task independent."""
import subprocess

from colab.run_deep_width import _git, files_to_publish, publish, run_argv
from deep_width.report import one_device


def test_a_rejected_token_does_not_ask_for_a_username(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-b", "main"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "nalint2@illinois.edu"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "NalinTiwaryUIUC"], cwd=repo)
    (repo / "note.txt").write_text("token check\n")
    subprocess.check_call(["git", "add", "note.txt"], cwd=repo)
    subprocess.check_call(["git", "commit", "-m", "token check"], cwd=repo)
    subprocess.check_call(["git", "remote", "add", "origin",
                           "https://github.com/NalinTiwary/Width-Scaling-Posterior-Geometry.git"],
                          cwd=repo)
    try:
        _git(repo, ["push", "origin", "HEAD:main"], token="dummy-token")
    except SystemExit as exc:
        message = str(exc)
    else:
        raise AssertionError("a dummy token was accepted")
    assert "GIT_FAILED" in message
    assert "dummy-token" not in message
    assert "could not read Username" not in message


def test_colab_command_records_the_assigned_gpu():
    command = run_argv(21, "results/deep_width_colab", 0.16, 1)
    assert "--allow-any-device" in command
    assert "--device" in command and "cuda" in command
    assert command[command.index("--task") + 1] == "21"


def test_one_gpu_name_is_required():
    assert one_device(["Tesla T4", "Tesla T4"]) == "Tesla T4"
    try:
        one_device(["Tesla T4", "NVIDIA A10"])
    except ValueError as exc:
        assert "one GPU" in str(exc)
    else:
        raise AssertionError("mixed GPUs were accepted")


def test_publish_commits_tables_and_leaves_checkpoints(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "nalint2@illinois.edu"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "NalinTiwaryUIUC"], cwd=repo)
    out = repo / "results" / "deep_width_colab" / "run" / "pcnl_h_0.16" / "deep_m4096_r0"
    out.mkdir(parents=True)
    (out / "metrics_8192.csv").write_text("probe,value\nV,1\n")
    (out / "run_8192.json").write_text("{}")
    (out / "checkpoint.pt").write_bytes(b"weights")
    (out / "chain.h5").write_bytes(b"trace")
    (out / "huge.csv").write_bytes(b"x" * (50 * 1024 * 1024 + 1))
    committed = publish(repo, repo / "results" / "deep_width_colab", "Add a Colab task.")
    assert committed == [
        "results/deep_width_colab/run/pcnl_h_0.16/deep_m4096_r0/metrics_8192.csv",
        "results/deep_width_colab/run/pcnl_h_0.16/deep_m4096_r0/run_8192.json",
    ]
    tracked = subprocess.check_output(["git", "ls-files"], cwd=repo, text=True)
    assert "checkpoint.pt" not in tracked
    assert "chain.h5" not in tracked
    assert "huge.csv" not in tracked
    assert files_to_publish(repo / "results" / "deep_width_colab")
    assert publish(repo, repo / "results" / "deep_width_colab", "Add a Colab task.") == []
