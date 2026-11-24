"""Colab publishes the small result files and keeps each task independent."""
import subprocess

from colab.run_deep_width import files_to_publish, git_env, publish, run_argv
from deep_width.report import one_device


def test_pull_stays_anonymous_and_push_carries_the_token():
    pull = git_env()
    assert pull["GIT_TERMINAL_PROMPT"] == "0"
    assert pull["GIT_CONFIG_VALUE_0"] == ""
    assert "x-access-token" not in "".join(pull.values())
    push = "\n".join(git_env("secret-token").values())
    assert "url.https://x-access-token:secret-token@github.com/.insteadOf" in push
    assert "https://github.com/" in push


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
