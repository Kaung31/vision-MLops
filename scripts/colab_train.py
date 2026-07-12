#!/usr/bin/env python3
"""Self-contained Colab bootstrap — the ONE file `colab run` uploads to a fresh VM.

`colab run` uploads only this file and forwards everything after it to sys.argv; the repo,
deps, and data are NOT on the VM. So this script provisions all three, then hands off to
`python -m src.train.run`. Every result (metrics + best.pt) is logged to DagsHub MLflow, so
nothing depends on the ephemeral VM surviving — `colab run` tears it down on exit.

From the Mac (absolute script path; hours-long --timeout since colab run defaults to 30s):

  colab run --gpu T4 --timeout 36000 /ABS/PATH/scripts/colab_train.py \
      --token <DAGSHUB_TOKEN> \
      --config configs/training/base.yaml --device 0 --fraction 0.25 --run-name lc-25

The bootstrap consumes its own flags (--token, --repo, ...) and forwards every OTHER arg
verbatim to `python -m src.train.run` — no `--` separator (colab run's click parser may eat
it). The DagsHub token is an arg because colab run forwards no env; use a scoped, revocable
token (DagsHub → Settings → Tokens), never your account password.

This file has no repo imports on purpose: it must run standalone on a bare VM.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REPO_DEFAULT = "https://github.com/Kaung31/vision-MLops.git"
DAGSHUB_USER_DEFAULT = "Kaung31"
MLFLOW_URI_DEFAULT = "https://dagshub.com/Kaung31/vision-MLops.mlflow"
WORKDIR_DEFAULT = "/content/vision-MLops"
# torch/torchvision are preinstalled on Colab GPU VMs (CUDA-matched) — do NOT reinstall them.
PIP_PACKAGES = ["ultralytics", "mlflow>=2.14", "dvc[http]>=3.55", "pyyaml"]
TUNE_PACKAGES = ["ray[tune]", "optuna"]  # only for --entry src.train.tune


def _masked(cmd: list[str]) -> str:
    """Render a command for logging with any secret (arg after 'password'/'--token') hidden."""
    out, hide_next = [], False
    for a in cmd:
        out.append("********" if hide_next else a)
        hide_next = a in ("password", "--token")
    return " ".join(out)


def sh(
    cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None, check: bool = True
) -> int:
    """Run a command, RELAYING its output through Python prints.

    colab run executes this script under a Jupyter kernel: only Python-level sys.stdout is
    forwarded to the local terminal — a subprocess writing to the raw fd is invisible. So we
    pipe and re-print every line (this is also why the first dry-run looked silent).
    """
    print(f"[colab-boot] $ {_masked(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line.rstrip("\n"), flush=True)
    rc = proc.wait()
    if check and rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    return rc


def main() -> int:
    # allow_abbrev=False so a train flag like --device is never mistaken for a bootstrap flag;
    # parse_known_args returns everything else as the train passthrough (no `--` separator).
    p = argparse.ArgumentParser(description="Colab training bootstrap", allow_abbrev=False)
    p.add_argument("--repo", default=REPO_DEFAULT)
    p.add_argument("--ref", default="main")
    p.add_argument("--user", default=DAGSHUB_USER_DEFAULT)
    p.add_argument("--token", required=True, help="DagsHub access token (scoped, revocable)")
    p.add_argument("--mlflow-uri", default=MLFLOW_URI_DEFAULT)
    p.add_argument("--workdir", default=WORKDIR_DEFAULT)
    p.add_argument("--dvc-target", default="data/processed/train-v1.dvc")
    p.add_argument("--entry", default="src.train.run", help="module to exec (src.train.tune)")
    args, train_argv = p.parse_known_args(sys.argv[1:])

    # 1. code
    if os.path.isdir(os.path.join(args.workdir, ".git")):
        sh(["git", "-C", args.workdir, "fetch", "--depth", "1", "origin", args.ref])
        sh(["git", "-C", args.workdir, "checkout", "-f", args.ref])
    else:
        sh(["git", "clone", "--depth", "1", "--branch", args.ref, args.repo, args.workdir])

    # 2. deps (torch stays as Colab shipped it)
    packages = PIP_PACKAGES + (TUNE_PACKAGES if args.entry.endswith(".tune") else [])
    sh([sys.executable, "-m", "pip", "install", "-q", *packages])

    # 3. data — dvc pull the pinned train-v1 from DagsHub (creds are --local, so set them here)
    for k, v in (("auth", "basic"), ("user", args.user), ("password", args.token)):
        sh(["dvc", "remote", "modify", "--local", "origin", k, v], cwd=args.workdir)
    sh(["dvc", "pull", "-r", "origin", "-j", "16", args.dvc_target], cwd=args.workdir)

    # 4. tracking creds -> env (colab run forwards none), then hand off to the real script
    env = {
        **os.environ,
        "MLFLOW_TRACKING_URI": args.mlflow_uri,
        "MLFLOW_TRACKING_USERNAME": args.user,
        "MLFLOW_TRACKING_PASSWORD": args.token,
        "PYTHONUNBUFFERED": "1",  # epoch lines must arrive live through the relay
    }
    print(f"[colab-boot] launching {args.entry}: {' '.join(train_argv)}", flush=True)
    return sh(
        [sys.executable, "-m", args.entry, *train_argv],
        cwd=args.workdir,
        env=env,
        check=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
