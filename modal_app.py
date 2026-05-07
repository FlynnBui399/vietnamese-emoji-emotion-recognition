"""Modal entrypoint for ViGoEmotions baseline training on an A100.

Usage (local CLI, after data has been uploaded once via scripts/upload_data.py):

    modal run modal_app.py::train \\
        --config-path configs/visobert_baseline.yaml \\
        --run-name visobert-baseline-v1

    Override RNG seed (otherwise YAML seed is used):

        modal run modal_app.py::train ... --seed 42

Optional flags:
    --use-wandb         enable W&B logging. Set WANDB_API_KEY locally and it
                        will be forwarded to the Modal container.

Volumes:
    vigoemotions-data   /data                       (read-only intent, populated by scripts/upload_data.py)
    vigoemotions-runs   /runs                       (TensorBoard logs + checkpoints + metrics.json)
    hf-cache            /root/.cache/huggingface    (model weights cache between runs)
"""
from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "vigoemotions-baseline"
DATA_VOLUME_NAME = "vigoemotions-data"
RUNS_VOLUME_NAME = "vigoemotions-runs"
HF_CACHE_VOLUME_NAME = "hf-cache"

ROOT = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11", force_build=True)
    .apt_install("git")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"), force_build=False)
    .add_local_dir(str(ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_dir(str(ROOT / "configs"), remote_path="/root/configs", copy=True) 
)

# Optionally ship preprocessing resources (patterns.json, emojis.json,
# teencode4.txt). Without them clean_text still runs but the pattern /
# teencode steps become no-ops.
_DOCS_LOCAL = ROOT / "docs"
if _DOCS_LOCAL.exists():
    image = image.add_local_dir(str(_DOCS_LOCAL), remote_path="/root/docs", copy=True)

image = image.workdir("/root")

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
runs_vol = modal.Volume.from_name(RUNS_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name(HF_CACHE_VOLUME_NAME, create_if_missing=True)


_wandb_secret = modal.Secret.from_dict(
    {
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "WANDB_PROJECT": os.environ.get("WANDB_PROJECT", "vigoemotions"),
        "HF_TOKEN": os.environ.get("HF_TOKEN", "")
    }
)


@app.function(
    gpu="A100",
    volumes={
        "/data": data_vol,
        "/runs": runs_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[_wandb_secret],
    timeout=60 * 60 * 4,
)
def train_remote(
    config_path: str,
    run_name: str,
    use_wandb: bool = False,
    seed: int | None = None,
) -> dict:
    """Remote training entrypoint. Returns the summary dict written to metrics.json."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from src.train import run_training

    summary = run_training(
        config_path=config_path,
        run_name=run_name,
        use_wandb=use_wandb,
        seed_override=seed,
    )
    runs_vol.commit()
    return summary


@app.local_entrypoint()
def train(
    config_path: str = "configs/visobert_baseline.yaml",
    run_name: str = "visobert-baseline-v1",
    use_wandb: bool = False,
    seed: int | None = None,
) -> None:
    """Local CLI -> remote A100 training run."""
    remote_config = config_path if config_path.startswith("/") else f"/root/{config_path}"

    print(f"[modal_app] Submitting training run '{run_name}' (use_wandb={use_wandb}, seed={seed})")
    print(f"[modal_app] Config (remote): {remote_config}")

    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        print("[modal_app] WARN: --use-wandb passed but WANDB_API_KEY not in local env; "
              "W&B logging will be skipped inside the container.")

    summary = train_remote.remote(
        config_path=remote_config,
        run_name=run_name,
        use_wandb=use_wandb,
        seed=seed,
    )

    print("[modal_app] Run finished.")
    print(
        f"  best epoch        : {summary.get('best_epoch')}\n"
        f"  best val macroF1  : {summary.get('best_val_macro_f1'):.4f}\n"
        f"  test macroF1      : {summary['test']['macro_f1']:.4f}\n"
        f"  test weightedF1   : {summary['test']['weighted_f1']:.4f}\n"
        f"  test microF1      : {summary['test']['micro_f1']:.4f}\n"
        f"  test hamming      : {summary['test']['hamming']:.4f}\n"
        f"  artifacts         : Volume '{RUNS_VOLUME_NAME}' under /runs/{run_name}\n"
    )


@app.function(volumes={"/runs": runs_vol}, timeout=300)
def list_runs() -> list[str]:
    """List run directories on the runs volume."""
    return sorted(os.listdir("/runs")) if os.path.isdir("/runs") else []


@app.function(
    volumes={"/data": data_vol},
    timeout=300,
)
def debug_preprocess() -> None:
    import pandas as pd
    from src.config import TrainConfig
    from src.data import load_split
    from src.preprocess import build_preprocessor

    cfg = TrainConfig.from_yaml("/root/configs/visobert_baseline.yaml")
    clean_fn = build_preprocessor(
        apply_clean_text=cfg.apply_clean_text,
        apply_pyvi=cfg.apply_pyvi,
        docs_dir=cfg.docs_dir,
    )
    df = load_split("/data/train.csv", clean_text_fn=clean_fn)

    # In 3 samples để xác nhận preprocess đang chạy
    for i in range(3):
        print(f"ROW {i}: {df['text'].iloc[i][:150]}")


@app.local_entrypoint()
def runs() -> None:
    """`modal run modal_app.py::runs` to list completed runs on the volume."""
    items = list_runs.remote()
    if not items:
        print("(no runs yet)")
        return
    for item in items:
        print(item)


@app.local_entrypoint()
def debug() -> None:
    """`modal run modal_app.py::debug` to verify preprocessing on the remote volume."""
    print("[modal_app] Running debug_preprocess...")
    debug_preprocess.remote()
    print("[modal_app] Done.")

