"""
Multi-model sweep for is_checkable evaluation.

For each model: launch SGLang server, run few-shot configs, collect results, kill server.
Aggregates cross-model comparison at the end.
"""

import subprocess
import time
import json
import sys
import os
import signal
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

# --- Resolve model snapshot paths from HF cache ---
HF_CACHE_HUB = "/data/nfs/.cache/hub"


WRITABLE_CACHE_HUB = "/data/nfs/ruggsea/huggingface/hub"


def resolve_snapshot(model_id):
    """Resolve HF model ID to local snapshot path.
    Returns (model_path, tokenizer_path) - may differ if tokenizer
    was downloaded separately to writable cache."""
    model_key = model_id.replace('/', '--')

    # Check shared cache first (has model weights)
    shared_dir = f"{HF_CACHE_HUB}/models--{model_key}"
    shared_ref_file = f"{shared_dir}/refs/main"
    model_path = None
    if os.path.exists(shared_ref_file):
        with open(shared_ref_file) as f:
            ref = f.read().strip()
        snapshot = f"{shared_dir}/snapshots/{ref}"
        if os.path.isdir(snapshot):
            model_path = snapshot

    # Check writable cache (may have full model OR just tokenizer files)
    writable_dir = f"{WRITABLE_CACHE_HUB}/models--{model_key}"
    writable_ref_file = f"{writable_dir}/refs/main"
    tokenizer_path = None
    if os.path.exists(writable_ref_file):
        with open(writable_ref_file) as f:
            ref = f.read().strip()
        snapshot = f"{writable_dir}/snapshots/{ref}"
        if os.path.isdir(snapshot):
            files = os.listdir(snapshot)
            has_weights = any(f.endswith('.safetensors') for f in files)
            has_tokenizer = any('tokenizer' in f for f in files)
            # If writable cache has full model weights, use it as model_path
            if has_weights and model_path is None:
                model_path = snapshot
            # Track tokenizer location
            if has_tokenizer:
                tokenizer_path = snapshot

    # If model_path has tokenizer, no need for separate tokenizer_path
    if model_path:
        try:
            has_tok = any('tokenizer' in f for f in os.listdir(model_path))
            if has_tok:
                tokenizer_path = None  # not needed
        except OSError:
            pass

    return model_path, tokenizer_path


# --- Model configs ---
# Round 2: Newer, bigger models
MODELS = [
    {
        "id": "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "tp": 2,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "is_thinking": False,
    },
    {
        "id": "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "tp": 4,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "is_thinking": False,
    },
]

# Few-shot configs to test per model (trimmed from full round 2)
CONFIGS = [
    # (name, n_shots, strategy, use_strict)
    ("zero_shot", 0, None, False),
    ("zero_shot_strict", 0, None, True),
    ("4shot_diverse_strict", 4, "diverse", True),
    ("8shot_diverse_strict", 8, "diverse", True),
    ("10shot_diverse_strict", 10, "diverse", True),
    ("12shot_diverse_strict", 12, "diverse", True),
    ("8shot_diverse", 8, "diverse", False),
    ("10shot_diverse", 10, "diverse", False),
]

OUTPUT_DIR = "results/multi_model_sweep"
ANNOTATED_CSV = "data/samples/labeling_sample_200_annotated.csv"


def notify(title, message):
    """Send email notification via ntfy.sh."""
    try:
        subprocess.run([
            "curl", "-s",
            "-H", "Email: rmlazza@gmail.com",
            "-H", f"Title: {title}",
            "-d", message,
            "ntfy.sh/infini-news-build-done"
        ], capture_output=True, timeout=10)
    except Exception:
        pass


def kill_sglang():
    """Kill any running SGLang server processes owned by current user."""
    try:
        result = subprocess.run(
            ["pgrep", "-u", os.environ["USER"], "-f", "sglang.launch_server"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid.strip():
                try:
                    os.kill(int(pid.strip()), signal.SIGTERM)
                except (ProcessLookupError, ValueError):
                    pass
        # Also kill scheduler processes
        result = subprocess.run(
            ["pgrep", "-u", os.environ["USER"], "-f", "sglang::scheduler"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid.strip():
                try:
                    os.kill(int(pid.strip()), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
        time.sleep(5)
    except Exception as e:
        print(f"Warning: Error killing SGLang: {e}")


def launch_sglang(model_id, tp, port=7501, local_path=None, tokenizer_path=None):
    """Launch SGLang server and return (process, actual_port).
    Uses local_path (snapshot dir) if provided, else model_id from HF."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    # Use a writable cache for any tokenizer/config downloads
    env["HF_HOME"] = "/data/nfs/ruggsea/huggingface"
    env["HF_HUB_CACHE"] = "/data/nfs/ruggsea/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = "/data/nfs/ruggsea/huggingface"

    model_arg = local_path if local_path else model_id
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model", model_arg,
        "--served-model-name", model_id,
        "--host", "0.0.0.0",
        "--trust-remote-code",
        "--tp", str(tp),
        "--port", str(port),
    ]
    if tokenizer_path:
        cmd.extend(["--tokenizer-path", tokenizer_path])

    print(f"  Launching: {' '.join(cmd)}")
    # Redirect stdout to log file, don't capture (avoids blocking)
    log_file = f"/tmp/sglang_server_{model_id.replace('/', '_')}.log"
    log_fh = open(log_file, 'w')
    proc = subprocess.Popen(
        cmd, env=env,
        stdout=log_fh, stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready via health endpoint polling
    actual_port = port
    start = time.time()
    timeout = 600  # 10 minutes max
    ready = False

    while time.time() - start < timeout:
        # Try health check
        try:
            r = requests.get(f"http://localhost:{actual_port}/health", timeout=3)
            if r.status_code == 200:
                ready = True
                elapsed = time.time() - start
                print(f"  Server healthy on port {actual_port} ({elapsed:.0f}s)")
                break
        except Exception:
            pass

        # Check if proc died
        if proc.poll() is not None:
            log_fh.close()
            with open(log_file) as f:
                content = f.read()
            print(f"  Server died! Exit code: {proc.returncode}")
            print(f"  Last output: {content[-2000:]}")
            return None, None

        time.sleep(5)

    log_fh.close()
    if not ready:
        with open(log_file) as f:
            content = f.read()
        print(f"  Server failed to start within {timeout}s")
        print(f"  Last output: {content[-1000:]}")
        proc.terminate()
        return None, None

    return proc, actual_port


def run_fewshot_sweep(model_id, port, extra_body, output_dir, max_workers=20):
    """Run the fewshot_is_checkable.py sweep for a model."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["HF_HOME"] = "/data/nfs/ruggsea/huggingface"
    env["HF_HUB_CACHE"] = "/data/nfs/ruggsea/huggingface/hub"
    env["TRANSFORMERS_CACHE"] = "/data/nfs/ruggsea/huggingface"
    env["WANDB_MODE"] = "disabled"

    # Run round 2 sweep (the one with diverse+strict configs)
    cmd = [
        sys.executable, "scripts/fewshot_is_checkable.py",
        "--model", model_id,
        "--port", str(port),
        "--max_workers", str(max_workers),
        "--output_dir", output_dir,
        "--round", "2",
        "--annotated_csv", ANNOTATED_CSV,
    ]

    print(f"  Running sweep: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, env=env,
        capture_output=True, text=True,
        timeout=1800,  # 30 min max
    )

    print(result.stdout[-3000:] if result.stdout else "No stdout")
    if result.stderr:
        print(f"  STDERR (last 1000): {result.stderr[-1000:]}")

    return result.returncode == 0


def aggregate_results(base_dir):
    """Aggregate all model comparison CSVs into a single cross-model table."""
    all_dfs = []
    for model_dir in Path(base_dir).iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for csv_file in model_dir.glob("comparison_*.csv"):
            df = pd.read_csv(csv_file)
            df['model'] = model_name
            all_dfs.append(df)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined['avg_kappa'] = (combined['kappa_vs_ruggero'] + combined['kappa_vs_ilaria']) / 2
    return combined.sort_values('avg_kappa', ascending=False)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(OUTPUT_DIR)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Previous results from round 1 sweep + baseline
    previous_results = [
        {"model": "Qwen--Qwen3-32B", "best_config": "10shot_diverse_strict",
         "kappa_vs_ruggero": 0.6824, "kappa_vs_ilaria": 0.6484, "avg_kappa": 0.6654, "source": "round1"},
        {"model": "Qwen--Qwen3-14B", "best_config": "10shot_diverse_strict",
         "kappa_vs_ruggero": 0.59, "kappa_vs_ilaria": 0.6544, "avg_kappa": 0.6222, "source": "round1"},
        {"model": "NousResearch--Hermes-4.3-36B", "best_config": "8shot_diverse_strict",
         "kappa_vs_ruggero": 0.5666, "kappa_vs_ilaria": 0.5913, "avg_kappa": 0.5790, "source": "round1"},
        {"model": "Qwen--Qwen2.5-14B-Instruct", "best_config": "zero_shot_strict",
         "kappa_vs_ruggero": 0.5161, "kappa_vs_ilaria": 0.5588, "avg_kappa": 0.5375, "source": "round1"},
        {"model": "google--gemma-3-27b-it", "best_config": "4shot_diverse_strict",
         "kappa_vs_ruggero": 0.4082, "kappa_vs_ilaria": 0.4041, "avg_kappa": 0.4062, "source": "round1"},
    ]

    model_results = list(previous_results)
    failed_models = []
    total = len(MODELS)

    print(f"\n{'='*80}")
    print(f"MULTI-MODEL SWEEP - {timestamp}")
    print(f"Testing {total} NEW models + {len(previous_results)} previous results")
    print(f"{'='*80}\n")

    notify("Multi-Model Sweep Round 2",
           f"Testing {total} newer models for is_checkable task on GPUs 0-3.\n"
           f"Previous best: Qwen3-32B avg_kappa=0.665")

    for i, model_cfg in enumerate(MODELS):
        model_id = model_cfg["id"]
        model_short = model_id.replace("/", "--")
        tp = model_cfg["tp"]
        extra_body = model_cfg["extra_body"]

        # Resolve local snapshot path to avoid HF cache permission issues
        local_path, tokenizer_path = resolve_snapshot(model_id)

        print(f"\n{'='*80}")
        print(f"[{i+1}/{total}] {model_id} (tp={tp})")
        if local_path:
            print(f"  Model path: {local_path}")
        else:
            print(f"  WARNING: No local snapshot found, skipping")
            failed_models.append(model_id)
            notify(f"Model Skipped: {model_id}",
                   f"[{i+1}/{total}] No cached weights for {model_id}")
            continue
        if tokenizer_path:
            print(f"  Tokenizer path: {tokenizer_path}")
        print(f"{'='*80}")

        # Kill any existing server
        print("  Killing existing SGLang processes...")
        kill_sglang()
        time.sleep(3)

        # Launch server with local path
        print(f"  Launching SGLang server for {model_id}...")
        proc, port = launch_sglang(model_id, tp, local_path=local_path,
                                   tokenizer_path=tokenizer_path)

        if proc is None:
            print(f"  FAILED to launch {model_id}, skipping")
            failed_models.append(model_id)
            notify(f"Model Failed: {model_id}",
                   f"[{i+1}/{total}] Failed to launch SGLang for {model_id}")
            continue

        # Run sweep
        model_output_dir = str(base_dir / model_short)
        Path(model_output_dir).mkdir(parents=True, exist_ok=True)

        try:
            success = run_fewshot_sweep(model_id, port, extra_body, model_output_dir)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT running sweep for {model_id}")
            success = False
        except Exception as e:
            print(f"  ERROR: {e}")
            success = False

        # Kill server
        print(f"  Stopping SGLang server...")
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        kill_sglang()

        if success:
            # Read results
            comparison_files = list(Path(model_output_dir).glob("comparison_*.csv"))
            if comparison_files:
                df = pd.read_csv(comparison_files[-1])  # latest
                df['avg_kappa'] = (df['kappa_vs_ruggero'] + df['kappa_vs_ilaria']) / 2
                best_row = df.loc[df['avg_kappa'].idxmax()]
                result = {
                    "model": model_short,
                    "best_config": best_row['config'],
                    "kappa_vs_ruggero": best_row['kappa_vs_ruggero'],
                    "kappa_vs_ilaria": best_row['kappa_vs_ilaria'],
                    "avg_kappa": best_row['avg_kappa'],
                    "source": "sweep",
                }
                model_results.append(result)
                print(f"\n  Best for {model_id}: {best_row['config']} "
                      f"(avg_kappa={best_row['avg_kappa']:.3f})")
                notify(f"Model Done: {model_id}",
                       f"[{i+1}/{total}] {model_id}\n"
                       f"Best: {best_row['config']} avg_kappa={best_row['avg_kappa']:.3f}\n"
                       f"vs Ruggero: {best_row['kappa_vs_ruggero']:.3f}\n"
                       f"vs Ilaria: {best_row['kappa_vs_ilaria']:.3f}")
            else:
                failed_models.append(model_id)
                notify(f"Model Failed: {model_id}",
                       f"[{i+1}/{total}] No results for {model_id}")
        else:
            failed_models.append(model_id)
            notify(f"Model Failed: {model_id}",
                   f"[{i+1}/{total}] Sweep failed for {model_id}")

    # Final aggregation
    print(f"\n{'='*80}")
    print("FINAL CROSS-MODEL COMPARISON")
    print(f"{'='*80}\n")

    results_df = pd.DataFrame(model_results)
    results_df = results_df.sort_values('avg_kappa', ascending=False)
    print(results_df.to_string(index=False))

    results_file = base_dir / f"cross_model_comparison_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nSaved to {results_file}")

    if failed_models:
        print(f"\nFailed models: {', '.join(failed_models)}")

    # Final notification
    best = results_df.iloc[0]
    summary = (
        f"MULTI-MODEL SWEEP COMPLETE\n\n"
        f"Best overall: {best['model']}\n"
        f"  Config: {best['best_config']}\n"
        f"  avg_kappa: {best['avg_kappa']:.3f}\n"
        f"  vs Ruggero: {best['kappa_vs_ruggero']:.3f}\n"
        f"  vs Ilaria: {best['kappa_vs_ilaria']:.3f}\n\n"
        f"All results:\n"
    )
    for _, row in results_df.iterrows():
        summary += f"  {row['model']}: {row['avg_kappa']:.3f} ({row['best_config']})\n"
    if failed_models:
        summary += f"\nFailed: {', '.join(failed_models)}"

    notify("Multi-Model Sweep DONE", summary)
    print(f"\nDone! Best model: {best['model']} with avg_kappa={best['avg_kappa']:.3f}")


if __name__ == "__main__":
    main()
