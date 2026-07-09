"""Modal app -- run the J-lens bridge experiment (jlens_bridge.py) on a GPU.

  modal run modal_jlens.py --smoke     # shape/path validation (2 contexts, 4 tokens, 2 records)
  modal run --detach modal_jlens.py    # full run: 32 contexts x 11 tokens, 40 records x 7 conditions

Outputs land on the 'jlens-bridge' volume:
  modal volume get jlens-bridge /jlens_bridge_out ./jlens_bridge_out
"""
import modal

app = modal.App("jlens-bridge")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.8.0", "transformers==5.6.2", "numpy", "jinja2>=3.1", "accelerate", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
    .add_local_file("jlens_bridge.py", "/root/jlens/jlens_bridge.py")
    .add_local_file("bridge_inputs.json", "/root/jlens/bridge_inputs.json")
)

vol = modal.Volume.from_name("jlens-bridge", create_if_missing=True)
hf_cache = modal.Volume.from_name("jlens-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
)
def run(smoke: bool = False):
    import os
    import subprocess

    env = dict(
        os.environ,
        BRIDGE_INPUTS="/root/jlens/bridge_inputs.json",
        OUT_DIR="/vol/jlens_bridge_out_v2" + ("_smoke" if smoke else ""),
        SMOKE="1" if smoke else "0",
        PHASE="all",
        DEVICE="cuda",
        DTYPE="float32",
        JLENS_BATCH="8",
    )
    subprocess.run(["python", "/root/jlens/jlens_bridge.py"], env=env, check=True)
    vol.commit()
    out = env["OUT_DIR"] + "/bridge_results.json"
    if os.path.exists(out):
        print("==== bridge_results.json ====")
        print(open(out).read())


@app.local_entrypoint()
def main(smoke: bool = False):
    run.remote(smoke=smoke)
