# Stable Audio HF Local

This tool is a browser-only Stable Audio 3 workbench. It loads ONNX files from Hugging Face and runs them with ONNX Runtime Web using WebGPU first, then WASM fallback.

## Current model status

- This folder does not store ONNX files in git.
- The cloned `stable-audio-3` repository is kept under `vendor/` and ignored by git.
- The official `stable-audio-3` source repository does not currently expose a generic ONNX export script.
- The browser build uses Stability AI's official pre-converted ONNX artifacts from `stabilityai/stable-audio-3-optimized`, or a Hugging Face repo you mirror with the same file paths.

Required browser model paths:

- `onnx/t5gemma/encoder.onnx`
- `onnx/sa3-sm-music/dit.onnx`
- `onnx/sa3-sm-sfx/dit.onnx`
- `onnx/same-s/dec_dynamic_bf16.onnx`
- `tensorRT/sm_90/t5gemma/tokenizer.json`

## Build and upload model pack

Install the small Python dependency once:

```powershell
python -m pip install -U huggingface_hub
```

Clone the official repo, download the official ONNX pack, write the web manifest, and upload it to your own Hugging Face model repo:

```powershell
$env:HF_TOKEN = "hf_xxx"
python tools/stable-audio-hf-local/scripts/build_hf_onnx_pack.py all --target-repo YOUR_NAME/stable-audio-3-web-onnx
```

If you are already logged in with `hf auth login` or `huggingface-cli login`, omit `$env:HF_TOKEN`.

After upload, open the web page and set `HF model repository` to your mirrored repo id. The repo must preserve the same ONNX paths.

## Local-only preparation

To only clone and download files without upload:

```powershell
python tools/stable-audio-hf-local/scripts/build_hf_onnx_pack.py prepare
```

The downloaded files go to `tools/stable-audio-hf-local/artifacts/onnx-pack/` and are ignored by git because they are large.