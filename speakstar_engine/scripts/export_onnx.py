"""One-time explicit network operation. Runtime inference is local-only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from speakstar.model import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("models/wav2vec2"))
    parser.add_argument("--model-id", default="facebook/wav2vec2-xlsr-53-espeak-cv-ft")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import Wav2Vec2ForCTC
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading phoneme model (first run downloads weights)...", flush=True)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_id, revision=args.revision,
                                         local_files_only=args.local_files_only,
                                         attn_implementation="eager").eval()
    revision = getattr(model.config, "_commit_hash", None) or args.revision
    vocab_path = hf_hub_download(args.model_id, "vocab.json", revision=revision, local_files_only=args.local_files_only)
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    (args.output / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    class Wrapper(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, input_values):
            return self.wrapped(input_values).logits

    wrapped = Wrapper(model).eval()
    example = torch.randn(1, 16000)
    fp32, int8 = args.output / "model.fp32.onnx", args.output / "model.int8.onnx"
    print("Exporting fp32 ONNX...", flush=True)
    with torch.inference_mode():
        torch.onnx.export(wrapped, example, str(fp32), input_names=["input_values"], output_names=["logits"],
                          dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
                          opset_version=17, dynamo=False)
    # Leave convolutions floating-point: dynamic ConvInteger lacks some CPU kernels.
    print("Quantizing transformer linear layers (experimental int8)...", flush=True)
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul", "Gemm"], per_channel=True)
    checks = {}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    for precision, path in [("fp32", fp32), ("int8", int8)]:
        t = time.perf_counter()
        session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        load = time.perf_counter() - t
        errors = []
        for size in [16000, 48000]:
            rng = np.random.default_rng(42)
            x = rng.normal(size=(1, size)).astype(np.float32)
            with torch.inference_mode():
                expected = wrapped(torch.from_numpy(x)).numpy()
            t = time.perf_counter()
            actual = session.run(None, {"input_values": x})[0]
            elapsed = time.perf_counter() - t
            if actual.shape != expected.shape or not np.isfinite(actual).all():
                raise RuntimeError("ONNX output shape/values failed verification")
            error = float(np.max(np.abs(actual - expected)))
            if precision == "fp32" and not np.allclose(actual, expected, atol=0.005, rtol=0.005):
                raise RuntimeError(f"fp32 export disagrees with PyTorch: {error}")
            errors.append({"audio_seconds": size / 16000, "inference_seconds": elapsed, "max_logit_error": error})
        checks[precision] = {"size_bytes": path.stat().st_size, "load_seconds": load, "checks": errors}
        del session
    stride, receptive = 1, 1
    for kernel, step in zip(model.config.conv_kernel, model.config.conv_stride, strict=True):
        receptive += (kernel - 1) * stride
        stride *= step
    metadata = {"model_id": args.model_id, "revision": revision, "blank_id": model.config.pad_token_id,
                "stride_samples": stride, "receptive_samples": receptive,
                "sha256": {"fp32": sha256(fp32), "int8": sha256(int8)},
                "validation": "export-numerics-only; no pronunciation accuracy validation", "benchmark": checks}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
