#!/usr/bin/env python
"""
Merge the three machine-readable outputs of the model preflight into one file:
  audit/preflight/model_cpu_facts.json            (job 245675, M1-M5, CPU)
  audit/preflight/model_gpu_facts.json            (job 245676, M6-M7, one A100)
  audit/preflight/model_gpu_followup_facts.json   (follow-up job, LoRA+checkpointing, combined
                                                   step, full fine-tuning probe)
into audit/preflight/model_facts.json, with a short "summary" block on top that the report
(audit/2026-10-02_preflight_model.md) quotes. Light work: small JSON files only.
"""
import json
from pathlib import Path

D = Path("/projects/bdbk/liv/repos/taxa_maze/audit/preflight")
cpu = json.loads((D / "model_cpu_facts.json").read_text())
gpu = json.loads((D / "model_gpu_facts.json").read_text())
fu_path = D / "model_gpu_followup_facts.json"
fu = json.loads(fu_path.read_text()) if fu_path.exists() else None

m6 = gpu["M6"]
cfg_head = "train_py_HEAD_flags(flash=1,mem_eff=0,math=0,matmul=medium)"
cfg_def = "default_flags(all_backends_enabled,matmul=highest)"


def ok_map(cfg):
    return {k: (True if r["ok"] else f"FAIL: {r['error_type']}: {r['error']}") for k, r in m6[cfg]["cases"].items()}


def thr(block):
    return {k: {"ok": r.get("ok"), "items_per_s": r.get("items_per_s"), "step_s": r.get("step_s"),
                "peak_allocated_GiB": r.get("peak_allocated_GiB"), "error": r.get("error")}
            for k, r in block.items()}


summary = {
    "M1_weight_sha256": cpu["M1"]["files"]["open_clip_pytorch_model.bin"]["sha256_computed"],
    "M1_equal_SHA256SUMS": cpu["M1"]["files"]["open_clip_pytorch_model.bin"]["equal_to_SHA256SUMS"],
    "M1_equal_manifest": cpu["M1"]["weight_equal_manifest"],
    "M2_visual_output_dim": cpu["M2"]["visual_output_dim"],
    "M2_n_params_total": cpu["M2"]["n_params_total"],
    "M2_n_params_visual_text_logit": [cpu["M2"]["n_params_visual"], cpu["M2"]["n_params_text"],
                                      cpu["M2"]["n_params_logit"]],
    "M3_attn_class_visual": cpu["M3"]["visual"]["block0"]["attn_class"],
    "M3_attn_class_text": cpu["M3"]["text"]["block0"]["attn_class"],
    "M3_in_proj_weight_visual_text": [cpu["M3"]["visual"]["block0"]["in_proj_weight_shape"],
                                      cpu["M3"]["text"]["block0"]["in_proj_weight_shape"]],
    "M3_row_order_qkv_max_abs_err": {t: v["max_abs_diff_vs_module"]["eval: q,k,v (assumed)"]
                                     for t, v in cpu["M3"]["row_order_check"].items()},
    "M3_logit_scale_raw_exp": [cpu["M3"]["logit_scale_raw"], cpu["M3"]["logit_scale_exp"]],
    "M3_logit_bias_is_None": cpu["M3"]["logit_bias_is_None"],
    "M4_trainable": cpu["M4"]["n_trainable"],
    "M4_total_frozen_plus_adapters": cpu["M4"]["n_total_frozen_plus_adapters"],
    "M4_fraction": cpu["M4"]["fraction_trainable_of_total_with_adapters"],
    "M4_arm_ii_fraction": cpu["M4"]["arm_ii_fraction_of_total_with_adapters"],
    "M5_repr_equal_after_masking_address": cpu["M5"]["repr_equal_after_masking_function_address"],
    "M6_train_py_HEAD_flags": ok_map(cfg_head),
    "M6_default_flags": ok_map(cfg_def),
    "M6_default_flags_attention_ops": m6["default_flags_attention_ops"],
    "M7_node": gpu["host"],
    "M7_gpu": gpu["gpu"],
    "M7a_inference_image": thr(gpu["M7"]["a_inference_image"]),
    "M7b_image_train_proxy": thr(gpu["M7"]["b_image_train_proxy"]),
    "M7b_extra_image_merged_lora_job1": thr(gpu["M7"]["b_extra_image_train_merged_lora"]),
    "M7c_text_throughput_bs2048": thr(gpu["M7"]["c_text"]["throughput_bs2048"]),
    "M7c_truncation_77_vs_32": gpu["M7"]["c_text"]["truncation_77_vs_32"],
    "followup_probes": thr(fu["probes"]) if fu else None,
}

# ------------------------------------------------------------------ section 6 feasibility arithmetic
# Inputs: spec numbers (train split size, global batch, epoch counts) and measured step times.
# Everything here is compute-only: it assumes the data loader never stalls and ignores DDP
# communication, checkpoint writes, smoke runs, queueing and restarts.
N_TRAIN = 5_908_775          # spec line 41 (not re-verified here; data preflight owns it)
N_VAL = 310_899              # spec line 42, before deduplication
B_GLOBAL = 8192              # spec lines 144, 160
GPUS = 4                     # spec line 159
EPOCHS = {"lr_sweep_2_runs_x_3": 6, "finish_chosen_lr_run_to_10": 7, "arms_d_a_b_x_10": 30}
REFRESHES = 2 * 10           # arms (b) and (d), one exact bank recomputation after each epoch (spec line 91)
steps_per_epoch = N_TRAIN / B_GLOBAL
fp = fu["probes"] if fu else {}
step_s = {
    "image_only_lora_ckpt_2048": fp.get("lora_image/bs2048/ckpt/uncached", {}).get("step_s"),
    "img2048_ckpt+txt2048_L32": fp.get("lora_combined/img2048_ckpt+txt2048_L32_no_ckpt", {}).get("step_s"),
    "img2048_ckpt+txt2048_L77": fp.get("lora_combined/img2048_ckpt+txt2048_L77_no_ckpt", {}).get("step_s"),
}
infer_bf16 = gpu["M7"]["a_inference_image"]["encode_image/bf16_autocast/bs1024"]["items_per_s"]
infer_fp16 = gpu["M7"]["a_inference_image"]["encode_image/fp16_autocast/bs1024"]["items_per_s"]
refresh_wall_s = N_TRAIN / (GPUS * infer_bf16)
val_eval_wall_s = N_VAL / (GPUS * infer_fp16)
n_epochs = sum(EPOCHS.values())
feas = {"inputs": {"N_train": N_TRAIN, "N_val": N_VAL, "B_global": B_GLOBAL, "gpus": GPUS,
                   "epochs": EPOCHS, "n_epochs_total": n_epochs, "bank_refreshes": REFRESHES,
                   "inference_img_per_s_bf16": infer_bf16, "inference_img_per_s_fp16": infer_fp16},
        "steps_per_epoch": steps_per_epoch,
        "bank_refresh_gpu_hours_each": N_TRAIN / infer_bf16 / 3600,
        "bank_refresh_wall_min_each_on_4_gpus": refresh_wall_s / 60,
        "tol_val_eval_image_wall_min_each_on_4_gpus": val_eval_wall_s / 60,
        "scenarios": {}}
for name, s in step_s.items():
    if s is None:
        continue
    epoch_min = steps_per_epoch * s / 60
    train_h = n_epochs * epoch_min / 60
    total_h = train_h + REFRESHES * refresh_wall_s / 3600 + n_epochs * val_eval_wall_s / 3600
    feas["scenarios"][name] = {"step_s": s, "epoch_min_on_4_gpus": epoch_min,
                               "training_h_all_epochs": train_h,
                               "total_h_with_refresh_and_val_image_eval": total_h,
                               "total_h_if_2_gpus_same_per_gpu_rate": 2 * total_h}
summary["feasibility_section6"] = feas

merged = {
    "report": "/projects/bdbk/liv/repos/taxa_maze/audit/2026-10-02_preflight_model.md",
    "sources": {"cpu": str(D / "model_cpu_facts.json"), "gpu": str(D / "model_gpu_facts.json"),
                "gpu_followup": str(fu_path) if fu else None,
                "logs": sorted(str(p) for p in D.glob("model_*.log"))},
    "summary": summary,
    "cpu_job": cpu,
    "gpu_job": gpu,
    "gpu_followup_job": fu,
}
(D / "model_facts.json").write_text(json.dumps(merged, indent=2, default=str))
print("wrote", D / "model_facts.json")
