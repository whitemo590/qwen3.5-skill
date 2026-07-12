#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  convert_qwen3_5_0.8B_weights.sh hf-to-dcp <hf_dir> <dcp_dir>
  convert_qwen3_5_0.8B_weights.sh dcp-to-hf <dcp_release_dir> <origin_hf_dir> <save_hf_dir>
EOF
}

if ! command -v mm-convert >/dev/null 2>&1; then
    echo "[ERROR] mm-convert is not available. Install MindSpeed-MM first." >&2
    exit 1
fi

mode="${1:-}"
case "$mode" in
    hf-to-dcp)
        [[ $# -eq 3 ]] || { usage; exit 2; }
        hf_dir="$2"
        dcp_dir="$3"
        [[ -f "$hf_dir/config.json" ]] || { echo "[ERROR] HF config not found: $hf_dir/config.json" >&2; exit 1; }
        mkdir -p "$dcp_dir"
        mm-convert Qwen35Converter hf_to_dcp \
            --hf_dir "$hf_dir" \
            --dcp_dir "$dcp_dir" \
            --tie_weight_mapping '{"lm_head.weight":"model.language_model.embed_tokens.weight"}'
        [[ -d "$dcp_dir/release" ]] || { echo "[ERROR] DCP release directory missing" >&2; exit 1; }
        echo "[PASS] HF -> DCP: $dcp_dir"
        ;;
    dcp-to-hf)
        [[ $# -eq 4 ]] || { usage; exit 2; }
        dcp_release_dir="$2"
        origin_hf_dir="$3"
        save_hf_dir="$4"
        [[ -d "$dcp_release_dir" ]] || { echo "[ERROR] DCP directory not found: $dcp_release_dir" >&2; exit 1; }
        [[ -f "$origin_hf_dir/config.json" ]] || { echo "[ERROR] Original HF config missing" >&2; exit 1; }
        mm-convert Qwen35Converter dcp_to_hf \
            --dcp_dir "$dcp_release_dir" \
            --origin_hf_dir "$origin_hf_dir" \
            --save_hf_dir "$save_hf_dir"
        [[ -f "$save_hf_dir/config.json" ]] || { echo "[ERROR] Exported HF config missing" >&2; exit 1; }
        echo "[PASS] DCP -> HF: $save_hf_dir"
        ;;
    *) usage; exit 2 ;;
esac
