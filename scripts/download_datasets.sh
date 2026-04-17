#!/usr/bin/env bash
# download_datasets.sh — Download pico-banana-400k and GPT-Image-Edit-1.5M datasets
#
# Usage:
#   bash scripts/download_datasets.sh --data-dir /path/to/datasets [OPTIONS]
#
# Options:
#   --data-dir DIR       Root directory for downloaded datasets (required)
#   --workers N          Number of parallel download workers (default: 8)
#   --skip-pico-banana   Skip pico-banana-400k download
#   --skip-gpt-image     Skip GPT-Image-Edit-1.5M download
#   -h, --help           Show this help message
#
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DATA_DIR="datasets"
WORKERS=8
SKIP_PICO_BANANA=false
SKIP_GPT_IMAGE=false

# ── Argument parsing ─────────────────────────────────────────────────────────
usage() {
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)   DATA_DIR="$2"; shift 2 ;;
        --workers)    WORKERS="$2"; shift 2 ;;
        --skip-pico-banana) SKIP_PICO_BANANA=true; shift ;;
        --skip-gpt-image)   SKIP_GPT_IMAGE=true; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [[ -z "$DATA_DIR" ]]; then
    echo "Error: --data-dir is required"
    usage
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
log_info()  { echo "[$(date '+%H:%M:%S')] INFO  $*"; }
log_error() { echo "[$(date '+%H:%M:%S')] ERROR $*" >&2; }

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        log_error "Required command '$1' not found. Please install it."
        exit 1
    fi
}

# ── Preflight checks ────────────────────────────────────────────────────────
check_cmd python
check_cmd wget
check_cmd parallel

mkdir -p "$DATA_DIR"

# ── Download pico-banana-400k ────────────────────────────────────────────────
download_pico_banana() {
    local dest="$DATA_DIR/pico-banana-400k"
    mkdir -p "$dest"/{jsonl,manifest,openimage_source_images}

    local CDN="https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb"

    # 1. Download JSONL metadata
    log_info "Downloading pico-banana-400k JSONL metadata..."
    for split in sft preference multi-turn; do
        local out="$dest/jsonl/${split}.jsonl"
        if [[ -f "$out" ]]; then
            log_info "  $split.jsonl already exists, skipping"
        else
            wget -q --show-progress -O "$out" "$CDN/jsonl/${split}.jsonl"
        fi
    done

    # 2. Download source images from Open Images (AWS S3, no credentials needed)
    log_info "Downloading source images from Open Images S3..."
    check_cmd aws
    for i in 0 1; do
        local tarfile="$dest/openimage_source_images/train_${i}.tar.gz"
        if [[ -f "$tarfile" ]]; then
            log_info "  train_${i}.tar.gz already exists, skipping"
        else
            aws s3 --no-sign-request --endpoint-url https://s3.amazonaws.com \
                cp "s3://open-images-dataset/tar/train_${i}.tar.gz" "$tarfile"
        fi
    done

    # 4. Extract source images
    log_info "Extracting source images..."
    for i in 0 1; do
        tar -xzf "$dest/openimage_source_images/train_${i}.tar.gz" -C "$dest/openimage_source_images"
    done

    log_info "pico-banana-400k download complete"
}

# ── Download GPT-Image-Edit-1.5M ────────────────────────────────────────────
# Uses multi-process download (parallel wget) instead of huggingface-cli for
# significantly faster throughput on large split archives.
# See: https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M#multi-process-download
download_gpt_image() {
    local dest="$DATA_DIR/gptimage15m/gpt-edit"
    local BASE_URL="https://huggingface.co/datasets/UCSC-VLAA/GPT-Image-Edit-1.5M/resolve/main/gpt-edit"

    mkdir -p "$dest"

    for dataset in hqedit omniedit ultraedit; do
        local ddir="$dest/$dataset"
        mkdir -p "$ddir"

        # Determine part range per dataset
        local range
        case "$dataset" in
            hqedit)    range=$(seq -w 001 100) ;;
            omniedit)  range=$(seq -w 001 175) ;;
            ultraedit) range=$(seq -w 001 004) ;;
        esac

        # Check if already extracted
        if [[ -f "$ddir/.extracted" ]]; then
            log_info "$dataset already downloaded and extracted, skipping"
            continue
        fi

        # Download parts in parallel with resume support
        log_info "Downloading $dataset parts ($WORKERS parallel jobs)..."
        echo "$range" | parallel --lb -j "$WORKERS" \
            "wget --progress=bar:force -c '${BASE_URL}/${dataset}.tar.gz.part{}?download=true' -O '${ddir}/${dataset}.tar.gz.part{}'"

        # Merge and extract
        log_info "Merging and extracting $dataset..."
        cat "$ddir/${dataset}.tar.gz.part"* > "$ddir/${dataset}.tar.gz"
        tar -xzf "$ddir/${dataset}.tar.gz" -C "$ddir"
        rm -f "$ddir/${dataset}.tar.gz.part"* "$ddir/${dataset}.tar.gz"
        touch "$ddir/.extracted"

        log_info "$dataset download and extraction complete"
    done

    log_info "GPT-Image-Edit-1.5M download complete"
}

# ── Main ─────────────────────────────────────────────────────────────────────
log_info "Data directory: $DATA_DIR"
log_info "Workers: $WORKERS"

PIDS=()

if [[ "$SKIP_PICO_BANANA" == false ]]; then
    download_pico_banana &
    PIDS+=($!)
    log_info "Started pico-banana-400k download (PID ${PIDS[-1]})"
fi

if [[ "$SKIP_GPT_IMAGE" == false ]]; then
    download_gpt_image &
    PIDS+=($!)
    log_info "Started GPT-Image-Edit-1.5M download (PID ${PIDS[-1]})"
fi

# Wait for all background downloads
FAILED=false
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        log_error "Download process $pid failed"
        FAILED=true
    fi
done

if [[ "$FAILED" == true ]]; then
    log_error "One or more downloads failed. Check errors above."
    exit 1
fi

log_info "All downloads complete."
