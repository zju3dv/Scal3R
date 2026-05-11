#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

target_dir="data/result/custom/run"
force="${1:-}"

log() {
    echo "[clean] $*"
}

if [[ ! -e "$target_dir" ]]; then
    log "Nothing to clean. ${target_dir} does not exist."
    exit 0
fi

if [[ "$force" != "--force" ]]; then
    echo "This will permanently remove:"
    echo "  $repo_root/$target_dir"
    read -r -p "Continue? [y/N] " reply
    case "$reply" in
        y|Y|yes|YES)
            ;;
        *)
            log "Aborted."
            exit 1
            ;;
    esac
fi

rm -rf "$target_dir"
log "Removed ${target_dir}"
