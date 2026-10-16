#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
src="$here/target/release/libvocal_clean.so"
[ -f "$src" ] || src="$here/Vocal Clean.clap"
[ -f "$src" ] || { echo "Build first: cargo build --release"; exit 1; }
mkdir -p "$HOME/.clap"
cp "$src" "$HOME/.clap/Vocal Clean.clap"
echo "Installed to $HOME/.clap/Vocal Clean.clap. Re-scan plugins in REAPER."
