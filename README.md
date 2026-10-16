# Vocal Clean

AI vocal noise reduction for Linux, as a **CLAP plugin** for REAPER (and any other CLAP host).
The engine is [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet) (MIT/Apache-2.0), run
on the CPU in pure Rust. No Python, no GPU, no extra runtime libraries; the model is built into
the plugin file.

## Install (prebuilt)

```bash
mkdir -p ~/.clap
cp "Vocal Clean.clap" ~/.clap/
```

Then in REAPER (6.71 or newer, for CLAP support): **Options > Preferences > Plug-ins > CLAP**,
make sure `~/.clap` is in the path list, and click **Re-scan** (or Clear cache / re-scan).
Add it from the FX browser as **CLAP: Vocal Clean (Mahadev Joshi)**.

Needs a 64-bit x86 Linux with glibc 2.35 or newer (Fedora 36 and later are fine).

## Controls

| Control | What it does |
|---|---|
| **Reduction** (0 to 100 dB, default 30) | The most noise the plugin may remove. Lower = more natural, keeps a little room tone. "Max" = unlimited. Start at 20 to 35 dB. |
| **Post-Filter** (0 to 100 %, default 0) | Extra clean-up of leftover noise. Helps with stubborn hiss, can sound slightly thinner. |
| **Mix** (default 100 %) | Dry/wet balance. Dry signal is delayed to match, so there is no comb filtering. |
| **Output** (-12 to +12 dB) | Makeup gain. |
| **Bypass** | Click-free crossfade, still latency compensated. |

REAPER shows these in its generic plugin window (there is no custom GUI yet).

## Things to know

* **Latency:** 40 ms at 48 kHz, about 50 ms at 44.1 / 96 kHz. The plugin reports it, so REAPER's
  plugin delay compensation lines everything up. It is meant for cleaning recorded vocals; for
  live monitoring while tracking, use your interface's direct monitoring instead.
* **CPU:** roughly 10 to 17 % of one core per instance on the 2-core cloud machine it was tested
  on, so a 10-track session is realistic on a normal desktop. Processing runs inside REAPER's
  audio thread, so use a buffer of 256 samples or more when many instances are loaded.
* **Mono and stereo** tracks are supported. Stereo tracks share one noise mask, so the image does
  not wobble.
* **Sample rates:** any, via internal resampling. 48 kHz is the cheapest and cleanest.
* The model is trained on speech. Heavily effected singing or very loud music in the background
  can sound artificial; use less Reduction or Mix some dry back in.

## Build from source (Fedora)

```bash
sudo dnf install gcc git
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh   # if you have no Rust yet
cargo build --release
./install.sh
```

The first build takes a few minutes (the neural-network runtime is large). `install.sh` copies
`target/release/libvocal_clean.so` to `~/.clap/Vocal Clean.clap`.

Test the engine on a WAV without a host:

```bash
cargo run --release --example check -- noisy.wav [clean_reference.wav] [out.wav]
```

It reports speed, checks that the output is identical for any host buffer size, and measures the
real delay against the one the plugin reports.

## Layout

* `src/engine.rs`: streaming wrapper (frames, resampling, latency).
* `src/lib.rs`: the CLAP plugin (parameters, mix, bypass, latency reporting).
* `examples/check.rs`: offline test harness.

## Licensing

DeepFilterNet is MIT/Apache-2.0 and nih-plug is ISC. This build is CLAP only: nih-plug's VST3
export pulls in GPLv3 bindings, so it is deliberately switched off. Vocal Clean itself is
MIT OR Apache-2.0. The plugin has its own name and has no connection to any commercial product.

## Pinned versions

`tract` is pinned to exactly 0.21.4 in `Cargo.toml`. DeepFilterNet does not compile against newer
0.21.x releases. Keep `Cargo.lock` when building.
