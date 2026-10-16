//! Offline test harness: WAV in -> engine -> WAV out, with latency, determinism and speed checks.
//!
//! usage: check <noisy.wav> [clean_reference.wav] [out.wav]

use std::time::Instant;
use vocal_clean::engine::Engine;

fn read_wav(path: &str) -> (Vec<Vec<f32>>, usize) {
    let mut r = hound::WavReader::open(path).expect("open wav");
    let spec = r.spec();
    let ch = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            r.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect()
        }
    };
    let mut out = vec![Vec::with_capacity(samples.len() / ch); ch];
    for (i, s) in samples.iter().enumerate() {
        out[i % ch].push(*s);
    }
    (out, spec.sample_rate as usize)
}

fn write_wav(path: &str, data: &[Vec<f32>], sr: usize) {
    let spec = hound::WavSpec {
        channels: data.len() as u16,
        sample_rate: sr as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..data[0].len() {
        for c in data {
            w.write_sample((c[i].clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
        }
    }
    w.finalize().unwrap();
}

/// Streams `input` through a fresh engine using the given block-size pattern.
fn run(
    input: &[Vec<f32>],
    sr: usize,
    reduction_db: f32,
    pf: f32,
    blocks: &[usize],
) -> (Vec<Vec<f32>>, usize, f64) {
    let ch = input.len();
    let mut e = Engine::new(ch, sr).expect("engine");
    e.set_reduction_db(reduction_db);
    e.set_post_filter(pf);
    let n = input[0].len();
    let mut out = vec![vec![0.0f32; n]; ch];
    let (mut pos, mut bi) = (0usize, 0usize);
    let mut worst_ms = 0.0f64;
    let t_all = Instant::now();
    while pos < n {
        let b = blocks[bi % blocks.len()].min(n - pos);
        bi += 1;
        let t0 = Instant::now();
        let ins: Vec<&[f32]> = input.iter().map(|c| &c[pos..pos + b]).collect();
        let mut outs: Vec<&mut [f32]> = out.iter_mut().map(|c| &mut c[pos..pos + b]).collect();
        e.process(&ins, &mut outs);
        worst_ms = worst_ms.max(t0.elapsed().as_secs_f64() * 1000.0);
        pos += b;
    }
    let total = t_all.elapsed().as_secs_f64();
    let audio = n as f64 / sr as f64;
    println!(
        "    rtf {:.3} (x{:.1} realtime), worst single block {:.2} ms",
        total / audio,
        audio / total,
        worst_ms
    );
    (out, e.latency(), worst_ms)
}

fn best_lag(a: &[f32], b: &[f32], max_lag: usize) -> isize {
    // lag such that b[i + lag] ~ a[i]
    let len = a.len().min(b.len()).min(200_000) - max_lag - 1;
    let mut best = (f64::MIN, 0isize);
    for lag in 0..=max_lag as isize {
        let mut acc = 0.0f64;
        for i in 0..len {
            acc += a[i] as f64 * b[i + lag as usize] as f64;
        }
        if acc > best.0 {
            best = (acc, lag);
        }
    }
    best.1
}

fn snr_db(clean: &[f32], test: &[f32]) -> f64 {
    let n = clean.len().min(test.len());
    let (mut s, mut e) = (0.0f64, 0.0f64);
    for i in 0..n {
        s += (clean[i] as f64).powi(2);
        e += ((test[i] - clean[i]) as f64).powi(2);
    }
    10.0 * (s / e.max(1e-20)).log10()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (input, sr) = read_wav(&args[1]);
    let n = input[0].len();
    println!("input: {} ch, {} Hz, {:.2} s", input.len(), sr, n as f64 / sr as f64);

    println!("[1] speed, block size 512");
    let (ref_out, latency, _) = run(&input, sr, 100.0, 0.0, &[512]);
    println!("    reported latency: {} samples = {:.2} ms", latency, latency as f64 * 1000.0 / sr as f64);

    println!("[2] block-size independence (output must match for any host buffer size)");
    let patterns: [(&str, Vec<usize>); 3] = [
        ("64", vec![64]),
        ("odd mix", vec![1, 7, 33, 480, 1023, 129, 2048, 5]),
        ("huge 20000", vec![20000]),
    ];
    for (name, p) in patterns.iter() {
        let (o, _, _) = run(&input, sr, 100.0, 0.0, p);
        let mut max_diff = 0.0f32;
        for c in 0..input.len() {
            for i in 0..n {
                max_diff = max_diff.max((o[c][i] - ref_out[c][i]).abs());
            }
        }
        println!("    blocks {name:>10}: max |diff| vs 512-blocks = {max_diff:.2e}  {}", if max_diff < 1e-5 { "OK" } else { "MISMATCH" });
    }

    println!("[3] measured delay (cross-correlation of output against input)");
    let lag = best_lag(&input[0], &ref_out[0], latency + 2000);
    println!("    measured lag {} samples, reported {} -> diff {}", lag, latency, lag - latency as isize);

    if args.len() > 2 && sr == 48000 {
        let (clean, _) = read_wav(&args[2]);
        println!("[4] quality vs clean reference (48 kHz only)");
        let lag = latency;
        let aligned: Vec<f32> = ref_out[0][lag..].to_vec();
        println!("    input SNR   : {:.2} dB", snr_db(&clean[0], &input[0]));
        println!("    output SNR  : {:.2} dB (max reduction)", snr_db(&clean[0], &aligned));
        let (o30, _, _) = run(&input, sr, 30.0, 0.0, &[512]);
        println!("    output SNR  : {:.2} dB (30 dB reduction)", snr_db(&clean[0], &o30[0][lag..]));
        let (o12, _, _) = run(&input, sr, 12.0, 0.0, &[512]);
        println!("    output SNR  : {:.2} dB (12 dB reduction)", snr_db(&clean[0], &o12[0][lag..]));
    }

    if args.len() > 3 {
        let lag = latency.min(n);
        let trimmed: Vec<Vec<f32>> = ref_out.iter().map(|c| c[lag..].to_vec()).collect();
        write_wav(&args[3], &trimmed, sr);
        println!("wrote {} (latency trimmed)", args[3]);
    }
}
