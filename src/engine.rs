//! Streaming wrapper around DeepFilterNet3.
//!
//! DeepFilterNet works on 48 kHz audio in 480-sample hops. A plugin host hands us
//! blocks of arbitrary size at an arbitrary sample rate, so this module:
//!   1. collects host samples into fixed "super-chunks",
//!   2. resamples them to 48 kHz when needed,
//!   3. runs DeepFilterNet one 480-sample hop at a time,
//!   4. resamples back, and
//!   5. hands out exactly as many samples as the host asked for.
//!
//! Output is always delayed by a constant `latency()` samples, which the plugin reports to the
//! host and also applies to its dry signal so the Mix control stays phase-aligned.

use std::collections::VecDeque;

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::Array2;
use rubato::{FftFixedInOut, Resampler};

pub const DF_SR: usize = 48_000;
pub const DF_HOP: usize = 480;

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

struct ResamplePath {
    to_df: FftFixedInOut<f32>,
    from_df: FftFixedInOut<f32>,
    /// scratch buffers, one per channel
    in_host: Vec<Vec<f32>>,
    df_in: Vec<Vec<f32>>,
    df_out: Vec<Vec<f32>>,
    out_host: Vec<Vec<f32>>,
}

pub struct Engine {
    ch: usize,
    host_sr: usize,
    df: DfTract,
    /// Host-rate samples per super-chunk (480 when the host runs at 48 kHz).
    chunk_host: usize,
    /// DF hops per super-chunk.
    hops_per_chunk: usize,
    resample: Option<ResamplePath>,
    in_fifo: Vec<VecDeque<f32>>,
    out_fifo: Vec<VecDeque<f32>>,
    frame_in: Array2<f32>,
    frame_out: Array2<f32>,
    latency: usize,
    pristine: Box<DfTract>,
}

// SAFETY: tract's runtime state uses `Rc` internally, which is why `DfTract` is not `Send`. Every
// `Rc` reachable from an `Engine` (including the ones shared between `df` and `pristine`) is owned
// by that same `Engine`; no clone of them is ever handed out. The engine is therefore always moved
// between threads as one whole unit and never used from two threads at once (the plugin host calls
// initialize/reset/process serially), so the non-atomic reference counts cannot race.
unsafe impl Send for Engine {}

#[derive(Debug)]
pub struct EngineError(pub String);

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Engine {
    /// Builds the engine. Slow (loads and optimises the ONNX models): call from `initialize()`,
    /// never from the audio thread.
    pub fn new(channels: usize, host_sr: usize) -> Result<Self, EngineError> {
        let err = |e: &dyn std::fmt::Display| EngineError(e.to_string());
        if channels == 0 || channels > 2 {
            return Err(EngineError(format!("unsupported channel count {channels}")));
        }
        let mut rp = RuntimeParams::default_with_ch(channels);
        rp.atten_lim_db = 30.0;
        let df = DfTract::new(DfParams::default(), &rp).map_err(|e| err(&e))?;
        if df.sr != DF_SR || df.hop_size != DF_HOP {
            return Err(EngineError(format!(
                "unexpected model format: sr={} hop={}",
                df.sr, df.hop_size
            )));
        }
        // delay of the STFT plus the model's look-ahead, same formula the deep-filter CLI uses
        let model_delay = df.fft_size - df.hop_size + df.lookahead * df.hop_size;

        let (chunk_host, hops_per_chunk, resample, extra_delay_host) = if host_sr == DF_SR {
            (DF_HOP, 1usize, None, 0usize)
        } else {
            let g = gcd(host_sr, DF_SR);
            let in_per = host_sr / g;
            let out_per = DF_SR / g;
            // smallest k with out_per * k a multiple of one DF hop
            let mut k = 1;
            while (out_per * k) % DF_HOP != 0 {
                k += 1;
            }
            let chunk_host = in_per * k;
            let chunk_df = out_per * k;
            let to_df = FftFixedInOut::<f32>::new(host_sr, DF_SR, chunk_df, channels)
                .map_err(|e| err(&e))?;
            let from_df = FftFixedInOut::<f32>::new(DF_SR, host_sr, chunk_host, channels)
                .map_err(|e| err(&e))?;
            if to_df.input_frames_next() != chunk_host
                || to_df.output_frames_next() != chunk_df
                || from_df.input_frames_next() != chunk_df
                || from_df.output_frames_next() != chunk_host
            {
                return Err(EngineError(format!(
                    "resampler chunking mismatch for {host_sr} Hz"
                )));
            }
            // delays reported in *output* frames of each resampler
            let d_in_host = to_df.output_delay() * host_sr / DF_SR;
            let d_out_host = from_df.output_delay();
            let path = ResamplePath {
                in_host: vec![vec![0.0; chunk_host]; channels],
                df_in: vec![vec![0.0; chunk_df]; channels],
                df_out: vec![vec![0.0; chunk_df]; channels],
                out_host: vec![vec![0.0; chunk_host]; channels],
                to_df,
                from_df,
            };
            (chunk_host, chunk_df / DF_HOP, Some(path), d_in_host + d_out_host)
        };

        let model_delay_host = model_delay * host_sr / DF_SR;
        // `chunk_host` zeros are pre-queued on the output, so buffering adds one chunk
        let latency = chunk_host + model_delay_host + extra_delay_host;

        let mut e = Self {
            ch: channels,
            host_sr,
            chunk_host,
            hops_per_chunk,
            resample,
            in_fifo: (0..channels)
                .map(|_| VecDeque::with_capacity(chunk_host * 64))
                .collect(),
            out_fifo: (0..channels)
                .map(|_| VecDeque::with_capacity(chunk_host * 64))
                .collect(),
            frame_in: Array2::zeros((channels, DF_HOP)),
            frame_out: Array2::zeros((channels, DF_HOP)),
            latency,
            pristine: Box::new(df.clone()),
            df,
        };
        e.prime_fifos();
        Ok(e)
    }

    fn prime_fifos(&mut self) {
        for q in self.in_fifo.iter_mut() {
            q.clear();
        }
        for q in self.out_fifo.iter_mut() {
            q.clear();
            q.extend(std::iter::repeat(0.0).take(self.chunk_host));
        }
    }

    /// Total delay in host samples between input and (wet) output.
    pub fn latency(&self) -> usize {
        self.latency
    }

    pub fn host_sample_rate(&self) -> usize {
        self.host_sr
    }

    pub fn channels(&self) -> usize {
        self.ch
    }

    /// Resets all streaming state (called on transport restart / re-activation).
    pub fn reset(&mut self) {
        self.df = (*self.pristine).clone();
        // keep the user's current settings after the reset
        self.prime_fifos();
        if let Some(rs) = self.resample.as_mut() {
            // fresh resamplers: rebuild from known-good parameters, falling back to flushing
            // with silence if construction ever failed
            let chunk_df = self.hops_per_chunk * DF_HOP;
            if let (Ok(a), Ok(b)) = (
                FftFixedInOut::<f32>::new(self.host_sr, DF_SR, chunk_df, self.ch),
                FftFixedInOut::<f32>::new(DF_SR, self.host_sr, self.chunk_host, self.ch),
            ) {
                rs.to_df = a;
                rs.from_df = b;
            }
        }
    }

    /// Reduction in dB (0 = off, 100 = unlimited).
    pub fn set_reduction_db(&mut self, db: f32) {
        // Below ~0.1 dB the library short-circuits and returns the un-delayed input, which would
        // change the latency. Clamp so the signal always goes through the same path.
        self.df.set_atten_lim(db.clamp(0.1, 100.0));
    }

    /// Post-filter strength in the range 0..=0.05 (0 = off).
    pub fn set_post_filter(&mut self, beta: f32) {
        self.df.set_pf_beta(beta.clamp(0.0, 0.05));
    }

    /// Processes a host block. `input` and `output` hold `ch` slices of identical length.
    /// The output is the *wet* signal delayed by `latency()` samples.
    pub fn process(&mut self, input: &[&[f32]], output: &mut [&mut [f32]]) {
        let n = input[0].len();
        for c in 0..self.ch {
            self.in_fifo[c].extend(input[c].iter().copied());
        }

        while self.in_fifo[0].len() >= self.chunk_host {
            self.process_chunk();
        }

        for c in 0..self.ch {
            let q = &mut self.out_fifo[c];
            for o in output[c][..n].iter_mut() {
                // The pre-queued chunk guarantees data is always available.
                *o = q.pop_front().unwrap_or(0.0);
            }
        }
    }

    fn process_chunk(&mut self) {
        let ch = self.ch;
        let hops = self.hops_per_chunk;
        let chunk_host = self.chunk_host;

        match self.resample.as_mut() {
            None => {
                // 48 kHz host: one hop per chunk, straight through
                for c in 0..ch {
                    for i in 0..DF_HOP {
                        self.frame_in[[c, i]] = self.in_fifo[c].pop_front().unwrap_or(0.0);
                    }
                }
                if self
                    .df
                    .process(self.frame_in.view(), self.frame_out.view_mut())
                    .is_err()
                {
                    self.frame_out.assign(&self.frame_in);
                }
                for c in 0..ch {
                    for i in 0..DF_HOP {
                        self.out_fifo[c].push_back(self.frame_out[[c, i]]);
                    }
                }
            }
            Some(rs) => {
                for c in 0..ch {
                    for i in 0..chunk_host {
                        rs.in_host[c][i] = self.in_fifo[c].pop_front().unwrap_or(0.0);
                    }
                }
                let _ = rs
                    .to_df
                    .process_into_buffer(&rs.in_host, &mut rs.df_in, None);
                for h in 0..hops {
                    for c in 0..ch {
                        for i in 0..DF_HOP {
                            self.frame_in[[c, i]] = rs.df_in[c][h * DF_HOP + i];
                        }
                    }
                    if self
                        .df
                        .process(self.frame_in.view(), self.frame_out.view_mut())
                        .is_err()
                    {
                        self.frame_out.assign(&self.frame_in);
                    }
                    for c in 0..ch {
                        for i in 0..DF_HOP {
                            rs.df_out[c][h * DF_HOP + i] = self.frame_out[[c, i]];
                        }
                    }
                }
                let _ = rs
                    .from_df
                    .process_into_buffer(&rs.df_out, &mut rs.out_host, None);
                for c in 0..ch {
                    for i in 0..chunk_host {
                        self.out_fifo[c].push_back(rs.out_host[c][i]);
                    }
                }
            }
        }
    }
}
