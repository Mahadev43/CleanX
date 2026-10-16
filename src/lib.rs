//! Vocal Clean: an AI vocal noise reduction CLAP plugin for Linux, powered by DeepFilterNet3.

pub mod engine;

use engine::Engine;
use nih_plug::prelude::*;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Longest block processed in one go; larger host blocks are split.
const SCRATCH_LEN: usize = 8192;

#[derive(Params)]
struct VocalCleanParams {
    /// How much noise may be removed, in dB. 100 dB = unlimited.
    #[id = "reduction"]
    reduction: FloatParam,
    /// Extra suppression of residual noise (DeepFilterNet post-filter).
    #[id = "postfilter"]
    post_filter: FloatParam,
    /// Dry/wet balance, latency compensated.
    #[id = "mix"]
    mix: FloatParam,
    /// Output level.
    #[id = "outgain"]
    output_gain: FloatParam,
    /// Click-free bypass.
    #[id = "bypass"]
    bypass: BoolParam,
}

impl Default for VocalCleanParams {
    fn default() -> Self {
        Self {
            reduction: FloatParam::new(
                "Reduction",
                30.0,
                FloatRange::Linear { min: 0.0, max: 100.0 },
            )
            .with_unit(" dB")
            .with_step_size(0.5)
            .with_value_to_string(Arc::new(|v| {
                if v >= 99.5 {
                    "Max".to_string()
                } else {
                    format!("{v:.1}")
                }
            }))
            .with_string_to_value(Arc::new(|s| {
                let t = s.trim().trim_end_matches("dB").trim();
                if t.eq_ignore_ascii_case("max") {
                    Some(100.0)
                } else {
                    t.parse().ok()
                }
            })),
            post_filter: FloatParam::new(
                "Post-Filter",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_unit(" %")
            .with_value_to_string(formatters::v2s_f32_percentage(0))
            .with_string_to_value(formatters::s2v_f32_percentage()),
            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit(" %")
                .with_value_to_string(formatters::v2s_f32_percentage(0))
                .with_string_to_value(formatters::s2v_f32_percentage()),
            output_gain: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Linear { min: -12.0, max: 12.0 },
            )
            .with_unit(" dB")
            .with_step_size(0.1)
            .with_smoother(SmoothingStyle::Linear(20.0)),
            bypass: BoolParam::new("Bypass", false).make_bypass(),
        }
    }
}

struct DryDelay {
    buf: Vec<Vec<f32>>,
    pos: usize,
}

impl DryDelay {
    fn new(channels: usize, delay: usize) -> Self {
        Self {
            buf: vec![vec![0.0; delay.max(1)]; channels],
            pos: 0,
        }
    }

    fn clear(&mut self) {
        for b in self.buf.iter_mut() {
            b.fill(0.0);
        }
        self.pos = 0;
    }
}

struct VocalClean {
    params: Arc<VocalCleanParams>,
    engine: Option<Engine>,
    dry: DryDelay,
    in_scratch: Vec<Vec<f32>>,
    wet_scratch: Vec<Vec<f32>>,
    mix_state: f32,
    mix_coeff: f32,
    last_reduction: f32,
    last_post_filter: f32,
    force_param_push: bool,
}

impl Default for VocalClean {
    fn default() -> Self {
        Self {
            params: Arc::new(VocalCleanParams::default()),
            engine: None,
            dry: DryDelay::new(2, 1),
            in_scratch: Vec::new(),
            wet_scratch: Vec::new(),
            mix_state: 1.0,
            mix_coeff: 0.99,
            last_reduction: f32::NAN,
            last_post_filter: f32::NAN,
            force_param_push: true,
        }
    }
}

impl Plugin for VocalClean {
    const NAME: &'static str = "Vocal Clean";
    const VENDOR: &'static str = "Mahadev Joshi";
    const URL: &'static str = "https://github.com/Mahadev43";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    const SAMPLE_ACCURATE_AUTOMATION: bool = false;

    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn initialize(
        &mut self,
        audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl InitContext<Self>,
    ) -> bool {
        let channels = audio_io_layout
            .main_input_channels
            .map(NonZeroU32::get)
            .unwrap_or(2) as usize;
        let sr = buffer_config.sample_rate.round() as usize;

        // Rebuild only when the configuration changed; model loading is slow.
        let needs_new = match self.engine.as_ref() {
            Some(e) => e.channels() != channels || e.host_sample_rate() != sr,
            None => true,
        };
        if needs_new {
            match Engine::new(channels, sr) {
                Ok(e) => self.engine = Some(e),
                Err(e) => {
                    nih_error!("Vocal Clean could not start: {e}");
                    return false;
                }
            }
        }

        // The audio thread must never print: keep only errors from here on.
        log::set_max_level(log::LevelFilter::Error);

        let latency = self.engine.as_ref().map(Engine::latency).unwrap_or(0);
        context.set_latency_samples(latency as u32);

        self.dry = DryDelay::new(channels, latency);
        self.in_scratch = vec![vec![0.0; SCRATCH_LEN]; channels];
        self.wet_scratch = vec![vec![0.0; SCRATCH_LEN]; channels];
        // ~15 ms one-pole smoothing for the mix / bypass crossfade
        self.mix_coeff = (-1.0 / (0.015 * buffer_config.sample_rate)).exp();
        self.mix_state = if self.params.bypass.value() {
            0.0
        } else {
            self.params.mix.value()
        };
        self.force_param_push = true;
        true
    }

    fn reset(&mut self) {
        if let Some(e) = self.engine.as_mut() {
            e.reset();
        }
        self.dry.clear();
        self.force_param_push = true;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let Some(engine) = self.engine.as_mut() else {
            return ProcessStatus::Normal;
        };
        let channels = buffer.channels().min(engine.channels());
        if channels == 0 {
            return ProcessStatus::Normal;
        }

        // push control changes to the engine
        let reduction = self.params.reduction.value();
        if self.force_param_push || reduction != self.last_reduction {
            engine.set_reduction_db(reduction);
            self.last_reduction = reduction;
        }
        let pf = self.params.post_filter.value();
        if self.force_param_push || pf != self.last_post_filter {
            // 0..100 % maps to the library's useful beta range 0..0.05
            engine.set_post_filter(pf * 0.05);
            self.last_post_filter = pf;
        }
        self.force_param_push = false;

        let mix_target = if self.params.bypass.value() {
            0.0
        } else {
            self.params.mix.value()
        };

        let total = buffer.samples();
        let slices = buffer.as_slice();
        let delay = self.dry.buf[0].len();

        let mut start = 0;
        while start < total {
            let n = (total - start).min(SCRATCH_LEN);

            for c in 0..channels {
                self.in_scratch[c][..n].copy_from_slice(&slices[c][start..start + n]);
            }

            // fixed-size arrays avoid allocating on the audio thread (max 2 channels)
            let (in0, in1) = self.in_scratch.split_at(1);
            let (wet0, wet1) = self.wet_scratch.split_at_mut(1);
            if channels == 1 {
                let input = [&in0[0][..n]];
                let mut output = [&mut wet0[0][..n]];
                engine.process(&input, &mut output);
            } else {
                let input = [&in0[0][..n], &in1[0][..n]];
                let mut output = [&mut wet0[0][..n], &mut wet1[0][..n]];
                engine.process(&input, &mut output);
            }

            for i in 0..n {
                self.mix_state = mix_target + (self.mix_state - mix_target) * self.mix_coeff;
                let m = self.mix_state;
                let gain = util::db_to_gain(self.params.output_gain.smoothed.next());
                let pos = self.dry.pos;
                for c in 0..channels {
                    let dry = self.dry.buf[c][pos];
                    self.dry.buf[c][pos] = self.in_scratch[c][i];
                    let wet = self.wet_scratch[c][i];
                    slices[c][start + i] = (wet * m + dry * (1.0 - m)) * gain;
                }
                self.dry.pos = if pos + 1 >= delay { 0 } else { pos + 1 };
            }
            start += n;
        }

        ProcessStatus::Normal
    }

    fn deactivate(&mut self) {}
}

impl ClapPlugin for VocalClean {
    const CLAP_ID: &'static str = "io.github.mahadev43.vocal-clean";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("AI vocal noise reduction powered by DeepFilterNet3");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Restoration,
        ClapFeature::Mono,
        ClapFeature::Stereo,
    ];
}

nih_export_clap!(VocalClean);
