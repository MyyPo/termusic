// mod scaletempo_1;
mod sonic;

use std::collections::VecDeque;

use super::mix_source::MixSource;
use super::Source;
use sonic_sys::{
    sonicCreateStream, sonicDestroyStream, sonicFlushStream, sonicReadFloatFromStream,
    sonicSamplesAvailable, sonicSetSpeed, sonicWriteFloatToStream,
};

use sonic::Sonic;

#[allow(clippy::cast_sign_loss)]
pub fn tempo_stretch<I: Source<Item = f32>>(mut input: I, rate: f32) -> TempoStretch<I>
where
    I: Source<Item = f32>,
{
    let channels = input.channels();
    let sample_rate = input.sample_rate();

    let mut stream_local = Sonic::new(sample_rate as i32, channels as i32);

    let min_samples = 6000;
    let initial_latency = 8000;
    let mut out_buffer = VecDeque::new();
    out_buffer.resize(initial_latency, 0.0);
    out_buffer.make_contiguous();
    let mut initial_input: VecDeque<f32> = input.by_ref().take(initial_latency).collect();
    let samples = initial_input.make_contiguous();
    let len_input = samples.len();
    let mut out_buf: Vec<f32> = Vec::new();
    unsafe {
        let stream = sonicCreateStream(sample_rate as i32, channels as i32);
        sonicSetSpeed(stream, rate as f32);
        sonicWriteFloatToStream(stream, samples.as_ptr(), len_input as i32);
        sonicFlushStream(stream);
        let num_samples = sonicSamplesAvailable(stream);
        if num_samples <= 0 {
            return TempoStretch {
                input,
                min_samples,
                out_buffer,
                in_buffer: initial_input,
                mix: 1.0,
                factor: f64::from(rate),
                stream_local,
            };
        }
        out_buf.reserve_exact(num_samples as usize * channels as usize);
        sonicReadFloatFromStream(
            stream,
            out_buf.spare_capacity_mut().as_mut_ptr().cast(),
            num_samples,
        );
        sonicDestroyStream(stream);
        out_buf.set_len(num_samples as usize);
    }

    out_buf
        .iter()
        .copied()
        .for_each(|x| out_buffer.push_back(x));
    TempoStretch {
        input,
        min_samples,
        // soundtouch: st,
        stream_local,
        out_buffer,
        in_buffer: initial_input,
        mix: 1.0,
        factor: f64::from(rate),
    }
}

pub struct TempoStretch<I> {
    input: I,
    min_samples: usize,
    out_buffer: VecDeque<f32>,
    in_buffer: VecDeque<f32>,
    mix: f32,
    factor: f64,
    stream_local: Sonic,
}

impl<I> Iterator for TempoStretch<I>
where
    I: Source<Item = f32>,
{
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        // This is to skip calculation if speed is not changed
        if (self.factor - 1.0).abs() < 0.05 {
            return self.input.next();
        }

        if self.out_buffer.is_empty() {
            let channels = self.input.channels();
            let sample_rate = self.input.sample_rate();
            self.in_buffer.clear();
            self.input
                .by_ref()
                .take(self.min_samples)
                .for_each(|x| self.in_buffer.push_back(x));
            let samples = self.in_buffer.make_contiguous();
            let len_input = samples.len();
            // let len_input = samples.len() / channels as usize;
            let mut out_buf: Vec<f32> = Vec::new();
            unsafe {
                let stream = sonicCreateStream(sample_rate as i32, channels as i32);
                sonicSetSpeed(stream, self.factor as f32);
                sonicWriteFloatToStream(stream, samples.as_ptr(), len_input as i32);
                sonicFlushStream(stream);
                let num_samples = sonicSamplesAvailable(stream);
                if num_samples <= 0 {
                    return None;
                }
                out_buf.reserve_exact(num_samples as usize * channels as usize);
                sonicReadFloatFromStream(
                    stream,
                    out_buf.spare_capacity_mut().as_mut_ptr().cast(),
                    num_samples,
                );
                sonicDestroyStream(stream);
                out_buf.set_len(num_samples as usize);
            }

            out_buf
                .iter()
                .copied()
                .for_each(|x| self.out_buffer.push_back(x));
        }
        self.out_buffer.pop_front()

        // match (
        //     self.out_buffer.pop_front().map(|x| x * self.mix),
        //     self.in_buffer.pop_front().map(|x| x * (1.0 - self.mix)),
        // ) {
        //     (Some(a), Some(b)) => Some(a + b),
        //     (None, None) => None,
        //     (None, Some(v)) | (Some(v), None) => Some(v),
        // }
    }
}

impl<I> ExactSizeIterator for TempoStretch<I> where I: Source<Item = f32> + ExactSizeIterator {}

impl<I> Source for TempoStretch<I>
where
    I: Source<Item = f32>,
{
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.min_samples)
    }

    fn channels(&self) -> u16 {
        self.input.channels()
    }

    fn sample_rate(&self) -> u32 {
        self.input.sample_rate()
    }

    fn total_duration(&self) -> Option<std::time::Duration> {
        self.input.total_duration()
    }

    fn seek(&mut self, time: std::time::Duration) -> Option<std::time::Duration> {
        self.input.seek(time)
    }

    fn elapsed(&mut self) -> std::time::Duration {
        self.input.elapsed()
    }
}

impl<I> MixSource for TempoStretch<I>
where
    I: Source<Item = f32>,
{
    fn set_mix(&mut self, mix: f32) {
        self.mix = mix;
    }
}

#[allow(unused)]
impl<I> TempoStretch<I>
where
    I: Source<Item = f32>,
{
    /// Modifies the speed factor.
    #[inline]
    pub fn set_factor(&mut self, factor: f64) {
        self.factor = factor;
    }
}
