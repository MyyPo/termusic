// This filter was ported from Chromium
// (https://chromium.googlesource.com/chromium/chromium/+/51ed77e3f37a9a9b80d6d0a8259e84a8ca635259/media/filters/audio_renderer_algorithm.cc)
//
// Copyright 2015 The Chromium Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Algorithm overview (from chromium):
// Waveform Similarity Overlap-and-add (WSOLA).
//
// One WSOLA iteration
//
// 1) Extract |target_block| as input frames at indices
//    [|target_block_index|, |target_block_index| + |ola_window_size|).
//    Note that |target_block| is the "natural" continuation of the output.
//
// 2) Extract |search_block| as input frames at indices
//    [|search_block_index|,
//     |search_block_index| + |num_candidate_blocks| + |ola_window_size|).
//
// 3) Find a block within the |search_block| that is most similar
//    to |target_block|. Let |optimal_index| be the index of such block and
//    write it to |optimal_block|.
//
// 4) Update:
//    |optimal_block| = |transition_window| * |target_block| +
//    (1 - |transition_window|) * |optimal_block|.
//
// 5) Overlap-and-add |optimal_block| to the |wsola_output|.
//
// 6) Update:write

use std::alloc::{alloc, dealloc, Layout};
use std::mem::size_of;
use std::ptr::null_mut;
use std::slice;

const MP_NUM_CHANNELS: usize = 64;

struct MpScaletempo2Opts {
    // Max/min supported playback rates for fast/slow audio. Audio outside of these
    // ranges are muted.
    // Audio at these speeds would sound better under a frequency domain algorithm.
    min_playback_rate: f32,
    max_playback_rate: f32,
    // Overlap-and-add window size in milliseconds.
    ola_window_size_ms: f32,
    // Size of search interval in milliseconds. The search interval is
    // [-delta delta] around |output_index| * |playback_rate|. So the search
    // interval is 2 * delta.
    wsola_search_interval_ms: f32,
}

struct MpScaletempo2 {
    opts: MpScaletempo2Opts,
    // Number of channels in audio stream.
    channels: i32,
    // Sample rate of audio stream.
    samples_per_second: i32,
    // If muted, keep track of partial frames that should have been skipped over.
    muted_partial_frame: f64,
    // Book keeping of the current time of generated audio, in frames.
    // Corresponds to the center of |search_block|. This is increased in
    // intervals of |ola_hop_size| multiplied by the current playback_rate,
    // for every WSOLA iteration. This tracks the number of advanced frames as
    // a double to achieve accurate playback rates beyond the integer precision
    // of |search_block_index|.
    // Needs to be adjusted like any other index when frames are evicted from
    // |input_buffer|.
    output_time: f64,
    // The offset of the center frame of |search_block| w.r.t. its first frame.
    search_block_center_offset: i32,
    // Index of the beginning of the |search_block|, in frames. This may be
    // negative, which is handled by |peek_audio_with_zero_prepend|.
    search_block_index: i32,
    // Number of Blocks to search to find the most similar one to the target
    // frame.
    num_candidate_blocks: i32,
    // Index of the beginning of the target block, counted in frames.
    target_block_index: i32,
    // Overlap-and-add window size in frames.
    ola_window_size: i32,
    // The hop size of overlap-and-add in frames. This implementation assumes 50%
    // overlap-and-add.
    ola_hop_size: i32,
    // Number of frames in |wsola_output| that overlap-and-add is completed for
    // them and can be copied to output if fill_buffer() is called. It also
    // specifies the index where the next WSOLA window has to overlap-and-add.
    num_complete_frames: i32,
    // Whether |wsola_output| contains an additional |ola_hop_size| of overlap
    // frames for the next iteration.
    wsola_output_started: bool,
    // Overlap-and-add window.
    ola_window: Vec<f32>,
    // Transition window, used to update |optimal_block| by a weighted sum of
    // |optimal_block| and |target_block|.
    transition_window: Vec<f32>,
    // This stores a part of the output that is created but couldn't be rendered.
    // Output is generated frame-by-frame which at some point might exceed the
    // number of requested samples. Furthermore, due to overlap-and-add,
    // the last half-window of the output is incomplete, which is stored in this
    // buffer.
    wsola_output: Vec<Vec<f32>>,
    wsola_output_size: i32,
    // Auxiliary variables to avoid allocation in every iteration.
    // Stores the optimal block in every iteration. This is the most
    // similar block to |target_block| within |search_block| and it is
    // overlap-and-added to |wsola_output|.
    optimal_block: Vec<Vec<f32>>,
    // A block of data that search is performed over to find the |optimal_block|.
    search_block: Vec<Vec<f32>>,
    search_block_size: i32,
    // Stores the target block, denoted as |target| above. |search_block| is
    // searched for a block (|optimal_block|) that is most similar to
    // |target_block|.
    target_block: Vec<Vec<f32>>,
    // Buffered audio data.
    input_buffer: Vec<Vec<f32>>,
    input_buffer_size: i32,
    input_buffer_frames: i32,
    // How many frames in |input_buffer| need to be flushed by padding with
    // silence to process the final packet. While this is nonzero, the filter
    // appends silence to |input_buffer| until these frames are processed.
    input_buffer_final_frames: i32,
    // How many additional frames of silence have been added to |input_buffer|
    // for padding after the final packet.
    input_buffer_added_silence: i32,
    energy_candidate_blocks: Vec<f32>,
    // output buffer
    output_buffer: Vec<Vec<f32>>,
    output_buffer_size: i32,
    output_buffer_frames: i32,
}

impl MpScaletempo2 {
    fn mp_scaletempo2_reset(&mut self) {
        self.target_block_index = 0;
        self.search_block_index = 0;
        self.input_buffer_frames = 0;
        self.input_buffer_final_frames = 0;
        self.input_buffer_added_silence = 0;
        self.output_buffer_frames = 0;
        self.num_complete_frames = 0;
    }

    pub fn mp_scaletempo2_create(channels: i32, ola_window_size: i32, ola_hop_size: i32) -> Self {
        let target_block_size = ola_window_size * 2;
        let search_block_size = ola_window_size * 2;
        let optimal_block_size = ola_window_size * 2;
        let wsola_output_size = ola_window_size * 2;

        let input_buffer_size = 0;
        let input_buffer_frames = 0;
        let input_buffer_final_frames = 0;
        let input_buffer_added_silence = 0;
        let output_buffer_frames = 0;
        let num_complete_frames = 0;

        let target_block_index = 0;
        let search_block_index = 0;

        let search_block_center_offset = 0.0;

        let target_block = vec![null_mut(); channels as usize];
        let search_block = vec![null_mut(); channels as usize];
        let optimal_block = vec![null_mut(); channels as usize];
        let wsola_output = vec![null_mut(); channels as usize];

        let energy_candidate_blocks = vec![0.0; (search_block_size * channels) as usize];

        Self {
            channels,
            ola_window_size,
            ola_hop_size,
            target_block_size,
            search_block_size,
            optimal_block_size,
            wsola_output_size,
            input_buffer_size,
            input_buffer_frames,
            input_buffer_final_frames,
            input_buffer_added_silence,
            output_buffer_frames,
            num_complete_frames,
            target_block_index,
            search_block_index,
            search_block_center_offset,
            target_block,
            search_block,
            optimal_block,
            wsola_output,
            energy_candidate_blocks,
        }
    }

    fn mp_scaletempo2_destroy(p: &mut MpScaletempo2) {
        for i in 0..p.channels {
            unsafe {
                dealloc(
                    p.target_block[i as usize] as *mut u8,
                    Layout::from_size_align_unchecked(
                        (p.target_block_size * size_of::<f32>()) as usize,
                        1,
                    ),
                );
                dealloc(
                    p.search_block[i as usize] as *mut u8,
                    Layout::from_size_align_unchecked(
                        (p.search_block_size * size_of::<f32>()) as usize,
                        1,
                    ),
                );
                dealloc(
                    p.optimal_block[i as usize] as *mut u8,
                    Layout::from_size_align_unchecked(
                        (p.optimal_block_size * size_of::<f32>()) as usize,
                        1,
                    ),
                );
                dealloc(
                    p.wsola_output[i as usize] as *mut u8,
                    Layout::from_size_align_unchecked(
                        (p.wsola_output_size * size_of::<f32>()) as usize,
                        1,
                    ),
                );
            }
        }
        dealloc(
            p.input_buffer as *mut u8,
            Layout::from_size_align_unchecked(
                (p.input_buffer_size * size_of::<*mut f32>()) as usize,
                1,
            ),
        );
    }
}

struct Interval {
    lo: i32,
    hi: i32,
}

impl Interval {
    fn in_interval(&self, n: i32) -> bool {
        n >= self.lo && n <= self.hi
    }
}

fn realloc_2d(p: *mut *mut f32, x: i32, y: i32) -> *mut *mut f32 {
    let array_size =
        size_of::<*mut f32>() * x as usize + size_of::<f32>() * x as usize * y as usize;
    let array_layout = Layout::from_size_align(array_size, 1).unwrap();
    let array = unsafe { alloc(array_layout) as *mut *mut f32 };
    let data = unsafe { array.add(x as usize) as *mut f32 };
    for i in 0..x {
        unsafe {
            *array.add(i as usize) = data.add((i * y) as usize);
        }
    }
    array
}

fn zero_2d(a: *mut *mut f32, x: i32, y: i32) {
    let size = size_of::<f32>() * x as usize * y as usize;
    unsafe {
        std::ptr::write_bytes(a.add(x as usize) as *mut u8, 0, size);
    }
}

fn zero_2d_partial(a: Vec<Vec<f32>>, x: i32, y: i32) {
    for i in 0..x {
        let size = size_of::<f32>() * y as usize;
        unsafe {
            std::ptr::write_bytes(a.add(i as usize) as *mut u8, 0, size);
        }
    }
}

fn multi_channel_moving_block_energies(
    input: &[*mut f32],
    input_frames: i32,
    channels: i32,
    frames_per_block: i32,
    energy: &mut [f32],
) {
    let num_blocks = input_frames - (frames_per_block - 1);

    for k in 0..channels {
        let input_channel =
            unsafe { slice::from_raw_parts(input[k as usize], input_frames as usize) };

        energy[k as usize] = 0.0;

        // First block of channel |k|.
        for m in 0..frames_per_block {
            energy[k as usize] += input_channel[m as usize] * input_channel[m as usize];
        }

        let mut slide_out = input_channel;
        let mut slide_in = &input_channel[frames_per_block as usize..];
        for n in 1..num_blocks {
            energy[(k + n * channels) as usize] = energy[(k + (n - 1) * channels) as usize]
                - slide_out[0] * slide_out[0]
                + slide_in[0] * slide_in[0];
            slide_out = &slide_out[1..];
            slide_in = &slide_in[1..];
        }
    }
}

fn multi_channel_similarity_measure(
    dot_prod_a_b: &[f32],
    energy_a: &[f32],
    energy_b: &[f32],
    channels: i32,
) -> f32 {
    const EPSILON: f32 = 1e-12;
    let mut similarity_measure = 0.0;
    for n in 0..channels {
        similarity_measure += dot_prod_a_b[n as usize]
            / (energy_a[n as usize] * energy_b[n as usize] + EPSILON).sqrt();
    }
    similarity_measure
}

// #[cfg(feature = "vector")]
// type V8sf = [f32; 8];

// #[cfg(feature = "vector")]
// fn multi_channel_dot_product(
//     a: &[*mut f32],
//     frame_offset_a: i32,
//     b: &[*mut f32],
//     frame_offset_b: i32,
//     channels: i32,
//     num_frames: i32,
//     dot_product: &mut [f32],
// ) {
//     assert!(frame_offset_a >= 0);
//     assert!(frame_offset_b >= 0);

//     for k in 0..channels {
//         let ch_a = unsafe { slice::from_raw_parts(a[k as usize].add(frame_offset_a as usize), num_frames as usize) };
//         let ch_b = unsafe { slice::from_raw_parts(b[k as usize].add(frame_offset_b as usize), num_frames as usize) };
//         let mut sum = 0.0;
//         if num_frames < 32 {
//             goto rest;
//         }

//         let va = unsafe { &*(ch_a.as_ptr() as *const V8sf) };
//         let vb = unsafe { &*(ch_b.as_ptr() as *const V8sf) };
//         let mut vsum = [
//             va[0] * vb[0],
//             va[1] * vb[1],
//             va[2] * vb[2],
//             va[3] * vb[3],
//         ];
//         let mut va = unsafe { va.add(4) };
//         let mut vb = unsafe { vb.add(4) };

//         // Process `va` and `vb` across four vertical stripes
//         for _ in 1..num_frames / 32 {
//             vsum[0] += va[0] * vb[0];
//             vsum[1] += va[1] * vb[1];
//             vsum[2] += va[2] * vb[2];
//             vsum[3] += va[3] * vb[3];
//             va = unsafe { va.add(4) };
//             vb = unsafe { vb.add(4) };
//         }

//         // Vertical sum across `vsum` entries
//         vsum[0] += vsum[1];
//         vsum[2] += vsum[3];
//         vsum[0] += vsum[2];

//         // Horizontal sum across `vsum[0]`, could probably be done better but
//         // this section is not super performance critical
//         let vf = unsafe { &*(vsum[0].as_ptr() as *const [f32; 8]) };
//         sum = vf.iter().sum();
//         let ch_a = unsafe { va as *const f32 };
//         let ch_b = unsafe { vb as *const f32 };

//     rest:
//         // Process the remainder
//         for n in 0..num_frames % 32 {
//             sum += unsafe { *ch_a.add(n as usize) } * unsafe { *ch_b.add(n as usize) };
//         }

//         dot_product[k as usize] = sum;
//     }
// }

// #[cfg(not(feature = "vector"))]
fn multi_channel_dot_product(
    a: &[*mut f32],
    frame_offset_a: i32,
    b: &[*mut f32],
    frame_offset_b: i32,
    channels: i32,
    num_frames: i32,
    dot_product: &mut [f32],
) {
    assert!(frame_offset_a >= 0);
    assert!(frame_offset_b >= 0);

    for k in 0..channels {
        let ch_a = unsafe {
            slice::from_raw_parts(
                a[k as usize].add(frame_offset_a as usize),
                num_frames as usize,
            )
        };
        let ch_b = unsafe {
            slice::from_raw_parts(
                b[k as usize].add(frame_offset_b as usize),
                num_frames as usize,
            )
        };
        let mut sum = 0.0;
        for n in 0..num_frames {
            sum += ch_a[n as usize] * ch_b[n as usize];
        }
        dot_product[k as usize] = sum;
    }
}

fn quadratic_interpolation(y_values: &[f32], extremum: &mut f32, extremum_value: &mut f32) {
    let a = 0.5 * (y_values[2] + y_values[0]) - y_values[1];
    let b = 0.5 * (y_values[2] - y_values[0]);
    let c = y_values[1];

    if a == 0.0 {
        *extremum = 0.0;
        *extremum_value = y_values[1];
    } else {
        *extremum = -b / (2.0 * a);
        *extremum_value = a * (*extremum) * (*extremum) + b * (*extremum) + c;
    }
}

fn decimated_search(
    decimation: i32,
    exclude_interval: Interval,
    target_block: &[*mut f32],
    target_block_frames: i32,
    search_segment: &[*mut f32],
    search_segment_frames: i32,
    channels: i32,
    energy: &[f32],
) -> i32 {
    let num_candidate_blocks = search_segment_frames - (target_block_frames - 1);
    let mut dot_prod = vec![0.0; channels as usize];
    let mut similarity = vec![0.0; 3];

    let mut n = 0;
    multi_channel_dot_product(
        target_block,
        0,
        search_segment,
        n,
        channels,
        target_block_frames,
        &mut dot_prod,
    );
    similarity[0] = multi_channel_similarity_measure(
        &dot_prod,
        &energy[0..channels as usize],
        &energy[(n * channels) as usize..],
        channels,
    );

    let mut best_similarity = similarity[0];
    let mut optimal_index = 0;

    n += decimation;
    if n >= num_candidate_blocks {
        return 0;
    }

    multi_channel_dot_product(
        target_block,
        0,
        search_segment,
        n,
        channels,
        target_block_frames,
        &mut dot_prod,
    );
    similarity[1] = multi_channel_similarity_measure(
        &dot_prod,
        &energy[0..channels as usize],
        &energy[(n * channels) as usize..],
        channels,
    );

    n += decimation;
    if n >= num_candidate_blocks {
        return if similarity[1] > similarity[0] {
            decimation
        } else {
            0
        };
    }

    for _ in n..num_candidate_blocks {
        multi_channel_dot_product(
            target_block,
            0,
            search_segment,
            n,
            channels,
            target_block_frames,
            &mut dot_prod,
        );

        similarity[2] = multi_channel_similarity_measure(
            &dot_prod,
            &energy[0..channels as usize],
            &energy[(n * channels) as usize..],
            channels,
        );

        if (similarity[1] > similarity[0] && similarity[1] >= similarity[2])
            || (similarity[1] >= similarity[0] && similarity[1] > similarity[2])
        {
            let mut normalized_candidate_index = 0.0;
            let mut candidate_similarity = 0.0;
            quadratic_interpolation(
                &similarity,
                &mut normalized_candidate_index,
                &mut candidate_similarity,
            );

            let candidate_index =
                n - decimation + (normalized_candidate_index * decimation as f32 + 0.5) as i32;
            if candidate_similarity > best_similarity
                && !exclude_interval.in_interval(candidate_index)
            {
                optimal_index = candidate_index;
                best_similarity = candidate_similarity;
            }
        } else if n + decimation >= num_candidate_blocks
            && similarity[2] > best_similarity
            && !exclude_interval.in_interval(n)
        {
            optimal_index = n;
            best_similarity = similarity[2];
        }
        similarity.copy_within(1.., 0);
    }
    optimal_index
}

fn full_search(
    low_limit: i32,
    high_limit: i32,
    exclude_interval: Interval,
    target_block: &[*mut f32],
    target_block_frames: i32,
    search_block: &[*mut f32],
    search_block_frames: i32,
    channels: i32,
    energy_target_block: &[f32],
    energy_candidate_blocks: &[f32],
) -> i32 {
    let mut best_similarity = f32::NEG_INFINITY;
    let mut optimal_index = 0;

    for n in low_limit..=high_limit {
        if exclude_interval.in_interval(n) {
            continue;
        }
        let mut dot_prod = vec![0.0; channels as usize];
        multi_channel_dot_product(
            target_block,
            0,
            search_block,
            n,
            channels,
            target_block_frames,
            &mut dot_prod,
        );

        let similarity = multi_channel_similarity_measure(
            &dot_prod,
            &energy_target_block[0..channels as usize],
            &energy_candidate_blocks[(n * channels) as usize..],
            channels,
        );

        if similarity > best_similarity {
            best_similarity = similarity;
            optimal_index = n;
        }
    }

    optimal_index
}

fn compute_optimal_index(
    search_block: &[*mut f32],
    search_block_frames: i32,
    target_block: &[*mut f32],
    target_block_frames: i32,
    energy_candidate_blocks: &mut [f32],
    channels: i32,
    exclude_interval: Interval,
) -> i32 {
    let num_candidate_blocks = search_block_frames - (target_block_frames - 1);

    // This is a compromise between complexity reduction and search accuracy. I
    // don't have a proof that down sample of order 5 is optimal.
    // One can compute a decimation factor that minimizes complexity given
    // the size of |search_block| and |target_block|. However, my experiments
    // show the rate of missing the optimal index is significant.
    // This value is chosen heuristically based on experiments.
    const SEARCH_DECIMATION: i32 = 5;

    let mut energy_target_block = [0.0; MP_NUM_CHANNELS];

    // Energy of all candid frames.
    multi_channel_moving_block_energies(
        search_block,
        search_block_frames,
        channels,
        target_block_frames,
        energy_candidate_blocks,
    );

    // Energy of target frame.
    multi_channel_dot_product(
        target_block,
        0,
        target_block,
        0,
        channels,
        target_block_frames,
        &mut energy_target_block,
    );

    let optimal_index = decimated_search(
        SEARCH_DECIMATION,
        exclude_interval,
        target_block,
        target_block_frames,
        search_block,
        search_block_frames,
        channels,
        &energy_target_block,
    );

    let lim_low = std::cmp::max(0, optimal_index - SEARCH_DECIMATION);
    let lim_high = std::cmp::min(num_candidate_blocks - 1, optimal_index + SEARCH_DECIMATION);
    full_search(
        lim_low,
        lim_high,
        exclude_interval,
        target_block,
        target_block_frames,
        search_block,
        search_block_frames,
        channels,
        &energy_target_block,
        energy_candidate_blocks,
    )
}

fn peek_buffer(
    p: &mut MpScaletempo2,
    frames: i32,
    read_offset: i32,
    write_offset: i32,
    dest: Vec<Vec<f32>>,
) {
    assert!(p.input_buffer_frames >= frames);
    for i in 0..p.channels {
        unsafe {
            std::ptr::copy_nonoverlapping(
                p.input_buffer[i as usize].add(read_offset as usize),
                dest[i as usize].add(write_offset as usize),
                frames as usize,
            );
        }
    }
}

fn seek_buffer(p: &mut MpScaletempo2, frames: i32) {
    assert!(p.input_buffer_frames >= frames);
    p.input_buffer_frames -= frames;
    if p.input_buffer_final_frames > 0 {
        p.input_buffer_final_frames = MPMAX(0, p.input_buffer_final_frames - frames);
    }
    for i in 0..p.channels {
        unsafe {
            std::ptr::copy(
                p.input_buffer[i as usize].add(frames as usize),
                p.input_buffer[i as usize],
                p.input_buffer_frames as usize,
            );
        }
    }
}

fn write_completed_frames_to(
    p: &mut MpScaletempo2,
    requested_frames: i32,
    dest_offset: i32,
    dest: &mut [*mut f32],
) -> i32 {
    let rendered_frames = MPMIN(p.num_complete_frames, requested_frames);

    if rendered_frames == 0 {
        return 0;
    }

    for i in 0..p.channels {
        unsafe {
            std::ptr::copy_nonoverlapping(
                p.wsola_output[i as usize],
                dest[i as usize].add(dest_offset as usize),
                rendered_frames as usize,
            );
        }
    }

    let frames_to_move = p.wsola_output_size - rendered_frames;
    for k in 0..p.channels {
        let ch = p.wsola_output[k as usize];
        unsafe {
            std::ptr::copy(
                ch.add(rendered_frames as usize),
                ch,
                frames_to_move as usize,
            );
        }
    }
    p.num_complete_frames -= rendered_frames;
    rendered_frames
}

fn get_updated_time(p: &MpScaletempo2, playback_rate: f64) -> f64 {
    p.output_time + p.ola_hop_size * playback_rate
}

fn get_search_block_index(p: &MpScaletempo2, output_time: f64) -> i32 {
    (output_time - p.search_block_center_offset + 0.5) as i32
}

fn frames_needed(p: &MpScaletempo2, playback_rate: f64) -> i32 {
    let search_block_index = get_search_block_index(p, get_updated_time(p, playback_rate));
    MPMAX(
        0,
        MPMAX(
            p.target_block_index + p.ola_window_size - p.input_buffer_frames,
            search_block_index + p.search_block_size - p.input_buffer_frames,
        ),
    )
}

fn can_perform_wsola(p: &MpScaletempo2, playback_rate: f64) -> bool {
    frames_needed(p, playback_rate) <= 0
}

fn resize_input_buffer(p: &mut MpScaletempo2, size: i32) {
    p.input_buffer_size = size;
    let array_size = size_of::<*mut f32>() * size as usize;
    let array_layout = Layout::from_size_align(array_size, 1).unwrap();
    let array = unsafe { alloc(array_layout) as *mut *mut f32 };
    let data = unsafe { array.add(size as usize) as *mut f32 };
    for i in 0..size {
        unsafe {
            *array.add(i as usize) = data.add((i * p.channels) as usize);
        }
    }
    p.input_buffer = array;
}

fn add_input_buffer_final_silence(p: &mut MpScaletempo2, playback_rate: f64) {
    let needed = frames_needed(p, playback_rate);
    if needed <= 0 {
        return;
    }

    let required_size = needed + p.input_buffer_frames;
    if required_size > p.input_buffer_size {
        resize_input_buffer(p, required_size);
    }

    for i in 0..p.channels {
        let ch_input = unsafe {
            slice::from_raw_parts_mut(p.input_buffer[i as usize], p.input_buffer_frames as usize)
        };
        for j in 0..needed {
            ch_input[p.input_buffer_frames as usize + j as usize] = 0.0;
        }
    }

    p.input_buffer_added_silence += needed;
    p.input_buffer_frames += needed;
}

fn mp_scaletempo2_set_final(p: &mut MpScaletempo2) {
    if p.input_buffer_final_frames <= 0 {
        p.input_buffer_final_frames = p.input_buffer_frames;
    }
}

fn mp_scaletempo2_fill_input_buffer(
    p: &mut MpScaletempo2,
    planes: &[*mut u8],
    frame_size: i32,
    playback_rate: f64,
) -> i32 {
    let needed = frames_needed(p, playback_rate);
    let read = MPMIN(needed, frame_size);
    if read == 0 {
        return 0;
    }

    let required_size = read + p.input_buffer_frames;
    if required_size > p.input_buffer_size {
        resize_input_buffer(p, required_size);
    }

    for i in 0..p.channels {
        let ch_input = unsafe {
            slice::from_raw_parts_mut(p.input_buffer[i as usize], p.input_buffer_frames as usize)
        };
        let ch_planes = unsafe { slice::from_raw_parts(planes[i as usize], frame_size as usize) };
        ch_input[p.input_buffer_frames as usize..].copy_from_slice(ch_planes[..read as usize]);
    }

    p.input_buffer_frames += read;
    read
}

fn target_is_within_search_region(p: &MpScaletempo2) -> bool {
    p.target_block_index >= p.search_block_index
        && p.target_block_index + p.ola_window_size <= p.search_block_index + p.search_block_size
}

fn peek_audio_with_zero_prepend(
    p: &mut MpScaletempo2,
    read_offset_frames: i32,
    dest: Vec<Vec<f32>>,
    dest_frames: i32,
) {
    assert!(read_offset_frames + dest_frames <= p.input_buffer_frames);

    let mut write_offset = 0;
    let mut num_frames_to_read = dest_frames;
    if read_offset_frames < 0 {
        let num_zero_frames_appended = MPMIN(-read_offset_frames, num_frames_to_read);
        read_offset_frames = 0;
        num_frames_to_read -= num_zero_frames_appended;
        write_offset = num_zero_frames_appended;
        zero_2d_partial(dest, p.channels, num_zero_frames_appended);
    }
    peek_buffer(
        p,
        num_frames_to_read,
        read_offset_frames,
        write_offset,
        dest,
    );
}

fn get_optimal_block(p: &mut MpScaletempo2) {
    let optimal_index: i32;

    let exclude_interval_length_frames = 160;
    if target_is_within_search_region(p) {
        optimal_index = p.target_block_index;
        peek_audio_with_zero_prepend(p, optimal_index, &mut p.optimal_block, p.ola_window_size);
    } else {
        peek_audio_with_zero_prepend(
            p,
            p.target_block_index,
            &mut p.target_block,
            p.ola_window_size,
        );
        peek_audio_with_zero_prepend(
            p,
            p.search_block_index,
            &mut p.search_block,
            p.search_block_size,
        );
        let last_optimal = p.target_block_index - p.ola_hop_size - p.search_block_index;
        let exclude_iterval = Interval {
            lo: last_optimal - exclude_interval_length_frames / 2,
            hi: last_optimal + exclude_interval_length_frames / 2,
        };

        optimal_index = compute_optimal_index(
            &p.search_block,
            p.search_block_size,
            &p.target_block,
            p.ola_window_size,
            &p.energy_candidate_blocks,
            p.channels,
            exclude_iterval,
        );

        optimal_index += p.search_block_index;
        peek_audio_with_zero_prepend(p, optimal_index, &mut p.optimal_block, p.ola_window_size);
    }

    let last_optimal = p.target_block_index - p.ola_hop_size;
    let transition_window_size = p.optimal_block_size * 2;
    let transition_window = vec![0.0; transition_window_size as usize];
    let transition_window_center = transition_window_size / 2;
    for i in 0..transition_window_size {
        let weight = (i - transition_window_center) as f32 / transition_window_center as f32;
        let weight = 0.5 * (1.0 - weight.cos());
        let target_weight = 1.0 - weight;
        let optimal_weight = weight;
        for k in 0..p.channels {
            let target = unsafe { *p.target_block[k as usize].add(i as usize) };
            let optimal = unsafe { *p.optimal_block[k as usize].add(i as usize) };
            transition_window[i as usize] += target * target_weight + optimal * optimal_weight;
        }
    }

    for i in 0..p.ola_window_size {
        for k in 0..p.channels {
            let target = unsafe { *p.target_block[k as usize].add(i as usize) };
            let optimal = unsafe { *p.optimal_block[k as usize].add(i as usize) };
            let transition = transition_window[i as usize];
            unsafe {
                *p.target_block[k as usize].add(i as usize) =
                    target * (1.0 - transition) + optimal * transition;
            }
        }
    }
}

fn mp_scaletempo2_output(
    p: &mut MpScaletempo2,
    playback_rate: f64,
    frames: i32,
    dest: &mut [*mut f32],
) -> i32 {
    let mut frames_written = 0;
    while frames_written < frames {
        if can_perform_wsola(p, playback_rate) {
            get_optimal_block(p);
            let frames_to_write = MPMIN(
                frames - frames_written,
                p.target_block_size - p.target_block_index,
            );
            for i in 0..p.channels {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        p.target_block[i as usize].add(p.target_block_index as usize),
                        dest[i as usize].add((frames_written + p.output_buffer_frames) as usize),
                        frames_to_write as usize,
                    );
                }
            }
            p.target_block_index += frames_to_write;
            frames_written += frames_to_write;
            p.output_buffer_frames += frames_to_write;
            if p.target_block_index >= p.target_block_size {
                p.target_block_index = 0;
                p.num_complete_frames += p.target_block_size;
            }
        } else {
            let frames_to_write = MPMIN(
                frames - frames_written,
                p.num_complete_frames - p.output_buffer_frames,
            );
            for i in 0..p.channels {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        p.wsola_output[i as usize],
                        dest[i as usize].add((frames_written + p.output_buffer_frames) as usize),
                        frames_to_write as usize,
                    );
                }
            }
            p.output_buffer_frames += frames_to_write;
            frames_written += frames_to_write;
            if p.output_buffer_frames >= p.num_complete_frames {
                p.output_buffer_frames = 0;
                p.num_complete_frames = 0;
            }
        }
    }
    frames_written
}

fn main() {
    let channels = 2;
    let ola_window_size = 1024;
    let ola_hop_size = 256;
    let mut p = mp_scaletempo2_create(channels, ola_window_size, ola_hop_size);
    let playback_rate = 1.0;
    let frames: i32 = 1024;
    let mut dest = vec![null_mut(); channels as usize];
    for i in 0..channels {
        let size = frames as usize * size_of::<f32>();
        let layout = Layout::from_size_align(size, 1).unwrap();
        dest[i as usize] = unsafe { alloc(layout) as *mut f32 };
    }
    let frames_written = mp_scaletempo2_output(&mut p, playback_rate, frames, &mut dest);
    for i in 0..channels {
        unsafe {
            dealloc(
                dest[i as usize] as *mut u8,
                Layout::from_size_align_unchecked(frames as usize * size_of::<f32>(), 1),
            );
        }
    }
    mp_scaletempo2_destroy(&mut p);
}
