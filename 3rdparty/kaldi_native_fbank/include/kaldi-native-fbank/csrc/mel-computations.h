/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// This file is copied/modified from kaldi/src/feat/mel-computations.h
#ifndef KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
#define KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_

#include <cmath>
#include <cstdint>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {
struct FrameExtractionOptions;

struct MelBanksOptions {
  int32_t num_bins = 25;  // e.g. 25; number of triangular bins
  float low_freq = 20;    // e.g. 20; lower frequency cutoff

  // an upper frequency cutoff; 0 -> no cutoff, negative
  // ->added to the Nyquist frequency to get the cutoff.
  float high_freq = 0;

  float vtln_low = 100;  // vtln lower cutoff of warping function.

  // vtln upper cutoff of warping function: if negative, added
  // to the Nyquist frequency to get the cutoff.
  float vtln_high = -500;

  bool debug_mel = false;
  // htk_mode is a "hidden" config, it does not show up on command line.
  // Enables more exact compatibility with HTK, for testing purposes.  Affects
  // mel-energy flooring and reproduces a bug in HTK.
  bool htk_mode = false;

  // Note that if you set is_librosa, you probably need to set
  // low_freq to 0.
  // Please see
  // https://librosa.org/doc/main/generated/librosa.filters.mel.html
  bool is_librosa = false;

  // used only when is_librosa=true
  // Possible values: "", slaney. We don't support a numeric value here, but
  // it can be added on demand.
  // See https://librosa.org/doc/main/generated/librosa.filters.mel.html
  std::string norm = "slaney";

  std::string ToString() const {
    std::ostringstream os;
    os << "num_bins: " << num_bins << "\n";
    os << "low_freq: " << low_freq << "\n";
    os << "high_freq: " << high_freq << "\n";
    os << "vtln_low: " << vtln_low << "\n";
    os << "vtln_high: " << vtln_high << "\n";
    os << "debug_mel: " << debug_mel << "\n";
    os << "htk_mode: " << htk_mode << "\n";
    os << "is_librosa: " << is_librosa << "\n";
    os << "norm: " << norm << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const MelBanksOptions &opts);

class MelBanks {
 public:
  // see also https://en.wikipedia.org/wiki/Mel_scale
  // htk, mel to hz
  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  // htk, hz to mel
  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  // slaney, mel to hz
  static inline float InverseMelScaleSlaney(float mel_freq) {
    if (mel_freq <= 15) {
      return 200.0f / 3 * mel_freq;
    }

    // return 1000 * expf((mel_freq - 15) * logf(6.4f) / 27);

    // Note: log(6.4)/27 = 0.06875177742094911

    return 1000 * expf((mel_freq - 15) * 0.06875177742094911f);
  }

  // slaney, hz to mel
  static inline float MelScaleSlaney(float freq) {
    if (freq <= 1000) {
      return freq * 3 / 200.0f;
    }

    // return 15 + 27 * logf(freq / 1000) / logf(6.4f)
    //
    // Note: 27/log(6.4) = 14.545078505785561

    return 15 + 14.545078505785561f * logf(freq / 1000);
  }

  static float VtlnWarpFreq(
      float vtln_low_cutoff,
      float vtln_high_cutoff,  // discontinuities in warp func
      float low_freq,
      float high_freq,  // upper+lower frequency cutoffs in
      // the mel computation
      float vtln_warp_factor, float freq);

  static float VtlnWarpMelFreq(float vtln_low_cutoff, float vtln_high_cutoff,
                               float low_freq, float high_freq,
                               float vtln_warp_factor, float mel_freq);

  // TODO(fangjun): Remove vtln_warp_factor
  MelBanks(const MelBanksOptions &opts,
           const FrameExtractionOptions &frame_opts, float vtln_warp_factor);

  // Initialize with a 2-d weights matrix
  // @param weights Pointer to the start address of the matrix
  // @param num_rows It equls to number of mel bins
  // @param num_cols It equals to (number of fft bins)/2+1
  MelBanks(const float *weights, int32_t num_rows, int32_t num_cols);

  /// Compute Mel energies (note: not log energies).
  /// At input, "fft_energies" contains the FFT energies (not log).
  ///
  /// @param fft_energies 1-D array of size num_fft_bins/2+1
  /// @param mel_energies_out  1-D array of size num_mel_bins
  void Compute(const float *fft_energies, float *mel_energies_out) const;

  int32_t NumBins() const { return bins_.size(); }

 private:
  // for kaldi-compatible
  void InitKaldiMelBanks(const MelBanksOptions &opts,
                         const FrameExtractionOptions &frame_opts,
                         float vtln_warp_factor);

  // for librosa-compatible
  // See https://librosa.org/doc/main/generated/librosa.filters.mel.html
  void InitLibrosaMelBanks(const MelBanksOptions &opts,
                           const FrameExtractionOptions &frame_opts,
                           float vtln_warp_factor);

 private:
  // the "bins_" vector is a vector, one for each bin, of a pair:
  // (the first nonzero fft-bin), (the vector of weights).
  std::vector<std::pair<int32_t, std::vector<float>>> bins_;

  // TODO(fangjun): Remove debug_ and htk_mode_
  bool debug_ = false;
  bool htk_mode_ = false;
};

// Compute liftering coefficients (scaling on cepstral coeffs)
// coeffs are numbered slightly differently from HTK: the zeroth
// index is C0, which is not affected.
void ComputeLifterCoeffs(float Q, std::vector<float> *coeffs);

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_MEL_COMPUTATIONS_H_
