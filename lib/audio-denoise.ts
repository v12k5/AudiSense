import FFT from "fft.js"

type DenoiseOptions = {
  frameMs?: number
  hopMs?: number
  nfft?: number
  alpha?: number
  gMin?: number
}

const EPS = 1e-10

function nextPow2(x: number) {
  let p = 1
  while (p < x) p <<= 1
  return p
}

function hamming(N: number) {
  const w = new Float64Array(N)
  for (let n = 0; n < N; n++) {
    w[n] = 0.54 - 0.46 * Math.cos((2 * Math.PI * n) / (N - 1))
  }
  return w
}

function frameSignal(sig: Float32Array | Float64Array, frameSize: number, hop: number) {
  const n = sig.length
  if (n < frameSize) {
    return new Float64Array(0)
  }
  const numFrames = 1 + Math.floor((n - frameSize) / hop)
  const frames = new Float64Array(numFrames * frameSize)
  for (let i = 0; i < numFrames; i++) {
    const start = i * hop
    for (let j = 0; j < frameSize; j++) {
      frames[i * frameSize + j] = sig[start + j]
    }
  }
  return frames
}

function computeSTFT(frames: Float64Array, frameLen: number, hop: number, window: Float64Array, nfft: number) {
  const numFrames = Math.floor(frames.length / frameLen)
  const fft = new FFT(nfft)
  const outFrames = Array<Float64Array>(numFrames)
  for (let m = 0; m < numFrames; m++) {
    const start = m * frameLen
    // prepare complex input
    const input = fft.createComplexArray() as Float64Array
    for (let i = 0; i < nfft; i++) {
      const re = i < frameLen ? frames[start + i] * window[i] : 0
      input[2 * i] = re
      input[2 * i + 1] = 0
    }
    const out = fft.createComplexArray() as Float64Array
    fft.transform(out, input)
    outFrames[m] = out // length = 2 * nfft (interleaved complex)
  }
  return outFrames
}

function powerSpectrum(cspec: Float64Array, nfft: number, out?: Float64Array) {
  const p = out ?? new Float64Array(nfft)
  for (let k = 0; k < nfft; k++) {
    const re = cspec[2 * k]
    const im = cspec[2 * k + 1]
    p[k] = re * re + im * im
  }
  return p
}

function energyVAD(framed: Float64Array, frameLen: number, window: Float64Array, thresholdScale = 0.5) {
  const numFrames = Math.floor(framed.length / frameLen)
  const energy = new Float64Array(numFrames)
  for (let m = 0; m < numFrames; m++) {
    let acc = 0
    for (let i = 0; i < frameLen; i++) {
      const v = framed[m * frameLen + i] * window[i]
      acc += v * v
    }
    energy[m] = acc / frameLen
  }
  // median and std
  const sorted = Array.from(energy).sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  const median = sorted.length % 2 === 0 ? 0.5 * (sorted[mid - 1] + sorted[mid]) : sorted[mid]
  let mean = 0
  for (let i = 0; i < energy.length; i++) mean += energy[i]
  mean /= energy.length || 1
  let varsum = 0
  for (let i = 0; i < energy.length; i++) {
    const d = energy[i] - mean
    varsum += d * d
  }
  const std = Math.sqrt(varsum / (energy.length || 1))
  const thresh = median + thresholdScale * std
  const isSpeech = new Array<boolean>(numFrames)
  for (let i = 0; i < numFrames; i++) isSpeech[i] = energy[i] > thresh
  return { isSpeech, energy, thresh }
}

function estimateNoisePSD(stftFrames: Float64Array[], isSpeech: boolean[], nfft: number) {
  const numFrames = stftFrames.length
  const noisePower = new Float64Array(nfft)
  let noiseCount = 0
  const tmp = new Float64Array(nfft)
  for (let m = 0; m < numFrames; m++) {
    if (!isSpeech[m]) {
      powerSpectrum(stftFrames[m], nfft, tmp)
      for (let k = 0; k < nfft; k++) noisePower[k] += tmp[k]
      noiseCount++
    }
  }
  if (noiseCount === 0) {
    // Fallback: take minimum over time
    // First compute all powers then min
    const minPow = new Float64Array(nfft).fill(Number.POSITIVE_INFINITY)
    for (let m = 0; m < numFrames; m++) {
      powerSpectrum(stftFrames[m], nfft, tmp)
      for (let k = 0; k < nfft; k++) if (tmp[k] < minPow[k]) minPow[k] = tmp[k]
    }
    for (let k = 0; k < nfft; k++) noisePower[k] = Math.max(minPow[k], EPS)
  } else {
    for (let k = 0; k < nfft; k++) noisePower[k] = Math.max(noisePower[k] / noiseCount, EPS)
  }
  return noisePower
}

function wienerFilterDecisionDirected(
  stftFrames: Float64Array[],
  noisePSD: Float64Array,
  nfft: number,
  alpha = 0.98,
  gMin = 0.01,
) {
  const numFrames = stftFrames.length
  const enhanced: Float64Array[] = new Array(numFrames)
  const Sprev = new Float64Array(2 * nfft) // complex prev estimate

  for (let m = 0; m < numFrames; m++) {
    const X = stftFrames[m]
    const S = new Float64Array(2 * nfft)

    for (let k = 0; k < nfft; k++) {
      const xr = X[2 * k],
        xi = X[2 * k + 1]
      const powerX = xr * xr + xi * xi
      const np = noisePSD[k] + EPS

      const spr = Sprev[2 * k],
        spi = Sprev[2 * k + 1]
      const prevPow = spr * spr + spi * spi

      const gamma = powerX / np // a posteriori SNR
      const xiSnr = alpha * (prevPow / np) + (1 - alpha) * Math.max(gamma - 1.0, 0.0)
      const G = Math.max(xiSnr / (1.0 + xiSnr), gMin)

      S[2 * k] = G * xr
      S[2 * k + 1] = G * xi
    }

    enhanced[m] = S
    Sprev.set(S)
  }

  return enhanced
}

function istftOverlapAdd(
  stftFrames: Float64Array[],
  frameLen: number,
  hop: number,
  window: Float64Array,
  nfft: number,
) {
  const numFrames = stftFrames.length
  const fft = new FFT(nfft)
  const outLen = (numFrames - 1) * hop + frameLen
  const out = new Float64Array(outLen)
  const winSum = new Float64Array(outLen)

  const input = fft.createComplexArray() as Float64Array
  const time = new Float64Array(nfft)

  for (let m = 0; m < numFrames; m++) {
    const S = stftFrames[m]
    // inverse FFT
    // NOTE: fft.js expects separate input and output; inverseTransform(out, data)
    // We'll place S into a temp array and inverse into input (reusing buffers).
    const invOut = fft.createComplexArray() as Float64Array
    fft.inverseTransform(invOut, S)

    // Real part is the time signal; normalize by nfft (fft.js inverse is not guaranteed to scale; this keeps things safe)
    // Many FFT conventions differ; we normalize later overall anyway.
    for (let i = 0; i < nfft; i++) {
      time[i] = invOut[2 * i] // ignore imag residuals
    }

    const start = m * hop
    for (let i = 0; i < frameLen; i++) {
      out[start + i] += time[i] * window[i]
      winSum[start + i] += window[i] * window[i]
    }
  }

  for (let i = 0; i < outLen; i++) {
    if (winSum[i] > EPS) out[i] /= winSum[i]
  }
  return out
}

export async function denoiseWiener(input: Float32Array, sampleRate: number, opts: DenoiseOptions = {}) {
  const frameMs = opts.frameMs ?? 25
  const hopMs = opts.hopMs ?? 10
  let nfft = opts.nfft ?? 512
  const alpha = opts.alpha ?? 0.98
  const gMin = opts.gMin ?? 0.01

  // Normalize input to [-1, 1]
  let maxAbs = 1e-6
  for (let i = 0; i < input.length; i++) {
    const a = Math.abs(input[i])
    if (a > maxAbs) maxAbs = a
  }
  const norm = new Float64Array(input.length)
  for (let i = 0; i < input.length; i++) norm[i] = input[i] / (maxAbs + EPS)

  const frameLen = Math.round((frameMs / 1000) * sampleRate)
  const hop = Math.round((hopMs / 1000) * sampleRate)
  if (nfft < frameLen) nfft = nextPow2(frameLen)

  const window = hamming(frameLen)
  const framed = frameSignal(norm, frameLen, hop)
  if (framed.length === 0) {
    // Not enough samples for one frame; return original
    return input
  }

  const stftFrames = computeSTFT(framed, frameLen, hop, window, nfft) // complex frames
  const { isSpeech } = energyVAD(framed, frameLen, window, 0.5)
  const noisePSD = estimateNoisePSD(stftFrames, isSpeech, nfft)
  const enhancedSpec = wienerFilterDecisionDirected(stftFrames, noisePSD, nfft, alpha, gMin)
  const enhanced = istftOverlapAdd(enhancedSpec, frameLen, hop, window, nfft)

  // Align length to input; re-normalize to [-1, 1]
  const N = Math.min(enhanced.length, input.length)
  let maxE = 0
  for (let i = 0; i < N; i++) if (Math.abs(enhanced[i]) > maxE) maxE = Math.abs(enhanced[i])
  const out = new Float32Array(N)
  const scale = maxE > 0 ? 1 / (maxE + EPS) : 1
  for (let i = 0; i < N; i++) out[i] = Math.max(-1, Math.min(1, enhanced[i] * scale))
  return out
}

// Encode a Float32Array [-1,1] PCM to 16-bit WAV
export function encodeWavPCM16(samples: Float32Array, sampleRate: number) {
  const numChannels = 1
  const bytesPerSample = 2
  const blockAlign = numChannels * bytesPerSample
  const byteRate = sampleRate * blockAlign
  const dataSize = samples.length * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, "RIFF")
  view.setUint32(4, 36 + dataSize, true)
  writeString(view, 8, "WAVE")

  // fmt chunk
  writeString(view, 12, "fmt ")
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, numChannels, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true) // bits per sample

  // data chunk
  writeString(view, 36, "data")
  view.setUint32(40, dataSize, true)

  // PCM samples
  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }

  return new Blob([view], { type: "audio/wav" })
}

function writeString(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i))
  }
}
