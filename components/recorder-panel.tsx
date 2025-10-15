"use client"

import { useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { denoiseWiener, encodeWavPCM16 } from "@/lib/audio-denoise"

type Status = "idle" | "recording" | "recorded" | "processing" | "processed" | "error"

export default function RecorderPanel() {
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const [status, setStatus] = useState<Status>("idle")
  const [error, setError] = useState<string | null>(null)

  const [origBlob, setOrigBlob] = useState<Blob | null>(null)
  const [origUrl, setOrigUrl] = useState<string | null>(null)

  const [refinedBlob, setRefinedBlob] = useState<Blob | null>(null)
  const [refinedUrl, setRefinedUrl] = useState<string | null>(null)

  const audioCtxRef = useRef<AudioContext | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const pcmChunksRef = useRef<Float32Array[]>([])
  const totalSamplesRef = useRef(0)
  const sampleRateRef = useRef(48000)
  const origPcmRef = useRef<Float32Array | null>(null)
  // level for subtle mic pulse
  const [level, setLevel] = useState(0)

  useEffect(() => {
    return () => {
      if (origUrl) URL.revokeObjectURL(origUrl)
      if (refinedUrl) URL.revokeObjectURL(refinedUrl)
    }
  }, [origUrl, refinedUrl])

  async function startRecording() {
    setError(null)
    setRefinedBlob(null)
    setRefinedUrl(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          noiseSuppression: true,
          echoCancellation: true,
          autoGainControl: false,
          sampleRate: 48000,
        },
        video: false,
      })

      // Create AudioContext; browser may not honor requested rate exactly.
      const ac = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 })
      sampleRateRef.current = ac.sampleRate
      audioCtxRef.current = ac
      streamRef.current = stream

      const source = ac.createMediaStreamSource(stream)
      sourceRef.current = source

      // Use ScriptProcessor for widest support. Buffer of 2048 gives a decent latency/CPU balance.
      const proc = ac.createScriptProcessor(2048, 1, 1)
      procRef.current = proc

      // Reset buffers
      pcmChunksRef.current = []
      totalSamplesRef.current = 0
      origPcmRef.current = null
      setOrigBlob(null)
      setOrigUrl(null)

      let rafPending = false
      proc.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0)
        // Copy to avoid re-use of the internal buffer
        const copy = new Float32Array(input.length)
        copy.set(input)
        pcmChunksRef.current.push(copy)
        totalSamplesRef.current += copy.length

        // simple level meter (RMS) throttled with rAF
        if (!rafPending) {
          rafPending = true
          requestAnimationFrame(() => {
            rafPending = false
            let sum = 0
            for (let i = 0; i < input.length; i++) {
              const v = input[i]
              sum += v * v
            }
            const rms = Math.sqrt(sum / (input.length || 1))
            setLevel(Math.min(1, rms * 4)) // amplify for display
          })
        }
      }

      // Must connect to destination for ScriptProcessor to run.
      source.connect(proc)
      proc.connect(ac.destination)

      setStatus("recording")
    } catch (e: any) {
      setError(e?.message || "Failed to start recording")
      setStatus("error")
    }
  }

  function stopRecording() {
    try {
      if (procRef.current) {
        try {
          procRef.current.disconnect()
        } catch {}
        procRef.current.onaudioprocess = null as any
      }
      if (sourceRef.current) {
        try {
          sourceRef.current.disconnect()
        } catch {}
      }

      // Close audio context
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        audioCtxRef.current.close().catch(() => {})
      }

      // Stop tracks to release the mic
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
      }

      // Build mono PCM from chunks
      const total = totalSamplesRef.current
      const pcm = new Float32Array(total)
      let offset = 0
      for (const chunk of pcmChunksRef.current) {
        pcm.set(chunk, offset)
        offset += chunk.length
      }
      origPcmRef.current = pcm

      // Also present an "Original" player by encoding to WAV
      const wavBlob = encodeWavPCM16(pcm, sampleRateRef.current)
      const url = URL.createObjectURL(wavBlob)
      setOrigBlob(wavBlob)
      setOrigUrl(url)

      setStatus("recorded")
    } catch (e: any) {
      setError(e?.message || "Failed to stop recording")
      setStatus("error")
    }
  }

  async function refineAudio() {
    setError(null)
    setStatus("processing")
    try {
      let mono: Float32Array
      let sr: number

      if (origPcmRef.current) {
        mono = origPcmRef.current
        sr = sampleRateRef.current
      } else if (origBlob) {
        // Fallback: decode from blob if PCM path not available
        const arrayBuf = await origBlob.arrayBuffer()
        const ac = new AudioContext({ sampleRate: 48000 })
        const decoded = await ac.decodeAudioData(arrayBuf.slice(0))
        const ch0 = decoded.getChannelData(0)
        if (decoded.numberOfChannels > 1) {
          const ch1 = decoded.getChannelData(1)
          const N = Math.min(ch0.length, ch1.length)
          mono = new Float32Array(N)
          for (let i = 0; i < N; i++) mono[i] = 0.5 * (ch0[i] + ch1[i])
        } else {
          mono = ch0.slice(0)
        }
        sr = decoded.sampleRate
        await ac.close()
      } else {
        throw new Error("No audio to process")
      }

      // Denoise (decision-directed Wiener)
      const enhanced = await denoiseWiener(mono, sr, {
        frameMs: 25,
        hopMs: 10,
        nfft: 512,
        alpha: 0.98,
        gMin: 0.01,
      })

      // Encode to WAV (PCM16)
      const wavBlob = encodeWavPCM16(enhanced, sr)
      const url = URL.createObjectURL(wavBlob)
      setRefinedBlob(wavBlob)
      setRefinedUrl(url)
      setStatus("processed")
    } catch (e: any) {
      console.log("[v0] refineAudio error:", e?.message || e)
      setError(e?.message || "Failed to process audio")
      setStatus("error")
    }
  }

  return (
    <div className="grid gap-6">
      {/* Mic circle */}
      <div className="flex items-center justify-center">
        <button
          onClick={() => (status === "recording" ? stopRecording() : startRecording())}
          aria-label={status === "recording" ? "Stop recording" : "Start recording"}
          className="relative h-32 w-32 rounded-full outline-none ring-0 select-none"
          style={{
            // animated conic disco ring using theme tokens
            backgroundImage:
              "conic-gradient(from 0deg at 50% 50%, var(--color-primary), var(--color-accent), var(--color-primary))",
            filter: "saturate(120%)",
            boxShadow:
              "0 0 0 4px color-mix(in oklch, var(--color-background) 85%, transparent), 0 0 24px var(--color-accent), 0 0 48px color-mix(in oklch, var(--color-accent) 50%, transparent)",
          }}
        >
          {/* rotating halo */}
          <div className="absolute inset-0 rounded-full animate-[spin_6s_linear_infinite]" />
          {/* inner dark disc */}
          <div
            className="absolute inset-1 rounded-full flex items-center justify-center transition-transform duration-150"
            style={{
              background:
                "radial-gradient(60% 60% at 50% 50%, color-mix(in oklch, var(--color-background) 85%, transparent) 0%, var(--color-background) 70%)",
              transform: `scale(${1 + Math.min(0.15, level * 0.15)})`,
            }}
          >
            {/* mic glyph */}
            <div
              className={`h-10 w-10 rounded-[12px] ${status === "recording" ? "bg-destructive" : "bg-primary"}`}
              style={{
                boxShadow:
                  status === "recording"
                    ? "0 0 18px color-mix(in oklch, var(--color-destructive) 70%, transparent)"
                    : "0 0 18px color-mix(in oklch, var(--color-primary) 70%, transparent)",
              }}
            />
          </div>
        </button>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button
          onClick={startRecording}
          disabled={status === "recording"}
          className="bg-primary text-primary-foreground hover:opacity-90"
        >
          Start Recording
        </Button>
        <Button
          variant="secondary"
          onClick={stopRecording}
          disabled={status !== "recording"}
          className="bg-secondary text-secondary-foreground hover:opacity-90"
        >
          Stop Recording
        </Button>
        <Button
          variant="default"
          onClick={refineAudio}
          disabled={status !== "recorded"}
          className="bg-accent text-accent-foreground hover:opacity-90"
        >
          Refine Audio
        </Button>
      </div>

      {/* Players */}
      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border p-4">
          <h3 className="font-medium mb-2">Original</h3>
          {status === "recording" ? (
            <p className="text-muted-foreground">Recording in progress…</p>
          ) : origUrl ? (
            <audio controls src={origUrl} className="w-full" />
          ) : (
            <p className="text-muted-foreground">No recording yet.</p>
          )}
        </div>

        <div className="rounded-lg border p-4">
          <h3 className="font-medium mb-2">Refined (Noise Reduced)</h3>
          {status === "processing" ? (
            <p className="text-muted-foreground">Processing… please wait.</p>
          ) : refinedUrl ? (
            <div className="grid gap-3">
              <audio controls src={refinedUrl} className="w-full" />
              <div className="flex gap-2">
                <Button asChild className="bg-primary text-primary-foreground hover:opacity-90">
                  <a download="refined.wav" href={refinedUrl}>
                    Download Refined Audio
                  </a>
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => {
                    const a = new Audio(refinedUrl!)
                    a.play().catch(() => {})
                  }}
                  className="bg-secondary text-secondary-foreground hover:opacity-90"
                >
                  Play Refined Audio
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">No refined audio yet.</p>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-md border border-destructive/30 p-3 text-sm">
          <p className="text-destructive">Error: {error}</p>
        </div>
      )}
    </div>
  )
}
