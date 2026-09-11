export const MAX_RECORDING_SECONDS = 15
export const MIN_RECORDING_SECONDS = 1
export const MAX_AUDIO_BYTES = 4 * 1024 * 1024
export const MIN_PEAK_AMPLITUDE = 0.002
export const MIN_RMS_AMPLITUDE = 0.0005

const workletSource = `class MonoCaptureProcessor extends AudioWorkletProcessor { constructor(){ super(); this.port.onmessage=(event)=>{ if(event.data.type==='flush') this.port.postMessage({type:'flush-complete'}); }; } process(inputs){ const channels=inputs[0]; if(channels&&channels.length) this.port.postMessage({type:'samples',channels:channels.map((channel)=>new Float32Array(channel))}); return true; } } registerProcessor('mono-capture-processor',MonoCaptureProcessor);`
type WorkletMessage = { type: 'samples'; channels: Float32Array[] } | { type: 'flush-complete' }

export interface RecordingQuality { duration: number; sampleCount: number; peak: number; rms: number }
export class RecordingError extends Error {}

export function downmix(channels: Float32Array[]): Float32Array {
  if (!channels.length) return new Float32Array()
  const output = new Float32Array(channels[0].length)
  channels.forEach((channel) => { for (let index = 0; index < output.length; index += 1) output[index] += channel[index] / channels.length })
  return output
}

export function measureRecording(samples: Float32Array, sampleRate: number): RecordingQuality {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !samples.length) throw new RecordingError('No microphone samples were captured.')
  let peak = 0; let sumSquares = 0
  for (const sample of samples) { if (!Number.isFinite(sample)) throw new RecordingError('The microphone recording contains invalid audio samples.'); const absolute = Math.abs(sample); peak = Math.max(peak, absolute); sumSquares += sample * sample }
  return { duration: samples.length / sampleRate, sampleCount: samples.length, peak, rms: Math.sqrt(sumSquares / samples.length) }
}

export function validateRecordingQuality(quality: RecordingQuality): void {
  if (quality.duration < MIN_RECORDING_SECONDS) throw new RecordingError('Record at least one second of audio before analysing.')
  if (quality.peak < MIN_PEAK_AMPLITUDE || quality.rms < MIN_RMS_AMPLITUDE) throw new RecordingError('No clear microphone audio was detected. Check the microphone and record again.')
}

export function encodeMonoPcm16Wav(samples: Float32Array, sampleRate: number): Blob {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0 || !samples.length) throw new RecordingError('Recording samples are invalid.')
  const dataBytes = samples.length * 2; const buffer = new ArrayBuffer(44 + dataBytes); const view = new DataView(buffer)
  const text = (offset: number, value: string) => { for (let index = 0; index < value.length; index += 1) view.setUint8(offset + index, value.charCodeAt(index)) }
  text(0, 'RIFF'); view.setUint32(4, 36 + dataBytes, true); text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, dataBytes, true)
  for (let index = 0; index < samples.length; index += 1) {
    if (!Number.isFinite(samples[index])) throw new RecordingError('Recording samples are invalid.')
    view.setInt16(44 + index * 2, Math.round(Math.max(-1, Math.min(1, samples[index])) * 32767), true)
  }
  return new Blob([buffer], { type: 'audio/wav' })
}

export class RecordingSession {
  private context: AudioContext | null = null; private stream: MediaStream | null = null; private source: MediaStreamAudioSourceNode | null = null; private node: AudioWorkletNode | null = null; private timer: number | null = null; private chunks: Float32Array[] = []; private stopped = false; private flushResolver: (() => void) | null = null
  sampleRate = 0; lastQuality: RecordingQuality | null = null

  async start(onMaximumDuration: () => void): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) throw new RecordingError('No microphone device is available in this browser.')
    const Context = window.AudioContext ?? (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Context || !window.AudioWorkletNode) throw new RecordingError('AudioWorklet recording is not supported by this browser.')
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, autoGainControl: false } }); this.context = new Context(); this.sampleRate = this.context.sampleRate
      if (!this.context.audioWorklet) throw new RecordingError('AudioWorklet recording is not supported by this browser.')
      const moduleUrl = URL.createObjectURL(new Blob([workletSource], { type: 'application/javascript' })); try { await this.context.audioWorklet.addModule(moduleUrl) } finally { URL.revokeObjectURL(moduleUrl) }
      this.source = this.context.createMediaStreamSource(this.stream); this.node = new AudioWorkletNode(this.context, 'mono-capture-processor')
      this.node.port.onmessage = (event: MessageEvent<WorkletMessage>) => { if (event.data.type === 'flush-complete') { this.flushResolver?.(); this.flushResolver = null; return }; this.chunks.push(downmix(event.data.channels.map((channel) => new Float32Array(channel)))) }
      this.source.connect(this.node); this.node.connect(this.context.destination); this.stream.getTracks().forEach((track) => { track.onended = () => { if (!this.stopped) onMaximumDuration() } }); this.timer = window.setTimeout(onMaximumDuration, MAX_RECORDING_SECONDS * 1000)
    } catch (error) { await this.cleanup(); if (error instanceof RecordingError) throw error; if (error instanceof DOMException && error.name === 'NotAllowedError') throw new RecordingError('Microphone permission was denied.'); throw new RecordingError('Microphone recording could not be started.') }
  }

  async stop(): Promise<File> {
    if (this.stopped) throw new RecordingError('Recording is no longer active.')
    this.stopped = true
    try {
      await this.flushWorklet()
      const samples = this.joinChunks()
      this.lastQuality = measureRecording(samples, this.sampleRate)
      validateRecordingQuality(this.lastQuality)
      const wav = encodeMonoPcm16Wav(samples, this.sampleRate)
      if (wav.size > MAX_AUDIO_BYTES) throw new RecordingError('The recorded WAV exceeds the 4 MiB limit.')
      return new File([wav], 'microphone-recording.wav', { type: 'audio/wav' })
    } finally {
      await this.cleanup()
    }
  }

  async cleanup(): Promise<void> { if (this.timer !== null) window.clearTimeout(this.timer); this.timer = null; this.node?.disconnect(); this.source?.disconnect(); this.stream?.getTracks().forEach((track) => track.stop()); this.node = null; this.source = null; this.stream = null; if (this.context && this.context.state !== 'closed') await this.context.close(); this.context = null }
  private async flushWorklet(): Promise<void> {
    if (!this.node?.port) return
    await new Promise<void>((resolve) => {
      const timeout = window.setTimeout(() => {
        this.flushResolver = null
        resolve()
      }, 250)
      this.flushResolver = () => {
        window.clearTimeout(timeout)
        resolve()
      }
      this.node?.port.postMessage({ type: 'flush' })
    })
  }
  private joinChunks(): Float32Array { const length = this.chunks.reduce((sum, chunk) => sum + chunk.length, 0); const output = new Float32Array(length); let offset = 0; this.chunks.forEach((chunk) => { output.set(chunk, offset); offset += chunk.length }); return output }
}
