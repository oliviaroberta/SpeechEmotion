import { afterEach, describe, expect, it, vi } from 'vitest'
import { downmix, encodeMonoPcm16Wav, measureRecording, RecordingError, RecordingSession, validateRecordingQuality } from './audioRecording'

afterEach(() => vi.restoreAllMocks())

describe('browser WAV encoding', () => {
  it('downmixes channels using their arithmetic mean', () => { expect(Array.from(downmix([new Float32Array([1, -1]), new Float32Array([-1, 1])]))).toEqual([0, 0]) })
  it('encodes a mono PCM-16 RIFF/WAVE file with consistent metadata', async () => { const blob = encodeMonoPcm16Wav(new Float32Array([0, .5, -.5]), 16000); const view = new DataView(await blob.arrayBuffer()); const text = (offset: number) => String.fromCharCode(...new Uint8Array(view.buffer.slice(offset, offset + 4))); expect(text(0)).toBe('RIFF'); expect(text(8)).toBe('WAVE'); expect(view.getUint16(20, true)).toBe(1); expect(view.getUint16(22, true)).toBe(1); expect(view.getUint32(24, true)).toBe(16000); expect(view.getUint16(34, true)).toBe(16); expect(view.getUint32(40, true)).toBe(6); expect(blob.type).toBe('audio/wav') })
  it('preserves non-zero captured samples and creates different WAV bytes for different signals', async () => {
    const first = new Float32Array(16000).fill(0.1)
    const second = new Float32Array(16000).fill(0.2)
    const firstBuffer = await encodeMonoPcm16Wav(first, 16000).arrayBuffer()
    const secondBuffer = await encodeMonoPcm16Wav(second, 16000).arrayBuffer()
    const firstBytes = new Uint8Array(firstBuffer)
    const secondBytes = new Uint8Array(secondBuffer)
    expect(firstBytes.slice(44)).not.toEqual(new Uint8Array(firstBytes.length - 44))
    expect(Array.from(firstBytes)).not.toEqual(Array.from(secondBytes))
    const hash = async (bytes: Uint8Array) => {
      const input = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer
      return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', input))).map((value) => value.toString(16).padStart(2, '0')).join('')
    }
    expect(await hash(firstBytes)).not.toBe(await hash(secondBytes))
    const firstQuality = measureRecording(first, 16000)
    const secondQuality = measureRecording(second, 16000)
    expect(firstQuality.sampleCount).toBe(16000)
    expect(firstQuality.duration).toBe(1)
    expect(firstQuality.peak).toBeCloseTo(0.1)
    expect(firstQuality.rms).toBeCloseTo(0.1)
    expect(secondQuality.peak).toBeCloseTo(0.2)
    expect(secondQuality.rms).toBeCloseTo(0.2)
  })
  it('rejects silent or effectively silent recordings before API submission', () => {
    expect(() => validateRecordingQuality(measureRecording(new Float32Array(16000), 16000))).toThrow(/no clear microphone audio/i)
    expect(() => validateRecordingQuality(measureRecording(new Float32Array(16000).fill(0.0001), 16000))).toThrow(/no clear microphone audio/i)
  })
  it('joins captured chunks in order and writes the matching PCM payload length', async () => {
    const session = new RecordingSession() as unknown as { stopped: boolean; chunks: Float32Array[]; sampleRate: number; stop: () => Promise<File> }
    session.stopped = false
    session.chunks = [new Float32Array(8000).fill(0.1), new Float32Array(8000).fill(0.2)]
    session.sampleRate = 16000
    const view = new DataView(await (await session.stop()).arrayBuffer())
    expect(view.getUint32(40, true)).toBe(32000)
    expect(view.getInt16(44, true)).toBe(Math.round(0.1 * 32767))
    expect(view.getInt16(44 + (8000 * 2), true)).toBe(Math.round(0.2 * 32767))
  })
  it('rejects a recording shorter than the minimum duration', async () => { const session = new RecordingSession() as unknown as { stopped: boolean; chunks: Float32Array[]; sampleRate: number; stop: () => Promise<File> }; session.stopped = false; session.chunks = [new Float32Array(15999)]; session.sampleRate = 16000; await expect(session.stop()).rejects.toBeInstanceOf(RecordingError) })
  it('stops tracks, disconnects nodes, closes the context, and builds a WAV after stopping', async () => { const stop = vi.fn(); const disconnect = vi.fn(); const close = vi.fn().mockResolvedValue(undefined); const session = new RecordingSession() as unknown as { stopped: boolean; chunks: Float32Array[]; sampleRate: number; stream: { getTracks: () => { stop: () => void }[] }; node: { disconnect: () => void }; source: { disconnect: () => void }; context: { state: string; close: () => Promise<void> }; stop: () => Promise<File> }; session.stopped = false; session.chunks = [new Float32Array(16000).fill(0.1)]; session.sampleRate = 16000; session.stream = { getTracks: () => [{ stop }] }; session.node = { disconnect }; session.source = { disconnect }; session.context = { state: 'running', close }; const file = await session.stop(); expect(file.type).toBe('audio/wav'); expect(stop).toHaveBeenCalled(); expect(disconnect).toHaveBeenCalledTimes(2); expect(close).toHaveBeenCalled() })
  it('reports microphone permission denial without retaining a session', async () => { const previous = navigator.mediaDevices; const previousContext = window.AudioContext; const previousNode = window.AudioWorkletNode; Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: { getUserMedia: vi.fn().mockRejectedValue(new DOMException('', 'NotAllowedError')) } }); Object.defineProperty(window, 'AudioContext', { configurable: true, value: class {} }); Object.defineProperty(window, 'AudioWorkletNode', { configurable: true, value: class {} }); await expect(new RecordingSession().start(() => undefined)).rejects.toThrow(/permission was denied/i); Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: previous }); Object.defineProperty(window, 'AudioContext', { configurable: true, value: previousContext }); Object.defineProperty(window, 'AudioWorkletNode', { configurable: true, value: previousNode }) })
})
