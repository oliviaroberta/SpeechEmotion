import { afterEach, describe, expect, it, vi } from 'vitest'
import { PredictionApiError, predictAudio } from './predictionApi'

const file = new File([new Uint8Array([1])], 'voice.wav', { type: 'audio/wav' })
const response = { emotion: 'happy', class_index: 2, confidence: .7, probabilities: { neutral: .02, calm: .03, happy: .7, sad: .04, angry: .05, fearful: .04, disgust: .03, surprised: .09 }, model: { identifier: 'cnn', version: '1.0.0' } }
afterEach(() => vi.unstubAllEnvs())

describe('prediction API client', () => {
  it('posts FormData to the configured endpoint without a manual content type', async () => { vi.stubEnv('VITE_API_BASE_URL', 'http://api.test/'); const request = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 })); const result = await predictAudio(file, request); expect(result.emotion).toBe('happy'); expect(request).toHaveBeenCalledWith('http://api.test/api/v1/predict', expect.objectContaining({ method: 'POST' })); const options = request.mock.calls[0][1]; expect(options.headers).toBeUndefined(); expect(options.body.get('file')).toBe(file) })
  it('uses the safe fallback and rejects malformed output', async () => { const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...response, probabilities: { happy: 1 } }), { status: 200 })); await expect(predictAudio(file, request)).rejects.toThrow(/invalid response/i); expect(request.mock.calls[0][0]).toBe('http://127.0.0.1:8000/api/v1/predict') })
  it.each([[400, /could not be processed/i], [413, /size limit/i], [415, /valid WAV/i], [500, /unexpected error/i], [503, /temporarily unavailable/i]])('maps HTTP %i safely', async (status, message) => { const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'internal detail' }), { status })); await expect(predictAudio(file, request)).rejects.toThrow(message) })
  it('maps network errors and timeouts to helpful messages', async () => { await expect(predictAudio(file, vi.fn().mockRejectedValue(new TypeError('network')))).rejects.toThrow(/backend is running/i); const timeoutRequest: typeof fetch = vi.fn((_: RequestInfo | URL, options?: RequestInit) => new Promise<Response>((_, reject) => options?.signal?.addEventListener('abort', () => reject(new DOMException('', 'AbortError'))))); await expect(predictAudio(file, timeoutRequest, 1)).rejects.toThrow(/timed out/i) })
  it('returns a typed API error for invalid confidence data', async () => { const request = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...response, confidence: 2 }), { status: 200 })); await expect(predictAudio(file, request)).rejects.toBeInstanceOf(PredictionApiError) })
})
