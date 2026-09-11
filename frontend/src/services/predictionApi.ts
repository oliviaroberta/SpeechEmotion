import { EMOTIONS, type Emotion, type PredictionResult } from '../types'

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000'
const REQUEST_TIMEOUT_MS = 30_000

export class PredictionApiError extends Error {
  readonly status?: number
  constructor(message: string, status?: number) { super(message); this.name = 'PredictionApiError'; this.status = status }
}

const apiBaseUrl = () => (import.meta.env.VITE_API_BASE_URL?.trim() || DEFAULT_API_BASE_URL).replace(/\/$/, '')
const isProbability = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1

function parsePrediction(payload: unknown): PredictionResult {
  if (!payload || typeof payload !== 'object') throw new PredictionApiError('The prediction service returned an invalid response.')
  const data = payload as Record<string, unknown>
  if (!EMOTIONS.includes(data.emotion as Emotion) || !Number.isInteger(data.class_index) || data.class_index !== EMOTIONS.indexOf(data.emotion as Emotion) || !isProbability(data.confidence) || !data.probabilities || typeof data.probabilities !== 'object' || !data.model || typeof data.model !== 'object') throw new PredictionApiError('The prediction service returned an invalid response.')
  const probabilities = data.probabilities as Record<string, unknown>
  if (!EMOTIONS.every((emotion) => isProbability(probabilities[emotion])) || Object.keys(probabilities).length !== EMOTIONS.length) throw new PredictionApiError('The prediction service returned an invalid response.')
  const total = EMOTIONS.reduce((sum, emotion) => sum + Number(probabilities[emotion]), 0)
  const model = data.model as Record<string, unknown>
  if (Math.abs(total - 1) > 0.001 || typeof model.identifier !== 'string' || !model.identifier || typeof model.version !== 'string' || !model.version) throw new PredictionApiError('The prediction service returned an invalid response.')
  return { emotion: data.emotion as Emotion, confidence: data.confidence as number, probabilities: Object.fromEntries(EMOTIONS.map((emotion) => [emotion, probabilities[emotion]])) as Record<Emotion, number> }
}

function messageForStatus(status: number, detail: unknown): string {
  if (status === 400) return 'The WAV audio could not be processed. Choose another recording and try again.'
  if (status === 413) return 'This WAV file is larger than the 4 MiB upload size limit.'
  if (status === 415) return 'Only valid WAV audio files are supported.'
  if (status === 503) return 'The prediction service is temporarily unavailable. Please try again shortly.'
  if (status === 500) return 'The prediction service encountered an unexpected error. Please try again.'
  return typeof detail === 'string' && detail.length <= 180 ? detail : 'The prediction request could not be completed.'
}

export async function predictAudio(file: File, request: typeof fetch = fetch, timeoutMs = REQUEST_TIMEOUT_MS): Promise<PredictionResult> {
  const controller = new AbortController(); const timeout = window.setTimeout(() => controller.abort(), timeoutMs); const body = new FormData(); body.append('file', file)
  try {
    const response = await request(`${apiBaseUrl()}/api/v1/predict`, { method: 'POST', body, signal: controller.signal })
    const payload: unknown = await response.json().catch(() => null)
    if (!response.ok) throw new PredictionApiError(messageForStatus(response.status, (payload as { detail?: unknown } | null)?.detail), response.status)
    return parsePrediction(payload)
  } catch (error) {
    if (error instanceof PredictionApiError) throw error
    if (error instanceof DOMException && error.name === 'AbortError') throw new PredictionApiError('The prediction request timed out. Check that the backend is running and try again.')
    throw new PredictionApiError('The prediction service is unavailable. Check that the backend is running and try again.')
  } finally { window.clearTimeout(timeout) }
}
