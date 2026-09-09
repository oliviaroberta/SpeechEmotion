import { BarChart3, Sparkles } from 'lucide-react'
import { EMOTIONS, type PredictionResult } from '../types'

interface ResultCardProps { result?: PredictionResult }
const percent = (value: number) => `${Math.round(value * 100)}%`

export function ResultCard({ result }: ResultCardProps) {
  return <section className="result-card" aria-labelledby="result-title">
    <div className="section-heading"><div className="section-icon" aria-hidden="true"><BarChart3 size={18} /></div><div><p className="eyebrow">Prediction</p><h2 id="result-title">Emotion analysis</h2></div></div>
    {result ? <div className="result-summary"><div><span className="result-label">Detected emotion</span><strong className="emotion-result">{result.emotion}</strong></div><div className="confidence-block"><span className="result-label">Confidence</span><strong>{percent(result.confidence)}</strong></div></div> : <div className="empty-result" aria-live="polite"><Sparkles size={20} aria-hidden="true" /><p>Select a WAV recording and run an analysis to see the returned prediction.</p></div>}
    <div className="probability-list" aria-label="Emotion probability distribution">{EMOTIONS.map((emotion) => { const value = result?.probabilities[emotion] ?? 0; return <div className="probability-row" key={emotion}><span>{emotion}</span><div className="probability-track" aria-hidden="true"><div className="probability-fill" style={{ width: `${value * 100}%` }} /></div><span className="probability-value">{result ? percent(value) : '-'}</span></div> })}</div>
  </section>
}
