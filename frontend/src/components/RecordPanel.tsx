import { Mic, Play, RotateCcw, Square } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { RecordingError, RecordingSession } from '../services/audioRecording'

interface RecordPanelProps { analysing: boolean; error: string | null; onReady: (file: File) => void; onDiscard: () => void; onError: (message: string) => void; onAnalyse: () => void }
type RecordState = 'idle' | 'requesting' | 'recording' | 'ready'
const time = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`

export function RecordPanel({ analysing, error, onReady, onDiscard, onError, onAnalyse }: RecordPanelProps) {
  const [state, setState] = useState<RecordState>('idle'); const [duration, setDuration] = useState(0); const [recording, setRecording] = useState<File | null>(null); const [url, setUrl] = useState<string | null>(null); const [playing, setPlaying] = useState(false); const session = useRef<RecordingSession | null>(null); const timer = useRef<number | null>(null); const audio = useRef<HTMLAudioElement>(null); const urlRef = useRef<string | null>(null)
  const clearTimer = () => { if (timer.current !== null) window.clearInterval(timer.current); timer.current = null }
  const release = async () => { clearTimer(); await session.current?.cleanup(); session.current = null }
  const replaceUrl = (nextUrl: string | null) => { if (urlRef.current) URL.revokeObjectURL(urlRef.current); urlRef.current = nextUrl; setUrl(nextUrl) }
  const clearRecording = () => { replaceUrl(null); setRecording(null); setDuration(0); setPlaying(false) }
  useEffect(() => () => { void release(); if (urlRef.current) URL.revokeObjectURL(urlRef.current) }, [])
  const finish = async () => { try { const file = await session.current?.stop(); clearTimer(); if (!file) return; replaceUrl(URL.createObjectURL(file)); setRecording(file); setState('ready'); onReady(file) } catch (caught) { setState('idle'); onError(caught instanceof RecordingError ? caught.message : 'WAV encoding failed.') } finally { session.current = null } }
  const start = async () => { clearRecording(); onDiscard(); setState('requesting'); const next = new RecordingSession(); session.current = next; try { await next.start(() => { void finish() }); setState('recording'); const started = Date.now(); timer.current = window.setInterval(() => setDuration((Date.now() - started) / 1000), 250) } catch (caught) { setState('idle'); onError(caught instanceof RecordingError ? caught.message : 'Microphone recording could not be started.'); session.current = null } }
  const again = async () => { await release(); clearRecording(); onDiscard(); setState('idle') }
  const togglePlayback = async () => {
    if (!audio.current) return
    try {
      if (audio.current.paused) await audio.current.play()
      else audio.current.pause()
    } catch {
      setPlaying(false)
      onError('Recorded audio could not be played. Record again and try once more.')
    }
  }
  return <section className="record-panel" role="tabpanel" aria-labelledby="record-tab"><div className="section-heading"><div className="section-icon" aria-hidden="true"><Mic size={18} /></div><div><p className="eyebrow">Audio source</p><h2>Record from microphone</h2></div></div>
    <div className="recording-stage" aria-live="polite">{state === 'recording' ? <><span className="recording-dot" aria-hidden="true" /><strong>Recording live</strong><span>{time(duration)} / 00:15</span></> : state === 'requesting' ? <p>Requesting microphone permission...</p> : state === 'ready' ? <p>Recording ready: {time(duration)}. Play it or analyse it.</p> : <p>Record one to fifteen seconds of speech. Audio stays in memory until analysis.</p>}</div>
    {error && <p className="form-error" role="alert">{error}</p>}
    <div className="record-controls">{state === 'idle' && <button className="analyse-button" type="button" onClick={() => void start()}><Mic size={17} aria-hidden="true" /> Start recording</button>}{state === 'recording' && <button className="stop-button" type="button" onClick={() => void finish()}><Square size={16} aria-hidden="true" /> Stop recording</button>}{state === 'ready' && <><button className="secondary-button" type="button" onClick={() => void togglePlayback()} aria-label={playing ? 'Pause recording' : 'Play recording'}><Play size={16} aria-hidden="true" /> {playing ? 'Pause' : 'Play recording'}</button><button className="secondary-button" type="button" onClick={() => void again()}><RotateCcw size={16} aria-hidden="true" /> Record again</button><button className="analyse-button" type="button" disabled={analysing || !recording} onClick={onAnalyse}>{analysing ? 'Analysing...' : 'Analyse recording'}</button></>}</div>
    {url && <audio ref={audio} src={url} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} />}</section>
}
