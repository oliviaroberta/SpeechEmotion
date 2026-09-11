import { BrainCircuit, Info, Mic, Upload } from 'lucide-react'
import { useState } from 'react'
import { RecordPanel } from './components/RecordPanel'
import { ResultCard } from './components/ResultCard'
import { UploadPanel } from './components/UploadPanel'
import { MAX_AUDIO_BYTES } from './services/audioRecording'
import { PredictionApiError, predictAudio } from './services/predictionApi'
import { EMOTIONS, type PredictionResult } from './types'
import './recording.css'

type SourceTab = 'upload' | 'record'
type Phase = 'idle' | 'selected' | 'analysing' | 'success' | 'error'
const validate = (file: File) => !file.name.toLowerCase().endsWith('.wav') ? 'Please choose a WAV audio file.' : file.size === 0 ? 'This audio file is empty. Please choose another one.' : file.size > MAX_AUDIO_BYTES ? 'This file is larger than the 4 MiB limit. Please choose a smaller WAV file.' : null

function App() {
  const [tab, setTab] = useState<SourceTab>('upload'); const [file, setFile] = useState<File | null>(null); const [error, setError] = useState<string | null>(null); const [result, setResult] = useState<PredictionResult | undefined>(); const [phase, setPhase] = useState<Phase>('idle')
  const selectFile = (selected: File) => { const message = validate(selected); setResult(undefined); if (message) { setFile(null); setError(message); setPhase('error'); return }; setFile(selected); setError(null); setPhase('selected') }
  const removeFile = () => { setFile(null); setError(null); setResult(undefined); setPhase('idle') }
  const analyse = async () => { if (!file || phase === 'analysing') return; setError(null); setResult(undefined); setPhase('analysing'); try { setResult(await predictAudio(file)); setPhase('success') } catch (caught) { setError(caught instanceof PredictionApiError ? caught.message : 'The prediction request could not be completed.'); setPhase('error') } }
  const changeTab = (next: SourceTab) => { if (next !== tab && tab === 'record') removeFile(); setTab(next) }
  const status = phase === 'analysing' ? 'Listening to your recording. This may take a moment.' : phase === 'success' ? 'Your result is ready.' : phase === 'error' ? 'Please choose another file or try again.' : file ? `${file.name} is ready to check.` : 'Choose a WAV recording to get started.'
  return <div className="app-shell"><header className="site-header"><a className="brand" href="#analysis" aria-label="Vemotion home"><span className="brand-mark" aria-hidden="true"><BrainCircuit size={22} /></span><span>Vemotion</span></a><span className="project-tag">Final-year project</span></header>
    <main id="analysis"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Voice emotion checker</p><h1 id="page-title">Understand the emotion in a voice recording.</h1><p className="lead">Upload a WAV file or record a short message. We will show the emotion our AI thinks is most likely.</p><div className="emotion-chips" aria-label="Emotions this tool can recognise">{EMOTIONS.map((emotion) => <span key={emotion}>{emotion}</span>)}</div></section>
      <section className="workspace" aria-label="Audio analysis workspace"><div className="source-card"><div className="tab-list" role="tablist" aria-label="Audio source"><button role="tab" id="upload-tab" type="button" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'active' : ''} onClick={() => changeTab('upload')}><Upload size={16} aria-hidden="true" /> Upload</button><button role="tab" id="record-tab" type="button" aria-selected={tab === 'record'} className={tab === 'record' ? 'active' : ''} onClick={() => changeTab('record')}><Mic size={16} aria-hidden="true" /> Record</button></div>
      {tab === 'upload' ? <><UploadPanel file={file} error={error} onFileSelected={selectFile} onRemove={removeFile} /><div className="action-row"><p aria-live="polite" className="status-message" aria-busy={phase === 'analysing'}>{status}</p><button className="analyse-button" type="button" disabled={!file || phase === 'analysing'} onClick={analyse}>{phase === 'analysing' ? 'Checking...' : phase === 'error' && file ? 'Try again' : 'Check emotion'}</button></div></> : <RecordPanel analysing={phase === 'analysing'} error={error} onReady={selectFile} onDiscard={removeFile} onError={(message) => { setError(message); setPhase('error') }} onAnalyse={() => void analyse()} />}</div><ResultCard result={result} /></section>
      <aside className="estimate-note"><Info size={18} aria-hidden="true" /><p>This result is an AI estimate. It can be affected by your voice, accent, background noise, and recording quality. For best results, speak for at least one second in a quiet place. It is not a medical, legal, or safety assessment.</p></aside></main>
    <footer><span>Speech Emotion Recognition System</span></footer></div>
}
export default App
