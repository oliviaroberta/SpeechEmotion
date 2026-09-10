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
const validate = (file: File) => !file.name.toLowerCase().endsWith('.wav') ? 'Choose a WAV audio file to continue.' : file.size === 0 ? 'The selected WAV file is empty.' : file.size > MAX_AUDIO_BYTES ? 'The selected WAV file exceeds the 10 MiB limit.' : null

function App() {
  const [tab, setTab] = useState<SourceTab>('upload'); const [file, setFile] = useState<File | null>(null); const [error, setError] = useState<string | null>(null); const [result, setResult] = useState<PredictionResult | undefined>(); const [phase, setPhase] = useState<Phase>('idle')
  const selectFile = (selected: File) => { const message = validate(selected); setResult(undefined); if (message) { setFile(null); setError(message); setPhase('error'); return }; setFile(selected); setError(null); setPhase('selected') }
  const removeFile = () => { setFile(null); setError(null); setResult(undefined); setPhase('idle') }
  const analyse = async () => { if (!file || phase === 'analysing') return; setError(null); setResult(undefined); setPhase('analysing'); try { setResult(await predictAudio(file)); setPhase('success') } catch (caught) { setError(caught instanceof PredictionApiError ? caught.message : 'The prediction request could not be completed.'); setPhase('error') } }
  const changeTab = (next: SourceTab) => { if (next !== tab && tab === 'record') removeFile(); setTab(next) }
  const status = phase === 'analysing' ? 'Analysing recording. This may take a moment.' : phase === 'success' ? 'Analysis complete.' : phase === 'error' ? 'Analysis needs attention. You can select a new file or retry.' : file ? `${file.name} is ready for analysis.` : 'Choose a WAV recording to begin.'
  return <div className="app-shell"><header className="site-header"><a className="brand" href="#analysis" aria-label="Voice Emotion AI home"><span className="brand-mark" aria-hidden="true"><BrainCircuit size={22} /></span><span>Voice Emotion AI</span></a><span className="project-tag">Final-year project</span></header>
    <main id="analysis"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Speech emotion recognition</p><h1 id="page-title">Voice emotion analysis for recorded speech.</h1><p className="lead">Upload a WAV recording or capture a short voice sample to receive an AI-assisted emotion estimate.</p><div className="emotion-chips" aria-label="Supported emotions">{EMOTIONS.map((emotion) => <span key={emotion}>{emotion}</span>)}</div></section>
      <section className="workspace" aria-label="Audio analysis workspace"><div className="source-card"><div className="tab-list" role="tablist" aria-label="Audio source"><button role="tab" id="upload-tab" type="button" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'active' : ''} onClick={() => changeTab('upload')}><Upload size={16} aria-hidden="true" /> Upload</button><button role="tab" id="record-tab" type="button" aria-selected={tab === 'record'} className={tab === 'record' ? 'active' : ''} onClick={() => changeTab('record')}><Mic size={16} aria-hidden="true" /> Record</button></div>
      {tab === 'upload' ? <><UploadPanel file={file} error={error} onFileSelected={selectFile} onRemove={removeFile} /><div className="action-row"><p aria-live="polite" className="status-message" aria-busy={phase === 'analysing'}>{status}</p><button className="analyse-button" type="button" disabled={!file || phase === 'analysing'} onClick={analyse}>{phase === 'analysing' ? 'Analysing...' : phase === 'error' && file ? 'Retry analysis' : 'Analyse voice'}</button></div></> : <RecordPanel analysing={phase === 'analysing'} error={error} onReady={selectFile} onDiscard={removeFile} onError={(message) => { setError(message); setPhase('error') }} onAnalyse={() => void analyse()} />}</div><ResultCard result={result} /></section>
      <aside className="estimate-note"><Info size={18} aria-hidden="true" /><p>Results are model estimates and may vary with voice, accent, noise, and recording quality. Speak for at least one second in a quiet environment. They should not be used as a clinical, legal, or safety assessment.</p></aside></main>
    <footer><span>Speech Emotion Recognition System</span><span>RAVDESS research dataset | React and TypeScript frontend</span></footer></div>
}
export default App
