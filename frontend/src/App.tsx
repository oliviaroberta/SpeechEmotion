import { BrainCircuit, Info, Mic, Upload } from 'lucide-react'
import { useState } from 'react'
import { ResultCard } from './components/ResultCard'
import { UploadPanel } from './components/UploadPanel'
import { EMOTIONS } from './types'

const MAX_AUDIO_BYTES = 10 * 1024 * 1024
type SourceTab = 'upload' | 'record'
const validate = (file: File) => !file.name.toLowerCase().endsWith('.wav') ? 'Choose a WAV audio file to continue.' : file.size === 0 ? 'The selected WAV file is empty.' : file.size > MAX_AUDIO_BYTES ? 'The selected WAV file exceeds the 10 MiB limit.' : null

function App() {
  const [tab, setTab] = useState<SourceTab>('upload'); const [file, setFile] = useState<File | null>(null); const [error, setError] = useState<string | null>(null); const [status, setStatus] = useState('Choose a WAV recording to begin.')
  const chooseFile = (selected: File) => { const message = validate(selected); if (message) { setFile(null); setError(message); setStatus('File selection needs attention.'); return }; setFile(selected); setError(null); setStatus(`${selected.name} is ready for analysis.`) }
  const removeFile = () => { setFile(null); setError(null); setStatus('File removed. Choose another WAV recording.') }
  return <div className="app-shell"><header className="site-header"><a className="brand" href="#analysis" aria-label="Voice Emotion AI home"><span className="brand-mark" aria-hidden="true"><BrainCircuit size={22} /></span><span>Voice Emotion AI</span></a><span className="project-tag">Final-year project</span></header>
    <main id="analysis"><section className="intro" aria-labelledby="page-title"><p className="eyebrow">Speech emotion recognition</p><h1 id="page-title">Understand the emotion carried in a voice.</h1><p className="lead">Choose a WAV recording to prepare an AI-assisted emotion analysis. This interface is local-only until the prediction API is connected.</p><div className="emotion-chips" aria-label="Supported emotions">{EMOTIONS.map((emotion) => <span key={emotion}>{emotion}</span>)}</div></section>
      <section className="workspace" aria-label="Audio analysis workspace"><div className="source-card"><div className="tab-list" role="tablist" aria-label="Audio source"><button role="tab" type="button" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'active' : ''} onClick={() => setTab('upload')}><Upload size={16} aria-hidden="true" /> Upload</button><button role="tab" type="button" aria-selected={tab === 'record'} className={tab === 'record' ? 'active' : ''} onClick={() => setTab('record')}><Mic size={16} aria-hidden="true" /> Record</button></div>
      {tab === 'upload' ? <UploadPanel file={file} error={error} onFileSelected={chooseFile} onRemove={removeFile} /> : <section className="record-placeholder" role="tabpanel" aria-live="polite"><Mic size={30} aria-hidden="true" /><h2>Microphone recording</h2><p>The microphone interface is planned but is not connected yet.</p></section>}
      <div className="action-row"><p aria-live="polite" className="status-message">{status}</p><button className="analyse-button" type="button" disabled={!file || tab !== 'upload'} onClick={() => setStatus('Analysis will be available when the API connection is enabled.')}>Analyse voice</button></div></div><ResultCard /></section>
      <aside className="estimate-note"><Info size={18} aria-hidden="true" /><p>Predictions are AI estimates and may be incorrect. They should not be used as a clinical, legal, or safety assessment.</p></aside></main>
    <footer><span>Speech Emotion Recognition System</span><span>RAVDESS research dataset · React and TypeScript frontend</span></footer></div>
}
export default App
