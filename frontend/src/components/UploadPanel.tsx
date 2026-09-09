import { FileAudio, Trash2, UploadCloud } from 'lucide-react'
import { useId, type ChangeEvent, type DragEvent } from 'react'

interface UploadPanelProps { file: File | null; error: string | null; onFileSelected: (file: File) => void; onRemove: () => void }
const formatBytes = (size: number) => `${(size / (1024 * 1024)).toFixed(size >= 1024 * 1024 ? 1 : 2)} MiB`

export function UploadPanel({ file, error, onFileSelected, onRemove }: UploadPanelProps) {
  const inputId = useId()
  const selectFile = (files: FileList | null) => { const selected = files?.item(0); if (selected) onFileSelected(selected) }
  const inputChange = (event: ChangeEvent<HTMLInputElement>) => { selectFile(event.target.files); event.target.value = '' }
  const onDrop = (event: DragEvent<HTMLDivElement>) => { event.preventDefault(); selectFile(event.dataTransfer.files) }
  return <section className="upload-panel" aria-labelledby="upload-title">
    <div className="section-heading"><div className="section-icon" aria-hidden="true"><UploadCloud size={18} /></div><div><p className="eyebrow">Audio source</p><h2 id="upload-title">Upload a WAV recording</h2></div></div>
    <div className="drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={onDrop}><FileAudio size={28} aria-hidden="true" /><p>Drag a WAV file here, or choose one from your device.</p><label className="file-picker" htmlFor={inputId}>Choose WAV file</label><input id={inputId} aria-label="Choose WAV file" type="file" accept=".wav,audio/wav" onChange={inputChange} /><span>Maximum file size: 10 MiB</span></div>
    <div className="upload-status" aria-live="polite" aria-atomic="true">{error && <p className="form-error" role="alert">{error}</p>}{file && <div className="selected-file"><FileAudio size={18} aria-hidden="true" /><div><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div><button className="icon-button" type="button" onClick={onRemove} aria-label="Remove selected file" title="Remove selected file"><Trash2 size={17} aria-hidden="true" /></button></div>}</div>
  </section>
}
