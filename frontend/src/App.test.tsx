import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it } from 'vitest'
import App from './App'
import { ResultCard } from './components/ResultCard'
import type { PredictionResult } from './types'

const wavFile = (name = 'voice.wav', size = 32) => new File([new Uint8Array(size)], name, { type: 'audio/wav' })
afterEach(cleanup)
describe('speech emotion frontend foundation', () => {
  it('renders the initial interface with accessible controls', () => { render(<App />); expect(screen.getByRole('heading', { name: /understand the emotion/i })).toBeInTheDocument(); expect(screen.getByRole('tab', { name: /upload/i })).toHaveAttribute('aria-selected', 'true'); expect(screen.getByLabelText(/choose wav file/i)).toBeInTheDocument(); expect(screen.getByRole('button', { name: /analyse voice/i })).toBeDisabled() })
  it('switches to the planned recording interface', async () => { const user = userEvent.setup(); render(<App />); await user.click(screen.getByRole('tab', { name: /record/i })); expect(screen.getByText(/not connected yet/i)).toBeInTheDocument(); expect(screen.getByRole('button', { name: /analyse voice/i })).toBeDisabled() })
  it('accepts a valid WAV file and supports removal', async () => { const user = userEvent.setup(); render(<App />); await user.upload(screen.getByLabelText(/choose wav file/i), wavFile('speech.wav')); expect(screen.getByText('speech.wav')).toBeInTheDocument(); expect(screen.getByRole('button', { name: /analyse voice/i })).toBeEnabled(); await user.click(screen.getByRole('button', { name: /remove selected file/i })); expect(screen.queryByText('speech.wav')).not.toBeInTheDocument(); expect(screen.getByRole('button', { name: /analyse voice/i })).toBeDisabled() })
  it('rejects unsupported and oversized files', async () => { const user = userEvent.setup(); render(<App />); const input = screen.getByLabelText(/choose wav file/i); await user.upload(input, wavFile('speech.mp3')); expect(screen.getByRole('alert')).toHaveTextContent(/choose a wav/i); await user.upload(input, wavFile('large.wav', 10 * 1024 * 1024 + 1)); expect(screen.getByRole('alert')).toHaveTextContent(/exceeds the 10 mib/i) })
  it('renders controlled result data without using it as a live prediction', () => { const result: PredictionResult = { emotion: 'happy', confidence: .72, probabilities: { neutral: .02, calm: .03, happy: .72, sad: .04, angry: .05, fearful: .04, disgust: .03, surprised: .07 } }; render(<ResultCard result={result} />); expect(screen.getByText('happy', { selector: 'strong' })).toBeInTheDocument(); expect(screen.getByText('72%', { selector: 'strong' })).toBeInTheDocument(); expect(screen.getByLabelText(/emotion probability distribution/i)).toBeInTheDocument() })
})
