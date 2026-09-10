import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'

const sessions = vi.hoisted(() => ({ start: vi.fn(), stop: vi.fn(), cleanup: vi.fn() }))

vi.mock('../services/audioRecording', () => {
  class RecordingError extends Error {}
  class RecordingSession {
    start = sessions.start
    stop = sessions.stop
    cleanup = sessions.cleanup
  }
  return { RecordingError, RecordingSession }
})

import { RecordPanel } from './RecordPanel'

const renderPanel = () => {
  const props = { analysing: false, error: null, onReady: vi.fn(), onDiscard: vi.fn(), onError: vi.fn(), onAnalyse: vi.fn() }
  return { ...render(<RecordPanel {...props} />), props }
}

afterEach(() => {
  cleanup()
  sessions.start.mockReset()
  sessions.stop.mockReset()
  sessions.cleanup.mockReset()
  vi.restoreAllMocks()
})

describe('RecordPanel playback lifecycle', () => {
  it('keeps a playback URL until re-recording and calls audio play and pause', async () => {
    const user = userEvent.setup()
    const createUrl = vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:recording')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    sessions.start.mockResolvedValue(undefined)
    sessions.stop.mockResolvedValue(new File(['audio'], 'microphone-recording.wav', { type: 'audio/wav' }))
    const { props } = renderPanel()
    await user.click(screen.getByRole('button', { name: /start recording/i }))
    await user.click(await screen.findByRole('button', { name: /stop recording/i }))
    const audio = document.querySelector('audio') as HTMLAudioElement
    Object.defineProperty(audio, 'paused', { configurable: true, value: true })
    const play = vi.spyOn(audio, 'play').mockResolvedValue(undefined)
    const pause = vi.spyOn(audio, 'pause').mockImplementation(() => undefined)
    await user.click(screen.getByRole('button', { name: /play recording/i }))
    expect(play).toHaveBeenCalledOnce()
    expect(revokeUrl).not.toHaveBeenCalledWith('blob:recording')
    fireEvent.play(audio)
    await screen.findByRole('button', { name: /pause recording/i })
    Object.defineProperty(audio, 'paused', { configurable: true, value: false })
    await user.click(screen.getByRole('button', { name: /pause recording/i }))
    expect(pause).toHaveBeenCalledOnce()
    await user.click(screen.getByRole('button', { name: /record again/i }))
    await waitFor(() => expect(revokeUrl).toHaveBeenCalledWith('blob:recording'))
    expect(createUrl).toHaveBeenCalledOnce()
    expect(props.onDiscard).toHaveBeenCalled()
  })

  it('shows a safe error when the browser rejects playback', async () => {
    const user = userEvent.setup()
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:recording')
    vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    sessions.start.mockResolvedValue(undefined)
    sessions.stop.mockResolvedValue(new File(['audio'], 'microphone-recording.wav', { type: 'audio/wav' }))
    const { props } = renderPanel()
    await user.click(screen.getByRole('button', { name: /start recording/i }))
    await user.click(await screen.findByRole('button', { name: /stop recording/i }))
    const audio = document.querySelector('audio') as HTMLAudioElement
    Object.defineProperty(audio, 'paused', { configurable: true, value: true })
    vi.spyOn(audio, 'play').mockRejectedValue(new DOMException('blocked', 'NotAllowedError'))
    await user.click(screen.getByRole('button', { name: /play recording/i }))
    await waitFor(() => expect(props.onError).toHaveBeenCalledWith('Recorded audio could not be played. Record again and try once more.'))
  })

  it('releases the URL when ready and recording session when unmounted', async () => {
    const user = userEvent.setup()
    vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:recording')
    const revokeUrl = vi.spyOn(URL, 'revokeObjectURL').mockImplementation(() => undefined)
    sessions.start.mockResolvedValue(undefined)
    sessions.stop.mockResolvedValue(new File(['audio'], 'microphone-recording.wav', { type: 'audio/wav' }))
    const { unmount } = renderPanel()
    await user.click(screen.getByRole('button', { name: /start recording/i }))
    await user.click(await screen.findByRole('button', { name: /stop recording/i }))
    unmount()
    expect(revokeUrl).toHaveBeenCalledWith('blob:recording')

    sessions.start.mockResolvedValue(undefined)
    const active = renderPanel()
    await user.click(screen.getByRole('button', { name: /start recording/i }))
    active.unmount()
    expect(sessions.cleanup).toHaveBeenCalled()
  })
})
