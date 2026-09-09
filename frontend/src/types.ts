export const EMOTIONS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'] as const
export type Emotion = (typeof EMOTIONS)[number]
export interface PredictionResult { emotion: Emotion; confidence: number; probabilities: Record<Emotion, number> }
