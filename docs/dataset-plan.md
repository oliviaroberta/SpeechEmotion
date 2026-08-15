# RAVDESS Dataset Plan

## 1. Dataset Selection

This project will use only the official RAVDESS audio-only speech archive from Zenodo. The project predicts emotion from spoken voice, so it does not require the song archive, the video archives, or the full multimodal collection with facial information.

The selected archive is:

- `Audio_Speech_Actors_01-24.zip`

The project deliberately uses the original official 48 kHz speech archive. Recordings will be resampled to 16 kHz later during preprocessing so that the transformation is controlled, reproducible, and documented by this project.

## 2. Official Source Metadata

- Dataset: The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- Creators: Steven R. Livingstone and Frank A. Russo
- Version: 1.0.0
- Official Zenodo record: `https://zenodo.org/records/1188976`
- DOI: `10.5281/zenodo.1188976`
- Selected archive filename: `Audio_Speech_Actors_01-24.zip`
- Archive size: `208.5 MB` as shown on the official Zenodo record page
- Archive size in bytes: exact byte count could not be retrieved in this environment on August 15, 2026 because the official Zenodo API endpoint was unreachable here; this should be rechecked from the official endpoint before download
- MD5 checksum: `bc696df654c87fed845eb13823edef8a`
- Official download URL: `https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1`
- Date metadata was verified: August 15, 2026

Only the official Zenodo record was used for these metadata values. The inaccessible exact byte count is the only field still needing direct API confirmation in a network-permitted environment.

## 3. Licence and Permitted Use

RAVDESS is released under:

`Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)`

In plain language:

- Attribution is required.
- This project uses the dataset for a non-commercial academic final-year project.
- The dataset must not be repackaged or used commercially without checking the licence terms or obtaining the appropriate commercial licence.
- The raw recordings will not be committed to Git or redistributed with the source code.

This document records the licence context for project planning only. It does not provide legal guarantees.

## 4. Required Academic Citation

Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLOS ONE, 13(5), e0196391. https://doi.org/10.1371/journal.pone.0196391

Zenodo dataset DOI:

`10.5281/zenodo.1188976`

## 5. Dataset Composition

The selected speech archive contains:

- 1,440 speech recordings
- 24 professional actors
- 12 male and 12 female actors
- 60 speech recordings per actor
- Two lexically matched spoken statements
- Audio-only, 16-bit, 48 kHz WAV recordings
- Acted emotions in a neutral North American English accent

Emotion labels:

- `01 = neutral`
- `02 = calm`
- `03 = happy`
- `04 = sad`
- `05 = angry`
- `06 = fearful`
- `07 = disgust`
- `08 = surprised`

Neutral has only normal intensity. The other seven emotions have normal and strong intensities. The emotion classes are therefore not perfectly balanced.

## 6. Filename Data Dictionary

RAVDESS audio filenames follow this seven-part convention:

`modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav`

Field definitions:

Modality:

- `03 = audio-only`

Vocal channel:

- `01 = speech`

Emotion:

- `01 = neutral`
- `02 = calm`
- `03 = happy`
- `04 = sad`
- `05 = angry`
- `06 = fearful`
- `07 = disgust`
- `08 = surprised`

Intensity:

- `01 = normal`
- `02 = strong`
- Neutral has no strong-intensity recording.

Statement:

- `01 = "Kids are talking by the door"`
- `02 = "Dogs are sitting by the door"`

Repetition:

- `01 = first repetition`
- `02 = second repetition`

Actor:

- `01` through `24`
- Odd actor IDs = male
- Even actor IDs = female

Example filename:

`03-01-05-02-01-01-03.wav`

Decoded:

- `03` = audio-only
- `01` = speech
- `05` = angry
- `02` = strong intensity
- `01` = "Kids are talking by the door"
- `01` = first repetition
- `03` = actor 03
- actor `03` is male because it is an odd actor ID

## 7. Planned Local Storage

The intended local raw-data structure is:

```text
ml/data/raw/ravdess/
|-- archive/
|   `-- Audio_Speech_Actors_01-24.zip
`-- extracted/
    |-- Actor_01/
    |-- Actor_02/
    `-- ...
```

The archive file and all extracted audio contents are intended to remain ignored by Git.

This planning step does not create the archive file or the 24 actor directories.

## 8. Planned Preprocessing

The original files will remain unchanged in the raw directory.

Planned processed copies will later be:

- loaded as mono audio
- resampled from 48 kHz to 16 kHz
- trimmed using a controlled silence-removal rule
- normalised
- padded or truncated to a consistent duration selected after dataset inspection
- stored or transformed reproducibly
- used for MFCC and Mel-spectrogram extraction

These are planned preprocessing decisions only. No preprocessing has been performed yet.

## 9. Speaker-Independent Dataset Split

Initial deterministic split:

- Training actors: `01-16`
- Validation actors: `17-20`
- Testing actors: `21-24`

Rationale:

- No actor appears in more than one split.
- This prevents the model from being tested on the same voices it learned during training.
- Each split contains equal numbers of male and female actors.
- With 60 recordings per actor, the expected totals are:
  - Training: 960 recordings
  - Validation: 240 recordings
  - Testing: 240 recordings
- This split may only be changed later if there is a documented technical reason.
- No random file-level split should be used because that could leak the same speaker's voice across training and testing.

## 10. Known Limitations

- Acted rather than naturally occurring emotions
- Neutral North American English accent
- Only 24 actors
- Fixed statements
- Class imbalance caused by fewer neutral recordings
- Possible generalisation problems with Ghanaian and other accents
- Sensitivity to background noise and recording equipment

## 11. Integrity Checks Required After Download

The following checks must be completed in the next step after download:

- archive exists
- file size matches metadata
- MD5 equals `bc696df654c87fed845eb13823edef8a`
- ZIP archive passes integrity testing
- safe extraction prevents path traversal
- exactly 24 actor directories
- exactly 1,440 `.wav` files
- exactly 60 files per actor
- every filename contains seven valid fields
- all emotion IDs are valid
- all audio files can be read
- raw files remain unmodified
- dataset contents remain untracked by Git
