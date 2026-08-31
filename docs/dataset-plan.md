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
- Archive size in bytes: `208,468,073` (downloaded and verified on August 31, 2026)
- MD5 checksum: `bc696df654c87fed845eb13823edef8a` (verified after download)
- Official download URL: `https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1`
- Date metadata was verified: August 15, 2026

Only the official Zenodo record was used for these metadata values.

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

The official archive was downloaded on August 31, 2026 and retained at `ml/data/raw/ravdess/archive/Audio_Speech_Actors_01-24.zip`. Its MD5 matches the official checksum, ZIP CRC validation passed, and inspection found 1,440 WAV entries across 24 actors. The archive remains ignored by Git and has not been extracted.

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

## 12. Extraction and Audio Inspection Status

Extraction and inspection were completed on August 31, 2026.

- Extraction destination: `ml/data/raw/ravdess/extracted/`
- The archive was safely extracted through a temporary sibling directory and promoted to the final destination only after all expected files were written and validated.
- The original ZIP remains unchanged and its MD5 is still `bc696df654c87fed845eb13823edef8a`.
- ZIP CRC validation passed before extraction and again after audio inspection.
- The extracted dataset contains 24 actor directories, 1,440 readable WAV files, and 60 recordings per actor.
- All raw audio files remain ignored by Git.

### Verified Audio Properties

Every recording was opened and read in full in chunks without rewriting it. All 1,440 files are WAV, 48,000 Hz, and `PCM_16` (16-bit PCM), with a positive frame count and duration. The original official archive has a mixed channel layout:

- Mono files: 1,435
- Stereo files: 5
- Channel distribution by statement: statement `01` has 718 mono and 2 stereo files; statement `02` has 717 mono and 3 stereo files.
- Channel distribution by actor: `Actor_01` has 58 mono and 2 stereo files; `Actor_05` has 59 mono and 1 stereo file; `Actor_20` has 58 mono and 2 stereo files; every other actor has 60 mono files.
- Channel distribution by emotion: neutral 96 mono; calm 190 mono and 2 stereo; happy 191 mono and 1 stereo; sad 192 mono; angry 192 mono; fearful 191 mono and 1 stereo; disgust 192 mono; surprised 191 mono and 1 stereo.

For the five stereo files, the left and right channels are bit-identical. No stereo file had differing channels, and the maximum absolute left-versus-right sample difference was `0`. This comparison was diagnostic only; no channel conversion or other audio modification occurred.

### Duration Statistics

Overall duration in seconds: minimum `2.936271`, maximum `5.271937`, mean `3.700665`, median `3.670333`, total `5328.957333`.

Per-emotion duration statistics in seconds:

- Neutral: minimum `3.069729`, maximum `4.137458`, mean `3.503153`, median `3.503500`
- Calm: minimum `2.936271`, maximum `4.771438`, mean `3.795806`, median `3.770437`
- Happy: minimum `3.103104`, maximum `4.404396`, mean `3.638183`, median `3.620281`
- Sad: minimum `3.103104`, maximum `4.738062`, mean `3.694490`, median `3.670333`
- Angry: minimum `3.236562`, maximum `5.105104`, mean `3.871404`, median `3.837167`
- Fearful: minimum `3.069729`, maximum `5.005000`, mean `3.574231`, median `3.553552`
- Disgust: minimum `3.136458`, maximum `5.271937`, mean `3.941785`, median `3.903896`
- Surprised: minimum `2.969625`, maximum `4.637979`, mean `3.487512`, median `3.503500`

No preprocessing has occurred. In particular, the raw recordings have not been downmixed, resampled, trimmed, normalised, padded, or used for feature extraction. A consistent mono conversion is planned for a later preprocessing step.

## 13. Deterministic Metadata Manifest and Speaker-Independent Split

The following tracked, portable outputs were generated without copying or modifying audio:

- `ml/metadata/ravdess_manifest.csv`
- `ml/metadata/ravdess_split_summary.json`

The CSV has one row per recording and uses this exact schema:

`relative_path, filename, modality_id, vocal_channel_id, emotion_id, emotion, intensity_id, intensity, statement_id, repetition_id, actor_id, gender, split, sample_rate, channels, subtype, frames, duration_seconds`

All paths are repository-relative, use forward slashes, and contain no local Windows path. Duration values are stored to six decimal places. The generated JSON has stable key ordering, fixed indentation, no timestamp, and no environment-specific information.

### Completed Actor Split

- Train: actors `01-16`, 16 actors, 960 recordings, 480 male and 480 female, 957 mono and 3 stereo.
- Validation: actors `17-20`, 4 actors, 240 recordings, 120 male and 120 female, 238 mono and 2 stereo.
- Test: actors `21-24`, 4 actors, 240 recordings, 120 male and 120 female, 240 mono and 0 stereo.

Each split has the expected emotion distribution: neutral has 64 train recordings and 16 validation/test recordings; every other emotion has 128 train recordings and 32 validation/test recordings. Intensity counts are train normal 512/strong 448, validation normal 128/strong 112, and test normal 128/strong 112.

Split intersections are empty: train/validation, train/test, and validation/test share no actor IDs. Split assignment uses only actor identity, never file-level randomisation or recording properties. This prevents the model from receiving recordings from the same speaker during training and evaluation: test voices are absent from train and validation, and validation voices are absent from train and test.

The manifest and summary were rebuilt and independently verified twice with identical SHA-256 hashes:

- `ravdess_manifest.csv`: `079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D`
- `ravdess_split_summary.json`: `5020FC97D7061A113B399A292653B393645DCA7DEDCB283544A9EF55EE23D19C`

The generated manifest summary uses six-decimal duration values from the CSV, so its aggregate total is `5328.957256` seconds. The direct raw-audio inspection total remains documented above. No preprocessing or model training has occurred. Neutral remains underrepresented in every split; class-imbalance handling will be decided during modelling without modifying the test distribution.
