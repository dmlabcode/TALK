# TALK - Toolkit for Assessing Language in Kids
Open source code and data for Sci. Adv. 2026 Paper: Automated Measurement of Child and Teacher Language in Preschool

## Download and set up

Download this repository using **Code → Download ZIP**, extract it, and open a terminal in the extracted folder. Keep the following files together at the repository root.

| File | Purpose |
| --- | --- |
| `whisper_env.yml` | Conda environment for the three TALK scripts |
| `whisper_transcribe.py` | Transcribe audio recordings |
| `whisper_vtc1_alignment.py` | Combine transcripts with speaker labels |
| `transcript_data_analysis.py` | Produce a summary CSV |
| `README.md` | Setup and usage instructions |
| `LICENSE` | Repository license |

Install Conda through Miniconda or Anaconda if needed, then create and activate the environment. [Conda installation instructions](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)

```bash
conda env create -f whisper_env.yml
conda activate talk
python -m pip install --no-build-isolation openai-whisper==20231117
```

These commands install Python, pandas, Whisper, PyTorch, and FFmpeg. It retains Python 3.9 and the core package versions from the project's environment. Supporting dependencies are resolved for the local platform. The separate Whisper command uses the build tools included in the environment. Whisper downloads the selected model on first use, so an internet connection is needed for that step.

Check the installation before processing recordings.

```bash
python -c "import pandas, whisper, torch; print('TALK dependencies imported successfully')"
ffmpeg -version
```

VTC 1.0 is installed and run separately in its own environment. Follow its repository instructions to generate an RTTM file containing speaker labels for the same recordings. Then reactivate `talk` before running the commands below. [VTC 1.0 setup and usage](https://github.com/MarvinLvn/voice-type-classifier)

## Organize your files

The examples use one session dated November 13, 2026. Replace `Example_2026`, the date, and the recording names with your own values. All paths below are relative to the repository root.

| Path | Contents |
| --- | --- |
| `data/Example_2026/11-13-2026/audio/` | Input recordings |
| `data/Example_2026/11-13-2026/diarization_output.rttm` | Speaker labels generated separately by VTC |
| `data/Example_2026/11-13-2026/Whisper/` | Generated transcription CSVs |
| `data/Example_2026/11-13-2026/Whisper-VTC-Aligned/` | Generated speaker-labeled CSVs |
| `results/` | Generated summary CSVs |

For batch analysis, retain the layout `data/<group>_<year>/<mm-dd-yyyy>/Whisper-VTC-Aligned/*.csv`. The group and year folder supplies the output's classroom and year metadata. Each date directory represents a session.

Use the same recording names and time origin for Whisper and VTC. Include speaker labels for every recording selected for alignment.

## 1. Transcribe recordings

Place your `.wav` or `.mp3` files directly inside the session's `audio` folder. Use lowercase file extensions. Create the transcription output directory, then run Whisper.

```bash
python -c "from pathlib import Path; Path('data/Example_2026/11-13-2026/Whisper').mkdir(parents=True, exist_ok=True)"
python whisper_transcribe.py data/Example_2026/11-13-2026/audio data/Example_2026/11-13-2026/Whisper
```

Each recording produces a CSV containing transcript text, start and end times in seconds, and language information. With the default settings, `recording.wav` produces `recording_AST_2min_large-v2.csv`.

| Optional argument | Default | Purpose |
| --- | --- | --- |
| `--model_name` | `large-v2` | Select `large-v2` or `large-v3` |
| `--cut_minutes` | `2` | Set the audio chunk length in minutes |
| `--overlap_minutes` | `0` | Set chunk overlap in whole minutes |

Use a positive chunk length and an overlap smaller than that length.

## 2. Add speaker labels

Place the corresponding VTC RTTM file at the session path shown above, then run the alignment script.

```bash
python whisper_vtc1_alignment.py data/Example_2026/11-13-2026/Whisper data/Example_2026/11-13-2026/diarization_output.rttm
```

The script creates the sibling `Whisper-VTC-Aligned` directory. Each output CSV contains segment start and end times, transcript text, and a speaker label. Filenames end in `_VTC_Aligned.csv`.

The first argument can also be a single Whisper CSV. Use `--outpath PATH` to choose an output directory. Keep the default labels when continuing to analysis, including `FEM` and `KCHI`.

## 3. Generate a summary

To summarize all sessions in the example group, pass the folder containing the date directories.

```bash
python transcript_data_analysis.py data/Example_2026 results/summary.csv
```

Batch input follows `<input_path>/*/Whisper-VTC-Aligned/*.csv`. In this example, the input is `data/Example_2026`.

To analyze a single recording, pass its aligned CSV directly.

```bash
python transcript_data_analysis.py data/Example_2026/11-13-2026/Whisper-VTC-Aligned/recording_AST_2min_large-v2_VTC_Aligned.csv results/recording_summary.csv
```

The output contains one row per input file, with file and session information, utterance and question counts, response and turn measures, word counts, mean utterance length, and lexical diversity. Speaker-specific measures use `FEM` and `KCHI`. The optional `--response-window SECONDS` argument sets the response window, which defaults to `2.5` seconds.

The script creates the output directory as needed. If the summary filename already exists, it adds `_v2`, `_v3`, and so on. Transcription and alignment outputs with existing filenames are overwritten when rerun.

## Command-line help

Run any script with `--help` for its full argument list.

```bash
python whisper_transcribe.py --help
python whisper_vtc1_alignment.py --help
python transcript_data_analysis.py --help
```

## License

See `LICENSE` for the MIT license. External dependencies retain their respective licenses.
