"""Transcribe a directory of recordings using the TALK Whisper workflow."""

import pandas as pd
import os
from pathlib import Path
from argparse import ArgumentParser


def process_AST(
    audio_file_pth: Path,
    output_file_pth: Path,
    cut_minutes: float = 2,
    overlap_minutes: int = 0,
    model_name: str = "large-v2",
):
    """
    Processes audio files by loading them, applying Whisper's model for transcription,
    and splitting them into segments based on the provided cut and overlap times.

    Parameters:
    audio_file_pth (Path): Directory path containing the audio files to be processed.
    output_file_pth (Path): Directory path where the output should be saved.
    cut_minutes (float): Length of each audio segment in minutes. Defaults to 2 minutes.
    overlap_minutes (int): Overlap duration between consecutive segments. Defaults to 0.
    model_name (str): Whisper model to use ('large-v3' or 'large-v2'). Defaults to 'large-v2'.
    """

    import whisper

    # Load the Whisper model based on the provided model name
    model = whisper.load_model(model_name)
    audio_files = os.listdir(audio_file_pth)

    # Set the model dimension based on the model type
    if model_name == "large-v3":
        n_dim = 128
    elif model_name == "large-v2":
        n_dim = 80

    for audio_file in audio_files:

        if audio_file.endswith(".wav") or audio_file.endswith(".mp3"):
            print("Process ", audio_file)
            audio_file_name = audio_file
            audio = whisper.load_audio(os.path.join(audio_file_pth, audio_file_name))

            get_cut_minutes = int(cut_minutes * 60 * 16000)  # SAMPLE_RATE = 16000
            get_overlap_minutes = int(overlap_minutes * 60 * 16000)
            total_data = []
            length = len(audio)
            i = 0
            print(
                "Total "
                + str(round(length / (60 * 16000), 2))
                + " minutes audio loaded."
            )

            while (i + 2) * (get_cut_minutes - get_overlap_minutes) < length:
                get_cut_minutes_audio = audio[
                    i
                    * (get_cut_minutes - get_overlap_minutes) : (
                        i * (get_cut_minutes - get_overlap_minutes) + get_cut_minutes
                    )
                ]
                result = model.transcribe(get_cut_minutes_audio, language="en")

                for sen in result["segments"]:
                    start_time = int(sen["start"] * 16000)
                    end_time = int(sen["end"] * 16000)
                    audio_cut = audio[start_time:end_time]
                    audio_trim = whisper.pad_or_trim(audio_cut)

                    # make log-Mel spectrogram and move to the same device as the model
                    mel = whisper.log_mel_spectrogram(audio_trim, n_mels=n_dim).to(
                        model.device
                    )

                    # detect the spoken language
                    _, probs = model.detect_language(mel)
                    top_langs = sorted(probs, key=probs.get, reverse=True)[:3]
                    lang = None
                    for item in top_langs:
                        if item in ["en", "es"]:
                            lang = item
                            break
                    top_lang_probs = {lang: probs[lang] for lang in top_langs}

                    total_data.append(
                        [
                            round(
                                float(sen["start"])
                                + (i * 60 * (cut_minutes - overlap_minutes)),
                                1,
                            ),
                            round(
                                float(sen["end"])
                                + (i * 60 * (cut_minutes - overlap_minutes)),
                                1,
                            ),
                            lang,
                            top_langs,
                            top_lang_probs,
                            sen["text"],
                        ]
                    )

                print(
                    "Finish "
                    + str(i * (cut_minutes - overlap_minutes))
                    + " minutes to "
                    + str(i * (cut_minutes - overlap_minutes) + cut_minutes)
                )

                i += 1
                print(result["text"])

            get_cut_minutes_audio = audio[i * (get_cut_minutes - get_overlap_minutes) :]

            result = model.transcribe(get_cut_minutes_audio)

            for sen in result["segments"]:
                start_time = int(sen["start"] * 16000)
                end_time = int(sen["end"] * 16000)
                audio_cut = audio[start_time:end_time]
                audio_trim = whisper.pad_or_trim(audio_cut)

                # make log-Mel spectrogram and move to the same device as the model
                mel = whisper.log_mel_spectrogram(audio_trim, n_mels=n_dim).to(
                    model.device
                )

                # detect the spoken language
                _, probs = model.detect_language(mel)
                top_langs = sorted(probs, key=probs.get, reverse=True)[:3]
                lang = None
                for item in top_langs:
                    if item in ["en", "es"]:
                        lang = item
                        break
                top_lang_probs = {lang: probs[lang] for lang in top_langs}

                total_data.append(
                    [
                        round(
                            float(sen["start"])
                            + (i * 60 * (cut_minutes - overlap_minutes)),
                            1,
                        ),
                        round(
                            float(sen["end"])
                            + (i * 60 * (cut_minutes - overlap_minutes)),
                            1,
                        ),
                        lang,
                        top_langs,
                        top_lang_probs,
                        sen["text"],
                    ]
                )
            print(
                "Finish "
                + str(i * (cut_minutes - overlap_minutes))
                + " minutes to "
                + str(round(length / (60 * 16000), 2))
            )

            print(result["text"])

            print("Finished")
            total_df = pd.DataFrame(
                total_data,
                columns=[
                    "start_sec",
                    "end_sec",
                    "lang",
                    "language_t3",
                    "probability_t3",
                    "sentence",
                ],
            )
            if "wav" in audio_file_name:
                total_df.to_csv(
                    os.path.join(
                        output_file_pth,
                        audio_file_name.replace(
                            ".wav",
                            "_AST_{}min_{}.csv".format(str(cut_minutes), model_name),
                        ),
                    )
                )
            else:
                total_df.to_csv(
                    os.path.join(
                        output_file_pth,
                        audio_file_name.replace(
                            ".mp3",
                            "_AST_{}min_{}.csv".format(str(cut_minutes), model_name),
                        ),
                    )
                )
    return


def main():
    """
    Main function to handle argument parsing and invoke the process_AST function.
    """
    parser = ArgumentParser(
        description="Audio Segmentation and Transcription (AST) using Whisper"
    )
    parser.add_argument(
        "audio_file_pth", type=str, help="Path to the directory containing audio files."
    )
    parser.add_argument(
        "output_file_pth",
        type=str,
        help="Path to the directory to save the transcription results.",
    )
    parser.add_argument(
        "--cut_minutes",
        type=float,
        default=2,
        help="Length of each audio segment in minutes. Default is 2 minutes.",
    )
    parser.add_argument(
        "--overlap_minutes",
        type=int,
        default=0,
        help="Overlap between consecutive segments in minutes. Default is 0 minutes.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="large-v2",
        help="Whisper model to use (e.g., large-v3 or large-v2). Default is 'large-v2'.",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects
    audio_file_pth = Path(args.audio_file_pth)
    output_file_pth = Path(args.output_file_pth)

    # Call the process function with parsed arguments
    process_AST(
        audio_file_pth,
        output_file_pth,
        args.cut_minutes,
        args.overlap_minutes,
        args.model_name,
    )


if __name__ == "__main__":
    main()
