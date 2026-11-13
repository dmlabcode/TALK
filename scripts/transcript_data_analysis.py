"""Calculate TALK language and conversational measures from aligned CSVs."""

import argparse
import os
import glob
import pandas as pd
import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def clean_text(text):
    """
    Removes non-alphanumeric characters (preserving Spanish accents)
    and converts to lowercase for accurate word counting.
    """
    if pd.isna(text) or str(text).lower() == "nan":
        return ""
    # Strip everything except letters, numbers, and spaces
    cleaned = re.sub(r"[^a-zA-ZñÑáéíóúÁÉÍÓÚüÜ0-9\s]", "", str(text))
    return cleaned.strip().lower()


def get_word_stats(sentence_list):
    """
    Takes a list of sentences, cleans them, and returns (total_words, unique_words).
    """
    all_text = " ".join([clean_text(s) for s in sentence_list])
    words = all_text.split()
    return len(words), len(set(words))


def safe_divide(numerator, denominator):
    """Safely divide two numbers, returning 0.0 if the denominator is 0."""
    return numerator / denominator if denominator else 0.0


def school_round(value):
    """
    Rounds a number to the thousandths place (3 decimal places)
    using traditional school book rounding (ROUND_HALF_UP).
    """
    # Convert to string first to avoid initial floating-point inaccuracies
    d = Decimal(str(value))
    # Quantize forces 3 decimal places, applying HALF_UP rounding
    rounded_val = d.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return float(rounded_val)


def get_natural_sort_key(text):
    """
    Splits a string into text and integers for natural alphanumeric sorting.
    Allows 'Subject_2' to be sorted before 'Subject_10'.
    """
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", str(text))
    )


def extract_metadata_from_path(file_path):
    """
    Extracts Classroom, Year, and Date from a file path matching:
    .../Classroom/Classroom_YYYY/mm-dd-yyyy/Whisper-VTC-Aligned/...
    Returns "n/a" for any missing elements.
    """
    classroom, year, date_str = "n/a", "n/a", "n/a"
    parts = Path(file_path).parts

    # Regex to match a date folder like 05-04-2026 or 05-04-26
    date_pattern = re.compile(r"^\d{2}-\d{2}-\d{2,4}$")

    for i, part in enumerate(parts):
        if date_pattern.match(part):
            date_str = part
            # The folder right before the date folder should be Classroom_YYYY
            if i >= 1:
                cls_year_part = parts[i - 1]
                # Split from the right on the underscore to separate name from year
                if "_" in cls_year_part:
                    classroom, year = cls_year_part.rsplit("_", 1)
                else:
                    classroom = cls_year_part  # Fallback if there is no year
            break

    return classroom, year, date_str


# ---------------------------------------------------------
# Main Processing Function
# ---------------------------------------------------------
def process_transcript(file_path, response_window=2.5):
    """
    Processes a single transcript file and returns a dictionary of metrics.
    """
    df = pd.read_csv(file_path)

    # Ensure dataframe is sorted chronologically
    if "start_sec" in df.columns:
        df = df.sort_values(by="start_sec").reset_index(drop=True)

    # 1. Base Variables & Dynamic Metadata
    subject_id = Path(file_path).stem
    classroom, year, date_str = extract_metadata_from_path(file_path)

    # Determine Context based on SubjectID
    context = "Therapy" if "slp" in subject_id.lower() else "Classroom"

    total_sentences = len(df)

    # Calculate duration safely
    if not df.empty and "start_sec" in df.columns and "end_sec" in df.columns:
        duration = df["end_sec"].max() - df["start_sec"].min()
    else:
        duration = 0.0

    # 2. Initialize Counters
    metrics = {
        "Q Sent KCHI": 0,
        "Q Sent FEM": 0,
        "NQ Sent KCHI": 0,
        "NQ Sent FEM": 0,
        "Q Resp f KCHI": 0,
        "Q Resp f FEM": 0,
        "NQ Resp f KCHI": 0,
        "NQ Resp f FEM": 0,
        "KCHI Turns": 0,
        "FEM Turns": 0,
    }

    # Storage for lexical calculations
    sentences = {"KCHI": [], "FEM": [], "ALL": []}

    # Memory for turn-taking to prevent double-counting floor transfers
    claimed_turn_indices = set()

    # 3. Main Loop: Iterate through rows to calculate occurrences and responses
    for idx, row in df.iterrows():
        speaker = str(row.get("speaker", "")).strip()
        sentence = str(row.get("sentence", ""))
        start_sec = row.get("start_sec", 0)
        end_sec = row.get("end_sec", 0)

        sentences["ALL"].append(sentence)
        is_question = "?" in sentence

        if speaker in ["KCHI", "FEM"]:
            sentences[speaker].append(sentence)

            # Tally Sentences
            if speaker == "KCHI":
                if is_question:
                    metrics["Q Sent KCHI"] += 1
                else:
                    metrics["NQ Sent KCHI"] += 1
            else:  # FEM
                if is_question:
                    metrics["Q Sent FEM"] += 1
                else:
                    metrics["NQ Sent FEM"] += 1

            # Look ahead for responses within the time window
            target_responder = "FEM" if speaker == "KCHI" else "KCHI"

            for j in range(idx + 1, len(df)):
                resp_row = df.iloc[j]
                resp_start = resp_row.get("start_sec", 0)
                resp_speaker = str(resp_row.get("speaker", "")).strip()

                # Break lookahead if we've passed the time window
                if resp_start > end_sec + response_window:
                    break

                # Check if it's a valid response from the target speaker
                if resp_start >= end_sec and resp_speaker == target_responder:

                    # A reply may count toward multiple prompts.
                    if speaker == "KCHI":
                        if is_question:
                            metrics["Q Resp f FEM"] += 1
                        else:
                            metrics["NQ Resp f FEM"] += 1
                    elif speaker == "FEM":
                        if is_question:
                            metrics["Q Resp f KCHI"] += 1
                        else:
                            metrics["NQ Resp f KCHI"] += 1

                    # Count each qualifying reply row once as a turn.
                    if j not in claimed_turn_indices:
                        claimed_turn_indices.add(
                            j
                        )  # Add to memory so it can't be claimed again
                        if speaker == "KCHI":
                            # KCHI yielded the floor to FEM
                            metrics["FEM Turns"] += 1
                        elif speaker == "FEM":
                            # FEM yielded the floor to KCHI
                            metrics["KCHI Turns"] += 1

                    break  # Use the first qualifying response for this prompt.

    # 4. Word Counts & Lexical Diversity
    kchi_total_words, kchi_uni_words = get_word_stats(sentences["KCHI"])
    fem_total_words, fem_uni_words = get_word_stats(sentences["FEM"])
    all_total_words, all_uni_words = get_word_stats(sentences["ALL"])

    # 5. Derived Metrics
    kchi_total_utt = metrics["Q Sent KCHI"] + metrics["NQ Sent KCHI"]
    fem_total_utt = metrics["Q Sent FEM"] + metrics["NQ Sent FEM"]
    total_conversational_turns = metrics["KCHI Turns"] + metrics["FEM Turns"]

    # Final Output Dictionary
    return {
        "SubjectID": subject_id,
        "Context": context,
        "Classroom": classroom,
        "Year": year,
        "Date": date_str,
        "Total Sentences": total_sentences,
        # Turn-Taking Metrics (Strict Transfer)
        "Total Conversational Turns": total_conversational_turns,
        "KCHI Turns": metrics["KCHI Turns"],
        "FEM Turns": metrics["FEM Turns"],
        # Sentence Type & Responsivity Metrics (Density)
        "Q Sent KCHI": metrics["Q Sent KCHI"],
        "Q Sent FEM": metrics["Q Sent FEM"],
        "NQ Sent KCHI": metrics["NQ Sent KCHI"],
        "NQ Sent FEM": metrics["NQ Sent FEM"],
        "Q Resp f KCHI": metrics["Q Resp f KCHI"],
        "Q Resp f FEM": metrics["Q Resp f FEM"],
        "NQ Resp f KCHI": metrics["NQ Resp f KCHI"],
        "NQ Resp f FEM": metrics["NQ Resp f FEM"],
        # Rounded Floating Point Variables
        "Duration": school_round(duration),
        "KCHI Utt per Minute": school_round(safe_divide(kchi_total_utt, duration) * 60),
        "FEM Utt per Minute": school_round(safe_divide(fem_total_utt, duration) * 60),
        "KCHI Q Rate": school_round(
            safe_divide(metrics["Q Sent KCHI"], kchi_total_utt)
        ),
        "FEM Q Rate": school_round(safe_divide(metrics["Q Sent FEM"], fem_total_utt)),
        "R Rate f KCHI": school_round(
            safe_divide(metrics["Q Resp f KCHI"], metrics["Q Sent FEM"])
        ),
        "R Rate f FEM": school_round(
            safe_divide(metrics["Q Resp f FEM"], metrics["Q Sent KCHI"])
        ),
        "KCHI MLU": school_round(safe_divide(kchi_total_words, kchi_total_utt)),
        "FEM MLU": school_round(safe_divide(fem_total_words, fem_total_utt)),
        # Lexical Variables
        "FEM Uni Words": fem_uni_words,
        "FEM Total Number of Words": fem_total_words,
        "FEM LexDiv": school_round(safe_divide(fem_uni_words, fem_total_words)),
        "KCHI Uni Words": kchi_uni_words,
        "KCHI Total Number of Words": kchi_total_words,
        "KCHI LexDiv": school_round(safe_divide(kchi_uni_words, kchi_total_words)),
        "Total Unique Words": all_uni_words,
        "Num of Total Words": all_total_words,
        "Total LexDiv": school_round(safe_divide(all_uni_words, all_total_words)),
    }


# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
def process_data(input_path, output_csv_path, response_window=2.5):
    """
    Takes a CSV file or a root containing */Whisper-VTC-Aligned/*.csv,
    processes them, and saves the consolidated metrics to a single CSV.
    """
    all_metrics = []

    # Determine if input is a file or directory
    if os.path.isfile(input_path):
        files_to_process = [input_path]
    elif os.path.isdir(input_path):
        # Uses targeted pattern mapping to strictly enforce the folder hierarchy.
        search_pattern = os.path.join(input_path, "*", "Whisper-VTC-Aligned", "*.csv")
        files_to_process = glob.glob(search_pattern)
    else:
        print(f"Error: {input_path} is not a valid file or directory.")
        return

    print(f"Found {len(files_to_process)} file(s) to process. Starting...")

    for file in files_to_process:
        try:
            print(f"Processing: {os.path.basename(file)}...")
            file_metrics = process_transcript(file, response_window=response_window)
            all_metrics.append(file_metrics)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    if all_metrics:
        # Convert list of dictionaries to DataFrame
        final_df = pd.DataFrame(all_metrics)

        # Sort by date and then natural subject ID.
        # 1. Create a temporary column to parse strings into actual datetime objects for chronological sorting
        final_df["_sort_date"] = pd.to_datetime(final_df["Date"], errors="coerce")
        # 2. Create a temporary column for natural alphanumeric sorting tuples
        final_df["_sort_subj"] = final_df["SubjectID"].map(get_natural_sort_key)

        # 3. Sort by the datetimes first, and then the subject IDs second
        final_df = final_df.sort_values(by=["_sort_date", "_sort_subj"]).reset_index(
            drop=True
        )

        # 4. Drop the invisible temp columns so they don't appear in the CSV
        final_df = final_df.drop(columns=["_sort_date", "_sort_subj"])

        # Safe Directory Creation
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Safe Version Bumping
        original_output_path = output_csv_path
        base, ext = os.path.splitext(output_csv_path)
        counter = 2

        while os.path.exists(output_csv_path):
            output_csv_path = f"{base}_v{counter}{ext}"
            counter += 1

        if original_output_path != output_csv_path:
            print(f"\nNotice: Target file exists. Renaming output to avoid overwrite.")

        # Save the file
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nSuccess! Consolidated metrics saved to: {output_csv_path}")
    else:
        print("\nNo data was processed.")


def main():
    """Read command-line settings and export one summary row per input file."""
    parser = argparse.ArgumentParser(
        description="Calculate TALK measures from Whisper/VTC aligned CSV files."
    )
    parser.add_argument(
        "input_path",
        help="Aligned CSV file, or root containing */Whisper-VTC-Aligned/*.csv.",
    )
    parser.add_argument("output_csv_path", help="Destination summary CSV path.")
    parser.add_argument(
        "--response-window",
        type=float,
        default=2.5,
        metavar="SECONDS",
        help="Maximum gap from prompt end to response start (default: 2.5).",
    )
    args = parser.parse_args()
    process_data(args.input_path, args.output_csv_path, args.response_window)


if __name__ == "__main__":
    main()
