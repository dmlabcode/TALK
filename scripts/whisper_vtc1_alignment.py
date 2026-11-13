"""Align Whisper CSV segments with VTC/ALICE RTTM speaker classes."""

import os
import re
import argparse
import pandas as pd


def parse_rttm(rttm_path):
    """Parses the ALICE RTTM file"""
    with open(rttm_path, "r") as file:
        rttm_contents = file.readlines()

    rttm_data = []
    for line in rttm_contents:
        parts = line.split()
        if len(parts) < 8:
            continue

        speaker_class = parts[7]
        if speaker_class in ["CHI", "FEM", "KCHI", "MAL"]:
            file_name = parts[1]
            start_time = float(parts[3])
            duration = float(parts[4])
            end_time = start_time + duration
            rttm_data.append([file_name, start_time, end_time, speaker_class, duration])

    return pd.DataFrame(
        rttm_data,
        columns=["file_name", "start_time", "end_time", "speaker_class", "duration"],
    )


def match_rttm_filename(target_filename, rttm_filenames):
    """Aligns CSV filenames with RTTM filenames using a cascading matching strategy."""
    target_lower = target_filename.lower()

    # 1. Tier 1: Hardcoded exceptions
    if "starfish_32" in target_lower and "066" in target_lower:
        if "_45_" in target_lower:
            return "Starfish_32_066_01_04172023_clip_030650to035150"
        elif "_10_" in target_lower:
            return "Starfish_32_066_01_04172023_clip_022400to023400"

    # 2. Tier 2: Dynamic Substring Matching
    sorted_rttm = sorted(rttm_filenames, key=len, reverse=True)
    for rname in sorted_rttm:
        if rname.lower() in target_lower:
            return rname

    # 3. Tier 3: Strict Regex Pattern (3 digits followed by 2 digits)
    regex_pattern = r"(\d{3}_\d{2})"
    csv_match = re.search(regex_pattern, target_filename)

    if csv_match:
        core_id = csv_match.group(1)
        for rname in rttm_filenames:
            if core_id in rname:
                return rname

    # 4. Tier 4: Token-based Fallback
    match = re.search(
        r"Whisper(?:Coding)?2?_(.*?)_(?:01|02)", target_filename, re.IGNORECASE
    )
    if match:
        core_id_string = match.group(1)
        excel_tokens = [
            t.lower() for t in core_id_string.split("_") if t.lower() != "child"
        ]

        for rname in rttm_filenames:
            rttm_tokens = [t.lower() for t in re.split(r"[-_]", rname)]
            if all(token in rttm_tokens for token in excel_tokens):
                return rname

    return None


def get_dominant_speaker(start, end, df_rttm_filtered):
    """Calculates overlaps and returns the ALICE speaker_class with max time."""
    if pd.isna(start) or pd.isna(end) or df_rttm_filtered.empty:
        return None

    mask = (df_rttm_filtered["start_time"] < end) & (
        df_rttm_filtered["end_time"] > start
    )
    overlaps = df_rttm_filtered.loc[mask].copy()

    if overlaps.empty:
        return "Unidentified_Speaker"

    overlaps["overlap_start"] = overlaps["start_time"].clip(lower=start)
    overlaps["overlap_end"] = overlaps["end_time"].clip(upper=end)
    overlaps["overlap_duration"] = overlaps["overlap_end"] - overlaps["overlap_start"]

    cumulative_times = overlaps.groupby("speaker_class")["overlap_duration"].sum()

    if cumulative_times.empty or cumulative_times.max() == 0:
        return "Unidentified_Speaker"

    return cumulative_times.idxmax()


def filter_whisper_loops(df):
    """Filter repeated sentences and alternating repetition using the existing rules."""
    valid_indices = []
    sentences = df["sentence"].fillna("").astype(str).str.strip().tolist()

    for i in range(len(sentences)):
        curr = sentences[i]

        if curr == "":
            valid_indices.append(df.index[i])
            continue

        drop = False

        if i >= 2 and curr == sentences[i - 1] == sentences[i - 2]:
            drop = True
        elif (
            i >= 4
            and curr == sentences[i - 2] == sentences[i - 4]
            and sentences[i - 1] == sentences[i - 3]
        ):
            drop = True
        elif (
            i >= 5
            and curr == sentences[i - 2] == sentences[i - 4]
            and sentences[i - 1] == sentences[i - 3] == sentences[i - 5]
        ):
            drop = True

        if not drop:
            valid_indices.append(df.index[i])

    return df.loc[valid_indices].copy()


def process_transcripts(
    input_path, rttm_path, custom_outpath=None, use_alt_labels=False
):
    print(f"Loading ALICE RTTM file: {rttm_path}")
    rttm_df = parse_rttm(rttm_path)
    rttm_unique_files = rttm_df["file_name"].unique()

    # 1. Determine target files based on input (File vs Directory)
    target_files = []

    if os.path.isfile(input_path):
        if not (
            input_path.endswith("_ALICE.csv") or input_path.endswith("_VTC_Aligned.csv")
        ):
            target_files.append(os.path.abspath(input_path))
        else:
            print(
                "The provided file already has an ALICE or VTC_Aligned suffix. Exiting."
            )
            return

    elif os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.endswith(".csv") and not (
                f.endswith("_ALICE.csv") or f.endswith("_VTC_Aligned.csv")
            ):
                target_files.append(os.path.join(input_path, f))

    if not target_files:
        print(f"No valid target CSV files found for input: {input_path}")
        return

    # 2. Determine Output Directory
    if custom_outpath:
        output_dir = os.path.abspath(custom_outpath)
    else:
        # Default behavior: Create 'Whisper-VTC-Aligned' alongside the input
        if os.path.isfile(input_path):
            base_dir = os.path.dirname(os.path.abspath(input_path))
        else:
            base_dir = os.path.dirname(os.path.abspath(input_path))

        output_dir = os.path.join(base_dir, "Whisper-VTC-Aligned")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(target_files)} files to process.")
    print(f"Outputs will be saved to: {output_dir}\n")

    alt_label_map = {
        "FEM": "Teacher",
        "MAL": "Male_Adult",
        "CHI": "Other_Child",
        "KCHI": "Main_Child",
    }

    failed_matches = []
    success_count = 0

    # 3. Process the files
    for csv_path in target_files:
        csv_filename = os.path.basename(csv_path)

        if len(rttm_unique_files) == 1:
            matched_rttm_name = rttm_unique_files[0]
        else:
            matched_rttm_name = match_rttm_filename(csv_filename, rttm_unique_files)

        if not matched_rttm_name:
            failed_matches.append(csv_filename)
            continue

        file_rttm_df = rttm_df[rttm_df["file_name"] == matched_rttm_name].copy()

        try:
            df_csv = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")
            continue

        df_filtered = filter_whisper_loops(df_csv)

        possible_time_cols = ["time_start", "timestamp", "time_stamp"]
        found_time_col = next(
            (col for col in possible_time_cols if col in df_filtered.columns), None
        )

        new_alice_labels = []
        for idx, row in df_filtered.iterrows():
            start = row.get("start_sec")
            end = row.get("end_sec")

            dominant_speaker = get_dominant_speaker(start, end, file_rttm_df)

            if use_alt_labels and dominant_speaker in alt_label_map:
                dominant_speaker = alt_label_map[dominant_speaker]

            new_alice_labels.append(dominant_speaker)

        df_filtered["speaker"] = new_alice_labels

        cols_to_export = ["start_sec", "end_sec"]
        if found_time_col:
            cols_to_export.append(found_time_col)
        cols_to_export.extend(["sentence", "speaker"])

        final_df = df_filtered[cols_to_export]

        base_name = os.path.splitext(csv_filename)[0]
        out_name = f"{base_name}_VTC_Aligned.csv"

        out_path = os.path.join(output_dir, out_name)
        final_df.to_csv(out_path, index=False)

        print(f"[\u2713] {csv_filename} -> {matched_rttm_name}")
        success_count += 1

    # 4. Terminal Summary Log
    print("\n" + "=" * 50)
    print(" ALIGNMENT BATCH SUMMARY")
    print("=" * 50)
    print(f" Successfully processed: {success_count} files")
    print(f" Failed to match:        {len(failed_matches)} files")

    if failed_matches:
        print("\n[!] The following files could not be matched to an RTTM ID:")
        for failed_file in failed_matches:
            print(f"    - {failed_file}")
    print("=" * 50 + "\n")


def main():
    """Parse command-line arguments and align the selected transcripts."""
    parser = argparse.ArgumentParser(
        description="Process Whisper CSVs and add ALICE speaker labels."
    )

    # Updated to handle a single file OR a directory
    parser.add_argument(
        "input_path",
        help="Path to a single Whisper .csv file OR a directory containing multiple CSVs.",
    )
    parser.add_argument(
        "rttm_file", help="Path to the ALICE diarization_output.rttm file."
    )

    # Standard optional boolean flag
    parser.add_argument(
        "--alt-labels",
        action="store_true",
        help="Use alternate speaker labels. If omitted, defaults to False.",
    )

    # Custom output directory argument
    parser.add_argument(
        "--outpath",
        type=str,
        default=None,
        help="Optional custom output directory path. If omitted, saves to a sibling 'Whisper-VTC-Aligned' folder.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: The input path '{args.input_path}' does not exist.")
    elif not os.path.isfile(args.rttm_file):
        print(f"Error: The RTTM file '{args.rttm_file}' does not exist.")
    else:
        process_transcripts(
            args.input_path, args.rttm_file, args.outpath, args.alt_labels
        )


if __name__ == "__main__":
    main()
