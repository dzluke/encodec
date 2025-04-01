import argparse
from pathlib import Path
from encodec.__main__ import network_bending_main
import util

# Written with help of ChatGPT

# Toggle force rerun mode (set to True to redo all experiments, even if the files exist already)
FORCE_RERUN = True

# Define output folder for experiments and ensure it exists
OUTPUT_FOLDER = Path("./experiments")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Define test configurations with multiple r values per function
bending_tests = [
    {"func": util.add_full, "r_values": [0.5, 1, 2, -1]},
    {"func": util.multiply, "r_values": [1, 2, 4, -1]},
    {"func": util.multiply, "r_values": [1, 2, 4, -1]},
    {"func": util.threshold, "r_values": [1, 10, 100, 1000]},
    {"func": util.log, "r_values": [-1]},
    {"func": util.power, "r_values": [0, 2, 5]},
]

# Define test audio files
audio_files = ["test_48k.wav"]

# Define layers to apply bending at
layers = ["NA"]

# Run tests
for audio in audio_files:
    audio = Path(audio)
    audio_name = audio.stem  # Extract filename without extension
    audio_folder = OUTPUT_FOLDER / audio_name  # Create subfolder for each audio file
    audio_folder.mkdir(parents=True, exist_ok=True)  # Ensure subfolder exists

    for test in bending_tests:
        func = test["func"]
        func_name = func.__name__  # Infer function name dynamically

        for r in test["r_values"]:
            for layer in layers:
                # Create unique output filename for each experiment
                experiment_name = f"{func_name}_r{r}_layer{layer}.wav"
                output_path = audio_folder / experiment_name

                # Skip processing if file exists and FORCE_RERUN is False
                if output_path.exists() and not FORCE_RERUN:
                    print(f"Skipping existing experiment: {output_path}")
                    continue

                print(f"Processing: {output_path}")

                # Create bending function with the specific r value
                bending_fn = func(r)

                # Construct arguments
                args = argparse.Namespace(
                    input=audio,
                    output=output_path,
                    bending_fn=bending_fn,
                    layer=layer,
                    hq=True,  # Use stereo 48kHz model
                    lm=False,  # No advanced language model (faster)
                    force=True,  # Overwrite if file exists
                    rescale=True,
                    bandwidth=6
                )

                # Call the main processing function
                network_bending_main(args)

print("Done.")
