Okay, let's modify the script to accept the audio file name (or path) as a command-line input argument. We'll continue to use the `.env` file for the Hugging Face token and model configurations.

We'll use the `argparse` module, which is Python's standard way to handle command-line arguments.

### 1. Keep `.env` File for Secrets and Model Config

Your `.env` file should still contain:

```env
# .env file
HUGGING_FACE_TOKEN="your_actual_hugging_face_token_here"
WHISPER_MODEL_SIZE="base"
# Optional: Set HUGGING_FACE_DIARIZATION_MODEL if you want to use a specific one
HUGGING_FACE_DIARIZATION_MODEL="pyannote/speaker-diarization-3.1"
```

The `AUDIO_FILE_PATH` will now come from the command line.

---

### 2. Updated Python Code 🐍

```python
import whisper
from pyannote.audio import Pipeline
import os
import datetime
import torch # Ensure PyTorch is imported to check for GPU availability
from dotenv import load_dotenv
import argparse # Import argparse

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS.milliseconds format."""
    delta = datetime.timedelta(seconds=seconds)
    return str(delta)

def transcribe_and_diarize_audio(audio_path, hf_token=None, whisper_model_size="base", diarization_model_name=None):
    """
    Transcribes an audio file and breaks out the conversation by unique speakers.

    Args:
        audio_path (str): The path to the audio file.
        hf_token (str, optional): Your Hugging Face access token.
                                   Required if not logged in via huggingface-cli.
        whisper_model_size (str, optional): The Whisper model size (e.g., "tiny", "base", "small", "medium", "large").
                                    Defaults to "base".
        diarization_model_name (str, optional): The pyannote.audio model name.
                                                 Defaults to "pyannote/speaker-diarization-3.1".

    Returns:
        str: A string containing the transcribed conversation with speaker labels,
             or an error message if something goes wrong.
    """
    if not audio_path or not os.path.exists(audio_path):
        return f"Error: Audio file not found at '{audio_path}'. Please check the provided file path."

    print(f"Starting transcription with Whisper model: {whisper_model_size}...")
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 1. Load Whisper model and transcribe
        whisper_model = whisper.load_model(whisper_model_size, device=device)
        transcription_result = whisper_model.transcribe(audio_path, verbose=False)
        segments = transcription_result["segments"]
        language = transcription_result["language"]
        print(f"Transcription complete. Detected language: {language}")
    except Exception as e:
        return f"Error during Whisper transcription: {e}"

    print("Starting speaker diarization with pyannote.audio...")
    try:
        # 2. Load pyannote.audio pipeline for speaker diarization
        if diarization_model_name is None:
            diarization_model_name = "pyannote/speaker-diarization-3.1" # Default model
        print(f"Using diarization model: {diarization_model_name}")

        if not hf_token and "pyannote" in diarization_model_name:
            print("Warning: Hugging Face token not explicitly provided for a pyannote model.")
            print("Attempting to use model, which might rely on a global Hugging Face login.")
            print("If this fails, ensure HUGGING_FACE_TOKEN is set in your .env file or you are logged in via `huggingface-cli login`.")

        pipeline_args = {"use_auth_token": hf_token} if hf_token else {}
        diarization_pipeline = Pipeline.from_pretrained(diarization_model_name, **pipeline_args)

        if device == "cuda":
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

        diarization = diarization_pipeline(audio_path, num_speakers=None)
        print("Speaker diarization complete.")
    except Exception as e:
        return f"Error during pyannote.audio diarization: {e}"

    # 3. Combine transcription and diarization
    print("Combining transcription and diarization results...")
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    full_transcript_with_speakers = []
    for seg in segments:
        segment_start = seg['start']
        segment_end = seg['end']
        segment_text = seg['text'].strip()
        segment_midpoint = segment_start + (segment_end - segment_start) / 2
        assigned_speaker = "Unknown Speaker"

        for turn in speaker_turns:
            if turn['start'] <= segment_midpoint < turn['end']:
                assigned_speaker = turn['speaker']
                break
        full_transcript_with_speakers.append({
            "start": segment_start,
            "end": segment_end,
            "speaker": assigned_speaker,
            "text": segment_text
        })

    if not full_transcript_with_speakers:
        return "No speech segments found or could not align speakers."

    consolidated_output = []
    if full_transcript_with_speakers:
        current_speaker = full_transcript_with_speakers[0]['speaker']
        current_text = ""
        current_start_time = full_transcript_with_speakers[0]['start']

        for entry in full_transcript_with_speakers:
            if entry['speaker'] == current_speaker:
                current_text += " " + entry['text']
            else:
                consolidated_output.append(
                    f"[{format_timestamp(current_start_time)}] {current_speaker}: {current_text.strip()}"
                )
                current_speaker = entry['speaker']
                current_text = entry['text']
                current_start_time = entry['start']
        consolidated_output.append(
            f"[{format_timestamp(current_start_time)}] {current_speaker}: {current_text.strip()}"
        )

    print("Processing complete.")
    return "\n".join(consolidated_output)

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Transcribe an audio file and perform speaker diarization.")
    parser.add_argument("audio_file", help="Path to the audio file to process.")
    args = parser.parse_args()
    # --- End Argument Parsing ---

    # Load environment variables from .env file for other configurations
    load_dotenv()

    # --- CONFIGURATION FROM .env AND ARGS ---
    AUDIO_FILE_PATH = args.audio_file # Get audio file path from command-line argument
    HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
    WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
    DIARIZATION_MODEL = os.getenv("HUGGING_FACE_DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")
    # --- END CONFIGURATION ---

    if not HUGGING_FACE_TOKEN and "pyannote" in DIARIZATION_MODEL:
         print("Warning: 'HUGGING_FACE_TOKEN' is not set in .env file or environment variables.")
         print(f"Attempting to use diarization model '{DIARIZATION_MODEL}' which might require authentication.")
         print("If diarization fails, ensure the token is set or you are logged in via 'huggingface-cli login'.")

    print(f"\nProcessing audio file: {AUDIO_FILE_PATH}")
    print(f"Using Whisper model: {WHISPER_MODEL_SIZE}")
    print(f"Using Diarization model: {DIARIZATION_MODEL}")

    conversation = transcribe_and_diarize_audio(
        AUDIO_FILE_PATH,
        HUGGING_FACE_TOKEN,
        WHISPER_MODEL_SIZE,
        DIARIZATION_MODEL
    )

    print("\n--- Transcribed Conversation ---")
    print(conversation)
    print("--- End of Conversation ---")

    if AUDIO_FILE_PATH and not conversation.startswith("Error:") :
        output_filename = os.path.splitext(AUDIO_FILE_PATH)[0] + "_transcript.txt"
        try:
            with open(output_filename, "w") as f:
                f.write(f"Transcription for: {AUDIO_FILE_PATH}\n")
                f.write(f"Whisper model: {WHISPER_MODEL_SIZE}\n")
                f.write(f"Diarization model: {DIARIZATION_MODEL}\n")
                f.write("--- Conversation ---\n")
                f.write(conversation)
            print(f"\nTranscript saved to: {output_filename}")
        except Exception as e:
            print(f"\nError saving transcript to file: {e}")
    elif conversation.startswith("Error:"):
        print(f"\nSkipping saving transcript due to processing error: {conversation}")

```

---

### 3. How to Run the Application (with command-line argument) 🚀

1.  **Ensure `python-dotenv` is installed:**
    ```bash
    pip install python-dotenv
    ```
2.  **Create/Update `.env` file:** Make sure your `.env` file (in the same directory as the script) contains `HUGGING_FACE_TOKEN`, and optionally `WHISPER_MODEL_SIZE` and `HUGGING_FACE_DIARIZATION_MODEL`.
3.  **Save the Python Code:** Save the updated Python code as `transcriber_app.py` (or your preferred name).
4.  **Run from Terminal:**
    * Navigate to the directory where you saved the script and your `.env` file.
    * Execute the script, providing the audio file path as an argument:

    ```bash
    python transcriber_app.py "/path/to/your/audiofile.wav"
    ```
    or
    ```bash
    python transcriber_app.py meeting_audio.mp3
    ```
    (If `meeting_audio.mp3` is in the same directory as the script).

    **Replace `"/path/to/your/audiofile.wav"` or `meeting_audio.mp3` with the actual path to your audio file.**

### Key Changes and Explanations:

* **`import argparse`**: Imports the module.
* **`parser = argparse.ArgumentParser(...)`**: Creates an ArgumentParser object.
* **`parser.add_argument("audio_file", help="Path to the audio file to process.")`**:
    * This defines a *positional argument* named `audio_file`.
    * The script will expect one argument after `python transcriber_app.py`, and that will be treated as the `audio_file`.
    * The `help` string provides information if the user runs the script with `-h` or `--help`.
* **`args = parser.parse_args()`**: Parses the command-line arguments. If the required arguments are not provided, `argparse` will automatically show an error and the help message.
* **`AUDIO_FILE_PATH = args.audio_file`**: The value of the `audio_file` argument provided on the command line is assigned to the `AUDIO_FILE_PATH` variable.
* The rest of the script remains largely the same, using `AUDIO_FILE_PATH` as before, but now its value comes from the command line instead of the `.env` file.
* The `.env` file is still loaded using `load_dotenv()` to get `HUGGING_FACE_TOKEN`, `WHISPER_MODEL_SIZE`, and `DIARIZATION_MODEL`.

This makes the script more flexible for processing different audio files without needing to edit the `.env` file each time for the audio path.
