Okay, here's a Python application to transcribe an audio file and break out the conversation by unique speakers. This solution will leverage some powerful open-source libraries.

**Core Libraries:**

  * **`Whisper` by OpenAI:** For highly accurate speech-to-text transcription.
  * **`pyannote.audio`:** For speaker diarization (identifying who spoke when).

**Important Considerations for macOS:**

  * **`ffmpeg`:** Whisper and `pyannote.audio` often rely on `ffmpeg` for audio processing. You'll likely need to install this.
  * **PyTorch:** Whisper runs on PyTorch. The installation command will handle this, but it's a key dependency.
  * **Hugging Face Token:** `pyannote.audio` requires a Hugging Face token to download pre-trained models.

-----

## Python Application: Audio Transcription & Speaker Diarization

This application will take an audio file as input, transcribe the speech, identify different speakers, and then output the transcribed text attributed to each speaker.

### 1\. Installation ⚙️

First, you'll need to install the necessary libraries and tools. Open your Terminal on macOS and run the following commands:

**a. Install `ffmpeg` (if you don't have it):**

```bash
brew install ffmpeg
```

**b. Install Python Libraries:**

```bash
pip install openai-whisper pyannote.audio
```

  * This will also install PyTorch and other dependencies.

**c. Hugging Face Authentication (for `pyannote.audio`):**

1.  Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create an access token (it's free).
2.  You'll need to accept the terms of use for the speaker diarization models you intend to use (e.g., `pyannote/speaker-diarization-3.1`). Visit the model's page on Hugging Face and accept the terms.
3.  You can either log in via the Hugging Face CLI or set an environment variable. For simplicity in the script, we'll assume you might need to pass the token directly if the CLI login isn't persistent for the script's environment.

-----

### 2\. The Python Code 🐍

```python
import whisper
from pyannote.audio import Pipeline
import os
import datetime
import torch # Ensure PyTorch is imported to check for GPU availability

def format_timestamp(seconds):
    """Converts seconds to HH:MM:SS.milliseconds format."""
    delta = datetime.timedelta(seconds=seconds)
    return str(delta)

def transcribe_and_diarize_audio(audio_path, hf_token=None, model_size="base"):
    """
    Transcribes an audio file and breaks out the conversation by unique speakers.

    Args:
        audio_path (str): The path to the audio file.
        hf_token (str, optional): Your Hugging Face access token.
                                   Required if not logged in via huggingface-cli.
        model_size (str, optional): The Whisper model size (e.g., "tiny", "base", "small", "medium", "large").
                                    Larger models are more accurate but slower and require more resources.
                                    Defaults to "base".

    Returns:
        str: A string containing the transcribed conversation with speaker labels,
             or an error message if something goes wrong.
    """
    if not os.path.exists(audio_path):
        return "Error: Audio file not found."

    print(f"Starting transcription with Whisper model: {model_size}...")
    try:
        # Check for GPU availability
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # 1. Load Whisper model and transcribe
        whisper_model = whisper.load_model(model_size, device=device)
        transcription_result = whisper_model.transcribe(audio_path, verbose=False) # Set verbose=True for progress
        segments = transcription_result["segments"]
        language = transcription_result["language"]
        print(f"Transcription complete. Detected language: {language}")
    except Exception as e:
        return f"Error during Whisper transcription: {e}"

    print("Starting speaker diarization with pyannote.audio...")
    try:
        # 2. Load pyannote.audio pipeline for speaker diarization
        #    Make sure you have accepted the terms of use for these models on Hugging Face.
        #    Example model: 'pyannote/speaker-diarization-3.1'
        #    For older versions or other models, you might use: 'pyannote/speaker-diarization@2.1'
        #    Ensure the model name is correct and you have access.
        if hf_token:
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", # Or your preferred diarization model
                use_auth_token=hf_token
            )
        else: # Assumes you are logged in via huggingface-cli
             diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )

        # Move pipeline to GPU if available
        if device == "cuda":
            diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

        diarization = diarization_pipeline(audio_path, num_speakers=None) # Set num_speakers if known, otherwise None
        print("Speaker diarization complete.")
    except Exception as e:
        return f"Error during pyannote.audio diarization: {e}"

    # 3. Combine transcription and diarization
    print("Combining transcription and diarization results...")
    output_conversation = []
    current_speaker = None
    current_speech = ""

    # Create a list of speaker turns with start, end, and speaker label
    speaker_turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Match segments with speaker turns
    # This is a simplified matching logic. For highly overlapping speech or very short segments,
    # more sophisticated alignment might be needed.
    full_transcript_with_speakers = []

    for seg in segments:
        segment_start = seg['start']
        segment_end = seg['end']
        segment_text = seg['text'].strip()

        # Find which speaker was speaking during the midpoint of the segment
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

    # Consolidate continuous speech from the same speaker
    if not full_transcript_with_speakers:
        return "No speech segments found or could not align speakers."

    consolidated_output = []
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

    # Add the last speaker's text
    consolidated_output.append(
        f"[{format_timestamp(current_start_time)}] {current_speaker}: {current_text.strip()}"
    )

    print("Processing complete.")
    return "\n".join(consolidated_output)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    AUDIO_FILE_PATH = "YOUR_AUDIO_FILE.wav"  # Replace with your audio file path (e.g., .wav, .mp3, .m4a)
    HUGGING_FACE_TOKEN = "YOUR_HUGGING_FACE_TOKEN" # Optional: Replace with your token if needed, or set to None
    WHISPER_MODEL_SIZE = "base" # "tiny", "base", "small", "medium", "large"
                                # "base" is a good starting point.
    # --- END CONFIGURATION ---

    if AUDIO_FILE_PATH == "YOUR_AUDIO_FILE.wav":
        print("Please update 'AUDIO_FILE_PATH' in the script with the actual path to your audio file.")
    elif HUGGING_FACE_TOKEN == "YOUR_HUGGING_FACE_TOKEN" and not os.getenv("HUGGING_FACE_HUB_TOKEN"):
        # Check if the default model needs a token and it's not set
        # pyannote/speaker-diarization-3.1 requires a token
        print("Warning: 'HUGGING_FACE_TOKEN' is not set in the script, and no environment variable found.")
        print("pyannote.audio models may require authentication. The script might fail or ask you to log in.")
        print("You can create a token at https://huggingface.co/settings/tokens")
        user_input_token = input("Enter your Hugging Face Token (or press Enter to try without it): ").strip()
        if user_input_token:
            HUGGING_FACE_TOKEN = user_input_token
        else:
            HUGGING_FACE_TOKEN = None # Explicitly set to None if user skips

    print(f"\nProcessing audio file: {AUDIO_FILE_PATH}")
    conversation = transcribe_and_diarize_audio(AUDIO_FILE_PATH, HUGGING_FACE_TOKEN, WHISPER_MODEL_SIZE)
    print("\n--- Transcribed Conversation ---")
    print(conversation)
    print("--- End of Conversation ---")

    # Optionally, save to a file
    output_filename = os.path.splitext(AUDIO_FILE_PATH)[0] + "_transcript.txt"
    try:
        with open(output_filename, "w") as f:
            f.write(f"Transcription for: {AUDIO_FILE_PATH}\n")
            f.write(f"Whisper model: {WHISPER_MODEL_SIZE}\n")
            f.write("Diarization model: pyannote/speaker-diarization-3.1 (or similar)\n") # Update if you change model
            f.write("--- Conversation ---\n")
            f.write(conversation)
        print(f"\nTranscript saved to: {output_filename}")
    except Exception as e:
        print(f"\nError saving transcript to file: {e}")
```

-----

### 3\. How to Run the Application 🚀

1.  **Save the Code:** Save the Python code above as a `.py` file (e.g., `transcriber_app.py`).

2.  **Configure:**

      * Open the `transcriber_app.py` file.
      * **`AUDIO_FILE_PATH`**: Change `"YOUR_AUDIO_FILE.wav"` to the actual path of your audio file (e.g., `"/Users/yourname/Documents/meeting.mp3"`). Most common audio formats should work.
      * **`HUGGING_FACE_TOKEN`**:
          * If you've logged in using `huggingface-cli login` and it's recognized by your environment, you might be able to set this to `None`.
          * Otherwise, replace `"YOUR_HUGGING_FACE_TOKEN"` with the actual token you generated.
      * **`WHISPER_MODEL_SIZE`**: You can start with `"base"`. If you need higher accuracy and have a good Mac (especially one with M1/M2/M3 for decent CPU performance, or if you manage to get GPU acceleration working for PyTorch), you can try `"small"`, `"medium"`, or even `"large"`. Larger models are slower and use more memory.

3.  **Run from Terminal:**

      * Navigate to the directory where you saved the file.
      * Execute the script:

    <!-- end list -->

    ```bash
    python transcriber_app.py
    ```

### 4\. Output 📝

The application will print the transcribed conversation to the console, with each line prefixed by a speaker label (e.g., `SPEAKER_00`, `SPEAKER_01`) and a timestamp. It will also save this output to a `.txt` file in the same directory as your audio file.

**Example Output:**

```
--- Transcribed Conversation ---
[0:00:01.234000] SPEAKER_00: Hello, this is a test recording.
[0:00:03.456000] SPEAKER_01: Yes, I can hear you. We are testing the diarization.
[0:00:06.789000] SPEAKER_00: Great, it seems to be working. Let's see if it correctly identifies us.
[0:00:10.112000] SPEAKER_01: Hopefully, the overlap is handled well too.
--- End of Conversation ---

Transcript saved to: YOUR_AUDIO_FILE_transcript.txt
```

-----

### Important Notes and Potential Improvements:

  * **Accuracy:**
      * **Whisper Model:** Using larger Whisper models (`medium`, `large`) will significantly improve transcription accuracy but will be slower.
      * **Audio Quality:** Clear audio with minimal background noise and distinct speakers will yield the best results.
      * **Diarization Model:** `pyannote.audio` has different pre-trained models. `pyannote/speaker-diarization-3.1` is a good recent choice, but you can explore others on Hugging Face. Some models might be better for specific scenarios (e.g., few vs. many speakers).
  * **Performance:**
      * **CPU vs. GPU:** Whisper and `pyannote` can be computationally intensive. On a MacBook Pro, it will primarily use the CPU unless you have a setup that allows PyTorch to effectively use the M-series GPU (this can be complex and support varies). The script includes a basic check for CUDA (NVIDIA GPU), which isn't standard on Macs, but the CPU fallback will work.
      * **Long Audio Files:** For very long files, consider processing them in chunks if memory becomes an issue, though Whisper is quite robust.
  * **Speaker Identification:** `pyannote.audio` assigns generic labels like `SPEAKER_00`, `SPEAKER_01`. It doesn't identify *who* the speaker is by name. For that, you'd need speaker recognition technology, which is a more complex task requiring voice samples for each known speaker.
  * **Overlapping Speech:** Diarization of heavily overlapping speech is challenging. The accuracy of speaker attribution might decrease in such segments.
  * **Error Handling:** The provided script includes basic error handling, but you could expand it for more robustness.
  * **Dependencies Management:** For larger projects, consider using virtual environments (`venv` or `conda`) to manage dependencies.
  * **`pyannote.audio` Model Access:** Always ensure you have accepted the terms of use on Hugging Face for the specific `pyannote.audio` models you are using. If the script has issues downloading/using a diarization model, this is a common reason.
