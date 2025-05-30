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
