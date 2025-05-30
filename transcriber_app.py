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
