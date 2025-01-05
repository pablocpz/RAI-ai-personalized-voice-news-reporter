import os
import tempfile
import soundfile as sf
from openai import OpenAI
# openai_client = OpenAI()
from dotenv import load_dotenv
load_dotenv()

# def transcribe_audio(recording, fs=44100):  # sample rate):
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
#         sf.write(temp_audio.name, recording, fs)
#         temp_audio.close()
#         with open(temp_audio.name, "rb") as audio_file:
#             # transcript = openai.Audio.transcribe("whisper-1", audio_file)
#             transcription = openai_client.audio.transcriptions.create(
#                 model="whisper-1", 
#                 file=audio_file, 
#                 response_format="text"
#             )
#         os.remove(temp_audio.name)
#     return transcription



openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def transcribe_audio(recording, fs=44100):  # sample rate)
    # Load the audio file as a NumPy array
    audio_data, _ = sf.read(recording)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        os.remove(temp_audio.name)
    return transcription