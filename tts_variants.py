
from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
import os

load_dotenv()
openai_client = OpenAI()

eleven_client = ElevenLabs(
    api_key=os.getenv("XI_API_KEY")  # Load API key from environment variable
)
from pydub.playback import play as pydub_play

from elevenlabs import Voice, VoiceSettings, play
import asyncio
from io import BytesIO
from pydub import AudioSegment
import time
import os
import queue
import tempfile
import threading
import time
import pygame
import soundfile as sf
import pyaudio
from dotenv import load_dotenv
from elevenlabs import ElevenLabs
from openai import OpenAI

def streamed_oai_response_tts(streaming_response, tts_provider="elevenlabs"):
    """
    Generate and play audio from input text using the selected TTS provider (ElevenLabs or OpenAI).

    :param streaming_response: The streaming response from OpenAI's Chat API.
    :param tts_provider: The TTS provider to use ("elevenlabs" or "openai").
    """
    load_dotenv()
    openai_client = OpenAI()
    eleven_client = ElevenLabs()
    audio_generation_queue = queue.Queue()
    audio_playback_queue = queue.Queue()
    pygame.mixer.init()
                                                                        #stability=0.8, similarity_boost=0.75
    def generate_audio_elevenlabs(text):
        """Generate TTS audio using ElevenLabs' API."""
        try:
            # audio_iterator = eleven_client.generate(
            #     text=text,
            #     voice_id="9BWtsMINqrJLrRacOk9x",
            #     model_id="eleven_turbo_v2_5",
            #     stream=True,
            #     # voice_settings={
            #     #     "stability": stability,
            #     #     "similarity_boost": similarity_boost
            #     # }
            # )
            
            audio_iterator = eleven_client.text_to_speech.convert(
                text=text,
                voice_id="9BWtsMINqrJLrRacOk9x",
                model_id="eleven_turbo_v2_5",
                # stream=True,
                # voice_settings={
                #     "stability": stability,
                #     "similarity_boost": similarity_boost
                # }
            )
            # Write the streamed audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                for chunk in audio_iterator:
                    temp_file.write(chunk)
                return temp_file.name
        except Exception as e:
            print(f"Error generating TTS audio with ElevenLabs: {e}")
        return None

    def generate_audio_openai(text, model="tts-1", voice="nova"):
        """Generate TTS audio using OpenAI's API."""
        try:
            response = openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="opus"
            )
            if hasattr(response, 'content') and response.content:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as temp_file:
                    temp_file.write(response.content)
                    return temp_file.name
        except Exception as e:
            print(f"Error generating TTS audio with OpenAI: {e}")
        return None

    def play_audio(audio_file_path):
        """Play the audio file at the specified path."""
        if audio_file_path:
            with sf.SoundFile(audio_file_path, 'r') as sound_file:
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16, channels=sound_file.channels, 
                                    rate=sound_file.samplerate, output=True)
                data = sound_file.read(1024, dtype='int16')
                while len(data) > 0:
                    stream.write(data.tobytes())
                    data = sound_file.read(1024, dtype='int16')
                stream.stop_stream()
                stream.close()
                audio.terminate()

    def process_audio_generation_queue():
        """Process the audio generation queue by generating audio for each sentence."""
        while True:
            text = audio_generation_queue.get()
            if text is None:
                break
            if tts_provider == "elevenlabs":
                audio_file_path = generate_audio_elevenlabs(text)
            elif tts_provider == "openai":
                audio_file_path = generate_audio_openai(text)
            else:
                print("Unsupported TTS provider specified.")
                audio_file_path = None

            if audio_file_path:
                audio_playback_queue.put(audio_file_path)
            audio_generation_queue.task_done()

    def process_audio_playback_queue():
        """Process the audio playback queue by playing each audio file."""
        while True:
            audio_file_path = audio_playback_queue.get()
            if audio_file_path is None:
                break
            play_audio(audio_file_path)
            audio_playback_queue.task_done()

    def print_w_stream(streaming_response):
        """Process the streaming response and queue sentences for TTS generation."""
        sentence = ''
        sentences = []
        sentence_end_chars = {'.', '?', '!', '\n'}

        for chunk in streaming_response:
            content = chunk.content
            if content is not None:
                for char in content:
                    sentence += char
                    if char in sentence_end_chars:
                        sentence = sentence.strip()
                        if sentence and sentence not in sentences:
                            sentences.append(sentence)
                            audio_generation_queue.put(sentence)
                            print(f"Queued sentence: {sentence}") 
                        sentence = ''
        return sentences

    # Start threads
    audio_generation_thread = threading.Thread(target=process_audio_generation_queue)
    audio_playback_thread = threading.Thread(target=process_audio_playback_queue)
    audio_generation_thread.start()
    audio_playback_thread.start()

    start_time = time.time()
    print_w_stream(streaming_response)

    # Cleanup
    audio_generation_queue.join()
    audio_generation_queue.put(None)
    audio_playback_queue.join()
    audio_playback_queue.put(None)

    audio_generation_thread.join()
    audio_playback_thread.join()
    pygame.mixer.quit()






def text_to_tts(input_text, provider="elevenlabs"):
    """
    Generate and play audio from input text using the selected TTS provider (ElevenLabs or OpenAI).

    :param input_text: The input text to be converted into speech.
    :param provider: The TTS provider to use ("elevenlabs" or "openai").
    """
    load_dotenv()

    eleven_client = ElevenLabs()
    openai_client = OpenAI()
    audio_generation_queue = queue.Queue()
    audio_playback_queue = queue.Queue()
    pygame.mixer.init()

    def generate_audio_elevenlabs(text):
        """Generate TTS audio using ElevenLabs' API."""
        try:
            
            # audio_iterator = eleven_client.generate(
            #     text=text,
            #     voice_id="9BWtsMINqrJLrRacOk9x",
            #     model_id="eleven_turbo_v2_5",
            #     stream=True,
            #     # voice_settings={
            #     #     "stability": stability,
            #     #     "similarity_boost": similarity_boost
            #     # }
            # )
            audio_iterator = eleven_client.text_to_speech.convert(
                text=text,
                voice_id="9BWtsMINqrJLrRacOk9x",
                model_id="eleven_turbo_v2_5",
                # stream=True,
                # voice_settings={
                #     "stability": stability,
                #     "similarity_boost": similarity_boost
                # }
            )
            # Write the streamed audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                for chunk in audio_iterator:
                    temp_file.write(chunk)
                return temp_file.name
        except Exception as e:
            print(f"Error generating TTS audio with ElevenLabs: {e}")
        return None

    def generate_audio_openai(text, model="tts-1", voice="nova"):
        """Generate TTS audio using OpenAI's API."""
        try:
            response = openai_client.audio.speech.create(
                model=model,
                voice=voice,
                input=text,
                response_format="opus"
            )
            if hasattr(response, 'content') and response.content:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as temp_file:
                    temp_file.write(response.content)
                    return temp_file.name
        except Exception as e:
            print(f"Error generating TTS audio with OpenAI: {e}")
        return None

    def play_audio(audio_file_path):
        """Play the audio file at the specified path."""
        if audio_file_path:
            
            elapsed_time = time.time() - start_time
            print(f"Time taken to start playing audio clip: {elapsed_time} seconds")
            
            with sf.SoundFile(audio_file_path, 'r') as sound_file:
                audio = pyaudio.PyAudio()
                stream = audio.open(format=pyaudio.paInt16, channels=sound_file.channels, 
                                    rate=sound_file.samplerate, output=True)
                data = sound_file.read(1024, dtype='int16')
                while len(data) > 0:
                    stream.write(data.tobytes())
                    data = sound_file.read(1024, dtype='int16')
                stream.stop_stream()
                stream.close()
                audio.terminate()

    def process_audio_generation_queue():
        """Process the audio generation queue by generating audio for each sentence."""
        while True:
            text = audio_generation_queue.get()
            if text is None:
                break
            if provider == "elevenlabs":
                audio_file_path = generate_audio_elevenlabs(text)
            elif provider == "openai":
                audio_file_path = generate_audio_openai(text)
            else:
                print("Unsupported provider specified.")
                audio_file_path = None

            if audio_file_path:
                audio_playback_queue.put(audio_file_path)
            audio_generation_queue.task_done()

    def process_audio_playback_queue():
        """Process the audio playback queue by playing each audio file."""
        while True:
            audio_file_path = audio_playback_queue.get()
            if audio_file_path is None:
                break
            play_audio(audio_file_path)
            audio_playback_queue.task_done()

    def split_into_sentences(text):
        """Split input text into sentences based on punctuation."""
        sentence = ''
        sentences = []
        sentence_end_chars = {'.', '?', '!', '\n'}

        for char in text:
            sentence += char
            if char in sentence_end_chars:
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
                sentence = ''

        if sentence.strip():
            sentences.append(sentence.strip())

        return sentences

    # Split the input text into sentences and add them to the generation queue
    sentences = split_into_sentences(input_text)
    for sentence in sentences:
        audio_generation_queue.put(sentence)
        print(f"Queued sentence: {sentence}") 

    # Start threads
    audio_generation_thread = threading.Thread(target=process_audio_generation_queue)
    audio_playback_thread = threading.Thread(target=process_audio_playback_queue)
    audio_generation_thread.start()
    audio_playback_thread.start()

    # Cleanup
    start_time = time.time()
    
    audio_generation_queue.join()
    audio_generation_queue.put(None)
    audio_playback_queue.join()
    audio_playback_queue.put(None)

    audio_generation_thread.join()
    audio_playback_thread.join()
    pygame.mixer.quit()



