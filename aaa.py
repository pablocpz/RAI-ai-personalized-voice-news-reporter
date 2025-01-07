import os
import queue
import tempfile
import threading
import time
import pygame
import soundfile as sf
import pyaudio
from dotenv import load_dotenv
from openai import OpenAI

def text_to_tts(input_text):
    """
    Generate and play audio from input text using OpenAI's TTS API.

    :param input_text: The input text to be converted into speech.
    """
    load_dotenv()
    client = OpenAI()
    audio_generation_queue = queue.Queue()
    audio_playback_queue = queue.Queue()
    pygame.mixer.init()

    def generate_audio(text, model='tts-1', voice='nova'):
        """Generate TTS audio using OpenAI's API."""
        try:
            response = client.audio.speech.create(
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
            print(f"Error generating TTS audio: {e}")
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
            audio_file_path = generate_audio(text)
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

    start_time = time.time()

    # Cleanup
    audio_generation_queue.join()
    audio_generation_queue.put(None)
    audio_playback_queue.join()
    audio_playback_queue.put(None)

    audio_generation_thread.join()
    audio_playback_thread.join()
    pygame.mixer.quit()



# text_to_tts("Hello how are you. my name is pablo")