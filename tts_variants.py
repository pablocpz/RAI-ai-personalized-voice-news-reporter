
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


def sync_tts(text:str):
    """
    Reproduce the audio response given an input text 
    
    normal strategy, no chunking, no async, no pararell, threads..etc
    
    ps: using openai, it can be elevenlabs too, just check
    """
    
    response = openai_client.audio.speech.create(
        model="tts-1",
        voice="nova",
        input=text,
        # language="en" #harcoded output language for TTS
        
    )

    if hasattr(response, 'content') and response.content:
        # Use your play function to play the audio instantly
        play(response.content)
    elif hasattr(response, 'url'):
        # If the API provides a URL, use your play function to play the audio from URL
        audio_url = response.url
        pydub_play(audio_url)
    else:
        print("No valid audio content found in the response.")  


def sync_tts_eleven(text:str):
    """
    Reproduce the audio response given an input text 
    
    normal strategy, no chunking, no async, no pararell, threads..etc
    
    ELEVENLABS VERSION
    """
    
    audio = eleven_client.generate(
            text=text,
            voice="Aria",  # You can replace "Aria" with any other voice ID or name
            model="eleven_multilingual_v2",  # Specify the model version to use
            stream=True,
            voice_settings={
                "stability": 0.8,  # Adjust stability (0.0 - 1.0)
                "similarity_boost": 0.75  # Adjust the likeness to the selected voice
            }
        )
    
    play(audio=audio)
        
async def speak_chunk_openai(chunk):
    """Asynchronous function to get speech from OpenAI and play the audio."""
    start_time = time.time()

    # Using OpenAI's Python client to generate audio
    response = openai_client.audio.speech.create(
        model="tts-1",  # Ensure you're using the correct model
        voice="nova",  # Specify the voice
        input=chunk  # Text chunk to convert to speech
    )

    # Extract audio content from the response
    audio_data = response.content  # Correct way to access the content
    audio = AudioSegment.from_mp3(BytesIO(audio_data))  # Convert binary to AudioSegment
    pydub_play(audio)  # Play the audio

    end_time = time.time()
    print(f"Chunk processed in {end_time - start_time:.2f} seconds.")

async def async_chunking_tts(text, chunk_size=50):
    """Asynchronous function to process text in chunks and speak."""
    words = text.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Run the tasks concurrently
    tasks = [speak_chunk_openai(chunk) for chunk in chunks]
    await asyncio.gather(*tasks)  # Wait for all chunks to be processed concurrently


#example usage await chunked_speak_openai(text, chunk_size=10)  # Use await instead of asyncio.run()


import requests, pyaudio, time, pygame, threading, queue, tempfile
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

def streamed_tts(input_text):
    """
    Generate and play audio from input text using OpenAI's TTS API.

    :param input_text: The input text to be processed into audio.
    """
    load_dotenv()
    client = OpenAI()
    audio_generation_queue = queue.Queue()
    audio_playback_queue = queue.Queue()
    pygame.mixer.init()

    def generate_audio(text, model='tts-1', voice='nova'):
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
        while True:
            text = audio_generation_queue.get()
            if text is None:
                break
            audio_file_path = generate_audio(text)
            if audio_file_path:
                audio_playback_queue.put(audio_file_path)
            audio_generation_queue.task_done()

    def process_audio_playback_queue():
        while True:
            audio_file_path = audio_playback_queue.get()
            if audio_file_path is None:
                break
            play_audio(audio_file_path)
            audio_playback_queue.task_done()

    def print_w_stream(message):
        completion = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": "You are a friendly AI assistant."},
                {"role": "user", "content": message},
            ],
            stream=True,
            temperature=0,
            max_tokens=500,
        )

        sentence = ''
        sentences = []
        sentence_end_chars = {'.', '?', '!', '\n'}

        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content is not None:
                for char in content:
                    sentence += char
                    if char in sentence_end_chars:
                        sentence = sentence.strip()
                        if sentence and sentence not in sentences:
                            sentences.append(sentence)
                            audio_generation_queue.put(sentence)
                        sentence = ''
        return sentences

    # Start threads
    audio_generation_thread = threading.Thread(target=process_audio_generation_queue)
    audio_playback_thread = threading.Thread(target=process_audio_playback_queue)
    audio_generation_thread.start()
    audio_playback_thread.start()

    start_time = time.time()
    print_w_stream(input_text)

    # Cleanup
    audio_generation_queue.join()
    audio_generation_queue.put(None)
    audio_playback_queue.join()
    audio_playback_queue.put(None)

    audio_generation_thread.join()
    audio_playback_thread.join()
    pygame.mixer.quit()
    
    
    
from langchain_openai import ChatOpenAI

# def streamed_tts_langchain(model:ChatOpenAI, text:str):
#     """
#     Generate and play audio from input text using OpenAI's TTS API.

#     :param input_text: The input text to be processed into audio.
#     """
#     load_dotenv()
#     client = OpenAI()
#     audio_generation_queue = queue.Queue()
#     audio_playback_queue = queue.Queue()
#     pygame.mixer.init()

#     def generate_audio(text, model='tts-1', voice='nova'):
#         try:
#             response = client.audio.speech.create(
#                 model=model,
#                 voice=voice,
#                 input=text,
#                 response_format="opus"
#             )
#             if hasattr(response, 'content') and response.content:
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as temp_file:
#                     temp_file.write(response.content)
#                     return temp_file.name
#         except Exception as e:
#             print(f"Error generating TTS audio: {e}")
#         return None

#     def play_audio(audio_file_path):
#         if audio_file_path:
            
#             elapsed_time = time.time() - start_time
#             print(f"Time taken to start playing audio clip: {elapsed_time} seconds")
        
#             with sf.SoundFile(audio_file_path, 'r') as sound_file:
#                 audio = pyaudio.PyAudio()
#                 stream = audio.open(format=pyaudio.paInt16, channels=sound_file.channels, 
#                                     rate=sound_file.samplerate, output=True)
#                 data = sound_file.read(1024, dtype='int16')
#                 while len(data) > 0:
#                     stream.write(data.tobytes())
#                     data = sound_file.read(1024, dtype='int16')
#                 stream.stop_stream()
#                 stream.close()
#                 audio.terminate()

#     def process_audio_generation_queue():
#         while True:
#             text = audio_generation_queue.get()
#             if text is None:
#                 break
#             audio_file_path = generate_audio(text)
#             if audio_file_path:
#                 audio_playback_queue.put(audio_file_path)
#             audio_generation_queue.task_done()

#     def process_audio_playback_queue():
#         while True:
#             audio_file_path = audio_playback_queue.get()
#             if audio_file_path is None:
#                 break
#             play_audio(audio_file_path)
#             audio_playback_queue.task_done()

#     def print_w_stream(message):
#         # completion = client.chat.completions.create(
#         #     model='gpt-3.5-turbo',
#         #     messages=[
#         #         {"role": "system", "content": "You are a friendly AI assistant."},
#         #         {"role": "user", "content": message},
#         #     ],
#         #     stream=True,
#         #     temperature=0,
#         #     max_tokens=500,
#         # )
        
#         model_streaming = model.stream(message, max_completion_tokens=500,
#                                        temperature=0)

#         sentence = ''
#         sentences = []
#         sentence_end_chars = {'.', '?', '!', '\n'}

#         for chunk in model_streaming:
#             content = chunk.content
#             if content is not None:
#                 for char in content:
#                     sentence += char
#                     if char in sentence_end_chars:
#                         sentence = sentence.strip()
#                         if sentence and sentence not in sentences:
#                             sentences.append(sentence)
#                             audio_generation_queue.put(sentence)
#                             print(f"Queued sentence: {sentence}") 
                            
#                         sentence = ''
#         return sentences

#     # Start threads
#     audio_generation_thread = threading.Thread(target=process_audio_generation_queue)
#     audio_playback_thread = threading.Thread(target=process_audio_playback_queue)
#     audio_generation_thread.start()
#     audio_playback_thread.start()

#     start_time = time.time()
#     print_w_stream(text)

#     # Cleanup
#     audio_generation_queue.join()
#     audio_generation_queue.put(None)
#     audio_playback_queue.join()
#     audio_playback_queue.put(None)

#     audio_generation_thread.join()
#     audio_playback_thread.join()
#     pygame.mixer.quit()

def streamed_response_tts(streaming_response):
    """
    Generate and play audio from input text using OpenAI's TTS API.

    """
    load_dotenv()
    client = OpenAI()
    audio_generation_queue = queue.Queue()
    audio_playback_queue = queue.Queue()
    pygame.mixer.init()

    def generate_audio(text, model='tts-1', voice='nova'):
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
        while True:
            text = audio_generation_queue.get()
            if text is None:
                break
            audio_file_path = generate_audio(text)
            if audio_file_path:
                audio_playback_queue.put(audio_file_path)
            audio_generation_queue.task_done()

    def process_audio_playback_queue():
        while True:
            audio_file_path = audio_playback_queue.get()
            if audio_file_path is None:
                break
            play_audio(audio_file_path)
            audio_playback_queue.task_done()

    def print_w_stream(streaming_response):
        # completion = client.chat.completions.create(
        #     model='gpt-3.5-turbo',
        #     messages=[
        #         {"role": "system", "content": "You are a friendly AI assistant."},
        #         {"role": "user", "content": message},
        #     ],
        #     stream=True,
        #     temperature=0,
        #     max_tokens=500,
        # )
        
        # model_streaming = model.stream(message, max_completion_tokens=500,
        #                                temperature=0)

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