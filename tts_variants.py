
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