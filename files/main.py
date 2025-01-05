import keyboard
import os
import tempfile

import numpy as np
import openai
import sounddevice as sd
import soundfile as sf
import tweepy

# from elevenlabs import generate, play, set_api_key
# from langchain.agents import initialize_agent, load_tools
# from langchain.agents.agent_toolkits import ZapierToolkit
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# from langchain.tools import BaseTool
# from langchain.utilities.zapier import ZapierNLAWrapper

from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings, play

from dotenv import load_dotenv
import os

from openai import OpenAI
openai_client = OpenAI()

load_dotenv()

# set_api_key(os.getenv("ELEVEN_API_KEY"))
# openai.api_key = os.get("OPENAI_API_KEY")

# Set recording parameters
duration = 5  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels


def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording


def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            # transcript = openai.Audio.transcribe("whisper-1", audio_file)
            transcription = openai_client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                response_format="text"
            )
        os.remove(temp_audio.name)
    return transcription


client = ElevenLabs(
    api_key=os.getenv("XI_API_KEY")  # Load API key from environment variable
)

def play_generated_audio(text):

    # Generate the audio response using Eleven Labs API
    # audio = client.generate(
    #     text=text,
    #     voice="Aria",  # You can replace "Aria" with any other voice ID or name
    #     model="eleven_multilingual_v2",  # Specify the model version to use
    #     stream=True,
    #     voice_settings={
    #         "stability": 0.8,  # Adjust stability (0.0 - 1.0)
    #         "similarity_boost": 0.75  # Adjust the likeness to the selected voice
    #     }
    # )
    
    # play(audio)
    
    
    
    
        
# def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
#     audio = generate(text=text, voice=voice, model=model)
#     play(audio)


# Replace with your API keys
# consumer_key = "<CONSUMER_KEY>"
# consumer_secret = "<CONSUMER_SECRET>"
# access_token = "<ACCESS_TOKEN>"
# access_token_secret = "<ACCESS_TOKEN_SECRET>"

# client = tweepy.Client(
#     consumer_key=consumer_key, consumer_secret=consumer_secret,
#     access_token=access_token, access_token_secret=access_token_secret
# )


# class TweeterPostTool(BaseTool):
#     name = "Twitter Post Tweet"
#     description = "Use this tool to post a tweet to twitter."

#     def _run(self, text: str) -> str:
#         """Use the tool."""
#         return client.create_tweet(text=text)

#     async def _arun(self, query: str) -> str:
#         """Use the tool asynchronously."""
#         raise NotImplementedError("This tool does not support async")


if __name__ == '__main__':

    llm = ChatOpenAI(name="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history")

    # zapier = ZapierNLAWrapper(zapier_nla_api_key="<ZAPIER_NLA_API_KEY>")
    # toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)

    # tools = [TweeterPostTool()] + toolkit.get_tools() + load_tools(["human"])

    # tools = toolkit.get_tools() + load_tools(["human"])

    # agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)

    while True:
        print("Press spacebar to start recording.")
        keyboard.wait("space")  # wait for spacebar to be pressed
        recorded_audio = record_audio(duration, fs, channels)
        message = transcribe_audio(recorded_audio, fs)
        print(f"You: {message}")
        # assistant_message = agent.run(message)
        assistant_message = llm.invoke(message).content
        
        play_generated_audio(assistant_message)