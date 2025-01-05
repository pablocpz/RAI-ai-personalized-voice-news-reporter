
import os
from elevenlabs import ElevenLabs
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Initialize the ElevenLabs client with the API key
eleven_client = ElevenLabs(
    api_key=os.getenv("XI_API_KEY")  # Load API key from environment variable
)
openai_client = OpenAI()


def speak_audio_sync(provider:str, text:str, output_path:str):
    """
    Handles both transcription (OpenAI) and TTS generation (ElevenLabs or OpenAI).

    Parameters:
        provider (str): The service provider ('eleven' or 'openai').
        text (str): Text to convert to speech (required for TTS).
        output_path (str): Path to save the generated audio.
    """

    if provider == "eleven":

        try:
            # Generate the audio response using Eleven Labs API
            response = eleven_client.generate(
                text=text,
                voice="Aria",  # Replace with desired voice name or ID
                model="eleven_multilingual_v2",  # Specify the model version
                voice_settings={
                    "stability": 0.8,  # Adjust stability (0.0 - 1.0)
                    "similarity_boost": 0.75  # Adjust voice likeness
                }
            )

            # Save the audio data to the output file
            with open(output_path, "wb") as f:
                for chunk in response:
                    f.write(chunk)

            print("Audio generated and saved successfully!")
        except Exception as e:
            print(f"Error generating audio with ElevenLabs: {str(e)}")

    elif provider == "openai":
        
  
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text
        )
        
        
        if hasattr(response, 'content') and response.content:
            # Use your play function to play the audio instantly
            # play(response.content)
            print(type(response.content))  # Para identificar el tipo de datos

            
            
            # Save the audio data to the output file
            with open(output_path, "wb") as f:
                f.write(response.content)

        else:
            print("No valid audio content found in the response.")    
        # if n or not output_path:
        #     raise ValueError("Input_path and output_path are required for OpenAI transcription.")

        # try:
        #     # Read the audio file and transcribe it
        #     with open(input_path, "rb") as audio_file:
        #         transcription = openai_client.audio.transcriptions.create(
        #             model="whisper-1",
        #             file=audio_file,
        #             language='en'
        #         )

        #     # Save the transcription to the output file
        #     with open(output_path, "w") as f:
        #         f.write(transcription.text)

        #     print("Transcription completed and saved successfully!")
        # except Exception as e:
        #     print(f"Error transcribing audio with OpenAI: {str(e)}")
    else:
        raise ValueError("Unsupported provider. Use 'eleven' for TTS or 'openai' for transcription.")

# import os
# from elevenlabs.client import ElevenLabs
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# # Initialize the ElevenLabs client with the API key
# eleven_client = ElevenLabs(
#     api_key=os.getenv("XI_API_KEY")  # Load API key from environment variable
# )
# openai_client = OpenAI()



# def transcribe_audio(provider="openai", text, out_path=""):
    
#     """
#     TTS function (ElevenLabs or OpenAI)
#     """
    
    
#     if provider == "eleven":
        
#         try:
#             # Generate the audio response using Eleven Labs API
#             audio = eleven_client.generate(
#                 text=text,
#                 voice="Aria",  # You can replace "Aria" with any other voice ID or name
#                 model="eleven_multilingual_v2",  # Specify the model version to use
#                 stream=True,
#                 voice_settings={
#                     "stability": 0.8,  # Adjust stability (0.0 - 1.0)
#                     "similarity_boost": 0.75  # Adjust the likeness to the selected voice
#                 }
#             )

#             # Define the path where the audio will be saved
            
            
#             # Open the file for writing in binary mode
#             with open(out_path, "wb") as f:
#                 # Consume the generator and write the audio data to the file
#                 for chunk in audio:
#                     f.write(chunk)
            
#             print("Audio generated and saved successfully!")
        
#         except Exception as e:
#             print(f"Error generating audio: {str(e)}")

#     elif provider == "openai":
#         with open(out_path, "rb") as audio_file:
#             transcription = openai_client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_file,
#                 language='en'
#             )
#         print(f'Transcription: {transcription.text}')