from utils.news_narrator_chain import get_news_narrative  # News summary chain
from utils.reporter_graph import graph as reporter_graph  # Reporter Graph
from utils.chatbot_graph import app as chatbot_graph  # QA Chatbot
from utils.news_reports import run_reports_creation, get_news_data
from utils.news_tools import export_markdown_reports
import sounddevice as sd
import soundfile as sf
import keyboard
import tempfile
import numpy as np
import time
from threading import Thread, Event
from utils.sst import transcribe_audio
import asyncio
from tts_variants import streamed_oai_response_tts, text_to_tts
import logging

# Set logging level to WARNING to suppress INFO-level messages
logging.basicConfig(level=logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)


# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RED = "\033[91m"


def user_coloring(text):
    # Blue background for user input
    return f"\033[44m{text}\033[0m"

def bot_coloring(text):
    # Green background for bot response
    return f"\033[42m{text}\033[0m"

def get_inference(input_question: str, news_summary: str, thread_id=1, max_recursion=25):
    """Handle inference and return responses with optional streaming TTS."""
    thread = {"configurable": {"thread_id": thread_id, "max_recursion": max_recursion}}
    inputs = {"question": input_question, "news_summary": news_summary}

    try:
        content = chatbot_graph.invoke(inputs, thread)
        
        if content["streaming_avaiable"] == True:
            generator = content["response_stream"]
            streamed_oai_response_tts(streaming_response=generator, tts_provider="elevenlabs")
            print(f"{GREEN}[Bot Streaming Response]{RESET} Streaming audio response generated.")
            return None
        elif content["streaming_avaiable"] == False and content["generation"] != None:
        
            generation = content["generation"].content
            
            print(f"{BOLD}[-] Bot said:{RESET} {bot_coloring(generation)}")
            # print(f"{GREEN}[Bot Response]{RESET} {generation}")
            return generation
    except RecursionError:
        error_message = "Sorry, there was an error, please ask me again!"
        text_to_tts(error_message)
        print(f"{RED}[Error]{RESET} {error_message}")
        return error_message

# Global variables
stop_event = Event()
recorded_frames = []
sample_rate = 44100
channels = 2

def audio_callback(indata, frames, time, status):
    """Callback function for the audio stream."""
    if status:
        print(f"{RED}[Audio Status]{RESET} {status}")
    if not stop_event.is_set():
        recorded_frames.append(indata.copy())

def record_audio():
    """Record audio and save it to a temporary file."""
    global recorded_frames
    recorded_frames = []
    stop_event.clear()

    def recording_thread():
        with sd.InputStream(samplerate=sample_rate, channels=channels, callback=audio_callback):
            stop_event.wait()

    thread = Thread(target=recording_thread)
    thread.start()

    print(f"{CYAN}[Instruction]{RESET} Press Spacebar to stop recording.")
    keyboard.wait("space")
    stop_event.set()
    thread.join()

    if recorded_frames:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            recorded_data = np.concatenate(recorded_frames, axis=0)
            sf.write(temp_file.name, recorded_data, sample_rate)
            return temp_file.name
    return None

#---------------------------------
#STARTING THE PIPELINE
#1. DETAILED REPORT CREATOR
#2. NEWS SUMMARY CHAIN
#3. RAG-WebSearch QA Chatbot
#--------------------------------


async def main():
    """Main pipeline for news narrative and user interaction."""
    print(f"{CYAN}[Startup]{RESET} Initializing the pipeline...")

    #---------Running Step 1) detailed report designer---------------------
    ## Call this from an existing async context
    # picked_headlines, news_content = get_news_data()

    print("News data and headlines sucessfully selected & retrieved!")
    
    # reports_list = await run_reports_creation(reporter_graph,
    #                                         picked_headlines=picked_headlines,
    #                                         news_content=news_content)

    # export_markdown_reports(data_list=reports_list, headlines=picked_headlines.headlines, output_folder="rag_docs")



    #-----------Running Step 2) News Narrative chain--------------------------------------------
    # news_summary = get_news_narrative()
    
    
    # Placeholder news summary (testing purposes)
    news_summary = (
        "Ready to know what happened this week? Let's dive in. First up, fitness enthusiasts are buzzing about "
        "high-intensity interval training, which promises to keep you burning calories long after your workout ends."
    )

    print(f"{GREEN}[News Summary Ready]{RESET} Running TTS for the news summary...")

    # text_to_tts(news_summary)
    
    #----------------Running Step 3) RAG-WebSearch QA Chatbot---------------

    while True:
        print(f"{CYAN}[Instruction]{RESET} Press Spacebar to start speaking...")
        keyboard.wait("space")
        print(f"{CYAN}[Recording]{RESET} Recording started... Press Spacebar to stop.")

        audio_file_path = record_audio()

        if audio_file_path:
            print(f"{YELLOW}[Audio Saved]{RESET} Audio saved to a temporary file.")
            user_input = transcribe_audio(audio_file_path)
            # print(f"{BOLD}[-] User said:{RESET}", f"{user_input}")
            print(f"{BOLD}[-] User said:{RESET} {user_coloring(user_input)}")

            start_time = time.time()
            content = get_inference(input_question=user_input, news_summary=news_summary, max_recursion=25)

            if content !=None:
                # print(f"{BOLD}[*] Bot response:{RESET} {content}")
                inference_time = time.time() - start_time
                print(f"{YELLOW}[Timing]{RESET} Inference took {inference_time:.2f} seconds.")

                text_to_tts(content, provider="elevenlabs")
                time.sleep(1)
                
        else:
            print(f"{RED}[No Audio]{RESET} No audio recorded.")

if __name__ == "__main__":
    asyncio.run(main())