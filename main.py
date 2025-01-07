
from utils.news_narrator_chain import get_news_narrative #news summary chain
from utils.reporter_graph import graph as reporter_graph #Reporter Graph

from utils.chatbot_graph import app as chatbot_graph #QA Chatbot
from utils.news_reports import run_reports_creation
from utils.news_reports import get_news_data

from tts_variants import sync_tts
import sounddevice as sd
import soundfile as sf
import keyboard
import tempfile
import numpy as np
import time
from threading import Thread, Event
import keyboard
from tts_variants import sync_tts, sync_tts_eleven, streamed_response_tts
from utils.sst import transcribe_audio
import asyncio

from aaa import text_to_tts


def get_inference(input_question:str, news_summary:str, thread_id=1, max_recursion=25):
    
    """
    if we get recursion error, drop custom/default message: "Please, try again"
    """
    thread = {"configurable": {"thread_id":thread_id, "max_recursion":max_recursion}} 

    
    
    inputs = {"question": input_question, "news_summary": news_summary}
    
    try:
        # Attempt the invocation
        content = chatbot_graph.invoke(inputs, thread)
        
        if content.get("streaming_avaiable", []) == True:
            
            generator = content["response_stream"]
            
            streamed_response_tts(streaming_response=generator)
            print("generator streamed")
            
            return None
         
        else:
            generation = content["generation"].content
            print(generation)
            
            return generation
          
        
    except RecursionError:
        
        content = "Sorry, there was an error, please ask me again!"

        return content



# Global variables
stop_event = Event()
recorded_frames = []
sample_rate = 44100
channels = 2

def audio_callback(indata, frames, time, status):
    """Callback function for the audio stream"""
    if status:
        print(f"Status: {status}")
    if not stop_event.is_set():
        recorded_frames.append(indata.copy())

def record_audio():
    """Record audio and return the path to the temporary file"""
    global recorded_frames
    recorded_frames = []
    stop_event.clear()
    
    def recording_thread():
        with sd.InputStream(samplerate=sample_rate, 
                          channels=channels,
                          callback=audio_callback):
            stop_event.wait()
    
    # Start recording
    thread = Thread(target=recording_thread)
    thread.start()
    
    # Wait for spacebar to stop recording
    keyboard.wait('space')
    stop_event.set()
    thread.join()
    
    # Save to temporary file
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
    
    
    #Running Step 1) detailed report designer---------------------
    ## Call this from an existing async context
    # picked_headlines, news_content = get_news_data()

    # print("News data and headlines sucessfully selected & retrieved!")
    
    # reports_list = await run_reports_creation(reporter_graph,
    #                                         picked_headlines=picked_headlines,
    #                                         news_content=news_content)

    # export_markdown_reports(data_list=reports_list, headlines=picked_headlines.headlines, output_folder="rag_docs")


    print("Markdown Reports Sucessfully saved!")
    
    from utils.news_tools import export_markdown_reports

    #Running Step 2) News Narrative chain--------------------------------------------
    # news_summary = get_news_narrative()
    
    news_summary = "Ready to know what happened this week? Let's dive in. First up, fitness enthusiasts are buzzing about high-intensity interval training, which promises to keep you burning calories long after your workout ends. Meanwhile, in the world of international affairs, Ukraine made headlines by using a naval drone to take down a Russian helicopter, marking a significant moment in their ongoing conflict. As Asia celebrated the arrival of 2025 with spectacular fireworks, Elon Musk stirred controversy in Germany by endorsing a far-right party, leading to a media shake-up. Across the Atlantic, Donald Trump surprised many by supporting Elon Musk and the H-1B visa program, causing ripples within the Republican Party. In entertainment, Lily-Rose Depp is making waves with her role in the new Nasratu film, while Trump mourns the loss of former U.S. President Jimmy Carter, who passed away at 100, leaving behind a legacy of peace and humanitarian work. Spain is facing unrest as public employees deal with frozen salaries and healthcare disruptions, while in Brazil, a tragic poisoning incident involving a Christmas cake is under investigation. Azerbaijan and Russia are at odds over a plane crash, with calls for an independent probe. In South Korea, a devastating plane crash claimed 179 lives, marking the worst air disaster in the country in decades. New York is reeling from a scandal involving prison brutality, prompting calls for justice and reform. The WHO's Tedros Adhanom Ghebreyesus narrowly escaped a bombing in Yemen, highlighting the ongoing conflict in the region. Tensions are also high between Venezuela and Argentina over accusations of terrorism, while Ukraine continues to fend off Russian aerial attacks. In Madrid, a graffiti artist faces a hefty fine for his street art spree. And in New York, a shocking video of prison violence has sparked outrage and investigations. Meanwhile, Russia's use of a ghost freight to bypass sanctions is raising alarms in the Baltic region. In China, a tragic incident involving a driver plowing into a crowd has sparked discussions on public mourning and censorship. Back in New York, the Guardian Angels are back on patrol in the subway after a horrific attack, aiming to boost public safety. Finally, Spain's Muface is in turmoil as major health insurers withdraw, leaving many civil servants without private coverage. As the year wraps up, these events remind us of the complexities and challenges facing our world today. Stay tuned for more updates.\n"
    print("News summary sucessfully created, gonna run TTS...")

    # sync_tts_eleven(news_summary)
    
    #Running Step 3) RAG-WebSearch QA Chatbot---------------

    if news_summary:

        while True:
            print("Press Spacebar to start speaking...")
            # Wait for the user to press spacebar to start recording
            keyboard.wait("space")
            print("Recording started... Press Spacebar to stop.")
            
            # Record the user's speech
            audio_file_path = record_audio()
            
            if audio_file_path:
                print("Audio saved to temporary file.")
                # Transcribe the recorded audio into text
                user_input = transcribe_audio(audio_file_path)
                print(f"[-] User said: {user_input}")
                
                start_time = time.time()

                # Get bot response based on the transcribed text
                #we'll only get such text if the streaming is not avaiable)
                
                content = get_inference(input_question=user_input, news_summary=news_summary, max_recursion=25)
                
                if content !=None:
                #only if the function returns something (non-streaming response)
                
                    print(f"[*] Bot response: {content}")
                    
                    end_time = time.time()
                    inference_time = end_time - start_time

                    print(f"Inference took {inference_time} seconds")
                    #--------------------
                    
                    # sync_tts(content)
                    text_to_tts(content)
                    # await async_chunking_tts(content, chunk_size=50)  # Use await instead of asyncio.run()

                    #--------------------   
                    
                    # Pause briefly before the next iteration
                    time.sleep(1)
            else:
                print("No audio recorded.")
                
                
                
if __name__ == "__main__":
    # Run the async function with asyncio
    asyncio.run(main())
    