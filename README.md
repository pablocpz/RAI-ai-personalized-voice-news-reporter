## Personalized Voice AI Realtime Reporter

<p align="center">
  <img src="https://github.com/user-attachments/assets/f76edc19-9518-4422-a3d0-47388a641531" width="700">
</p>


### Twitter thread +demo: https://x.com/pablocpz_ai/status/1876669029961843019 

- Tailored to your background
- Customized for your interests
- Never miss the headlines again – Stay updated, effortlessly!
- Be curious, don’t let anything slip by – Stay informed, always! Ask anything, in real time, to your phone (to-do)!

### This approach combines Agentic Retrieval Augmented Generation (RAG) using Langraph and OpenAI with Real Time Web Search using TavilyAPI and ElevenLabs for TTS

- See `local_setup.md` to run the script

## 1. Pipeline

### 1.1 - News Reporter Agent Graph
### 1.2 - News Voice-enabled Narrator Chain

### 1.3 - RAG-WebSearch QA Realtime Voice Chatbot
to-do: implement memmory between conversations & twilio-websockets communication for phone calling - and reduce latency (improve architecture efficiency)

<p align="center">
  <img src="https://github.com/user-attachments/assets/92d9a46a-ffd4-46b2-aa3a-d1bcaba71e48" width="750">
</p>

## 2. Things to Improve

- Reduce the overall latency and response time
- Enhace the graph architecture, and make it more efficient
- Use different data sources for the news
- Implement a nice GUI to the user be able to select his/her news interests, and background knowledge
- Set up the twilio engine
