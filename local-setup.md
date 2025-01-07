* Local Setup
Setup a new pip env for python 3.11.9

```bash
python -m venv ai_curator

.\ai_curator\Scripts\activate

pip install -r requirements.txt
```

Set up a .env file with the following variables (optional: log in into Langsmith in case you wanna trace the results. You can also use ElevenLabs for TTS)
```bash
OPENAI_API_KEY=""
TAVILY_API_KEY=""
```
To run the main script, execute:
```bash
python main.py
```
