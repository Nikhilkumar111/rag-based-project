# he default setting (which selects the turbo model) works well for transcribing 
# English. However, the turbo model is not trained for translation tasks. If you
#  need to translate non-English speech into English, use one of the multilingual 
# models (tiny, base, small, medium, large) instead of turbo.


# thats why i used the large-v2 model from the wispher so that i am able to 
# convert the video form to text form in english langauge

# direct conversion used -->> directly to convert english to english
# but if you have to convert the non-english langauge to english language then
# you have to used any model in this list (tiny, base, small, medium, large)
# so we used the large model 

import whisper
import json
import os

model = whisper.load_model("small")

audios = os.listdir("audios")

os.makedirs("jsons", exist_ok=True)   # ✅ required

for audio in audios: 
    if "_" in audio:
        number = audio.split("_")[0]
        title = audio.split("_")[1][:-4]
        print(number, title)

        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",          # ✅ fixed spelling
            task="translate",
            fp16=False              # ✅ CPU safe
        )
        
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })
        
        chunks_with_metadata = {
            "chunks": chunks,
            "text": result["text"]
        }

        with open(f"jsons/{audio}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False)
