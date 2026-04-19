from gtts import gTTS
import os

# create audio folder
os.makedirs("audio", exist_ok=True)

# list of alphabets
alphabets = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# generate audio file for each alphabet
for letter in alphabets:

    print("Creating audio for:", letter)

    tts = gTTS(text=letter, lang="en")

    tts.save(f"audio/{letter}.mp3")

print("All audio files created successfully")