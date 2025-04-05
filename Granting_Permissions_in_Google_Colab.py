from IPython.display import display, Audio, Javascript
from gtts import gTTS
from mutagen.mp3 import MP3
import time

def text_to_speech(text):
    tts = gTTS(text=text, lang='en') # Creating an object tts for the class gTTS by passing the values for class members
    tts.save("tts.mp3")

    audio_info = MP3("tts.mp3")
    duration = audio_info.info.length  # Duration in seconds

    display(Audio("tts.mp3", autoplay=True)) # Embedded audio player widget in the notebook interface

    # Wait until the audio completes
    time.sleep(duration)

js = Javascript('''
  async function requestPermissions() {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      stream.getTracks().forEach(track => track.stop());
  }
  requestPermissions();
  ''')

text_to_speech('Enter the Enter key after 5 seconds')
display(js)
