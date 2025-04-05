import re
from groq import Groq
import google.generativeai as genai
from faster_whisper import WhisperModel
import os

from gtts import gTTS
from IPython.display import Audio

from IPython.display import display, Javascript
from IPython.display import Image as ipython_display_image #To avoid ambiguity
from google.colab.output import eval_js
from base64 import b64decode

from mutagen.mp3 import MP3
import time

from PIL import Image

wake_word = "hey"

# Initialize models and settings
groq_client = Groq(api_key="")  # Replace with your Groq API key
genai.configure(api_key='')  # Replace with your Generative AI API key
whisper_model = WhisperModel("base", device='cpu', compute_type='int8')

# System message for AI
sys_msg=(
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]


generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {
        'category': 'HARM_CATEGORY_HARASSMENT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_HATE_SPEECH',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
        'threshold': 'BLOCK_NONE'
    },
    {
        'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
        'threshold': 'BLOCK_NONE'
    },
]

model = genai.GenerativeModel('gemini-1.5-flash-latest',
                              generation_config=generation_config,
                              safety_settings=safety_settings)

num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)


def groq_prompt(prompt, img_context):
    if img_context:
        prompt = f'USER PROMPT: {prompt}\n\n  IMAGE CONTEXT: {img_context}'
    convo.append({'role': 'user', 'content': prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)
    return response.content

def vision_prompt(prompt, photo_path):
    img = Image.open(photo_path)
    prompt = (
        'You are a vision analysis AI designed to derive semantic meaning from images and provide contextual information.'
        'Your purpose is to analyze the image based on the user\'s prompt and extract all relevant details.'
        'Your output will serve as input for another AI that will respond to the user. Do not respond directly to the user.'
        f'Instead, focus on generating objective and detailed data about the image that aligns with the user\'s prompt. \nUSER PROMPT: {prompt}'

    )
    response = model.generate_content([prompt, img])
    return response.text

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether capturing the webcam or calling no functions is best for'
        'a voice assistant to respond to the user\'s prompt. If the user asks to capture the webcam then reply with the corresponding action.'
        'The webcam can be assumed to be a normal laptop webcam facing the user.'
        'You will respond with only one action from this list: ["capture webcam", "None"].'
        'Respond with any one of the function call name exactly as I listed based on the user\'s prompt.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg},
                      {'role': 'user', 'content': prompt}]

    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,.?!]*([A-Za-z0-9].*)'

    match = re.search(pattern, transcribed_text, re.IGNORECASE) #makes the search case insensitive

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None

def process_audio_file(audio_file_path):
    prompt_text = wav_to_text(audio_file_path)
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        action = function_call(clean_prompt)

        visual_context = None
        if 'capture webcam' in action:
            capture_webcam()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='photo.jpg')

        response = groq_prompt(prompt=clean_prompt, img_context=visual_context)
        return response
    else:
        print('No valid clean prompt found.')
        return None

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    tts.save("tts.mp3")

    audio_info = MP3("tts.mp3")
    duration = audio_info.info.length

    display(Audio("tts.mp3", autoplay=True))

    time.sleep(duration)

def record_audio(filename='recorded_audio.wav'):
    js = Javascript('''
    async function recordAudio() {
        const div = document.createElement('div');
        const instructions = document.createElement('p');
        instructions.textContent = "Press Enter to start recording. Type '.' to stop.";

        const textarea = document.createElement('textarea');
        textarea.rows = 2;
        textarea.cols = 30;
        textarea.placeholder = "Press Enter to start...";

        div.appendChild(instructions);
        div.appendChild(textarea);
        document.body.appendChild(div);

        textarea.focus();

        await new Promise(resolve => {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === "Enter") {
                    e.preventDefault();
                    resolve();
                }
            });
        });

        instructions.textContent = "Recording... Type '.' to stop.";

        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        const recorder = new MediaRecorder(stream);
        const chunks = [];

        recorder.ondataavailable = e => chunks.push(e.data);

        recorder.start();

        await new Promise(resolve => {
            textarea.addEventListener('keydown', (e) => {
                if (e.key === ".") {
                    e.preventDefault();
                    recorder.stop();
                    resolve();
                }
            });
        });

        const audioData = await new Promise(resolve => {
            recorder.onstop = async () => {
                const blob = new Blob(chunks);
                const reader = new FileReader();

                reader.onload = () => resolve(reader.result.split(',')[1]);

                reader.readAsDataURL(blob);
            };
        });

        stream.getTracks().forEach(track => track.stop());

        div.remove();
        return audioData;
    }
    ''')

    display(js)
    data = eval_js('recordAudio()')

    audio_bytes = b64decode(data)
    with open(filename, 'wb') as f:
        f.write(audio_bytes)

    print(f"Audio saved to {filename}")
    return filename

def capture_webcam(filename='photo.jpg', quality=1.0):
    text_to_speech("As per your request, now the image will be captured through the web camera. So be ready.")
    time.sleep(0.25)
    text_to_speech('3')
    time.sleep(0.25)
    text_to_speech('2')
    time.sleep(0.25)
    text_to_speech('1')

    js = Javascript('''
      async function takePhoto(quality) {
        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        document.body.appendChild(video);
        video.srcObject = stream;
        await video.play();

        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        await new Promise(resolve => setTimeout(resolve, 2000));

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        canvas.getContext('2d').drawImage(video, 0, 0);
        stream.getVideoTracks()[0].stop();
        video.remove();

        return canvas.toDataURL('image/jpeg', quality);
      }
    ''')

    display(js)
    data = eval_js('takePhoto({})'.format(quality))

    display(Audio("camera_shutter.mp3", autoplay=True))
    time.sleep(2)

    binary = b64decode(data.split(',')[1])

    with open(filename, 'wb') as f:
        f.write(binary)

    text_to_speech("Thank you! The image has been successfully saved.")
    display(ipython_display_image(filename))
    return filename

def main():
    '''
    vova = ('Hello! I am VOVA Voice Over Vision Assistant designed to serve visually challenged people. I am pleased to help you.'
            'If you want to query me, start by entering the ENTER key which will be usually present at the bottom right of QWERTY keyboard.'
            'If you are not interested in the conversation anymore you can quit by entering the dot . which is to the bottom left of ENTER key.'
           )#Implicit String Concatenation
    '''
    while True:
      user = input()
      if user=='.':
        text_to_speech('Since you entered dot, hope you don\'t have any questions. It was a nice time with you. Thank you so much.')
        break
      elif user=="":
        #text_to_speech('Since you entered the enter key, feel free to ask any questions by entering the Enter key and stop recording by entering the dot key')

        user = record_audio()
        audio = Audio('/content/recorded_audio.wav')
        display(audio)

        response = process_audio_file('/content/recorded_audio.wav')
        if response:
          print(response)
          text_to_speech(response)

      text_to_speech('I would like to respond you with some more questions. If you are interested in asking some more questions, please enter the Enter key, else if you are done with it, please enter dot, and the Enter key')


if __name__ == "__main__":
  main()
