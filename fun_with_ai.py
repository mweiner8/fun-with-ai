import os
import base64
import shutil
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
load_dotenv()
my_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=my_api_key)

# Global variables to store state
displayed_text = ""
selected_option = ""
selected_voice = "alloy"
image_filename = None
audio_filename = None
is_loading = False


def move_image_to_static(source_path, static_dir='static'):
    filename = os.path.basename(source_path)
    destination_path = os.path.join(static_dir, filename)

    try:
        shutil.move(source_path, destination_path)
        return destination_path
    except FileNotFoundError:
        return None
    except Exception as e:
        print(e)
        return None


@app.route('/')
def index():
    global displayed_text, selected_option, selected_voice, image_filename, audio_filename, is_loading

    # Initialize global variables if they don't exist
    if 'displayed_text' not in globals():
        displayed_text = ""
    if 'selected_option' not in globals():
        selected_option = ""
    if 'selected_voice' not in globals():
        selected_voice = "alloy"
    if 'image_filename' not in globals():
        image_filename = None
    if 'audio_filename' not in globals():
        audio_filename = None
    if 'is_loading' not in globals():
        is_loading = False

    return render_template('index.html', text=displayed_text, option=selected_option, image=image_filename, audio=audio_filename, voice=selected_voice, loading=is_loading)


@app.route('/submit', methods=['POST'])
def submit_text():
    global displayed_text, selected_option, selected_voice, image_filename, audio_filename, is_loading

    # Set loading state
    is_loading = True

    # Get form data
    user_input = request.form.get('user_text', '')
    selected_option = request.form.get('processing_type', '')
    selected_voice = request.form.get('voice_selection', 'alloy')
    uploaded_file = request.files.get('user_file')

    # Clear previous results based on selected option
    if selected_option != 'image_generation':
        image_filename = None
    if selected_option != 'text_to_speech':
        audio_filename = None
    displayed_text = ""

    try:
        # Process based on selected option
        if selected_option == 'vision_capabilities':
            if uploaded_file and uploaded_file.filename != '':
                filename = uploaded_file.filename
                file_path = os.path.join('static', filename)
                uploaded_file.save(file_path)
                displayed_text = ai_process_uploaded_image(file_path)
            else:
                displayed_text = "[Vision Capabilities] No file uploaded"

        elif selected_option == 'speech_to_text':
            if uploaded_file and uploaded_file.filename != '':
                filename = uploaded_file.filename
                file_path = os.path.join('static', filename)
                uploaded_file.save(file_path)
                displayed_text = ai_speech_to_text(file_path)
            else:
                displayed_text = "[Speech-to-Text] No audio/video file uploaded"

        elif user_input.strip():
            if selected_option == 'text_generation':
                displayed_text = ask_ai_for_text(user_input)
            elif selected_option == 'image_generation':
                image_path = ai_image_generator(user_input)
                if image_path:
                    image_filename = image_path
                else:
                    displayed_text = "An error occurred while generating the image"
            elif selected_option == 'structured_data':
                displayed_text = ai_return_json(user_input)
            elif selected_option == 'text_to_speech':
                audio_path = ai_text_to_speech(user_input, selected_voice)
                if audio_path:
                    audio_filename = "speech.mp3"
                    displayed_text = ""
                else:
                    displayed_text = "An error occurred while generating the audio"
            else:
                displayed_text = f"[Unknown Option] Processing: {user_input}"
        else:
            if selected_option not in ['vision_capabilities', 'speech_to_text']:
                displayed_text = "Please provide input for the selected option"

    except Exception as e:
        displayed_text = f"An error occurred: {str(e)}"

    finally:
        # Always reset loading state when processing is complete
        is_loading = False

    return redirect(url_for('index'))


def ask_ai_for_text(prompt):
    try:
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        return f"Error generating text: {str(e)}"


def ai_image_generator(prompt):
    try:
        image_name = "picture.png"
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        with open(image_name, "wb") as f:
            f.write(image_bytes)
        move_image_to_static(image_name)
        return image_name
    except Exception as e:
        print(f"Error generating image: {e}")
        return None


def ai_return_json(prompt):
    try:
        class ProductDesc(BaseModel):
            product: str
            description: str
            estimated_cost: float

        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {"role": "system", "content": "Extract the product information."},
                {"role": "user", "content": prompt},
            ],
            text_format=ProductDesc,
        )
        return response.output_parsed
    except Exception as e:
        return f"Error generating structured data: {str(e)}"


def create_file(file_path):
    try:
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id
    except Exception as e:
        print(f"Error creating file: {e}")
        return None


def ai_process_uploaded_image(filepath):
    try:
        file_id = create_file(filepath)
        if not file_id:
            return "Error processing uploaded file"

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "what's in this image?"},
                    {"type": "input_image", "file_id": file_id},
                ],
            }],
        )
        return response.output_text
    except Exception as e:
        return f"Error processing image: {str(e)}"


def ai_speech_to_text(filepath):
    try:
        audio_file = open(filepath, "rb")

        transcription = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=audio_file
        )

        return transcription.text
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return f"Error processing audio file: {str(e)}"


def ai_text_to_speech(prompt, voice):
    try:
        speech_file_path = Path(__file__).parent / "static/speech.mp3"
        with client.audio.speech.with_streaming_response.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=prompt
        ) as response:
            response.stream_to_file(speech_file_path)
        return str(speech_file_path)
    except Exception as e:
        print(f"Error generating speech: {e}")
        return None


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)