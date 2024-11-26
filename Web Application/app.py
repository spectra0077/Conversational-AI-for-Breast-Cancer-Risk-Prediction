import os
import torch
from flask import Flask, render_template, request, redirect, send_file
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pydub import AudioSegment
import markdown2
from weasyprint import HTML
from transformers import BitsAndBytesConfig

app = Flask(__name__)

# Set up device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

# Free any existing GPU memory
torch.cuda.empty_cache()

# Define the model name for the quantized version of LLaMA 3.1-8B-Instruct
model_name = "meta-llama/Llama-3.1-8b-instruct"  # Replace with the correct model name if different
print("Loading LLaMA model... This may take a few minutes.")

# Setup bitsandbytes configuration for loading in 8-bit precision
bnb_config = BitsAndBytesConfig(load_in_8bit=True)

# Load the tokenizer and quantized model using bitsandbytes
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically map model layers across available GPUs
    quantization_config=bnb_config,  # Apply the 8-bit quantization
    torch_dtype=torch.float16  # Use FP16 for reduced memory usage
)

# Set paths for ffmpeg
AudioSegment.converter = "/home/rishi_btp/miniconda3/envs/tf_mps/bin/ffmpeg"
AudioSegment.ffprobe = "/home/rishi_btp/miniconda3/envs/tf_mps/bin/ffmprobe"

# Function to convert audio to WAV format
def convert_audio(input_file_path, output_file_path):
    try:
        audio = AudioSegment.from_file(input_file_path)
        audio_mono = audio.set_channels(1).normalize()
        audio_mono.export(output_file_path, format="wav")
        return output_file_path
    except Exception as e:
        print(f"Error converting audio: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(converted_audio_path):
    try:
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=device,
            chunk_length_s=30
        )
        result = asr_pipeline(converted_audio_path)
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


# Function to generate a mammography report using LLaMA 3.1-8B-Instruct
import os

def generate_mammography_report(transcriptions):
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_file = os.path.join(current_dir, "prompt.txt")

    # Read the prompt from prompt.txt
    try:
        with open(prompt_file, "r") as file:
            prompt_template = file.read()
    except FileNotFoundError:
        raise FileNotFoundError("The file 'prompt.txt' was not found in the current directory.")

    # Insert the transcriptions into the prompt template
    prompt = prompt_template.format(transcriptions=transcriptions)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output = model.generate(
        **inputs,
        max_length=2048,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Parse the output after the $$$ delimiter
    delimiter = "$$$"
    report_start = generated_text.find(delimiter) + len(delimiter)
    
    if report_start < len(delimiter):  # Check if delimiter is not found
        # Return the entire output
        return generated_text.strip()

    # Extract and return content after the delimiter
    return generated_text[report_start:].strip()





# Function to convert markdown to PDF
def convert_markdown_to_pdf(markdown_file, pdf_file):
    with open(markdown_file, 'r') as file:
        markdown_content = file.read()
    html_content = markdown2.markdown(markdown_content)
    html_template = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Mammography Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
            h1 {{ color: #333; text-align: center; }}
            h2 {{ color: #007BFF; margin-top: 20px; }}
            h3 {{ color: #555; margin: 10px 0; }}
            p, li {{ margin: 0.5em 0; }}
            hr {{ border: 1px solid #ccc; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Mammography Report</h1>
        <hr>
        {html_content}
        <hr>
    </body>
    </html>
    """
    HTML(string=html_template).write_pdf(pdf_file)




import uuid  # Import UUID for generating unique identifiers

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')
# Function to ensure the reports directory exists
def ensure_reports_directory_exists():
    if not os.path.exists("reports"):
        os.makedirs("reports")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return redirect('/')
    
    # Generate a unique ID for the session
    session_id = str(uuid.uuid4())
    
    # Ensure the reports directory exists
    ensure_reports_directory_exists()
    
    audio_file = request.files['audio_file']
    original_audio_path = os.path.join("converted_audio", f"{session_id}_{audio_file.filename}")
    audio_file.save(original_audio_path)
    converted_audio_path = f"converted_audio/{session_id}_converted.wav"
    
    if not convert_audio(original_audio_path, converted_audio_path):
        return "Audio conversion failed.", 500
    
    transcription = transcribe_audio(converted_audio_path)
    if not transcription:
        return "Transcription failed.", 500
    
    report = generate_mammography_report(transcription)
    
    markdown_path = f"reports/{session_id}_mammography_report.md"
    with open(markdown_path, "w") as f:
        f.write(report)
    
    return render_template('index.html', transcription=transcription, report=report, session_id=session_id)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    edited_report = request.form['report']
    session_id = request.form['session_id']
    markdown_path = f"reports/{session_id}_edited_mammography_report.md"
    with open(markdown_path, "w") as f:
        f.write(edited_report)
    
    pdf_path = f"reports/{session_id}_mammography_report.pdf"
    convert_markdown_to_pdf(markdown_path, pdf_path)
    
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
