import os
import torch
from flask import Flask, render_template, request, redirect, send_file
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pydub import AudioSegment
import markdown2
import pdfkit

app = Flask(__name__)
torch_dtype = torch.float32


device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("device:",device)
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Load the Whisper turbo model
model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
)
model.to(device, dtype=torch.float32)


# Load the processor for the model
processor = AutoProcessor.from_pretrained(model_id)

# Initialize the pipeline for automatic speech recognition (ASR)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=device,
    chunk_length_s=30  # Process audio in 30-second chunks
)


# Set the paths for ffmpeg and ffprobe
AudioSegment.converter = "/home/rishi_btp/miniconda3/envs/tf_mps/bin/ffmpeg"
AudioSegment.ffprobe = "/home/rishi_btp/miniconda3/envs/tf_mps/bin/ffmprobe"

# Function to convert audio to WAV format
def convert_audio(input_file_path, output_file_path):
    try:
        print(f"Attempting to load audio file: {input_file_path}")
        audio = AudioSegment.from_file(input_file_path)
    except Exception as e:
        print(f"Error loading {input_file_path}: {e}")
        return None  # Return None if conversion fails

    # Process audio
    audio_mono = audio.set_channels(1)  # Convert to mono
    normalized_audio = audio_mono.normalize()  # Normalize volume
    normalized_audio.export(output_file_path, format="wav")  # Export as WAV
    print(f"Successfully converted and saved audio as: {output_file_path}")
    return output_file_path

# Function to transcribe the audio using Whisper
def transcribe_audio(converted_audio_path):
    print(f"Using converted audio for transcription: {converted_audio_path}")

    # Transcribe the converted WAV audio using Whisper
    try:
        result = pipe(converted_audio_path, generate_kwargs={"language": "hindi"})  # Transcribe in Hindi
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None
    
    return result["text"]

# Report generation function
from groq import Groq
import os
import datetime
from fpdf import FPDF
from IPython.display import Markdown, display

# Set up the Groq client with the API key from an environment variable
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
# export GROQ_API_KEY="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"


def generate_mammography_report(transcriptions):
    if isinstance(transcriptions, str):
        transcriptions = [transcriptions] 
    # Construct the prompt for Groq
    prompt = f"""
    You are a medical assistant specializing in mammography reports. Please generate a detailed mammography report in "markdown format" with each text in new line based on the following findings from the transcripts. Use the format provided below and ensure to summarize the findings clearly. Include horizontal lines between different sections.
    
    -**Patient Information**:
        - **Name**: Patient's name
        - **MRN**: Medical record number
        - **Age**: Patient's age
    ---  # Add horizontal line for clarity

    - **Procedure**: Bilateral Mammogram
    - **Clinical Background**: Include relevant details about the patient's condition and reasons for examination.
    
    ---  # Add horizontal line for clarity

    - **Right Breast**:
        - **Breast density**: Describe the density category.
        - **Findings**: Summarize the findings in detail, including masses, calcifications, and any architectural distortion.
        - **Axilla**: Report the status of axillary lymph nodes.
        - **Parenchymal pattern**: Mention any specific features of the parenchymal pattern.
        - **Areola and Subcutaneous tissues**: Areola and subcutaneous tissues details.
        
    ---  # Add horizontal line for clarity

    - **Left Breast**:
        - **Breast density**: Describe the density category.
        - **Findings**: Summarize the findings in detail.
        - **Axilla**: Report the status of axillary lymph nodes.
        - **Parenchymal pattern**: Mention any specific features of the parenchymal pattern.
        - **Areola and Subcutaneous tissues**: Areola and subcutaneous tissues details.

    ---  # Add horizontal line for clarity

    - **Comparison**: Note if there are any prior comparisons available.
    
    ---  # Add horizontal line for clarity

    - **Impression**: Summarize the key findings and BI-RADS category for each breast. Highlight any recommendations for follow-up.
    ---  # Add horizontal line for clarity

    Please ensure to clearly differentiate between the right and left breast findings and highlight any recommendations for follow-up.
    Also ensure the report is in corret MARKDOWN FORMAT with line breaks and headers in right place. do not include any text extra except report.

    Transcripts: {''.join(transcriptions)}
    """

    # Create a chat completion request to Groq
    try:
        completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        if completion.choices and len(completion.choices) > 0:
            return completion.choices[0].message.content
        else:
            return "No choices returned in the completion response."

    except Exception as e:
        return f"An error occurred while calling the API: {str(e)}"
    
from weasyprint import HTML

def convert_markdown_to_pdf(markdown_file, pdf_file):
    # Read the Markdown file
    with open(markdown_file, 'r') as file:
        markdown_content = file.read()

    # Convert Markdown to HTML
    html_content = markdown2.markdown(markdown_content)

    # Define HTML structure with added separation lines
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

    # Convert HTML to PDF using WeasyPrint
    HTML(string=html_template).write_pdf(pdf_file)

# import markdown2
# import pdfkit

# def convert_markdown_to_pdf(markdown_file, pdf_file):
#     # Read the Markdown file
#     with open(markdown_file, 'r') as file:
#         markdown_content = file.read()

#     # Convert Markdown to HTML
#     html_content = markdown2.markdown(markdown_content)

#     # Define HTML structure with added separation lines
#     html_template = f"""
#     <html>
#     <head>
#         <meta charset="utf-8">
#         <title>Mammography Report</title>
#         <style>
#             body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 20px; }}
#             h1 {{ color: #333; text-align: center; }}
#             h2 {{ color: #007BFF; margin-top: 20px; }}
#             h3 {{ color: #555; margin: 10px 0; }}
#             p, li {{ margin: 0.5em 0; }}
#             hr {{ border: 1px solid #ccc; margin: 20px 0; }}
#         </style>
#     </head>
#     <body>
#         <h1>Mammography Report</h1>
#         <hr>
#         {html_content}
#         <hr>
#     </body>
#     </html>
#     """

#     # Convert HTML to PDF
#     pdfkit.from_string(html_template, pdf_file)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audio_file' not in request.files:
        return redirect('/')
    
    audio_file = request.files['audio_file']
    if audio_file.filename == '':
        return redirect('/')

    # Save the original audio file
    original_audio_path = os.path.join("converted_audio", audio_file.filename)
    audio_file.save(original_audio_path)

    # Convert audio to WAV
    converted_audio_path = os.path.join("converted_audio", "converted.wav")  # Define the output path
    if convert_audio(original_audio_path, converted_audio_path) is None:
        return "Audio conversion failed.", 500

    # Check if converted audio exists
    if not os.path.exists(converted_audio_path):
        return "Converted audio file not found.", 500

    transcription = transcribe_audio(converted_audio_path)
    if not transcription:
        return "Transcription failed.", 500

    report = generate_mammography_report(transcription)
    
    # Save the markdown report to a file
    markdown_path = "mammography_report.md"
    with open(markdown_path, "w") as f:
        f.write(report)

    return render_template('index.html', transcription=transcription, report=report)

@app.route('/download_pdf', methods=['POST'])
def download_pdf():
    edited_report = request.form['report']

    # Save the edited report to a markdown file
    markdown_path = "edited_mammography_report.md"
    with open(markdown_path, "w") as f:
        f.write(edited_report)

    # Specify the path for the PDF
    pdf_path = "mammography_report.pdf"
    convert_markdown_to_pdf(markdown_path, pdf_path)  # Use the markdown file path
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)