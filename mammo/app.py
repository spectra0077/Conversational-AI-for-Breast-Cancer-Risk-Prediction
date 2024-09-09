import streamlit as st
from pydub import AudioSegment
import pandas as pd
import openai
import io
import os
import time
from fpdf import FPDF
import json

AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"
MODEL="gpt-4o"
client = openai.OpenAI(api_key="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")

def convert_audio(input_file_path, output_file_path):
    audio = AudioSegment.from_file(input_file_path, format="m4a")
    audio_mono = audio.set_channels(1)
    change_in_dBFS = -audio_mono.max_dBFS
    normalized_audio = audio_mono.apply_gain(change_in_dBFS)
    normalized_audio.export(output_file_path, format="ipod", codec="aac")
    return output_file_path


def transcribe_audio(audio_path):
    response = client.audio.transcriptions.create(
        model="whisper-1",
        file=open(audio_path, "rb"),
        language="hi",
        temperature=0.5,
        timestamp_granularities=["segment"],
        response_format="verbose_json",
        prompt="दायें, बायें, छाती, बनावट, कैटेगरी, चर्बी, साइज, घनापन, क्षमता, गांठ, टेढ़ा-मेढ़ा, धुंधला, उभार, काँटेदार, कैल्सिफिकेशन, बिंदियाँ, बारिक, बायोप्सी, त्वचा, निपल, लिम्फ नोड्स, बायराड्स, फिब्रोएडेनोमा, सिस्ट"
    )
    return response.text


def generate_report(transcriptions):
    prompt = """
        You are a helpful assistant to generate formal mammography report from transcripts of conversation between radiologist and patient. 
        You will be provided with transcripts from Hindi conversation between radiologist and patient. There will be four transcripts of the same conversation given to you generated using whisper ai model at different temperatures. You should integrate all the information from the given transcripts and generate the mammography report strictly according to the following instructions. The conversation in Hindi will follow this pattern: Patient symptoms / clinical indication for mammography, followed by Findings described by the radiologist for both breasts and axillae and finally, Discussion with patient about further management and patient concerns, if any. If the patient has undergone unilateral mastectomy, the same information will be provided in the conversation and you should not include that unilateral breast in the report at all and mention history of mastectomy in the clinical indication.
        The conversation will mostly use Hindi along with some English words. You will extract the information from the conversation and prepare a formal mammography report using the following template: 
        {
          "Procedure": "Bilateral / Left / Right Mammogram",
          "Clinical Background": "Mention the indication for mammogram",
          "Right Breast": {
            "Breast density": "ACR category a/b/c/d with one line descriptor as in BIRADS lexicon given below",
            "Parenchyma": "No mass or asymmetric density. No abnormal calcification. No architectural distortion." OR "mention any positive findings seen in the breast parenchyma strictly according to the BIRADS lexicon given below",
            "Areola and Subcutaneous tissues": "Normal" OR "mention any positive findings strictly according to the BIRADS lexicon given below",
            "Axilla": "No suspicious lymph nodes." OR "mention any significant lymph nodes seen in the scan"
          },
          "Left Breast": {
            "Breast density": "ACR category a/b/c/d with one line descriptor as in BIRADS lexicon given below",
            "Parenchyma": "No mass or asymmetric density. No abnormal calcification. No architectural distortion." OR "mention any positive findings seen in the breast parenchyma strictly according to the BIRADS lexicon given below",
            "Areola and Subcutaneous tissues": "Normal" OR "mention any positive findings strictly according to the BIRADS lexicon given below",
            "Axilla": "No suspicious lymph nodes." OR "mention any significant lymph nodes seen in the scan"
          },
          "Comparison": "None" OR "mention comparison with any previous mammogram available",
          "Impression": {
            "Right Breast": {
              "Summary": "Normal study" OR "mention summary of any positive findings strictly according to the BIRADS lexicon given below",
              "BI-RADS Category": "BI-RADS assessment according to findings and their descriptor in the BIRADS lexicon given below"
            },
            "Left Breast": {
              "Summary": "Normal study" OR "mention summary of any positive findings strictly according to the BIRADS lexicon given below",
              "BI-RADS Category": "BI-RADS assessment according to findings and their descriptor in the BIRADS lexicon given below"
            }
          },
          "Recommendation": "Give recommendation on the basis of BIRADS category assessment given below"
        }
        You will prepare the report in json format.

        While preparing the report, keep in mind the following instructions:
        1. Strictly use this mammography BIRADS lexicon in the reports:

        **Breast Composition:**
        - **A:** Entirely fatty
        - **B:** Scattered areas of fibroglandular density
        - **C:** Heterogeneously dense, which may obscure masses
        - **D:** Extremely dense, which lowers sensitivity

        **Mass:**
        - **Shape:** Oval - round - irregular
        - **Margin:** Circumscribed - obscured - microlobulated - indistinct - spiculated
        - **Density:** Fat - low - equal - high

        **Asymmetry:**
        - Asymmetry - global - focal - developing

        **Architectural Distortion:**
        - Distorted parenchyma with no visible mass

        **Calcifications:**
        - **Morphology:** 
          - Typically benign
          - Suspicious:
            1. Amorphous
            2. Coarse heterogeneous
            3. Fine pleomorphic
            4. Fine linear or fine linear branching
        - **Distribution:** Diffuse - regional - grouped - linear - segmental

        **Associated Features:**
        - Skin retraction - nipple retraction - skin thickening - trabecular thickening - axillary adenopathy - architectural distortion – calcifications

        2. The Hindi terms describing the mammogram findings in the transcript should be transformed strictly into BIRADS lexicon terms: for example, छाती बनावट should be transformed as Breast Composition, A: पूरी तरह से चर्बी से भरा should be transformed as A: Entirely fatty , B: ग्बिखरा हुआ घनापन should be transformed as B: Scattered areas of fibroglandular density , C: अलग-अलग रूप से घना, जो गांठों को छिपा सकता है should be transformed as C: Heterogeneously dense, which may obscure masses, D: बहुत ज्यादा घना, जो जांच की क्षमता को कम करता है should be transformed as D: Extremely dense, which lowers sensitivity, नीचे अंदर की तरफ should be transformed as lower inner quadrant, ऊपर बाहर की तरफ should be transformed as upper outer quadrant, गांठ should be transformed as Mass, आकार should be transformed as Shape, अंडे जैसा should be transformed as Oval, गोल should be transformed as Round, टेढ़ा-मेढ़ा should be transformed as Irregular, किनारा should be transformed as Margin, साफ़ should be transformed as Circumscribed, धुंधला should be transformed as Obscured, छोटे-छोटे उभार should be transformed as Microlobulated, साफ़ नहीं should be transformed as Indistinct, काँटेदार should be transformed as Spiculated, घनापन should be transformed as Density, चर्बी should be transformed as Fat, कम should be transformed as Low, बराबर should be transformed as Equal, ज्यादा should be transformed as High, दूसरे स्तन से फ़र्क should be transformed as Asymmetry, हर जगह दूसरे स्तन से फ़र्क should be transformed as Global Asymmetry, एक जगह पर दूसरे स्तन से फ़र्क should be transformed as Focal Asymmetry, बढ़ रहा दूसरे स्तन से फ़र्क should be transformed as Developing Asymmetry, बनावट में बदलाव should be transformed as Architectural distortion, सफेद बिंदियाँ should be transformed as Calcifications, आमतौर पर चिंताजनक नहीं should be transformed as Typically benign, चिंताजनक should be transformed as Suspicious, पाउडर जैसा should be transformed as Amorphous, मोटा अलग-अलग आकार का should be transformed as Coarse heterogeneous, बारिक अलग-अलग आकार का should be transformed as Fine pleomorphic, बारिक रेखा जैसा or बारिक टूटती हुई रेखा जैसा should be transformed as Fine linear or fine linear branching, फैलाव should be transformed as Distribution, फैला हुआ should be transformed as Diffuse, एक जगह पर should be transformed as Regional, इकट्ठा should be transformed as Grouped, सीधा फैलाव should be transformed as Linear distribution, टुकड़ों में should be transformed as Segmental, आस-पास के निशान should be transformed as Associated features, त्वचा का सिकुड़ना should be transformed as Skin retraction, निपल का सिकुड़ना should be transformed as Nipple retraction, त्वचा का मोटा होना should be transformed as Skin thickening, स्तन के अंदर की रेखाओं का मोटा होना should be transformed as Trabecular thickening, बगल में गांठ should be transformed as Axillary adenopathy.
        The Hindi terms in the transcript may vary slightly from the given examples but the BIRADS descriptors used by you should be strictly according to the lexicon provided here.

        3. Strictly follow these management recommendations in the reports:

        ### Final Assessment Categories

        **BIRADS 0:**
        - **Management:** Recall for additional imaging and/or await prior examinations

        **BIRADS 1:**
        - **Management:** Annual screening

        **BIRADS 2:**
        - **Management:** Annual screening

        **BIRADS 3:**
        - **Management:** Short interval follow-up mammography at 6 months

        **BIRADS 4:**
        - **Management:** Biopsy

        **BIRADS 5:**
        - **Management:** Biopsy

        **BIRADS 6:**
        - **Management:** "follow recommendation by the radiologist for the specific case"

        4. If in any case there is not data to be stored in a key for example if there is only one sided breast mammogram you should give that column a value of NONE
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": [
                {"type": "text",
                 "text": f"The audio transcriptions are:\n{''.join(transcriptions)}\nPrepare the mammography report according to the given instructions."}
            ],
             }
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content


def create_excel_buffer(report_data):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df = pd.DataFrame([report_data])
        df.to_excel(writer, index=False)
    output.seek(0)
    return output


def json_to_text(data):
    def check_none(value):
        return "No significant findings" if value is None else value

    text = []
    text.append(f"Procedure:\n {check_none(data['Procedure'])}")
    text.append(f"Clinical Background:\n {check_none(data['Clinical Background'])}")
    if not isinstance(data.get('Right Breast'), dict):
        text.append("Right Breast:\n  No significant findings")
    else:
        text.append(
            f"Right Breast:\n  Breast density: {check_none(data['Right Breast'].get('Breast density'))}\n  Parenchyma: {check_none(data['Right Breast'].get('Parenchyma'))}\n  Areola and Subcutaneous tissues: {check_none(data['Right Breast'].get('Areola and Subcutaneous tissues'))}\n  Axilla: {check_none(data['Right Breast'].get('Axilla'))}")
    if not isinstance(data.get('Left Breast'), dict):
        text.append("Left Breast:\n  No significant findings")
    else:
        text.append(
            f"Left Breast:\n  Breast density: {check_none(data['Left Breast'].get('Breast density'))}\n  Parenchyma: {check_none(data['Left Breast'].get('Parenchyma'))}\n  Areola and Subcutaneous tissues: {check_none(data['Left Breast'].get('Areola and Subcutaneous tissues'))}\n  Axilla: {check_none(data['Left Breast'].get('Axilla'))}")

    text.append(f"Comparison:\n {check_none(data['Comparison'])}")
    if isinstance(data['Impression'].get('Right Breast'), dict):
        text.append(
            f"Impression\n  Right Breast:\n  Summary: {check_none(data['Impression']['Right Breast'].get('Summary'))}\n    BI-RADS Category: {check_none(data['Impression']['Right Breast'].get('BI-RADS Category'))}")
    else:
        text.append("Impression\n  Right Breast:\n  No significant findings")

    if isinstance(data['Impression'].get('Left Breast'), dict):
        text.append(
            f"  Left Breast:\n  Summary: {check_none(data['Impression']['Left Breast'].get('Summary'))}\n    BI-RADS Category: {check_none(data['Impression']['Left Breast'].get('BI-RADS Category'))}")
    else:
        text.append("  Left Breast:\n  No significant findings")

    text.append(f"Recommendation:\n {check_none(data['Recommendation'])}")

    return "\n".join(text)


def generate_pdf_report(report, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=24)
    pdf.cell(200, 20, txt=f"Report of {patient_id}", ln=True, align='C')
    pdf.set_font("Arial", size=12)

    for line in report.split("\n"):
        if line.startswith(("Procedure:", "Clinical Background:", "Right Breast:", "Left Breast:", "Comparison:",
                            "Impression", "Recommendation:")):
            pdf.set_font("Arial", "B", size=16)
            pdf.cell(200, 10, txt=line, ln=True, align='L')
            pdf.set_font("Arial", size=12)
        else:
            pdf.cell(200, 10, txt=line, ln=True, align='L')

    pdf.output(filename)
st.title("Mammography Report Generator")

audio_file = st.file_uploader("Upload an audio file", type=["m4a"])

if audio_file is not None:
    file_name = os.path.splitext(os.path.basename(audio_file.name))[0]
    patient_id = file_name

    # Check if the audio file is new or different
    if "audio_file" not in st.session_state or st.session_state.audio_file != audio_file:
        st.session_state.audio_file = audio_file
        st.session_state.patient_id = patient_id

        start_time = time.time()
        st.write("Processing the audio file...")
        normalized_audio_path = "normalized_audio.m4a"
        st.session_state.converted_audio_path = convert_audio(audio_file, normalized_audio_path)
        st.session_state.conversion_time = time.time()

        st.write(f"Time taken for audio processing: {st.session_state.conversion_time - start_time:.2f} seconds")

        st.write("Transcribing the audio...")
        st.session_state.transcription = transcribe_audio(st.session_state.converted_audio_path)
        st.session_state.transcription_time = time.time()

        st.write(f"Time taken for transcription: {st.session_state.transcription_time - st.session_state.conversion_time:.2f} seconds")

        st.write("Generating the report...")
        st.session_state.report = generate_report(st.session_state.transcription)
        st.session_state.report_time = time.time()

        st.write(f"Time taken to generate report: {st.session_state.report_time - st.session_state.transcription_time:.2f} seconds")

    json_string = f"""{st.session_state.report}"""
    json_string = json_string.replace("```", "")
    json_string = json_string.replace("json", "")
    data = json.loads(json_string)
    edited_report = json_to_text(data)
    edited_report = st.text_area("Editable Report", edited_report, height=300)

    if st.button("Save Changes"):
        report_data = {"Patient ID": st.session_state.patient_id, "Report": edited_report}
        excel_buffer = create_excel_buffer(report_data)

        st.download_button(
            label="Download Excel",
            data=excel_buffer,
            file_name="report_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        pdf_filename = f"{st.session_state.patient_id}_report.pdf"
        generate_pdf_report(edited_report, pdf_filename)

        with open(pdf_filename, "rb") as pdf_file:
            st.download_button(
                label="Download PDF Report",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )
else:
    st.warning("Please upload an audio file to proceed.")