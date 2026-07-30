import io
import os
import re
import tempfile
import threading
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import streamlit as st
import pandas as pd
from PIL import Image
import docx
from pdf2image import convert_from_bytes
import pdfplumber
from pdf2docx import Converter
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Utility Libraries
import qrcode
import cv2

# --- Threading Semaphores for Concurrent Limits ---
if "bg_semaphore" not in st.session_state:
    st.session_state["bg_semaphore"] = threading.Semaphore(2)
bg_semaphore = st.session_state["bg_semaphore"]

if "audio_semaphore" not in st.session_state:
    st.session_state["audio_semaphore"] = threading.Semaphore(5)
audio_semaphore = st.session_state["audio_semaphore"]

if "pdf_semaphore" not in st.session_state:
    st.session_state["pdf_semaphore"] = threading.Semaphore(3)
pdf_semaphore = st.session_state["pdf_semaphore"]

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CA.Editor — Multi-Tool Studio",
    page_icon="⚡",
    layout="wide"
)

# --- Reliable Transparent Asset Links ---
HEADSET_URL = "https://images.rawpixel.com/image_png_800/pngsite-mkt/43e778ea-4ec7-4148-9fdb-069bc4efadac.png"
CAMERA_URL = "https://images.rawpixel.com/image_png_800/pngsite-mkt/96f5b9d2-36e2-4bd5-a131-7b0a3f9bb467.png"

# --- Custom CSS ---
custom_css = f"""
<style>
.stApp {{
    background-color: #01060a;
    color: #f0f6fc;
}}

.bg-image-container {{
    position: fixed;
    top: 50%;
    right: 5%;
    transform: translateY(-50%);
    width: 450px;
    height: 450px;
    z-index: 0;
    pointer-events: none;
    opacity: 0.6;
}}

.bg-image {{
    width: 100%;
    height: 100%;
    background-size: contain;
    background-repeat: no-repeat;
    background-position: center;
    filter: drop-shadow(0 0 35px rgba(0, 229, 255, 0.6));
}}

.headset-bg {{ background-image: url('{HEADSET_URL}'); }}
.camera-bg {{ background-image: url('{CAMERA_URL}'); }}

div.block-container {{
    background: rgba(13, 22, 33, 0.70);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 2.5rem !important;
    margin-top: 1rem;
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.8);
    z-index: 10;
    position: relative;
    border: 1.5px solid rgba(0, 229, 255, 0.3);
}}

.tool-banner {{
    background: linear-gradient(90deg, rgba(0, 229, 255, 0.15) 0%, rgba(0, 119, 255, 0.15) 100%);
    border: 1px solid rgba(0, 229, 255, 0.4);
    border-radius: 10px;
    padding: 10px 15px;
    margin-bottom: 20px;
    color: #00e5ff;
    font-weight: 500;
    font-size: 0.95rem;
}}

.stButton > button {{
    background: linear-gradient(135deg, #00e5ff 0%, #0077ff 100%);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: bold;
    transition: all 0.3s ease;
}}

.stButton > button:hover {{
    box-shadow: 0 0 15px rgba(0, 229, 255, 0.7);
    transform: translateY(-2px);
}}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# --- Sidebar Navigation ---
st.sidebar.markdown("### 🛠️ More Tools Available")
st.sidebar.info("Use the menu below to switch between utility tools.")

app_mode = st.sidebar.radio(
    "Select Utility Tool:", 
    ["🎧 Audio Studio", "🖼️ Image BG Remover", "📄 PDF Converter", "📱 QR Studio"]
)


# ==========================================
# TOOL 1: AUDIO STUDIO
# ==========================================
if app_mode == "🎧 Audio Studio":
    st.markdown(
        """
        <div class="bg-image-container">
            <div class="bg-image headset-bg"></div>
        </div>
        <div class="tool-banner">
            💡 <b>Looking for more tools?</b> Open the left menu (<b>More Tools ➔</b>) to access tools!
        </div>
        """, 
        unsafe_allow_html=True
    )

    def apply_reverb(y, sr, wet_level=0.3, delay_ms=40, decay=0.5):
        if wet_level <= 0:
            return y
        delay_samples = int(sr * (delay_ms / 1000.0))
        output = np.copy(y)
        if output.ndim == 1:
            for i in range(delay_samples, len(output)):
                output[i] += output[i - delay_samples] * decay
            return (1 - wet_level) * y + wet_level * output
        else:
            for ch in range(output.shape[0]):
                for i in range(delay_samples, output.shape[1]):
                    output[ch, i] += output[ch, i - delay_samples] * decay
            return (1 - wet_level) * y + wet_level * output

    def apply_bass_boost(y, sr, gain_db=6.0, cutoff=200):
        if gain_db == 0:
            return y
        b, a = scipy.signal.butter(2, cutoff / (sr / 2), btype='low')
        gain_linear = 10 ** (gain_db / 20)
        filtered = scipy.signal.lfilter(b, a, y)
        return y + (filtered * (gain_linear - 1))

    st.title("CA.Editor — Web-Based Audio Studio")
    st.write("Lightweight, in-memory real-time audio manipulation engine.")

    uploaded_file = st.file_uploader("Upload an Audio File (WAV, MP3, OGG, FLAC)", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        if "audio_data" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
            with st.spinner("Uploading and decoding audio array..."):
                y, sr = librosa.load(uploaded_file, sr=None, mono=False)
                st.session_state["audio_data"] = y
                st.session_state["sr"] = sr
                st.session_state["file_name"] = uploaded_file.name

        y = st.session_state["audio_data"]
        sr = st.session_state["sr"]

        total_samples = y.shape[1] if y.ndim > 1 else len(y)
        total_duration = float(total_samples / sr)

        st.subheader("✂️ Audio Range Trimmer")
        trim_range = st.slider(
            "Drag left handle to cut start, right handle to cut end:",
            min_value=0.0,
            max_value=total_duration,
            value=(0.0, total_duration),
            step=0.1,
            format="%.1fs"
        )

        st.info(f"⏱️ **Selected Audio Duration:** {trim_range[1] - trim_range[0]:.1f}s (from {trim_range[0]:.1f}s to {trim_range[1]:.1f}s)")

        st.subheader("🎛️ Studio Controls")
        col1, col2 = st.columns(2)
        with col1:
            speed = st.slider("Speed Modifier", 0.5, 2.0, 1.0, 0.05)
            pitch = st.slider("Pitch Shift (Semitones)", -8, 8, 0, 1)
        with col2:
            bass = st.slider("Bass Boost (dB)", 0, 12, 0, 1)
            reverb = st.slider("Reverb (Wet/Dry Mix)", 0.0, 0.8, 0.0, 0.05)

        acquired = audio_semaphore.acquire(blocking=False)
        if not acquired:
            st.info("⏳ Audio engine is at capacity (5 active users). Queuing your request...")
            audio_semaphore.acquire()

        try:
            with st.spinner("Applying trimming & digital signal processing..."):
                start_sample = int(trim_range[0] * sr)
                end_sample = int(trim_range[1] * sr)

                if y.ndim > 1:
                    processed_y = y[:, start_sample:end_sample].copy()
                else:
                    processed_y = y[start_sample:end_sample].copy()

                if pitch != 0:
                    processed_y = librosa.effects.pitch_shift(y=processed_y, sr=sr, n_steps=pitch)

                if speed != 1.0:
                    if processed_y.ndim > 1:
                        channels = [librosa.effects.time_stretch(y=processed_y[c], rate=speed) for c in range(processed_y.shape[0])]
                        processed_y = np.array(channels)
                    else:
                        processed_y = librosa.effects.time_stretch(y=processed_y, rate=speed)

                if bass > 0:
                    processed_y = apply_bass_boost(processed_y, sr, gain_db=bass)

                if reverb > 0:
                    processed_y = apply_reverb(processed_y, sr, wet_level=reverb)

                buffer = io.BytesIO()
                out_data = processed_y.T if processed_y.ndim > 1 else processed_y
                sf.write(buffer, out_data, sr, format='WAV')
                buffer.seek(0)
        finally:
            audio_semaphore.release()

        st.subheader("Preview & Export")
        st.audio(buffer, format="audio/wav")
        st.download_button(
            label="Download Processed Audio",
            data=buffer,
            file_name="CA_Editor_Output.wav",
            mime="audio/wav"
        )


# ==========================================
# TOOL 2: IMAGE BACKGROUND REMOVER
# ==========================================
elif app_mode == "🖼️ Image BG Remover":
    st.markdown(
        """
        <div class="bg-image-container">
            <div class="bg-image camera-bg"></div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    st.title("CA.Editor — Image Background Remover")
    st.write("Remove image backgrounds instantly in RAM using AI segmentation.")

    @st.cache_resource
    def load_rembg_session():
        from rembg import new_session
        return new_session("u2netp")

    uploaded_img = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "webp"])

    if uploaded_img is not None:
        input_image = Image.open(uploaded_img)

        st.subheader("Original Image")
        st.image(input_image, use_column_width=True, width=400)

        if st.button("✨ Remove Background"):
            from rembg import remove
            session = load_rembg_session()

            acquired = bg_semaphore.acquire(blocking=False)
            if not acquired:
                st.info("⏳ AI engine is processing 2 images right now. You are next in queue — please wait!")
                bg_semaphore.acquire()

            try:
                with st.spinner("Processing background removal via AI..."):
                    max_dim = 2048
                    if max(input_image.size) > max_dim:
                        input_image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

                    output_image = remove(input_image, session=session)

                    st.subheader("Background Removed")
                    st.image(output_image, use_column_width=True)

                    img_buffer = io.BytesIO()
                    output_image.save(img_buffer, format="PNG")
                    img_buffer.seek(0)

                    st.download_button(
                        label="Download Transparent PNG",
                        data=img_buffer,
                        file_name="CA_BG_Removed.png",
                        mime="image/png"
                    )
            finally:
                bg_semaphore.release()


# ==========================================
# TOOL 3: PDF CONVERTER
# ==========================================
elif app_mode == "📄 PDF Converter":
    st.title("CA.Editor — PDF Studio & Converter")
    st.write("Perform fast, memory-optimized conversion between PDFs, documents, spreadsheets, and images.")

    pdf_option = st.selectbox(
        "Select Tool Mode:",
        [
            "📄 Word / Excel ➔ PDF",
            "🔄 PDF ➔ Word / Excel",
            "🖼️ PDF ➔ Image",
            "📸 Image ➔ PDF"
        ]
    )

    if pdf_option == "📄 Word / Excel ➔ PDF":
        uploaded_doc = st.file_uploader("Upload Word (.docx) or Excel (.xlsx) File", type=["docx", "xlsx"])

        if uploaded_doc and st.button("🔄 Convert to PDF"):
            acquired = pdf_semaphore.acquire(blocking=False)
            if not acquired:
                st.info("⏳ Engine busy with other conversions (Limit: 3 parallel). Queuing your request...")
                pdf_semaphore.acquire()

            try:
                with st.spinner("Generating PDF document in memory..."):
                    pdf_buffer = io.BytesIO()

                    if uploaded_doc.name.endswith(".docx"):
                        doc = docx.Document(uploaded_doc)
                        doc_pdf = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                        styles = getSampleStyleSheet()
                        story = []

                        for p in doc.paragraphs:
                            if p.text.strip():
                                story.append(Paragraph(p.text, styles['Normal']))
                                story.append(Spacer(1, 10))

                        doc_pdf.build(story)

                    elif uploaded_doc.name.endswith(".xlsx"):
                        excel_data = pd.read_excel(uploaded_doc)
                        excel_data = excel_data.fillna("")
                        
                        doc_pdf = SimpleDocTemplate(pdf_buffer, pagesize=letter)
                        table_data = [excel_data.columns.values.tolist()] + excel_data.values.tolist()

                        pdf_table = Table(table_data)
                        pdf_table.setStyle(TableStyle([
                            ('BACKGROUND', (0,0), (-1,0), colors.grey),
                            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                            ('GRID', (0,0), (-1,-1), 1, colors.black)
                        ]))

                        doc_pdf.build([pdf_table])

                    pdf_buffer.seek(0)
                    st.download_button(
                        label="Download PDF File",
                        data=pdf_buffer,
                        file_name=f"{uploaded_doc.name.rsplit('.', 1)[0]}.pdf",
                        mime="application/pdf"
                    )
            finally:
                pdf_semaphore.release()

    elif pdf_option == "🔄 PDF ➔ Word / Excel":
        uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
        target_format = st.radio("Select Target Output Format:", ["Word (.docx)", "Excel (.xlsx)"], horizontal=True)

        if uploaded_pdf and st.button("🔄 Convert PDF"):
            acquired = pdf_semaphore.acquire(blocking=False)
            if not acquired:
                st.info("⏳ Engine busy with other conversions (Limit: 3 parallel). Queuing your request...")
                pdf_semaphore.acquire()

            try:
                with st.spinner("Extracting content and parsing structures..."):
                    pdf_bytes = uploaded_pdf.read()

                    if target_format == "Word (.docx)":
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                            tmp_pdf.write(pdf_bytes)
                            tmp_pdf_path = tmp_pdf.name

                        tmp_docx_path = tmp_pdf_path.replace(".pdf", ".docx")

                        try:
                            cv = Converter(tmp_pdf_path)
                            cv.convert(tmp_docx_path)
                            cv.close()

                            with open(tmp_docx_path, "rb") as f:
                                docx_bytes = f.read()

                            st.download_button(
                                label="Download Word (.docx)",
                                data=docx_bytes,
                                file_name=f"{uploaded_pdf.name.rsplit('.', 1)[0]}.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                        finally:
                            if os.path.exists(tmp_pdf_path):
                                os.remove(tmp_pdf_path)
                            if os.path.exists(tmp_docx_path):
                                os.remove(tmp_docx_path)

                    elif target_format == "Excel (.xlsx)":
                        excel_buffer = io.BytesIO()
                        all_tables = []

                        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                            for page in pdf.pages:
                                tables = page.extract_tables()
                                for table in tables:
                                    if table:
                                        df = pd.DataFrame(table[1:], columns=table[0])
                                        all_tables.append(df)

                        if all_tables:
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                for idx, df in enumerate(all_tables):
                                    df.to_excel(writer, sheet_name=f"Table_{idx+1}", index=False)
                            excel_buffer.seek(0)

                            st.download_button(
                                label="Download Excel (.xlsx)",
                                data=excel_buffer,
                                file_name=f"{uploaded_pdf.name.rsplit('.', 1)[0]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            st.warning("No structured tables could be detected in this PDF file.")
            finally:
                pdf_semaphore.release()

    elif pdf_option == "🖼️ PDF ➔ Image":
        uploaded_pdf = st.file_uploader("Upload PDF File", type=["pdf"])

        if uploaded_pdf and st.button("🖼️ Extract Pages as PNG"):
            acquired = pdf_semaphore.acquire(blocking=False)
            if not acquired:
                st.info("⏳ Engine busy with other conversions (Limit: 3 parallel). Queuing your request...")
                pdf_semaphore.acquire()

            try:
                with st.spinner("Rendering PDF pages to PNG images..."):
                    import zipfile
                    images = convert_from_bytes(uploaded_pdf.read())

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for i, img in enumerate(images):
                            img_byte_arr = io.BytesIO()
                            img.save(img_byte_arr, format='PNG')
                            zip_file.writestr(f"page_{i+1}.png", img_byte_arr.getvalue())

                    zip_buffer.seek(0)
                    st.download_button(
                        label="Download Pages ZIP",
                        data=zip_buffer,
                        file_name=f"{uploaded_pdf.name.rsplit('.', 1)[0]}_images.zip",
                        mime="application/zip"
                    )
            finally:
                pdf_semaphore.release()

    elif pdf_option == "📸 Image ➔ PDF":
        uploaded_imgs = st.file_uploader(
            "Upload Images (PNG, JPG, JPEG, WEBP)", 
            type=["png", "jpg", "jpeg", "webp"], 
            accept_multiple_files=True
        )

        if uploaded_imgs and st.button("📸 Convert Images to PDF"):
            acquired = pdf_semaphore.acquire(blocking=False)
            if not acquired:
                st.info("⏳ Engine busy with other conversions (Limit: 3 parallel). Queuing your request...")
                pdf_semaphore.acquire()

            try:
                with st.spinner("Downscaling and building PDF in RAM..."):
                    pil_images = []
                    max_dim = 2048

                    for img in uploaded_imgs:
                        im = Image.open(img)
                        if max(im.size) > max_dim:
                            im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        pil_images.append(im.convert("RGB"))

                    pdf_buffer = io.BytesIO()
                    if pil_images:
                        pil_images[0].save(
                            pdf_buffer, 
                            format="PDF", 
                            save_all=True,
                            append_images=pil_images[1:]
                        )
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="Download PDF Document",
                        data=pdf_buffer,
                        file_name="Converted_Images.pdf",
                        mime="application/pdf"
                    )
            finally:
                pdf_semaphore.release()


# ==========================================
# TOOL 4: QR CODE GENERATOR & DECODER
# ==========================================
elif app_mode == "📱 QR Studio":
    st.title("CA.Editor — QR Code Studio")
    st.write("Generate custom QR codes or decode existing ones (including Wi-Fi network credentials).")

    qr_sub_mode = st.radio("Select Action:", ["✨ Generate QR Code", "🔍 Decode QR Code"], horizontal=True)

    if qr_sub_mode == "✨ Generate QR Code":
        qr_type = st.selectbox("Content Type:", ["Plain Text / URL", "📶 Wi-Fi Access Point"])

        if qr_type == "Plain Text / URL":
            qr_text = st.text_area("Enter Text or Web Address (URL):", placeholder="https://example.com")
            
            if qr_text and st.button("⚡ Generate QR Code"):
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(qr_text)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                st.subheader("Your Generated QR Code:")
                st.image(img_buffer, width=300)

                st.download_button(
                    label="Download QR Code (PNG)",
                    data=img_buffer,
                    file_name="qrcode.png",
                    mime="image/png"
                )

        elif qr_type == "📶 Wi-Fi Access Point":
            col1, col2 = st.columns(2)
            with col1:
                ssid = st.text_input("Network Name (SSID):")
                security = st.selectbox("Security Type:", ["WPA", "WEP", "nopass"])
            with col2:
                password = st.text_input("Wi-Fi Password:", type="password" if security != "nopass" else "default")
                hidden = st.checkbox("Hidden Network?")

            if ssid and st.button("📶 Generate Wi-Fi QR Code"):
                wifi_str = f"WIFI:S:{ssid};T:{security};P:{password if security != 'nopass' else ''};H:{'true' if hidden else 'false'};;"
                
                qr = qrcode.QRCode(
                    version=1,
                    error_correction=qrcode.constants.ERROR_CORRECT_H,
                    box_size=10,
                    border=4,
                )
                qr.add_data(wifi_str)
                qr.make(fit=True)

                img = qr.make_image(fill_color="black", back_color="white")
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_buffer.seek(0)

                st.subheader("Wi-Fi Connection QR Code:")
                st.image(img_buffer, width=300)

                st.download_button(
                    label="Download Wi-Fi QR Code (PNG)",
                    data=img_buffer,
                    file_name="wifi_qrcode.png",
                    mime="image/png"
                )

    elif qr_sub_mode == "🔍 Decode QR Code":
        uploaded_qr = st.file_uploader("Upload Image Containing QR Code", type=["png", "jpg", "jpeg", "webp"])

        if uploaded_qr:
            pil_img = Image.open(uploaded_qr)
            st.image(pil_img, caption="Uploaded Image", width=300)

            if st.button("🔍 Scan & Decode"):
                with st.spinner("Decoding QR code using OpenCV..."):
                    img_bytes = np.asarray(bytearray(uploaded_qr.getvalue()), dtype=np.uint8)
                    cv_img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                    
                    detector = cv2.QRCodeDetector()
                    raw_data, _, _ = detector.detectAndDecode(cv_img)

                if raw_data:
                    st.success("✅ QR Code Successfully Decoded!")
                    
                    if raw_data.startswith("WIFI:"):
                        st.subheader("📶 Wi-Fi Network Credentials Detected")
                        
                        ssid_match = re.search(r"S:(.*?);", raw_data)
                        pass_match = re.search(r"P:(.*?);", raw_data)
                        type_match = re.search(r"T:(.*?);", raw_data)

                        ssid_val = ssid_match.group(1) if ssid_match else "N/A"
                        pass_val = pass_match.group(1) if pass_match else "(No Password / Open Network)"
                        type_val = type_match.group(1) if type_match else "Unknown"

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric(label="Network SSID Name", value=ssid_val)
                            st.metric(label="Security Type", value=type_val)
                        with col_b:
                            st.code(f"Password: {pass_val}", language="text")
                        
                        st.text_area("Raw Decoded String:", raw_data, height=70)
                    else:
                        st.subheader("Decoded Text Payload:")
                        st.code(raw_data, language="text")
                else:
                    st.error("❌ Could not detect or read any valid QR code in this image. Make sure the image is clear and well-lit.")
