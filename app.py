import io
import librosa
import numpy as np
import scipy.signal
import soundfile as sf
import streamlit as st
from PIL import Image

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CA.Editor — Multi-Tool Studio",
    page_icon="⚡",
    layout="wide"
)

# --- Reliable Transparent Asset Links ---
HEADSET_URL = "https://images.rawpixel.com/image_png_800/pngsite-mkt/43e778ea-4ec7-4148-9fdb-069bc4efadac.png"
CAMERA_URL = "https://images.rawpixel.com/image_png_800/pngsite-mkt/96f5b9d2-36e2-4bd5-a131-7b0a3f9bb467.png"

# --- Inject Custom CSS ---
custom_css = f"""
<style>
/* Main App Background */
.stApp {{
    background-color: #01060a;
    color: #f0f6fc;
}}

/* Stationary Background Image Container */
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

/* Specific background images (Stationary) */
.headset-bg {{
    background-image: url('{HEADSET_URL}');
}}

.camera-bg {{
    background-image: url('{CAMERA_URL}');
}}

/* Glassmorphism Main Container */
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

/* Tool Switching Banner at Top */
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

.stSlider > div {{ color: #ffffff; }}

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

# --- Enhanced Sidebar Navigation ---
st.sidebar.markdown("### 🛠️ More Tools Available")
st.sidebar.info("Use the menu below to switch between utility tools.")

app_mode = st.sidebar.radio(
    "Select Utility Tool:", 
    ["🎧 Audio Studio", "🖼️ Image BG Remover"]
)


# ==========================================
# TOOL 1: AUDIO STUDIO
# ==========================================
if app_mode == "🎧 Audio Studio":
    # Stationary Headset Background Container
    st.markdown(
        """
        <div class="bg-image-container">
            <div class="bg-image headset-bg"></div>
        </div>
        <div class="tool-banner">
            💡 <b>For more tools</b> Open the left menu <b>More Tools   ➔</b> Use the AI Image Background Remover!
        </div>
        """, 
        unsafe_allow_html=True
    )

    def apply_reverb(y, sr, wet_level=0.3, delay_ms=40, decay=0.5):
        if wet_level <= 0: return y
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
        if gain_db == 0: return y
        b, a = scipy.signal.butter(2, cutoff / (sr / 2), btype='low')
        gain_linear = 10 ** (gain_db / 20)
        filtered = scipy.signal.lfilter(b, a, y)
        return y + (filtered * (gain_linear - 1))

    st.title("CA.Editor — Web-Based Editor Studio — Audio editor")
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

        st.subheader("Studio Controls")
        col1, col2 = st.columns(2)
        with col1:
            speed = st.slider("Speed Modifier", 0.5, 2.0, 1.0, 0.05)
            pitch = st.slider("Pitch Shift (Semitones)", -8, 8, 0, 1)
        with col2:
            bass = st.slider("Bass Boost (dB)", 0, 12, 0, 1)
            reverb = st.slider("Reverb (Wet/Dry Mix)", 0.0, 0.8, 0.0, 0.05)

        with st.spinner("Applying digital signal processing..."):
            processed_y = y.copy()

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
    # Stationary Camera Background Container
    st.markdown(
        """
        <div class="bg-image-container">
            <div class="bg-image camera-bg"></div>
        </div>
        <div class="tool-banner">
            💡 <b>Need audio editing?</b> Switch back to <b>🎧 Audio Studio</b> using the left menu!
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

            with st.spinner("Removing background via AI..."):
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
                
