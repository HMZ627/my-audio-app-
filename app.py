import streamlit as st
from pydub import AudioSegment
import io

st.title("Free Audio Speed Changer")

# 1. User uploads a file directly into temporary RAM
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # 2. Python reads the audio file straight from RAM
    audio = AudioSegment.from_file(uploaded_file)
    
    # 3. Apply the effect (e.g., speed up the audio by 1.5x)
    fast_audio = audio._spawn(audio.raw_data, overrides={
        "frame_rate": int(audio.frame_rate * 1.5)
    }).set_frame_rate(audio.frame_rate)
    
    # 4. Save the new MP3 into an in-memory byte buffer (RAM)
    buffer = io.BytesIO()
    fast_audio.export(buffer, format="mp3")
    
    # 5. Provide a download button for the user
    st.audio(buffer.getvalue(), format="audio/mp3")
    st.download_button(
        label="Download Fast MP3",
        data=buffer.getvalue(),
        file_name="fast_effect.mp3",
        mime="audio/mp3"
    )
    # The moment the session ends, 'buffer' drops out of RAM automatically!
  
