import streamlit as st
import librosa
import soundfile as sf
import io

st.title("Free Modern Audio Speed Changer")

uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav"])

if uploaded_file is not None:
    # Librosa handles the file directly in Python memory
    y, sr = librosa.load(uploaded_file, sr=None)

    # Speed up the audio by 1.5x smoothly
    y_fast = librosa.effects.time_stretch(y, rate=1.5)

    # Convert the processed sound array back to a file buffer
    buffer = io.BytesIO()
    sf.write(buffer, y_fast, sr, format='wav')
    buffer.seek(0)

    # Show preview and download options
    st.audio(buffer.getvalue(), format="audio/wav")
    st.download_button(
        label="Download Fast Audio",
        data=buffer.getvalue(),
        file_name="fast_effect.wav",
        mime="audio/wav"
     )
