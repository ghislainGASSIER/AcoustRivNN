#https://fr.acervolima.com/un-guide-du-debutant-pour-streamlit/
from scipy.io import wavfile
import scipy.io
from scipy import signal
import numpy as np
import streamlit as st
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import matplotlib.pyplot as plt
import pydub
from PIL import Image

img = Image.open("logo_CEREGE.jpeg")
st.sidebar.image(img, width=200)

st.title('AcoustRivNN')
st.text(" Cette application.............")

#https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
#uploaded_file = st.file_uploader("Choose a wav file")#, accept_multiple_files=True
#print(type(list(uploaded_file)))
import pydub
from pathlib import Path
#import ffprobe

from pathlib import Path
def draw_sound(sig,t):
    import plotly.express as px
    import pandas as pd
    sig=sig[::10]
    t=t[::10]
    df = pd.DataFrame(dict(x=t,y=sig))
    fig = px.line(df, x="x", y="y", title="Signal",width=1000, height=400)
    st.plotly_chart(fig)

def handle_uploaded_audio_file(uploaded_file):
    sample = pydub.AudioSegment.from_wav(uploaded_file)
    #st.write(sample.sample_width)
    samples = sample.get_array_of_samples()
    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples.typecode).max
    #st.sidebar.write(fp_arr.shape)
    sample.export(out_f="Sound.wav",format="wav")
    st.sidebar.write('Téléchargement reussit')
    fic = "Sound.wav"
    samplerate, data = wavfile.read(fic)
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    draw_sound(data, time)
    return fp_arr, 22050

file_uploader = st.sidebar.file_uploader(label="", type=".wav")
if file_uploader is not None:
   #st.write(file_uploader)
   y, sr = handle_uploaded_audio_file(file_uploader)



from scipy.io.wavfile import write

#st.write("filename:", uploaded_file[0].name)
#print(uploaded_file)
#with open('myfile.csv') as f:
#   st.download_button('Download CSV', f)  # Defaults to 'text/plain'
#waveform = pd.DataFrame({"Amplitude": sound.values[0].T})
#st.line_chart(waveform)


fic="Sound.wav"
# Load sound into Praat
#sound = parselmouth.Sound("03-01-01-01-01-01-01.wav")
sampling_frequency,sound = wavfile.read(fic)
samplerate, data = wavfile.read(fic)
length = data.shape[0] / samplerate
time = np.linspace(0., length, data.shape[0])

named_colorscales = px.colors.named_colorscales()


def draw_spectrogram(spectrogram,time,f):
    sg_db = 10 * np.log10(spectrogram)
    # Plot with plotly
    data = [go.Heatmap(x=t, y=f, z=sg_db, zmin=vmin, zmax= vmax, colorscale=colours,)]
    layout = go.Layout(
        title='Spectrogram',
        yaxis=dict(title='Frequency (Hz)'),
        xaxis=dict(title='Time (s)'),
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig)
# Side Bar #######################################################
nyquist_frequency = int(sampling_frequency/2)
#maximum_frequency = st.sidebar.slider('Maximum frequency (Hz)', 5000, nyquist_frequency, 5500)

#default_ix = named_colorscales.index('turbo')
type_NN = st.sidebar.selectbox(('Choose a type of neuronal network'),('CNN', 'MLP'))
#colours = st.sidebar.selectbox(('Choose a colour pallete'), named_colorscales, index=default_ix)
#dynamic_range = st.sidebar.slider('Dynamic Range (dB)',  min_value=100, max_value=1000, value=200, step=10)
#window_length = st.sidebar.slider('Window length (s)',  min_value=100, max_value=1000, value=200, step=10)


# App ##################################################
# Load sound into Praat
#sampling_frequency,sound = wavfile.read(fic)
#sound = parselmouth.Sound("03-01-01-01-01-01-01.wav")
#sound.pre_emphasize()


from scipy.fft import fftshift
#f, t, Sxx = signal.spectrogram(sound,sampling_frequency,nperseg=int(window_length),nfft=dynamic_range)
#spectrogram = sound.to_spectrogram(window_length=window_length, maximum_frequency=maximum_frequency)
#sg_db = 10 * np.log10(Sxx)
#vmin = sg_db.max() - dynamic_range
#vmax = sg_db.max() #+ dynamic_range
#draw_spectrogram(Sxx, t, f)
#draw_sound(data,time)
