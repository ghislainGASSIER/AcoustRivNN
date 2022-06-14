from scipy.io import wavfile

import scipy.io

import numpy as np
import streamlit as st
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
import matplotlib.pyplot as plt

st.title('AcoustRivNN')
#https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader
uploaded_file = st.file_uploader("Choose a wav file")#, accept_multiple_files=True
from scipy.io.wavfile import write
write("example.wav", uploaded_file)
plt.plot(uploaded_file)
plt.show()
#st.write("filename:", uploaded_file[0].name)

#print(uploaded_file)

#with open('myfile.csv') as f:
#   st.download_button('Download CSV', f)  # Defaults to 'text/plain'



#waveform = pd.DataFrame({"Amplitude": sound.values[0].T})
#st.line_chart(waveform)
fic="5galets20g1mamontgainMax.wav"
# Load sound into Praat
#sound = parselmouth.Sound("03-01-01-01-01-01-01.wav")
sampling_frequency,sound = wavfile.read(fic)
plt.plot(sound)
plt.show()
named_colorscales = px.colors.named_colorscales()
print(sound)
def draw_spectrogram(spectrogram,t,f):
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
maximum_frequency = st.sidebar.slider('Maximum frequency (Hz)', 5000, nyquist_frequency, 5500)

default_ix = named_colorscales.index('turbo')
colours = st.sidebar.selectbox(('Choose a colour pallete'), named_colorscales, index=default_ix)
dynamic_range = st.sidebar.slider('Dynamic Range (dB)',  min_value=100, max_value=1000, value=200, step=10)
window_length = st.sidebar.slider('Window length (s)',  min_value=100, max_value=1000, value=200, step=10)


# App ##################################################
# Load sound into Praat
sampling_frequency,sound = wavfile.read(fic)
#sound = parselmouth.Sound("03-01-01-01-01-01-01.wav")
#sound.pre_emphasize()
from scipy import signal

from scipy.fft import fftshift
f, t, Sxx = signal.spectrogram(sound,sampling_frequency,nperseg=int(window_length),nfft=dynamic_range)
#spectrogram = sound.to_spectrogram(window_length=window_length, maximum_frequency=maximum_frequency)
sg_db = 10 * np.log10(Sxx)
vmin = sg_db.max() - dynamic_range
vmax = sg_db.max() #+ dynamic_range
draw_spectrogram(Sxx, t, f)
