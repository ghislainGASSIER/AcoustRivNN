#https://fr.acervolima.com/un-guide-du-debutant-pour-streamlit/
import keras as keras
import librosa as librosa
import tensorflow
import joblib

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
import pydub
from scipy import signal
from numpy import linalg as LA

img = Image.open("logo_CEREGE.jpeg")
st.sidebar.image(img, width=200)
st.title('AcoustRivNN')
st.text(" Cette application.............")

from pathlib import Path
def draw_Estimed_Masse(Masse,tps,cl_Masse):
    fig = go.Figure(go.Scatter(
        x=tps,
        y=Masse
    ))
    print(np.unique(Masse))
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.unique(Masse),
            ticktext=cl_Masse
        )
    )

    st.plotly_chart(fig)

def draw_Estimed_Granulo(Granulo,tps,cl_granulo):
    fig = go.Figure(go.Scatter(
        x=tps,
        y=Granulo
    ))
    print(np.unique(Granulo))
    fig.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=np.unique(Granulo),
            ticktext=cl_granulo
        )
    )

    st.plotly_chart(fig)
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
def ProxL1(Poids, invlipchitz, x):

    soft = lambda x, T: x - np.sign(x) * T * (x > T)
    sol = soft(Poids * x, ParametreTraitement["Regularisation"] * invlipchitz)
    # self.rho # gamma/0.33333333333
    return sol
def df(x, C, M):
    y = np.dot(M, x) - C
    return y
def f(x, A, b):
    y = LA.norm(np.dot(A, x) - b, 2)
    return y
def gradientProjeter(A, signal, lipchitz, M):
    print("    Appel de la fonction gradientProjeter")
    Poids = np.ones(A.shape[1])
    erreur = 0
    x0 = np.zeros(A.shape[1])
    C = np.dot(A.T, signal)
    xn = x0
    i = 0
    xnp1 = xn
    fxn = 0
    rho = 0
    dfxn = 0
    invlipchitz = np.divide(1.0, lipchitz)
    erreur = 0
    i = 0
    print("Parametre de régularisation :", ParametreTraitement["Regularisation"])
    print("    Pas de descente :", invlipchitz)
    while erreur == 0:  # boucle selection
        # print "i= ", i
        AtomeEstimer = np.argmax(xn)
        norm_l0 = sum(np.absolute(xn[:] > 0))
        if (
            np.mod(
                np.divide(i * 100, ParametreTraitement["NbMaxIteration"], dtype=float),
                1,
            )
            == 0
            and i > 0
        ):
            print(
                "    "
                + str(round((i * 100) / ParametreTraitement["NbMaxIteration"]))
                + " %"
            )
            # print '    atome estimer : ' ,AtomeEstimer
            # print '    Energie de l Estimation de xn :',LA.norm(xn,2)
            print("    valeur de l erreur quadratique : ", fxn)
            print("    Nombre de coefficient non null : ", norm_l0)
            # print '    Pas de descente :',self.rho
            print("\n")
        # descente de gradient
        dfxn = df(xn, C, M)
        fxn = f(xn, A, signal)
        rho = np.divide(
            fxn, np.vdot(dfxn, dfxn), dtype=np.float64)
        # descente de gradient  : min ||Ax-y||²
        #############confiance à l'attache quadratique : yn = xn -self.rho*dfxn
        xnp1 = xn + ParametreTraitement["relaxation"] * (
            ProxL1(Poids, invlipchitz, xn - invlipchitz * dfxn) - xn
        )
        # projection sur les positifs
        I0 = np.where(xnp1 < 0)
        xnp1[I0] = 0
        # Appriori de parcimnie
        xnp1 = xnp1 + ParametreTraitement["relaxation"] * (ProxL1(Poids, invlipchitz, xnp1) - xnp1)
        #######contrainte de parcimonie L0
        # xnp1=xn+ParametreTraitement['lambda']*(self.proxL0(yn)-xn);#self.rho Lambda
        if i == 0:
            fx = np.array([fxn])
            support = np.array([norm_l0])
            ArgMax = np.array([AtomeEstimer])
            RHO = np.array([rho])
        else:
            # x=np.concatenate((x,np.array([xnp1]).T),axis=1)
            fx = np.concatenate((fx, np.array([fxn])))
            support = np.concatenate((support, np.array([norm_l0])))
            ArgMax = np.concatenate((ArgMax, np.array([AtomeEstimer])))
            RHO = np.concatenate((RHO, np.array([rho])))
        if LA.norm(xnp1 - xn, 2) < ParametreTraitement["Stop"]:
            erreur = 1
            print(
                "       Atomes selectionner :  Arret sur l erreur entre deux iterations :",
                ParametreTraitement["Stop"],
            )
        if i > ParametreTraitement["NbMaxIteration"]:
            erreur = 2
            print(
                "       Atomes selectionner :  Arret sur nombre maximum d"
                "iterations : ",
                i,
            )
        xn = xnp1
        i = i + 1
    # show(block=False)
    S1 = ParametreDictionnaire["Vrayon"].shape[0]
    S3 = ParametreDictionnaire["VRET"].shape[0]
    Detection = np.copy(xn.reshape(S3, S1)).T
    return Detection


file_uploader = st.sidebar.file_uploader(label="", type=".wav")
if file_uploader is not None:
   st.write(file_uploader)
   y, sr = handle_uploaded_audio_file(file_uploader)
fic="Sound.wav"
signal, sr =librosa.load(fic,sr=192000)
#sampling_frequency,sound = wavfile.read(fic)
#samplerate, data = wavfile.read(fic)
length = signal.shape[0] / sr
time = np.linspace(0., length, signal.shape[0])
named_colorscales = px.colors.named_colorscales()
nyquist_frequency = int(sr/2)

draw_sound(signal,time)

# Découpage du signal audio en audio de 1s et calcul des descripteurs
nb_audio = int(np.floor(len(signal) / sr))

SignalDecoupe = []
for i in range(nb_audio):
    SignalDecoupe.append(signal[i * sr:(i + 1) * sr])
type_NN = st.sidebar.selectbox(('Choose a type of neuronal network'),('CNN-STFT','CNN-AcoustRiv', 'MLP'))
if(type_NN=='MlP'):
    model = keras.models.load_model("Model_MLP_Augmented_DataBase.h5")
    DescriptorList=[]
    for sig in SignalDecoupe:
        f, Pxx_den = signal.periodogram(sig, fs=192000)
        P = []
        for i in range(len(f)):
            if f[i] <= 40000 and f[i] >= 100:
                P.append(Pxx_den[i])
        P = P / np.max(P)
        DescriptorList.append(P)
if(type_NN=='CNN-STFT'):
    print('Begin')
    model = keras.models.load_model("Model_AcoustRivNN_STFT_flux_and_granulo.h5")
    DescriptorList = []
    for sig in SignalDecoupe:
        spec = np.abs(librosa.stft(sig, n_fft=1024, hop_length=512))
        DescriptorList.append(spec)
    encoder = joblib.load('encoder_CNN_STFT.joblib')
    Intervalle_Masse=np.load('intervals_masse_STFT.npy')
    Intervalle_Granulo=np.load('intervals_granulometrie_STFT.npy')
    print("end")
if(type_NN=='CNN-AcoustRiv'):
    model = keras.models.load_model("Model_AcoustRivNN_AcoustRiv_flux_and_granulo.h5")
    ParametreDictionnaire = {}
    ParametreDictionnaire['RateDiametre'] = 0.001  # 0.01#
    ParametreDictionnaire['AtomeShape'] = signal.shape[0]  # 960# 350
    ParametreDictionnaire['Vrayon'] = np.arange(0.003, 0.05, ParametreDictionnaire['RateDiametre'])
    ParametreDictionnaire['VRET'] = np.arange(1, 160000, 10000)  # ParametreCollision['n']
    ParametreTraitement = {}
    ParametreTraitement["thres"] = 0.9
    ParametreTraitement["relaxation"] = 0.09  # 0.09
    ParametreTraitement["Regularisation"] = 0.003  # 0.003#0.8#10**(150)#1
    ParametreTraitement["NbMaxIteration"] = 1000  # 10**40
    ParametreTraitement["Stop"] = 10 ** (-4)  # 0.0001#
    A = np.load("./Dictionnaire.npy")
    M = np.dot(A.T, A)
    lipchitz = LA.norm(M, 2)
    DescriptorList = []
    for sig in SignalDecoupe:
        rate, signal = scipy.io.wavfile.read(sig)
        PlanTempsRayon = gradientProjeter(A, sig, lipchitz, M)
        DescriptorList.append(PlanTempsRayon)

pred = model.predict(np.array(DescriptorList))

labels,classes=[],[]

for e in pred:
    e=np.reshape(e,(1,-1))
    labels.append((e>=0.5).astype(int))

for label in labels:
    cl=encoder.inverse_transform(label)
    classes.append(cl)

tps=np.arange(nb_audio)

Masse,Granulo=[],[]

for c in classes:

    if (c[0][0]==None or c[0][1]==None):
        Masse.append(0)
        Granulo.append(0)
    else:
        Masse.append(c[0][1])
        Granulo.append(c[0][0])
#draw_sound(signal,time)

cl_masse,cl_granulo=[],[]


for i in range(1,len(Intervalle_Masse)):
    cl_masse.append(str([Intervalle_Masse[i-1],Intervalle_Masse[i]]))

for i in range(1,len(Intervalle_Granulo)):
    cl_granulo.append(str([Intervalle_Granulo[i - 1], Intervalle_Granulo[i]]))



draw_Estimed_Masse(Masse,tps,cl_masse)
draw_Estimed_Granulo(Granulo,tps,cl_granulo)


