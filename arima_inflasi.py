# Import Libraries
import streamlit as st
import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import pmdarima
from pmdarima import auto_arima
import warnings
import time
warnings.filterwarnings('ignore')

features=[
    "Introduksi",
    "Upload File",
    "Diskripsi",
    "Plot historical inflation",
    "Check stationarity and autocorrelation",
    "Fit SARIMA model, parameter, prediksi dan evaluasi",
    ]
menu=pd.DataFrame(features)
#st.write(menu)
#[m,n] =menu.shape
#st.write(m,n)
#st.sidebar.image("logo_stiei.jpg", use_column_width=False)
st.sidebar.markdown('<h3 style="color: White;"> Author: Nasri </h3>', unsafe_allow_html=True)
st.sidebar.markdown('<h5 style="color: White;"> email: nasri@stieimlg.ac.id </h5>', unsafe_allow_html=True)
st.sidebar.markdown('<h1 style="color: White;">Prediksi Inflasi dengan model SARIMA</h1>', unsafe_allow_html=True)

model_analisis = st.sidebar.radio('Baca ketentuan penggunaan dengan seksama, Pilih Analisis Statistik:', menu)

def intro():
    st.write("## Selamat Datang di Dashboard Prediksi Inflasi di Indonesia.  👋  👋")
    st.write("##### author: m nasri aw, email: nasri@stieimlg.ac.id; lecturer at https://www.stieimlg.ac.id/; Des 2024.")
    st.write(f"##### - Ketentuan: ")
    '''
    1. Aplikasi ini menggunakan bahasa python dengan library utama statsmodels dan streamlit.
    2. Menggunakan model SARIMA dengan data training inflasi 18 Tahun @ 12 bulan, model dapat digunakan untuk prediksi inflasi.
    3. Diperlukan data dalam bentuk file csv (contoh dapat di download di https://github.com/nasriaw/inflation_indonesia_2006-2024/blob/main/inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv;
    4. File hanya terdiri dari kolom pertama kolom waktu (bersifat series bisa dalam bentuk bulanan dan musiman), kolom ke dua data inflasi atau sejenis, misal data perjalanan, penenan, dll.
    5. Analisis statistik regresi meliputi:
       1. Diskripsi.
       2. Plot historical inflation.
       3. Check stationarity and autocorrelation.
       4. Fit SARIMA model, parameter, prediksi dan evaluasi.
       ###### 👈 Pilih Menu di sebelah; Pastikan data telah di upload (langkah ke-2: Upload File)
    6. Untuk link demo silahkan klik https://huggingface.co/spaces; Selamat belajar semoga memudahkan untuk memahami statistik regresi.
    '''
    return intro

def open_file():
    if 'data' not in st.session_state:
        st.session_state.data = None

    def load_data():
        st.session_state.data = pd.read_csv(st.session_state.loader)

    file = st.file_uploader('Choose a file', type='csv', key='loader', on_change=load_data)

    df = st.session_state.data
    if df is not None:
        # Run program
        st.write('Gunakan Browse Files jika upload data baru.')
    return df
#df=open_file()

def descriptive():
    df=open_file()
    df= df.dropna()
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0])
    st.write("### 1. Data Head dan Statistik Diskripsi ") #{df.iloc[:,1]}")
    st.write(f"dimensi data: {df.shape}")
    st.write("Data Head : ")
    st.write(df.head())
    st.write("Data Diskripsi Inflasi : ")
    st.write(df.iloc[:,1].describe())#(include='all').fillna("").astype("str"))
    
def historical():
    df=open_file()
    df.set_index(df.iloc[:,0], inplace=True)
    st.line_chart(df.iloc[:,1]) #scatter, bar, line, area, altair

def check_stationarity():
    # S3: Check stationarity and autocorrelation
    # Plot ACF and PACF to identify ARIMA parameters
    df=open_file()
    df.set_index(df.iloc[:,0], inplace=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4))
    plot_acf(df.iloc[:,1], ax=ax1)
    plot_pacf(df.iloc[:,1], ax=ax2)
    st.pyplot(fig)
    
    #decomposition_montly = seasonal_decompose(df.iloc[:,1], model='additive', period=12)
    #decomposition_montly.plot()
    #st.pyplot(fig)
 
    result = adfuller(df.iloc[:,1])
    st.write("ADF Statistic:", result[0].round(4))
    st.write("p-value:", result[1].round(4))
    for key, value in result[4].items():
        st.write(f'Critical Value ({key}): {value.round(4)}')
 
    '''
        Non-Stationary Data: The ADF statistic -3.035 is less than all critical values at 5 % (-2.875), 
        meaning the null hypothesis (cannot be) rejected. The p-value (0.031) is significantly less than 0.05, 
        indicating a high probability that the data has a unit root. Hence, the data is non-stationary, 
        meaning it exhibits trends or seasonality and does not have constant mean (constant??) and variance over time.
    '''

def SARIMA_model():
    df=open_file()
    df.set_index(df.iloc[:,0], inplace=True)
    
    progress_text = "Tunggu sedang proses optimasi parameter, sekitar 4 menit..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(4)

    start_time = time.time()
    model = auto_arima(df.iloc[:,1], seasonal=True, m=12, trace=True, error_action='ignore', suppress_warnings=True)
    model.fit(df.iloc[:,1])
    end_time = time.time()
    time_lapsed =np.mean(end_time - start_time)
    st.write(f'waktu perhitungan optimasi parameter SARIMA : {time_lapsed.round(3)} detik')
    
    st.write("Optimal parameter : ")
    st.write(model)
    st.write(f'parameter order optimal, p,d,q : {model.order}')
    st.write(f'parameter seasonal order optimal P,D,Q,m : {model.seasonal_order}')
    #order=model.order
    #seasonal_order=model.seasonal_order
    #return (model.order, model.seasonal_order)
    #df.set_index(df.iloc[:,0], inplace=True)
    st.write("Output Optimized model SARIMA : ")
    optimized_model = SARIMAX(
        df.iloc[:,1],  
        order=model.order[:3], #(5, 1, 1),              # Non-seasonal parameters
        seasonal_order=model.seasonal_order[:4], #(1, 0, 1, 12),  # Seasonal parameters
        enforce_stationarity=True, # False if p>0.05,
        enforce_invertibility=False
    )
 
    optimized_sarima_fit = optimized_model.fit(disp=False)
    st.write(optimized_sarima_fit.summary())
    
#    optimized_sarima_fit.plot_diagnostics(figsize=(12, 8))
#    st.pyplot(fig)
#    return optimized_sarima_fit
    
    train = df.iloc[:-24] 
    test = df.iloc[-24:]

    forecast_test_optimized = optimized_sarima_fit.forecast(steps=24)
    forecast_test_optimized.index = test.index
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 4))
    ax1.plot(train.iloc[:,1],label="Train Data", color="black")
    ax2.plot(test.iloc[:,1], label="Test Data", color="blue")
    ax3.plot(forecast_test_optimized, label="Forecast", color="red" )
    st.pyplot(fig)

#def predicted_values():
    predicted_values = forecast_test_optimized.values 
    actual_values = test.iloc[:,1] #values #.flatten()

    prediction_df = pd.DataFrame({
        'actual': actual_values.round(4),
        'predicted': predicted_values.round(4),
        'deviation': predicted_values-actual_values.round(4),
        'deviation^2': (predicted_values-actual_values.round(4))**2
    })
    st.write(prediction_df)

    # 7. Evaluation Model
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_values - actual_values))
    st.write("MAE:", mae.round(4))

    # Root Mean Squared Error (RMSE)
    mse = np.mean((predicted_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse.round(4))
    st.write("MSE:", mse.round(4))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((predicted_values - actual_values) / actual_values)) * 100
    st.write("MAPE:", mape.round(4))

    # 8. Forecast for the next 3 month
    forecast_steps = 3
    forecast, stderr, conf_int = optimized_sarima_fit.forecast(steps=forecast_steps)

    # Convert forecast to a pandas Series for easier plotting
    forecast_series = pd.Series(forecast, index=pd.date_range('2025', periods=forecast_steps, freq='ME'))
    st.write
    

if model_analisis == "Introduksi":
    intro()
elif model_analisis == "Upload File":
    open_file()
elif model_analisis == "Diskripsi":
    descriptive()
elif model_analisis == "Plot historical inflation":
    historical()
elif model_analisis == "Check stationarity and autocorrelation":
    check_stationarity()
else:
    SARIMA_model()
