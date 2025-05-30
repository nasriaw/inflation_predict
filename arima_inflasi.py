# Import Libraries
import streamlit as st
import pingouin as pg
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from pandas_datareader import data as pdr
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
import statistics
import warnings
import time
from datetime import timedelta
warnings.filterwarnings('ignore')

features=[
    "Introduksi",
    "Diskripsi",
    "Plot historical inflation",
    "Check stationarity and autocorrelation",
    "SARIMAX model, Forecast, Evaluation model & Predict",
    ]
menu=pd.DataFrame(features)
#st.write(menu)
#[m,n] =menu.shape
#st.write(m,n)
#st.sidebar.image("logo_stiei.jpg", use_column_width=False)
st.sidebar.markdown('<h3 style="color: Black;"> Author: Nasri </h3>', unsafe_allow_html=True)
st.sidebar.markdown('<h5 style="color: Black;"> email: nasri@stieimlg.ac.id </h5>', unsafe_allow_html=True)
st.sidebar.markdown('<h1 style="color: Black;">Analisis Prediksi Inflasi Harga dengan model SARIMAX</h1>', unsafe_allow_html=True)

model_analisis = st.sidebar.radio('Baca ketentuan penggunaan dengan seksama, Pilih Analisis Prediksi Inflasi Harga:', menu)

def intro():
    st.write("## Selamat Datang di Dashboard Analisis Prediksi Inflasi Harga di Indonesia, Menggunakan Model SARIMAX.  👋  👋")
    st.write("##### author: m nasri aw, email: nasri@stieimlg.ac.id; lecturer at https://www.stieimlg.ac.id/; Des 2024.")
    st.write(f"##### - Pendekatan analisis: ")
    '''
    1. Aplikasi ini menggunakan bahasa python dengan library utama statsmodels dan streamlit.
    2. Menggunakan model SARIMAX dengan data series (musiman) sebagai data training.
    3. File hanya terdiri dari kolom pertama kolom waktu (bersifat series bisa dalam bentuk bulanan dan musiman), kolom ke dua data inflasi atau sejenis, misal data perjalanan, penenan, dll.
    4. Diperlukan data dalam bentuk file csv, sebagai contoh disini menggunakan data inflasi harga di Indonesia kurun 2006-2024 (sumber bps, 2025), untuk file csv dapat di download di https://github.com/nasriaw/inflation_indonesia_2006-2024/blob/main/inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv;
    5. Analisis data series prediksi inflasi ini meliputi:
       1. Diskripsi.
       2. Plot historical inflation.
       3. Check stationarity and autocorrelation.
       4. SARIMAX model, Forecast, Evaluation model & Predict.
       ###### 👈 Pilih Menu di sebelah.
    6. Link template, siapkan data dan upload ke https://predict-analysis-data-series.streamlit.app/ ; source code demo ini di link: https://nasriaw-aw-predict.streamlit.app/ ;  
    Selamat belajar semoga memudahkan untuk untuk memahami Analisis Prediksi Data Series.
    '''
    return intro

def descriptive():
    #df=open_file()
    df=pd.read_csv('inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv')
    df= df.dropna()
    df.iloc[:,0]=pd.to_datetime(df.iloc[:,0])
    st.write("### 1. Data Head dan Statistik Diskripsi ") #{df.iloc[:,1]}")
    st.write(f"#### dimensi data: {df.shape}")
    st.write("#### Data Head : ")
    st.write(df.head())
    st.write("#### Data Diskripsi Inflasi : ")
    st.write(df.iloc[:,1].describe())#(include='all').fillna("").astype("str"))
    
def historical():
    #df=open_file()
    df=pd.read_csv('inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv')
    df.set_index(df.iloc[:,0], inplace=True)
    st.write("### Inflation Data (2006-2024) Line Chart")
    st.line_chart(df.iloc[:,1], x_label="bulan", y_label="inflation, %") #scatter, bar, line, area, altair

def check_stationarity():
    # Check stationarity and autocorrelation
    # Plot ACF and PACF, given ARIMA parameters
    df=pd.read_csv('inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv')
    df.set_index(df.iloc[:,0], inplace=True)
    
    st.write("### Auto Correlation Function (ACF) Chart")
    fig, (ax1) = plt.subplots(1, 1, figsize=(8, 3))
    plot_acf(df.iloc[:,1], ax=ax1)
    st.pyplot(fig)
    st.write("### Partial Auto Correlation Function (PCF) Chart")
    #st.line_chart(df.iloc[:,1], x_label="periode", y_label="inflasi")
    
    fig1, (ax2) = plt.subplots(1, 1, figsize=(8, 3))
    plot_pacf(df.iloc[:,1], ax=ax2)
    st.pyplot(fig1)
    
    st.write("### Augmented Dickey-Fuller (ADF), P-value Result")
    result = adfuller(df.iloc[:,1])
    st.write("Augmented Dickey-Fuller (ADF) Statistic:", result[0].round(4))
    st.write("p-value:", result[1].round(4))
       
    for key, value in result[4].items():
        st.write(f'Critical Value ({key}): {value.round(4)}')
 
    #st.write(f'result[0] = {result[0]:.4f}')
    #st.write(f'result[4] = {result[4]["5%"]:.4f}')
    st.write('### Kesimpulan :')
    if result[0] < result[4]["5%"]:
        st.write(f'ADF result = {result[0]:.4f} < {result[4]["5%"]:.4f} (Critical value=5%) : Hipotesis stasioner ditolak, Pola data tidak stasioner, yang berarti menunjukkan tren atau musiman dan yang tidak memiliki rata-rata dan varians yang konstan dari waktu ke waktu.')
    else:
        st.write(f'ADF result = {result[0]:.4f} > {result[4]["5%"]:.4f} (Critical value=5%) : Hipotesis stasioner diterima, Pola data stasioner, yang berarti menunjukkan tren atau musiman dan yang memiliki rata-rata dan varians yang konstan dari waktu ke waktu.')
    
    if result[1].round(4)> 0.05:
        st.write(f'p-value = {result[1]:.4f} > 0.05 : Hipotesis stasioner diterima, Pola data stasioner, yang berarti menunjukkan tren atau musiman dan yang memiliki rata-rata dan varians yang konstan dari waktu ke waktu, parameter enforce_stationarity=False.')
    else:
        st.write(f'p-value = {result[1]:.4f} < 0.05 : Hipotesis stasioner ditolak, Pola data tidak stasioner, yang berarti menunjukkan tren atau musiman dan yang tidak memiliki rata-rata dan varians yang konstan dari waktu ke waktu, parameter enforce_stationarity=True.')


def SARIMAX_model():
    df=pd.read_csv('inflasi_harga_konsumen_nasional_bulanan_2006_2024.csv')
    df.set_index(df.iloc[:,0], inplace=True)
    start_time = time.time()
    with st.spinner("Tunggu proses Optimized Model SARIMAX", show_time=True):
        optimized_model = SARIMAX(
            df.iloc[:,1],  
            order=(5,1,1), #model.order[:3], Non-seasonal parameters
            seasonal_order=(1, 0, 1, 12), #model.seasonal_order[:4], Seasonal parameters
            enforce_stationarity=True, # False if p>0.05,
            enforce_invertibility=False
        )
        end_time = time.time()
        time.sleep=end_time
        time_lapsed =np.mean(end_time - start_time)
        st.success(f"Selesai !!, waktu optimasi parameter SARIMA : {str(timedelta(seconds=time_lapsed))} detik': ")
            
    optimized_sarima_fit = optimized_model.fit(disp=False)
    st.write("Telah dihitung untuk pola data inflasi, menggunakan  auto_arima, diperoleh paramater SARIMAX(p,d,q)(P,D,Q,m), yaitu parameter Non seasonal order p,d,q: (5,1,1), dan parameter seasonal_order P,D,Q,m:(1,0,1,12)")
    st.write("### SARIMAX RESULT")
    st.write(optimized_sarima_fit.summary())
    
    st.write('### Standardized Residuals')
    fig=optimized_sarima_fit.plot_diagnostics(figsize=(12, 8))
    st.pyplot(fig)
    '''
    Residual tampak acak dan berfluktuasi di sekitar nol, yang menunjukkan tidak ada pola atau tren yang terlihat. Histogram dengan Kepadatan Diperkirakan, residual terdistribusi secara normal, karena histogram selaras dengan kurva kepadatan normal. Plot Q-Q Normal, residual sebagian besar mengikuti garis diagonal merah, yang memvalidasi bahwa residual tersebut hampir terdistribusi secara normal. Korelogram (ACF), tidak ada lonjakan signifikan dalam fungsi autokorelasi (ACF), yang menunjukkan residual tidak berkorelasi.
    '''
    
    train = df.iloc[:-24] 
    test = df.iloc[-24:]
    train.set_index(train.iloc[:,0], inplace=True)
    test.set_index(test.iloc[:,0], inplace=True)
    
    forecast_test_optimized = optimized_sarima_fit.forecast(steps=24)
    forecast_test_optimized.index = test.index
    
    st.write("### Data Train Line Chart")
    st.line_chart(train.iloc[:,1], x_label="bulan, train", y_label="inflasi, %")
    st.write("### Test Line Chart")
    st.line_chart(test.iloc[:,1],x_label="bulan, testing", y_label="inflasi testing, %")
    st.write("### Forecast Line Chart")
    st.line_chart(forecast_test_optimized, x_label="bulan forecast", y_label="inflasi Forecast, %")
    
    predicted_values = forecast_test_optimized.values 
    actual_values = test.iloc[:,1] #values #.flatten()

    prediction_df = pd.DataFrame({
        'actual': actual_values.round(4),
        'predicted': predicted_values.round(4),
        'deviation': predicted_values-actual_values.round(4),
        'deviation^2': (predicted_values-actual_values.round(4))**2
    })
    st.write("### Actual - Predicted - Deviation")
    st.write(prediction_df)
    
    data = {
        "Actual Inflation": actual_values,
        "Predicted Inflation": predicted_values,}
    df = pd.DataFrame(data) #, index=dates)
    column_names=list(df)
    #column_names[0]
    #column_names[1]
    
    st.write("### Actual - Predicted Line Chart")
    st.line_chart(df, x_label="bulan", y_label="inflation, %") #.iloc[:,1], x_label=column_names[0], y_label=column_names[1])

    # Evaluation Model
    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(predicted_values - actual_values))
    st.write("### Evaluation Indicator")
    st.write("MAE:", mae.round(4))

    # Root Mean Squared Error (RMSE)
    mse = np.mean((predicted_values - actual_values) ** 2)
    rmse = np.sqrt(mse)
    st.write("RMSE:", rmse.round(4))
    st.write("MSE:", mse.round(4))

    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((actual_values-predicted_values) / actual_values))/len(actual_values) * 100
    st.write("MAPE:", mape.round(4))
    
    # Standard Deviation Predicted
    std=np.mean(statistics.stdev(predicted_values))
    st.write(f'Standard Deviation : {(std.round(4))}')

    # Forecast for the next 3 month
    st.write("### Forecast next 3 month")
    forecast_steps = 3
    forecast, stderr, conf_int = optimized_sarima_fit.forecast(steps=forecast_steps)

    # Convert forecast to a pandas Series for easier plotting
    forecast_series = pd.Series(forecast, index=pd.date_range('2025', periods=forecast_steps, freq='ME'))
    st.write(f'Inflation Predicted next 3 month: {forecast:3f} %') #, stderr: {stderr}, conf: {conf_int}')
    

if model_analisis == "Introduksi":
    intro()
elif model_analisis == "Diskripsi":
    descriptive()
elif model_analisis == "Plot historical inflation":
    historical()
elif model_analisis == "Check stationarity and autocorrelation":
    check_stationarity()
else:
    SARIMAX_model()
