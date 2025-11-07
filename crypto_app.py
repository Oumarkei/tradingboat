import streamlit as st
import pandas as pd
import numpy as np
import joblib
import ccxt # Pour récupérer les données de Binance
import plotly.graph_objects as go 
import os
from dotenv import load_dotenv
from feature_eng import feature_engineering # Ta fonction originale
from datetime import datetime, timedelta
import time # <-- 1. IMPORT NÉCESSAIRE

import warnings
warnings.filterwarnings("ignore")


# Let's load env variables
load_dotenv()


# Configuration de la page Streamlit
st.set_page_config(layout="wide")
st.title("🤖 TradingBoat : Strategic ENA/USDT Altcoin Advisor 🧠 - HLC Prices forecating")

# Chargement des modèles et sceler en cache
@st.cache_resource
def load_models():
    scaler_path = os.getenv("SCALER_PATH", "models/mm_scaler.pkl")
    linear_model_path = os.getenv("LINEAR_MODEL_PATH", "models/linear_model.pkl")
    rfr_model_path = os.getenv("RFR_MODEL_PATH", "models/randomforest_model.pkl")
    
    try :
        scaler = joblib.load(scaler_path)
        linear_model = joblib.load(linear_model_path)
        rfr_model = joblib.load(rfr_model_path)
        return scaler, linear_model, rfr_model
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles ou du scaler: {e}")
        st.stop()

scaler, linear_model, rfr_model = load_models()

# --- 2. FONCTIONS SÉPARÉES ---

# FONCTION 1 : La grosse analyse (cache 1 heure)
@st.cache_data
def get_daily_analysis(date_key):
    """
    Fait la grosse analyse (graphiques, prédictions) une fois par heure.
    Ne retourne plus le prix live.
    """
    try:
        okx = ccxt.okx()
        ohlcv = okx.fetch_ohlcv("ENA/USDT", timeframe='1d', limit=500)
        
        data = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        data["date"] = pd.to_datetime(data["date"], unit="ms")
        data = data.set_index("date")
        
        # current_price est maintenant géré par get_live_price()
        
        eng_data = feature_engineering(data)
        eng_data = eng_data.dropna()
        
        today = pd.Timestamp("today").normalize()
        yesterday = today - pd.Timedelta(days=1)
        yesterday_str = yesterday.strftime('%Y-%m-%d')
        
        yesterday_close_price = eng_data.loc[yesterday_str]['close']
        feature_to_pred_today = eng_data.loc[yesterday_str].values.reshape(1, -1)
        
        feature_to_pred_today_scaled = scaler.transform(feature_to_pred_today)
        lr_pred = linear_model.predict(feature_to_pred_today_scaled)[0]
        rfr_pred = rfr_model.predict(feature_to_pred_today_scaled)[0]
        final_pred = (lr_pred + rfr_pred) / 2
        
        # On retourne les données pour les graphiques, le prix de J-1, et la prédiction
        return eng_data.reset_index(), yesterday_close_price, final_pred
    
    except Exception as e:
        st.error(f"Erreur lors de l'analyse journalière: {e}")
        st.exception(e) 
        st.stop()

# FONCTION 2 : Le prix live (cache 1 minute)
@st.cache_data(ttl=600)
def get_live_price():
    """
    Récupère juste le dernier prix tradé. Mis en cache 1 minute.
    """
    try:
        okx = ccxt.okx()
        ticker = okx.fetch_ticker("ENA/USDT")
        return ticker['last']
    except Exception as e:
        st.warning(f"Impossible de rafraîchir le prix live: {e}")
        return None


# --- 3. AFFICHAGE DU DASHBOARD ---

# Template de layout pour les graphiques Plotly
plotly_template = "plotly_white"
chart_height = 350

# Let's use UTC time for caching of our daily analysis (closing price are computed at UTC 00:00)
today_utc_str = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")

# On appelle la grosse fonction UNE SEULE FOIS au chargement
with st.spinner("Chargement de l'analyse journalière et des indicateurs..."):
    chart_data, yesterday_close_price, prediction = get_daily_analysis(date_key=today_utc_str)

# On récupère les prédictions
pred_high, pred_low, pred_close = prediction[0], prediction[1], prediction[2]

# --- RANGÉE 1 (Graphiques statiques, inchangés) ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Prix & Moyennes Mobiles (SMA)")
    fig_sma = go.Figure()
    fig_sma.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['close'], name='Prix de Clôture', line=dict(color='blue')))
    fig_sma.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['SMA7'], name='SMA7', line=dict(color='orange', dash='dash')))
    fig_sma.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['SMA14'], name='SMA14', line=dict(color='green', dash='dot')))
    fig_sma.update_layout(height=chart_height, template=plotly_template, xaxis_title='Date', yaxis_title='Prix', margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_sma, use_container_width=True)
    
with col2:
    st.subheader("Indicateur MACD")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['MACD'], name='MACD', line=dict(color='orange')))
    fig_macd.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['Signal_Line'], name='Signal', line=dict(color='purple', dash='dash')))
    colors = ['green' if val > 0 else 'red' for val in chart_data['Histogram']]
    fig_macd.add_trace(go.Bar(x=chart_data['date'], y=chart_data['Histogram'], name='Histogramme', marker_color=colors))
    fig_macd.update_layout(height=chart_height, template=plotly_template, xaxis_title='Date', yaxis_title='MACD', margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_macd, use_container_width=True)
    
# --- RANGÉE 2 (Graphique statique, inchangé) ---
col3, col4 = st.columns(2)
with col3:
    st.subheader("Bandes de Bollinger (BB)")
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['BB_Upper'], name='Bande Supérieure', line=dict(color='gray', dash='dash')))
    fig_bb.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['BB_Lower'], name='Bande Inférieure', line=dict(color='gray', dash='dash'),
                                fill='tonexty', fillcolor='rgba(128,128,128,0.2)'))
    fig_bb.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['close'], name='Prix de Clôture', line=dict(color='blue')))
    fig_bb.add_trace(go.Scatter(x=chart_data['date'], y=chart_data['BB_Midle'], name='Bande Milieu', line=dict(color='orange', dash='dot')))
    fig_bb.update_layout(height=chart_height, template=plotly_template, xaxis_title='Date', yaxis_title='Prix', margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_bb, use_container_width=True)

# --- QUADRANT 4 (Partiellement dynamique) ---
with col4:
    st.subheader(f"Espace de Prédiction pour Aujourd'hui ({datetime.today().strftime('%Y-%m-%d')})")
    
    ref_col1, ref_col2 = st.columns(2)
    with ref_col1:
        # Métrique J-1 (Statique, ne change pas)
        st.metric(label=f"Clôture d'Hier ({(datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')})", value=f"{yesterday_close_price:.4f} $")
    with ref_col2:
        # On crée un "placeholder" vide pour le prix J
        price_placeholder = st.empty() 

    st.divider()
    
    # Prédictions (Statiques, ne changent pas)
    delta_low = f"{(pred_low - yesterday_close_price) / yesterday_close_price * 100:.2f}%"
    delta_close = f"{(pred_close - yesterday_close_price) / yesterday_close_price * 100:.2f}%"
    delta_high = f"{(pred_high - yesterday_close_price) / yesterday_close_price * 100:.2f}%"

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric(label="Prédiction Basse 🔻", value=f"{pred_low:.4f} $", delta=delta_low)
    mcol2.metric(label="Prédiction Clôture ➡️", value=f"{pred_close:.4f} $", delta=delta_close)
    mcol3.metric(label="Prédiction Haute 🔺", value=f"{pred_high:.4f} $", delta=delta_high)
    
    
st.divider()
st.title("Recommendations Stratégiques de Trading 📈 avec Effet de Levier (Margin Trading)")
# Long position
long_entry_price = pred_low*1.0025
long_take_profit = long_entry_price*1.04
long_stop_loss = long_entry_price*(1-0.04)
# Short position
short_entry_price = pred_high*(1-0.005)
short_take_profit = short_entry_price*(1-0.04)
short_stop_loss = short_entry_price*1.04

st.subheader(f"- 📈 Poistion d'Entrée Longue : Entrez à prix <=  { long_entry_price:.4f} ------- TP = {long_take_profit:.4f} ------- SL = {long_stop_loss:.4f} (TP/SL à ajuster selon indicateurs techniques)")
st.subheader(f"- 📉 Poistion d'Entrée Short : Entrez à prix >=  { short_entry_price:.4f} ------- TP = {short_take_profit:.4f} ------- SL = {short_stop_loss:.4f} (TP/SL à ajuster selon indicateurs techniques)")

st.warning("⚠️ Le trading avec effet de levier comporte des risques élevés. Assurez-vous de bien comprendre ces risques avant de trader. Cette application ne constitue pas un conseil financier, les recommendations ne constituent qu'un outil d'aide à la décision et peuvent ne pas être valables en période de forte volatilité. Gérez vos propres risques ainsi que les effets de levier avec prudence.")
# --- 4. BOUCLE DE RAFRAÎCHISSEMENT ---
# (S'exécute à la fin du script)

while True:
    live_price = get_live_price()
    if live_price:
        # On remplit le placeholder avec la nouvelle valeur
        price_placeholder.metric(
            label=f"Prix Actuel (Dernier prix à {datetime.now().astimezone().time().strftime('%H:%M')})",
            value=f"{live_price:.4f} $"
)
    
    # On attend 60 secondes avant de redemander le prix
    time.sleep(60)