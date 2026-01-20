import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import poisson
import os

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Şampiyonlar Ligi Tahmincisi", layout="centered")

st.title("🏆 Şampiyonlar Ligi 25/26 Skor Tahmin Made By Mustafa Demirdaş")
st.markdown("💛❤️ BU ALEMDE EN BÜYÜK CİMBOM! 💛❤️")

# --- VERİ YÜKLEME VE ÖN İŞLEME ---
@st.cache_data # Performans için veriyi önbelleğe alıyoruz
def load_data():
    # Klasördeki CSV dosyasını otomatik bulmaya çalış
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not files:
        st.error("HATA: Klasörde .csv dosyası bulunamadı! Lütfen Kaggle veri setini proje klasörüne ekleyin.")
        return None
    
    filename = files[0]
    df = pd.read_csv(filename)
    
    # Sütun temizliği
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['score'])
    
    # Skor ayrıştırma
    df['score'] = df['score'].astype(str).str.replace('–', '-').str.replace(' ', '')
    try:
        df[['home_goals', 'away_goals']] = df['score'].str.split('-', expand=True).astype(int)
    except:
        st.warning("Bazı skorlar ayrıştırılamadı, veri formatını kontrol edin.")
        return None
        
    return df

df = load_data()

# --- MODEL EĞİTİMİ ---
@st.cache_resource # Modeli her seferinde tekrar eğitmemek için
def train_model(df):
    if df is None: return None
    
    # Veriyi modele uygun hale getir
    goal_model_data = pd.concat([
        df[['home_team', 'away_team', 'home_goals']].assign(home=1).rename(
            columns={'home_team': 'team', 'away_team': 'opponent', 'home_goals': 'goals'}),
        df[['away_team', 'home_team', 'away_goals']].assign(home=0).rename(
            columns={'away_team': 'team', 'home_team': 'opponent', 'away_goals': 'goals'})
    ])
    
    # Poisson Modeli
    model = smf.glm(formula="goals ~ home + team + opponent", 
                    data=goal_model_data, 
                    family=sm.families.Poisson()).fit()
    return model

if df is not None:
    model = train_model(df)
    
    # Takım Listesi (Alfabetik)
    teams = sorted(df['home_team'].unique())

    # --- KULLANICI ARAYÜZÜ ---
    col1, col2 = st.columns(2)
    
    with col1:
        Home_team = st.selectbox("Ev Sahibi Takım", teams, index=0)
    
    with col2:
        # Deplasman takımı listesinde ev sahibini otomatik seçtirmemek için basit mantık
        away_options = [t for t in teams if t != Home_team]
        Away_team = st.selectbox("Deplasman Takımı", away_options, index=0)

    # TAHMİN BUTONU
    if st.button("MAÇI TAHMİN ET", type="primary"):
        
        # --- HESAPLAMA MOTORU ---
        # Beklenen Goller (xG)
        home_xg = model.predict(pd.DataFrame(data={'team': [Home_team], 'opponent': [Away_team], 'home': [1]})).values[0]
        away_xg = model.predict(pd.DataFrame(data={'team': [Away_team], 'opponent': [Home_team], 'home': [0]})).values[0]
        
        # Olasılık Matrisi
        max_goals = 6
        home_probs = [poisson.pmf(i, home_xg) for i in range(max_goals)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(max_goals)]
        prob_matrix = np.outer(home_probs, away_probs)
        
        # En olası skor
        most_likely = np.unravel_index(np.argmax(prob_matrix), prob_matrix.shape)
        
        # Kazanma İhtimalleri
        home_win_prob = np.sum(np.tril(prob_matrix, -1))
        draw_prob = np.sum(np.diag(prob_matrix))
        away_win_prob = np.sum(np.triu(prob_matrix, 1))

        # --- SONUÇLARI GÖSTERME ---
        st.divider()
        
        # Skor Tahmini Kartı
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            st.markdown(f"<h1 style='text-align: center; color: #d3d3d3;'>{Home_team} vs {Away_team}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #4CAF50;'>{most_likely[0]} - {most_likely[1]}</h2>", unsafe_allow_html=True)
            st.caption(f"En olası skor (xG: {home_xg:.2f} - {away_xg:.2f})")

        st.divider()

        # Olasılık Grafiği
        st.subheader("Kazanma İhtimalleri")
        chart_data = pd.DataFrame({
            "Sonuç": [f"{Home_team} Kazanır", "Beraberlik", f"{Away_team} Kazanır"],
            "Olasılık": [home_win_prob, draw_prob, away_win_prob]
        })
        
        st.bar_chart(chart_data.set_index("Sonuç"))
        
        # Detaylı İstatistikler
        with st.expander("Detaylı İstatistikleri Gör"):
            st.write(f"**{Home_team} Galibiyet:** %{home_win_prob*100:.2f}")
            st.write(f"**Beraberlik:** %{draw_prob*100:.2f}")
            st.write(f"**{Away_team} Galibiyet:** %{away_win_prob*100:.2f}")
            
else:
    st.info("Lütfen .csv dosyasını klasöre yükleyip sayfayı yenileyin.")