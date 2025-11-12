import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
from io import BytesIO
from fpdf import FPDF
import os.path

# --- Konfigurace a názvy souborů ---
FILE_T = "mly-0-20000-0-11723-T.csv"
FILE_F = "mly-0-20000-0-11723-F.csv"
FILE_SRA = "mly-0-20000-0-11723-SRA.csv"
FONT_FILE = "DejaVuSans.ttf" 
FONT_BOLD_FILE = "DejaVuSans-Bold.ttf" 

# Ignorování varování
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# --- Funkce pro zpracování dat ---

@st.cache_data
def nacti_a_filtruj_data_z_cesty(filepath, time_func, md_func, nova_value_col):
    """Načte CSV z cesty (filename) a vyfiltruje potřebné řádky."""
    try:
        df = pd.read_csv(
            filepath, 
            usecols=['YEAR', 'MONTH', 'TIMEFUNCTION', 'MDFUNCTION', 'VALUE']
        )
        df_filtrovany = df[
            (df['TIMEFUNCTION'] == time_func) & 
            (df['MDFUNCTION'] == md_func)
        ].copy()
        df_final = df_filtrovany[['YEAR', 'MONTH', 'VALUE']]
        df_final = df_final.rename(columns={'VALUE': nova_value_col})
        df_final[nova_value_col] = pd.to_numeric(df_final[nova_value_col], errors='coerce')
        return df_final
    except FileNotFoundError:
        st.error(f"Chyba: Soubor nenalezen: `{filepath}`. Ujisti se, že je ve stejném repozitáři jako `app.py`.")
        return None
    except Exception as e:
        st.error(f"Chyba při zpracování souboru `{filepath}`: {e}")
        return None

@st.cache_data
def zpracuj_data_z_githubu():
    """Hlavní funkce pro zpracování dat a trénink modelu."""
    with st.spinner("Načítám a zpracovávám data z GitHub repozitáře..."):
        df_temp = nacti_a_filtruj_data_z_cesty(FILE_T, 'AVG', 'AVG', 't_avg')
        df_wind = nacti_a_filtruj_data_z_cesty(FILE_F, 'AVG', 'AVG', 'wspd_avg')
        df_precip = nacti_a_filtruj_data_z_cesty(FILE_SRA, '07:00', 'SUM', 'prcp_sum')

        if df_temp is None or df_wind is None or df_precip is None:
            return None, None, None, None

        df_monthly = pd.merge(df_temp, df_wind, on=['YEAR', 'MONTH'], how='outer')
        df_monthly = pd.merge(df_monthly, df_precip, on=['YEAR', 'MONTH'], how='outer')

        monthly_counts = df_monthly.dropna().groupby('YEAR').size().reset_index(name='month_count')
        complete_years = monthly_counts[monthly_counts['month_count'] == 12]['YEAR']
        df_monthly_complete = df_monthly[df_monthly['YEAR'].isin(complete_years)]

        data_yearly = df_monthly_complete.groupby('YEAR').agg(
            tavg=('t_avg', 'mean'),
            wspd=('wspd_avg', 'mean'),
            prcp=('prcp_sum', 'sum')
        ).reset_index().dropna()

        if data_yearly.empty:
            st.error("Po filtraci na kompletní roky nezbyla žádná data.")
            return None, None, None, None

        variables = ['tavg', 'wspd', 'prcp']
        models = {}
        results = {}
        X = data_yearly['YEAR'].values.reshape(-1, 1)

        for var in variables:
            y = data_yearly[var].values
            model = LinearRegression()
            model.fit(X, y)
            models[var] = model
            results[var] = {'slope': model.coef_[0], 'intercept': model.intercept_}
            data_yearly[f'{var}_trend'] = model.predict(X)
            
        st.success("Data úspěšně načtena a modely natrénovány.")
        return data_yearly, results, models, df_monthly

# --- Funkce pro generování Matplotlib grafu ---

def create_plot_for_pdf(var, info, data_yearly, df_predictions, results):
    """Vytvoří Matplotlib graf a vrátí ho jako buffer v paměti."""
    fig, ax = plt.subplots(figsize=(10, 6)) 
    
    ax.scatter(data_yearly['YEAR'], data_yearly[var], label=f'Skutečná roční data ({var})', alpha=0.7, s=10) 
    ax.plot(data_yearly['YEAR'], data_yearly[f'{var}_trend'], color='red', linestyle='--', label=f'Lineární trend ({results[var]["slope"]:.4f} {info["unit"]}/rok)')
    
    last_year_data = data_yearly['YEAR'].max()
    last_val_data = data_yearly.loc[data_yearly['YEAR'] == last_year_data, f'{var}_trend'].values[0]
    first_pred_year = df_predictions.index.min()
    first_pred_val = df_predictions.loc[first_pred_year, f'pred_{var}']
    
    ax.plot([last_year_data, first_pred_year], [last_val_data, first_pred_val], color='red', linestyle=':', label='Extrapolace')
    ax.plot(df_predictions.index, df_predictions[f'pred_{var}'], color='red', marker='o', linestyle=':', markersize=5)
    
    ax.set_title(f'Historický vývoj a lineární extrapolace - {info["label"]} (Brno)')
    ax.set_xlabel('Rok')
    ax.set_ylabel(f'{info["label"]} ({info["unit"]})')
    ax.legend()
    ax.grid(True)
    
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig) 
    img_buffer.seek(0)
    return img_buffer

# --- Funkce pro generování PDF ---

def generate_pdf_report(data_yearly, results, models, df_predictions, variables_to_plot):
    """Sestaví kompletní PDF report."""
    
    if not os.path.isfile(FONT_FILE) or not os.path.isfile(FONT_BOLD_FILE):
        st.error(f"Kritická chyba PDF: Chybí soubor `{FONT_FILE}` nebo `{FONT_BOLD_FILE}`! Nahrajte je prosím do stejné složky.")
        return None

    try:
        pdf = FPDF(orientation='P', unit='mm', format='A4')
        
        pdf.add_font('DejaVu', '', FONT_FILE, uni=True)
        pdf.add_font('DejaVu', 'B', FONT_BOLD_FILE, uni=True)
        
        # --- Stránka 1: Úvod a Metodika ---
        pdf.add_page()
        effective_width = pdf.w - pdf.l_margin - pdf.r_margin 
        
        pdf.set_font('DejaVu', 'B', 16)
        pdf.multi_cell(effective_width, 10, 'Analýza a predikce klimatu: Brno (stanice 11723)', 0, 'C', ln=1)
        pdf.ln(10) # Prázdný řádek

        # Metodika
        pdf.set_font('DejaVu', 'B', 12)
        pdf.multi_cell(effective_width, 10, '1. Metodika zpracování', 0, 'L', ln=1)
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(effective_width, 5, 
            "Data byla načtena z poskytnutých CSV souborů (T, F, SRA). Pro každou veličinu (teplota, vítr, srážky) "
            "byla vyfiltrována relevantní měsíční data (průměrná teplota, průměrná rychlost větru, suma srážek). "
            "Tato data byla následně agregována na roční bázi (průměry pro T a F, suma pro SRA). Zahrnuty byly "
            "pouze kompletní roky s 12 měsíčními záznamy, aby se předešlo zkreslení.\n"
            "Pro kvantifikaci trendu byla použita metoda lineární regrese, kde nezávislou proměnnou byl rok. "
            "Tento model byl následně použit pro extrapolaci scénářů do budoucnosti.",
            0, 'L', ln=1
        )
        pdf.ln(5)

        # Interpretace a Omezení
        pdf.set_font('DejaVu', 'B', 12)
        pdf.multi_cell(effective_width, 10, '2. Interpretace a Omezení (Kritické)', 0, 'L', ln=1)
        pdf.set_font('DejaVu', 'B', 10)
        pdf.multi_cell(effective_width, 5, 
            "Je absolutně klíčové chápat, že tento model NENÍ reálnou klimatickou predikcí, ale pouhou lineární extrapolací.",
            0, 'L', ln=1
        )
        pdf.set_font('DejaVu', '', 10)
        pdf.multi_cell(effective_width, 5,
            "Hlavní omezení jsou:\n"
            " - Lineární model: Klima je komplexní, nelineární systém. Předpoklad, že trend z posledních 60 let bude lineárně pokračovat dalších 1000 let, je statisticky platný, ale věcně téměř jistě nesprávný.\n"
            " - Fyzikální ignorance: Model neobsahuje žádnou fyziku klimatu (vliv CO2, oceánské proudy, body zvratu). Je to čistě statistické 'protahování čáry'.\n"
            " - Horizont extrapolace: Zatímco predikce na 10 let je nejistý odhad, predikce na 100 let je spíše cvičení a predikce na 1000 let je fikce. Slouží k demonstraci absurdit dlouhodobé lineární extrapolace.\n"
            " - Lokální vlivy: Data z jedné stanice mohou být ovlivněna např. 'městským tepelným ostrovem', který zkresluje globální klimatický signál.\n\n"
            "Závěr: Výsledky (zejména na 100 a 1000 let) nelze brát jako předpověď, ale jako ukázku toho, co by se stalo, kdyby se svět řídil jen jednoduchým pravítkem.",
            0, 'L', ln=1
        )
        
        # --- Stránka 2: Výsledky (Tabulky) ---
        pdf.add_page()
        effective_width = pdf.w - pdf.l_margin - pdf.r_margin
        
        pdf.set_font('DejaVu', 'B', 12)
        pdf.multi_cell(effective_width, 10, '3. Kvantifikované výsledky', 0, 'L', ln=1)
        pdf.ln(5)

        # Tabulka 1: Sklony přímek
        pdf.set_font('DejaVu', 'B', 11)
        pdf.multi_cell(effective_width, 10, 'Vypočtené trendy (sklony regresní přímky)', 0, 'L', ln=1)
        pdf.set_font('DejaVu', '', 10)
        
        pdf.cell(60, 7, 'Veličina', 1, 0)
        pdf.cell(60, 7, 'Trend (jednotka/rok)', 1, 1)
        
        pdf.cell(60, 7, 'Průměrná teplota', 1, 0)
        pdf.cell(60, 7, f"{results['tavg']['slope']:.4f} °C / rok", 1, 1)
        
        pdf.cell(60, 7, 'Průměrný vítr', 1, 0)
        pdf.cell(60, 7, f"{results['wspd']['slope']:.4f} m/s / rok", 1, 1)
        
        pdf.cell(60, 7, 'Roční srážky', 1, 0)
        pdf.cell(60, 7, f"{results['prcp']['slope']:.4f} mm / rok", 1, 1)
        pdf.ln(10)

        # Tabulka 2: Predikce
        pdf.set_font('DejaVu', 'B', 11)
        pdf.multi_cell(effective_width, 10, 'Extrapolované scénáře (zaokrouhleno)', 0, 'L', ln=1)
        
        pdf.set_font('DejaVu', 'B', 10)
        col_width = 45
        pdf.cell(col_width, 7, 'Rok', 1, 0, 'C')
        pdf.cell(col_width, 7, 'Teplota (°C)', 1, 0, 'C')
        pdf.cell(col_width, 7, 'Vítr (m/s)', 1, 0, 'C')
        pdf.cell(col_width, 7, 'Srážky (mm)', 1, 1, 'C')

        pdf.set_font('DejaVu', '', 10)
        for year, row in df_predictions.iterrows():
            pdf.cell(col_width, 7, str(year), 1, 0, 'C')
            pdf.cell(col_width, 7, f"{row['pred_tavg']:.1f}", 1, 0, 'C')
            pdf.cell(col_width, 7, f"{row['pred_wspd']:.1f}", 1, 0, 'C')
            pdf.cell(col_width, 7, f"{row['pred_prcp']:.0f}", 1, 1, 'C')
        
        # --- Stránky 3, 4, 5: Grafy ---
        for var, info in variables_to_plot.items():
            pdf.add_page()
            effective_width = pdf.w - pdf.l_margin - pdf.r_margin
            
            pdf.set_font('DejaVu', 'B', 12)
            pdf.multi_cell(effective_width, 10, f"4. Graf: {info['label']}", 0, 'L', ln=1)
            pdf.ln(5)
            
            img_buffer = create_plot_for_pdf(var, info, data_yearly, df_predictions, results)
            pdf.image(img_buffer, x=10, y=None, w=190)
            img_buffer.close()

        # Vrácení finálního PDF jako 'bytes'
        return bytes(pdf.output(dest='S'))

    except Exception as e:
        st.error(f"Došlo k chybě při generování PDF: {e}")
        return None

# --- Hlavní Rozhraní Aplikace Streamlit (s vylepšeným vzhledem) ---

st.set_page_config(layout="wide", page_title="Prediktor Klimatu Brno", initial_sidebar_state="collapsed")
st.title("☀️ Klimatická Analýza a Lineární Extrapolace - Brno")
st.caption("Tento nástroj provádí lineární regresi na historických datech a extrapoluje trendy do budoucnosti. Datová stanice: 11723.")

# Zpracování dat (volá se automaticky při startu)
data_yearly, results, models, df_monthly = zpracuj_data_z_githubu()

# Zobrazí se, jen když je vše v pořádku
if data_yearly is not None:
    
    # --- POSTROANNÍ PANEL / ZDROJ DAT ---
    st.sidebar.header("📊 Vypočtené Trendy")
    st.sidebar.metric(
        label="Růst Průměrné Teploty", 
        value=f"{results['tavg']['slope']:.4f} °C/rok", 
        delta="Zateplování" if results['tavg']['slope'] > 0 else "Ochlazování"
    )
    st.sidebar.metric(
        label="Změna Rychlosti Větru", 
        value=f"{results['wspd']['slope']:.4f} m/s/rok", 
        delta="Posilování" if results['wspd']['slope'] > 0 else "Slábnutí"
    )
    st.sidebar.metric(
        label="Změna Ročních Srážek", 
        value=f"{results['prcp']['slope']:.4f} mm/rok", 
        delta="Více srážek" if results['prcp']['slope'] > 0 else "Méně srážek"
    )
    st.sidebar.divider()
    st.sidebar.info(f"Analyzovaná data pokrývají roky {data_yearly['YEAR'].min()} až {data_yearly['YEAR'].max()}.")
    
    # --- HLAVNÍ STRÁNKA ---
    
    # 1. Nastavení Horizontů a Predikce
    st.header("🔮 Scénáře Lineární Extrapolace")
    
    col1, col2, col3 = st.columns(3)
    current_year = datetime.now().year

    with col1:
        h1 = st.slider("Horizont 1 (za X let)", 1, 50, 10)
    with col2:
        h2 = st.slider("Horizont 2 (za X let)", 51, 500, 100)
    with col3:
        h3 = st.slider("Horizont 3 (za X let)", 501, 2000, 1000)
    
    horizons_years = [current_year + h1, current_year + h2, current_year + h3]
    
    predictions = {}
    for var, model in models.items():
        future_years = np.array(horizons_years).reshape(-1, 1)
        future_predictions = model.predict(future_years)
        predictions[f'pred_{var}'] = future_predictions

    df_predictions = pd.DataFrame(predictions, index=horizons_years)
    df_predictions.index.name = 'Year'
    df_predictions_rounded = df_predictions.round(2)

    df_display = df_predictions_rounded.copy()
    df_display.index.name = "Rok Extrapolace"
    df_display = df_display.rename(
        columns={
            "pred_tavg": "Predikce teploty [°C]",
            "pred_wspd": "Predikce rychlost větru [m/s]",
            "pred_prcp": "Predikce množství srážek [mm]"
        }
    )

    st.subheader("Extrapolované Hodnoty")
    st.dataframe(df_display, use_container_width=True)

    with st.expander("🚨 Kritické Upozornění k Interpretaci Výsledků"):
        st.error(
            "Predikce na **100 a 1000 let** jsou čistě **hypotetická lineární extrapolace** "
            "a NEMAJÍ reálný vědecký smysl. Slouží k demonstraci toho, jak rychle by se "
            "veličiny změnily, kdyby aktuální, lineární trend pokračoval beze změny."
        )

    st.divider()

    # 2. Definice proměnných pro grafy a Zobrazení grafů v záložkách
    variables_to_plot = {
        'tavg': {'unit': '°C', 'label': 'Průměrná teplota'},
        'wspd': {'unit': 'm/s', 'label': 'Průměrná rychlost větru'},
        'prcp': {'unit': 'mm', 'label': 'Celkové roční srážky'}
    }

    st.header("📈 Vizuální Analýza Trendů")
    st.info("Grafy zobrazují historická data (body), lineární trend (čerchovaná) a extrapolaci (tečkovaná).")

    tab_t, tab_w, tab_p = st.tabs(["🌡️ Teplota", "🌬️ Vítr", "🌧️ Srážky"])

    # Graf Teplota
    with tab_t:
        with st.spinner("Generuji graf teploty..."):
            fig_t = create_plot_for_pdf('tavg', variables_to_plot['tavg'], data_yearly, df_predictions, results)
            st.image(fig_t, caption="Vývoj a extrapolace průměrné roční teploty", use_column_width=True)

    # Graf Vítr
    with tab_w:
        with st.spinner("Generuji graf větru..."):
            fig_w = create_plot_for_pdf('wspd', variables_to_plot['wspd'], data_yearly, df_predictions, results)
            st.image(fig_w, caption="Vývoj a extrapolace průměrné roční rychlosti větru", use_column_width=True)

    # Graf Srážky
    with tab_p:
        with st.spinner("Generuji graf srážek..."):
            fig_p = create_plot_for_pdf('prcp', variables_to_plot['prcp'], data_yearly, df_predictions, results)
            st.image(fig_p, caption="Vývoj a extrapolace celkových ročních srážek", use_column_width=True)
            
    st.divider()
    
    # 3. Generování PDF
    st.header("📄 Zpráva ve Formátu PDF")
    
    with st.spinner("Připravuji data pro PDF..."):
        pdf_data = generate_pdf_report(data_yearly, results, models, df_predictions, variables_to_plot)

    if pdf_data:
        st.download_button(
            label="Stáhnout kompletní zprávu jako PDF (včetně grafů)",
            data=pdf_data, 
            file_name=f"report_klima_brno_{current_year}.pdf",
            mime="application/pdf"
        )
        st.success("PDF připraveno ke stažení!")
    else:
        st.error("Nepodařilo se vygenerovat PDF. Zkontrolujte, zda máte ve správné složce fonty DejavuSans.")

else:
    # Zobrazí se, pokud selže načítání dat
    st.info("Čekání na data... Pokud se nic neděje, zkontrolujte chybové hlášky výše.")
