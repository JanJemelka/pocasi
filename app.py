# --- Hlavní Rozhraní Aplikace Streamlit (UPRAVENÁ VERZE) ---

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
        h1 = st.slider("Horizont 1 (do roku)", 1, 50, 10)
    with col2:
        h2 = st.slider("Horizont 2 (do roku)", 51, 500, 100)
    with col3:
        h3 = st.slider("Horizont 3 (do roku)", 501, 2000, 1000)
    
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
