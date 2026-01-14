import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import streamlit as st
import io

# Função auxiliar para evitar erro se houver menos de 20 strikes
def safe_get(lista, index, default=0.00):
    try:
        return lista[index]
    except IndexError:
        return default

def calcGammaEx_numpy(S, K, vol, T, r, q, optType, OI):
    with np.errstate(divide='ignore', invalid='ignore'):
        sqrtT = np.sqrt(T)
        dp = (np.log(S / K) + (r - q + 0.5 * vol ** 2) * T) / (vol * sqrtT)
        gamma = np.exp(-q * T) * norm.pdf(dp) / (S * vol * sqrtT)
        gamma[np.isnan(gamma)] = 0
        return OI * 100 * S * S * 0.01 * gamma

def isThirdFriday(d):
    return d.weekday() == 4 and 15 <= d.day <= 21

def main():
    st.set_page_config(layout="wide")
    st.title("Análise de Gamma Exposure")

    # Configuração da Sidebar
    with st.sidebar:
        st.header("Configurações")
        uploaded_file = st.file_uploader("Envie o arquivo quotedata.csv", type="csv")
        bar_width1 = st.number_input("Largura das barras do Gráfico 1", min_value=0.01, max_value=15.0, step=0.01, value=0.3)
        bar_width2 = st.number_input("Largura das barras do Gráfico 2", min_value=0.01, max_value=15.0, step=0.01, value=0.2)
        levels_input = st.slider("Quantidade de níveis (Gráfico 3)", min_value=20, max_value=100, step=5, value=60)

    if uploaded_file:
        optionsFileData = uploaded_file.getvalue().decode("utf-8").splitlines()

        spotLine = optionsFileData[1]
        spotPrice = float(spotLine.split('Last:')[1].split(',')[0])
        fromStrike = 0.8 * spotPrice
        toStrike = 1.2 * spotPrice

        dateLine = optionsFileData[2]
        monthDay = dateLine.split('Date: ')[1].split(',')[0].split(' ')

        nomes_meses = {'janeiro': 'January', 'fevereiro': 'February', 'março': 'March',
                       'abril': 'April', 'maio': 'May', 'junho': 'June',
                       'julho': 'July', 'agosto': 'August', 'setembro': 'September',
                       'outubro': 'October', 'novembro': 'November', 'dezembro': 'December'}

        nome_mes_pt = monthDay[2]
        nome_mes_en = nomes_meses[nome_mes_pt.lower()]
        num_mes = datetime.strptime(nome_mes_en, '%B').month

        year = int(monthDay[4])
        day = int(monthDay[0])
        todayDate = datetime(year=year, month=num_mes, day=day)

        df = pd.read_csv(uploaded_file, sep=",", header=None, skiprows=4)
        df.columns = ['ExpirationDate','Calls','CallLastSale','CallNet','CallBid','CallAsk','CallVol',
                      'CallIV','CallDelta','CallGamma','CallOpenInt','StrikePrice','Puts','PutLastSale',
                      'PutNet','PutBid','PutAsk','PutVol','PutIV','PutDelta','PutGamma','PutOpenInt']

        df['ExpirationDate'] = pd.to_datetime(df['ExpirationDate'], format='%a %b %d %Y') + timedelta(hours=16)
        df = df.astype({
            'StrikePrice': float, 'CallIV': float, 'PutIV': float, 'CallGamma': float,
            'PutGamma': float, 'CallOpenInt': float, 'PutOpenInt': float
        })

        df['CallGEX'] = df['CallGamma'] * df['CallOpenInt'] * 100 * spotPrice * spotPrice * 0.01
        df['PutGEX'] = df['PutGamma'] * df['PutOpenInt'] * 100 * spotPrice * spotPrice * 0.01 * -1
        df['TotalGamma'] = (df.CallGEX + df.PutGEX) / 10**9
        
        dfAgg = df.groupby(['StrikePrice']).sum(numeric_only=True)
        strikes = dfAgg.index.values

        # --- CÁLCULOS PARA O TXT ---
        top_calls = dfAgg['CallGEX'].sort_values(ascending=False).index.tolist()
        top_puts = dfAgg['PutGEX'].sort_values(ascending=True).index.tolist()
        dfAgg['AbsTotalGamma'] = dfAgg['TotalGamma'].abs()
        top_total = dfAgg['AbsTotalGamma'].sort_values(ascending=False).index.tolist()

        nextExpiry = df['ExpirationDate'].min()
        df['IsThirdFriday'] = df['ExpirationDate'].apply(isThirdFriday)
        
        df_0dte = df[df['ExpirationDate'] == nextExpiry]
        df_month = df[df['IsThirdFriday']]

        if not df_0dte.empty:
            agg_0dte = df_0dte.groupby('StrikePrice').sum(numeric_only=True)
            max_cw_0dte = agg_0dte['CallGEX'].idxmax() if not agg_0dte.empty else 0
            min_pw_0dte = agg_0dte['PutGEX'].idxmin() if not agg_0dte.empty else 0
        else:
            max_cw_0dte, min_pw_0dte = 0, 0

        if not df_month.empty:
            agg_month = df_month.groupby('StrikePrice').sum(numeric_only=True)
            max_cw_month = agg_month['CallGEX'].idxmax() if not agg_month.empty else 0
            min_pw_month = agg_month['PutGEX'].idxmin() if not agg_month.empty else 0
        else:
            max_cw_month, min_pw_month = 0, 0

        max_cw_all = top_calls[0] if top_calls else 0
        min_pw_all = top_puts[0] if top_puts else 0

        # --- PLOTS ---
        st.subheader("Gráfico 1: Total Gamma")
        fig1 = go.Figure(go.Bar(x=strikes, y=dfAgg['TotalGamma'].to_numpy(), width=bar_width1,
            marker_color='#1a76ff', marker_line_color='white', marker_line_width=0.15, name='Gamma Exposure'))
        fig1.add_shape(type='line', x0=spotPrice, y0=min(dfAgg['TotalGamma']), x1=spotPrice, y1=max(dfAgg['TotalGamma']), line=dict(color='red', width=2, dash='dash'))
        fig1.update_layout(title=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% ATIVO Move",
            xaxis_title='Strike', yaxis_title='Spot Gamma Exposure ($ billions/1% move)',
            xaxis=dict(range=[fromStrike, toStrike]), yaxis=dict(tickformat='$,.2f'),
            plot_bgcolor='black', font=dict(family='Arial', size=12, color='black'), width=1000, height=600)
        st.plotly_chart(fig1, use_container_width=True)
        st.download_button("Baixar Gráfico 1 como HTML", fig1.to_html(full_html=False), file_name="grafico1_total_gamma.html")

        st.subheader("Gráfico 2: Call e Put Gamma")
        fig2 = go.Figure()
        fig2.add_bar(x=strikes, y=dfAgg['CallGEX'].to_numpy() / 10**9, width=bar_width2, name="Call Gamma", marker_color='#00cc96')
        fig2.add_bar(x=strikes, y=dfAgg['PutGEX'].to_numpy() / 10**9, width=bar_width2, name="Put Gamma", marker_color='#ef553b')
        fig2.add_shape(dict(type="line", x0=spotPrice, y0=0, x1=spotPrice, y1=max(dfAgg['CallGEX'].to_numpy() / 10**9), line=dict(color="black", width=2)))
        fig2.update_layout(title_text=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% ATIVO Move",
            title_font=dict(size=20, family="Arial Black"), xaxis_title="Strike",
            yaxis_title="Spot Gamma Exposure ($ billions/1% move)", xaxis=dict(range=[fromStrike, toStrike]),
            width=1000, height=600)
        st.plotly_chart(fig2, use_container_width=True)
        st.download_button("Baixar Gráfico 2 como HTML", fig2.to_html(full_html=False), file_name="grafico2_call_put.html")

        st.subheader("Gráfico 3: Gamma Exposure Profile")
        levels = np.linspace(fromStrike, toStrike, levels_input)
        df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]
        nextMonthlyExp = df.loc[df['IsThirdFriday']]['ExpirationDate'].min()

        totalGamma, totalGammaExNext, totalGammaExFri = [], [], []

        for level in levels:
            df['callGammaEx'] = calcGammaEx_numpy(level, df['StrikePrice'].values, df['CallIV'].values, df['daysTillExp'].values, 0, 0, "call", df['CallOpenInt'].values)
            df['putGammaEx'] = calcGammaEx_numpy(level, df['StrikePrice'].values, df['PutIV'].values, df['daysTillExp'].values, 0, 0, "put", df['PutOpenInt'].values)
            totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())
            totalGammaExNext.append(df.loc[df['ExpirationDate'] != nextExpiry]['callGammaEx'].sum() - df.loc[df['ExpirationDate'] != nextExpiry]['putGammaEx'].sum())
            totalGammaExFri.append(df.loc[df['ExpirationDate'] != nextMonthlyExp]['callGammaEx'].sum() - df.loc[df['ExpirationDate'] != nextMonthlyExp]['putGammaEx'].sum())

        totalGamma = np.array(totalGamma) / 10**9
        totalGammaExNext = np.array(totalGammaExNext) / 10**9
        totalGammaExFri = np.array(totalGammaExFri) / 10**9

        zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
        if len(zeroCrossIdx) > 0:
            negGamma = totalGamma[zeroCrossIdx]
            posGamma = totalGamma[zeroCrossIdx+1]
            negStrike = levels[zeroCrossIdx]
            posStrike = levels[zeroCrossIdx+1]
            zeroGamma = posStrike - ((posStrike - negStrike) * posGamma / (posGamma - negGamma))
            zeroGamma = zeroGamma[0]
        else:
            zeroGamma = 0.00

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=levels, y=totalGamma, mode='lines', name='All Expiries', line=dict(color='blue')))
        fig3.add_trace(go.Scatter(x=levels, y=totalGammaExNext, mode='lines', name='Ex-Next Expiry', line=dict(color='orange')))
        fig3.add_trace(go.Scatter(x=levels, y=totalGammaExFri, mode='lines', name='Ex-Next Monthly Expiry', line=dict(color='yellow')))
        fig3.update_layout(title=f"Gamma Exposure Profile, ATIVO, {todayDate.strftime('%d %b %Y')}",
                           xaxis_title='Index Price', yaxis_title='Gamma Exposure ($ billions/1% move)', width=1000, height=600)
        fig3.add_shape(type="line", x0=spotPrice, y0=min(totalGamma), x1=spotPrice, y1=max(totalGamma), line=dict(color="red", width=1.5))
        if zeroGamma > 0:
            fig3.add_shape(type="line", x0=zeroGamma, y0=min(totalGamma), x1=zeroGamma, y1=max(totalGamma), line=dict(color="green", width=1.5))
        st.plotly_chart(fig3, use_container_width=True)
        st.download_button("Baixar Gráfico 3 como HTML", fig3.to_html(full_html=False), file_name="grafico3_gamma_profile.html")

        # --- GERAÇÃO TXT E BOTÃO NA SIDEBAR ---
        cw_lines = ""
        for i in range(20):
            val = safe_get(top_calls, i)
            suffix = "_HW" if i == 0 else ""
            cw_lines += f"CW{i+1}{suffix} ({val:.2f});\n"

        pw_lines = ""
        for i in range(20):
            val = safe_get(top_puts, i)
            suffix = "_LP" if i == 0 else ""
            pw_lines += f"PW{i+1}{suffix} ({val:.2f});\n"
        
        point_lines = ""
        for i in range(20):
            val = safe_get(top_total, i)
            point_lines += f"Point_I{i+1} ({val:.2f});\n"

        vol_triggers = ""
        vol_triggers += f"Vol_TriggerA ({safe_get(top_total, 0):.2f});\n"
        vol_triggers += f"Vol_TriggerB ({safe_get(top_total, 1):.2f});\n"
        vol_triggers += f"Vol_TriggerC ({safe_get(top_total, 2):.2f});\n"
        vol_triggers += f"Vol_TriggerD ({safe_get(top_total, 3):.2f});\n"

        txt_output = f"""{cw_lines}{pw_lines}{point_lines}GamaFlip_A ({zeroGamma:.2f});
GamaFlip_B (0.00);
GamaFlip_C (0.00);
{vol_triggers}Max_CW_0DTE ({max_cw_0dte:.2f});
Min_PW_0DTE ({min_pw_0dte:.2f});
Max_CW_MONTH ({max_cw_month:.2f});
Min_PW_MONTH ({min_pw_month:.2f});
Max_CW_ALL ({max_cw_all:.2f});
Min_PW_ALL ({min_pw_all:.2f});
Vol_Attack ({safe_get(top_total, 0):.2f});
A_Zero ({zeroGamma:.2f});"""

        # ADICIONA O BOTÃO DIRETAMENTE NA SIDEBAR AGORA
        st.sidebar.markdown("---")
        st.sidebar.header("Downloads")
        st.sidebar.download_button(
            label="📥 Baixar TXT (Gráfico 1)",
            data=txt_output,
            file_name="niveis_gamma.txt",
            mime="text/plain"
        )

if __name__ == "__main__":

    main()
