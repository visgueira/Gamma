import tempfile
import subprocess
import sys
import os

def run_streamlit_embutido():
    # Código Streamlit embutido como string
    streamlit_code = """import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import os
import streamlit as st


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

    with st.sidebar:
        uploaded_file = st.file_uploader("Envie o arquivo quotedata.csv", type="csv")
        bar_width1 = st.number_input("Largura das barras do Gráfico 1 (ex: 0.3)", min_value=0.01, max_value=1.0, step=0.01, value=0.3)
        bar_width2 = st.number_input("Largura das barras do Gráfico 2 (ex: 0.2)", min_value=0.01, max_value=1.0, step=0.01, value=0.2)
        levels_input = st.slider("Quantidade de níveis (resolução do gráfico 3)", min_value=20, max_value=100, step=5, value=60)
        bar_color1 = st.color_picker("Cor das barras do Gráfico 1", '#1a76ff')
        bar_color2 = st.color_picker("Cor das barras do Gráfico 2 (Call)", '#00cc96')
        bar_color3 = st.color_picker("Cor das barras do Gráfico 2 (Put)", '#ef553b')

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

        st.subheader("Gráfico 1: Total Gamma")
        fig1 = go.Figure(go.Bar(x=strikes, y=dfAgg['TotalGamma'].to_numpy(), width=bar_width1,
            marker_color=bar_color1, marker_line_color='black', marker_line_width=0.15,
            name='Gamma Exposure'))
        fig1.add_shape(type='line', x0=spotPrice, y0=min(dfAgg['TotalGamma']), x1=spotPrice,
            y1=max(dfAgg['TotalGamma']), line=dict(color='red', width=2, dash='dash'))
        fig1.update_layout(title=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% ATIVO Move",
            xaxis_title='Strike', yaxis_title='Spot Gamma Exposure ($ billions/1% move)',
            xaxis=dict(range=[fromStrike, toStrike]), yaxis=dict(tickformat='$,.2f'),
            plot_bgcolor='black', font=dict(family='Arial', size=12, color='black'), width=1000, height=600)
        st.plotly_chart(fig1, use_container_width=True)

        st.download_button("Baixar Gráfico 1 como HTML", fig1.to_html(full_html=False), file_name="grafico1_total_gamma.html")

        st.subheader("Gráfico 2: Call e Put Gamma")
        fig2 = go.Figure()
        fig2.add_bar(x=strikes, y=dfAgg['CallGEX'].to_numpy() / 10**9, width=bar_width2, name="Call Gamma", marker_color=bar_color2)
        fig2.add_bar(x=strikes, y=dfAgg['PutGEX'].to_numpy() / 10**9, width=bar_width2, name="Put Gamma", marker_color=bar_color3)
        fig2.add_shape(dict(type="line", x0=spotPrice, y0=0, x1=spotPrice,
            y1=max(dfAgg['CallGEX'].to_numpy() / 10**9), line=dict(color="black", width=2)))
        fig2.update_layout(title_text=f"Total Gamma: ${df['TotalGamma'].sum():,.2f} Bn per 1% ATIVO Move",
            title_font=dict(size=20, family="Arial Black"), xaxis_title="Strike",
            yaxis_title="Spot Gamma Exposure ($ billions/1% move)", xaxis=dict(range=[fromStrike, toStrike]),
            width=1000, height=600)
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button("Baixar Gráfico 2 como HTML", fig2.to_html(full_html=False), file_name="grafico2_call_put.html")

        st.subheader("Gráfico 3: Gamma Exposure Profile")
        levels = np.linspace(fromStrike, toStrike, levels_input)
        dtes = np.array([max(np.busday_count(todayDate.date(), d.date()), 1) / 262 for d in df['ExpirationDate']])
        df_call = df[['StrikePrice', 'CallIV', 'CallOpenInt']].to_numpy().T
        df_put = df[['StrikePrice', 'PutIV', 'PutOpenInt']].to_numpy().T
        strikes_call, iv_call, oi_call = df_call
        strikes_put, iv_put, oi_put = df_put

        totalGamma = []
        for level in levels:
            gamma_call = calcGammaEx_numpy(level, strikes_call, iv_call, dtes, 0, 0, "call", oi_call)
            gamma_put = calcGammaEx_numpy(level, strikes_put, iv_put, dtes, 0, 0, "put", oi_put)
            totalGamma.append(np.sum(gamma_call) - np.sum(gamma_put))

        totalGamma = np.array(totalGamma) / 10**9
        zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
        if len(zeroCrossIdx) > 0:
            negGamma = totalGamma[zeroCrossIdx]
            posGamma = totalGamma[zeroCrossIdx+1]
            negStrike = levels[zeroCrossIdx]
            posStrike = levels[zeroCrossIdx+1]
            zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
            zeroGamma = zeroGamma[0]
        else:
            zeroGamma = None

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=levels, y=totalGamma, mode='lines', name='All Expiries'))
        fig3.update_layout(title=f"Gamma Exposure Profile, ATIVO, {todayDate.strftime('%d %b %Y')}",
                           xaxis_title='Index Price', yaxis_title='Gamma Exposure ($ billions/1% move)',
                           width=1000, height=600)
        fig3.add_shape(type="line", x0=spotPrice, y0=min(totalGamma), x1=spotPrice, y1=max(totalGamma),
                       line=dict(color="red", width=1.5))
        if zeroGamma:
            fig3.add_shape(type="line", x0=zeroGamma, y0=min(totalGamma), x1=zeroGamma, y1=max(totalGamma),
                           line=dict(color="green", width=1.5))
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button("Baixar Gráfico 3 como HTML", fig3.to_html(full_html=False), file_name="grafico3_gamma_profile.html")


if __name__ == "__main__":
    main()
    """

    # Cria arquivo temporário e executa o app Streamlit
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
        temp_file.write(streamlit_code)
        temp_path = temp_file.name

    try:
        subprocess.run(["streamlit", "run", temp_path])
        

    finally:
        os.remove(temp_path)

if __name__ == "__main__":
    run_streamlit_embutido()
