import pandas as pd
import plotly.graph_objs as go
import numpy as np
from scipy.stats import norm
from datetime import datetime, timedelta
import streamlit as st

def calcGammaEx(S, K, vol, T, r, q, optType, OI):
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
        bar_width1 = st.number_input("Largura das barras do Gráfico 1 (ex: 0.3)", min_value=0.01, max_value=15.0, step=0.01, value=0.3)
        bar_width2 = st.number_input("Largura das barras do Gráfico 2 (ex: 0.2)", min_value=0.01, max_value=15.0, step=0.01, value=0.2)
        levels_input = st.slider("Quantidade de níveis (resolução do gráfico 3)", min_value=20, max_value=100, step=5, value=60)
        bar_color1 = st.color_picker("Cor das barras do Gráfico 1",  '#1a76ff')
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

        # Gráfico 3
        st.subheader("Gráfico 3: Gamma Exposure Profile")
        levels = np.linspace(fromStrike, toStrike, levels_input)

        df['daysTillExp'] = [1/262 if (np.busday_count(todayDate.date(), x.date())) == 0 else np.busday_count(todayDate.date(), x.date())/262 for x in df.ExpirationDate]
        nextExpiry = df['ExpirationDate'].min()

        df['IsThirdFriday'] = [isThirdFriday(x) for x in df.ExpirationDate]
        thirdFridays = df.loc[df['IsThirdFriday'] == True]
        nextMonthlyExp = thirdFridays['ExpirationDate'].min()

        totalGamma = []
        totalGammaExNext = []
        totalGammaExFri = []

        for level in levels:
            df['callGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['CallIV'], row['daysTillExp'], 0, 0, "call", row['CallOpenInt']), axis = 1)
            df['putGammaEx'] = df.apply(lambda row : calcGammaEx(level, row['StrikePrice'], row['PutIV'], row['daysTillExp'], 0, 0, "put", row['PutOpenInt']), axis = 1)

            totalGamma.append(df['callGammaEx'].sum() - df['putGammaEx'].sum())

            exNxt = df.loc[df['ExpirationDate'] != nextExpiry]
            totalGammaExNext.append(exNxt['callGammaEx'].sum() - exNxt['putGammaEx'].sum())

            exFri = df.loc[df['ExpirationDate'] != nextMonthlyExp]
            totalGammaExFri.append(exFri['callGammaEx'].sum() - exFri['putGammaEx'].sum())

        totalGamma = np.array(totalGamma) / 10**9
        totalGammaExNext = np.array(totalGammaExNext) / 10**9
        totalGammaExFri = np.array(totalGammaExFri) / 10**9

        zeroCrossIdx = np.where(np.diff(np.sign(totalGamma)))[0]
        zeroGamma = None
        if len(zeroCrossIdx) > 0:
            negGamma = totalGamma[zeroCrossIdx]
            posGamma = totalGamma[zeroCrossIdx+1]
            negStrike = levels[zeroCrossIdx]
            posStrike = levels[zeroCrossIdx+1]
            zeroGamma = posStrike - ((posStrike - negStrike) * posGamma/(posGamma-negGamma))
            zeroGamma = zeroGamma[0]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=levels, y=totalGamma, mode='lines', name='All Expiries'))
        fig3.add_trace(go.Scatter(x=levels, y=totalGammaExNext, mode='lines', name='Ex-Next Expiry'))
        fig3.add_trace(go.Scatter(x=levels, y=totalGammaExFri, mode='lines', name='Ex-Next Monthly Expiry'))

        chartTitle = f"Gamma Exposure Profile, ATIVO, {todayDate.strftime('%d %b %Y')}"
        fig3.update_layout(title_text=chartTitle, title_font=dict(size=20, family="Arial Black"),
                           xaxis_title='Index Price', yaxis_title='Gamma Exposure ($ billions/1% move)',
                           width=1400, height=700)

        fig3.add_trace(go.Scatter(
            x=[fromStrike, zeroGamma, toStrike],
            y=[min(totalGamma), min(totalGamma), min(totalGamma)],
            mode="none",
            fill="toself",
            fillcolor="red",
            opacity=0.1,
            showlegend=False,
            name="Negative Gamma"
        ))

        fig3.add_trace(go.Scatter(
            x=[fromStrike, zeroGamma, toStrike],
            y=[max(totalGamma), max(totalGamma), max(totalGamma)],
            mode="none",
            fill="toself",
            fillcolor="green",
            opacity=0.1,
            showlegend=False,
            name="Positive Gamma"
        ))

        fig3.add_trace(go.Scatter(
            x=[spotPrice, spotPrice],
            y=[min(totalGamma), max(totalGamma)],
            mode='lines',
            line=dict(color="red", width=1.5, dash="dash"),
            name="Spot Price"
        ))

        if zeroGamma:
            fig3.add_trace(go.Scatter(
                x=[zeroGamma, zeroGamma],
                y=[min(totalGamma), max(totalGamma)],
                mode='lines',
                line=dict(color="green", width=1.5, dash="dot"),
                name="Gamma Flip"
            ))

        st.plotly_chart(fig3, use_container_width=True)
        st.download_button("Baixar Gráfico 3 como HTML", fig3.to_html(full_html=False), file_name="grafico3_gamma_profile.html")

if __name__ == "__main__":
    main()
