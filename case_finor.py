import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gurobipy as gp
from gurobipy import GRB
import streamlit as st

def optimize_portfolio(returns_df, benchmark_df, K):

    # Cálculo das Médias dos Retornos Diários
    mean_returns = returns_df.mean()
    mean_benchmark = benchmark_df.mean().values[0]

    # Criação de uma Matriz de Covariância entre os Ativos e o IBOV
    cov_matrix = returns_df.cov()
    cov_matrix_benchmark = returns_df.apply(lambda x: np.cov(x, benchmark_df.squeeze())[0, 1])

    # Iniciação do Modelo pelo Gurobi
    model = gp.Model("Portfolio Optimization")
    model.Params.OutputFlag = 0

    # Adição das Variáveis Pesos (% entre 0 e 1) e Z (binária)
    weights = model.addVars(returns_df.columns, lb=0, ub=1, name="weights")
    Z = model.addVars(returns_df.columns, vtype=GRB.BINARY, name="y")

    # Adição de Restrições Soma dos Pesos = 1 e nº de Ativos = K
    model.addConstr(gp.quicksum(weights[i] for i in returns_df.columns) == 1, "budget")
    model.addConstr(gp.quicksum(Z[i] for i in returns_df.columns) == K, "cardinality")

    # Restrição que liga as variáveis binárias aos pesos dos ativos
    M = 1  
    for i in returns_df.columns:
        model.addConstr(weights[i] <= Z[i])
        model.addConstr(weights[i] >= Z[i] * 1e-6)  # Garante que se y[i] == 1, weights[i] >= 0

    # Iteração para calcular o Tracking Error
    portfolio_return = gp.quicksum(weights[i] * mean_returns[i] for i in returns_df.columns)
    tracking_error_var = (portfolio_return - mean_benchmark) ** 2
    for i in returns_df.columns:
        for j in returns_df.columns:
            tracking_error_var += weights[i] * weights[j] * cov_matrix.loc[i, j]
        tracking_error_var -= 2 * weights[i] * cov_matrix_benchmark[i]

    # Minimização do Tracking Error
    model.setObjective(tracking_error_var, GRB.MINIMIZE)

    # Configurações de parâmetros para melhorar a velocidade do Gurobi
    model.Params.TimeLimit = time_limit  # Limite de tempo (em segundos)
    model.Params.MIPGap = 0.01  # Tolerância de gap para soluções otimizadas

    # Otimização do Modelo
    model.optimize()

    if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
        optimized_weights = {i: weights[i].X for i in returns_df.columns}
        return optimized_weights
    else:
        raise RuntimeError("O modelo não encontrou uma solução ótima no tempo limite")

st.set_page_config(page_title="Case Finor | Index Tracking",
                   page_icon=":bar_chart:",layout="wide")
st.title('Bootcamp Fundo Amanhã | BAH - Index Tracking')

anos = ['2017','2018','2019','2020','2021','2022','2023','2024']
ano = st.sidebar.selectbox("Ano", anos)
N = st.sidebar.selectbox("N (nº de dias para otimização)", [120,90,60,30])
K = st.sidebar.selectbox("K (nº de ativos)", [5,10,15])
check = st.sidebar.checkbox('Mostrar Gráfico')

time_limit = 10
end = datetime.date(int(ano) - 1,12,31)
start =  end - datetime.timedelta(N)

if ano and N and K:
    componentes_ibov = pd.read_csv('https://raw.githubusercontent.com/arthurdiazb/case_finor/main/Componentes_IBOV.csv')
    returns = pd.read_csv('https://raw.githubusercontent.com/arthurdiazb/case_finor/main/Retornos_IBOV.csv',index_col='Date')
    returns.index = pd.to_datetime(returns.index)

    returns_filtered = returns.loc[start:end]
    returns_filtered = returns_filtered[list(componentes_ibov[ano].dropna())]
    returns_filtered.iloc[0] = 0

    ibov_return = pd.DataFrame(yf.download('^BVSP',start,end)['Adj Close'])
    ibov_return = ibov_return.apply(lambda x: (x / x.shift(1) - 1).fillna(0))
    ibov_return.columns = ['IBOV']
    ibov_return = pd.DataFrame(ibov_return['IBOV'],index=returns_filtered.index).fillna(0)

    optimized_weights = optimize_portfolio(returns_filtered, ibov_return, K)

    stocks_weights = pd.DataFrame(pd.DataFrame(optimized_weights,index=optimized_weights.keys()).iloc[0,0:])
    stocks_weights.columns = ['Pesos']
    stocks_weights = stocks_weights[stocks_weights > 0.0001].dropna()

    portfolio = pd.DataFrame(optimized_weights.values(),index=returns_filtered.columns).transpose()
    portfolio = pd.concat([portfolio,returns_filtered])
    portfolio = portfolio.loc[:, portfolio.iloc[0] != 0]
    portfolio.iloc[1] = (portfolio.iloc[0] * (100 - K))
    portfolio = portfolio.drop(0)
    portfolio = (1 + portfolio).cumprod()
    portfolio['Carteira Total'] = portfolio.sum(axis=1)

    ibov_acum = (1 + ibov_return).cumprod() * 100
    ibov_acum['Portfolio'] = portfolio['Carteira Total']

    start_forward = end
    end_forward = datetime.date(int(ano),12,31)

    returns_forward = returns.loc[start_forward:end_forward]
    returns_forward = returns_forward[list(componentes_ibov[ano].dropna())]

    portfolio_forward = pd.DataFrame(optimized_weights.values(),index=returns_forward.columns).transpose()
    portfolio_forward = pd.concat([portfolio_forward,returns_forward])
    portfolio_forward = portfolio_forward.loc[:, portfolio_forward.iloc[0] != 0]
    portfolio_forward.iloc[1] = (portfolio_forward.iloc[0] * (100 - K))
    portfolio_forward = portfolio_forward.drop(0)
    portfolio_forward = (1 + portfolio_forward).cumprod()
    portfolio_forward['Carteira Total'] = portfolio_forward.sum(axis=1)

    ibov_forward = pd.DataFrame(yf.download('^BVSP',start_forward,end_forward)['Adj Close'])
    ibov_forward = ibov_forward.apply(lambda x: (x / x.shift(1) - 1).fillna(0))
    ibov_forward.columns = ['IBOV']
    ibov_forward = pd.DataFrame(ibov_forward['IBOV'],index=returns_forward.index).fillna(0)
    ibov_forward = (1 + ibov_forward).cumprod() * 100
    ibov_forward['Portfolio'] = portfolio_forward['Carteira Total']

    tracking_error = np.sqrt((((ibov_forward['Portfolio'].pct_change().fillna(0) - ibov_forward['IBOV'].pct_change().fillna(0)) ** 2).sum()) / ibov_forward.shape[0])

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=ibov_forward.index,
    y=ibov_forward['IBOV'],
    mode='lines',
    name='IBOV',
    line=dict(color='#FFA500', width=2),  
    hovertemplate='<b>%{y}</b><extra>IBOV</extra>'))
fig.add_trace(go.Scatter(
    x=ibov_forward.index,
    y=ibov_forward['Portfolio'],
    mode='lines',
    name='Portfolio',
    line=dict(color='#FF4500', width=2), 
    hovertemplate='<b>%{y}</b><extra>Portfolio</extra>'))
fig.update_layout(
    title=f"Comparação de IBOV e Portfolio para o Ano {ano}",
    xaxis_title='Data',
    yaxis_title='Valores',
    template='plotly_dark',
    legend_title="Indicadores",
    legend=dict(yanchor="top", y=1.0, xanchor="right", x=0.0001),)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    x=stocks_weights.index,
    y=stocks_weights['Pesos'],
    text=[f'{value*100:.2f}%' for value in stocks_weights['Pesos']] ,
    textposition='outside',
    marker_color='#FF4500'))

fig2.update_layout(
    title=f"Pesos Sugeridos para o Ano {ano}",
    xaxis_title='Ativos',
    yaxis_title='Pesos',
    template='plotly_dark',)

if check:
    st.plotly_chart(fig,use_container_width=True)
    st.plotly_chart(fig2,use_container_width=True)
    st.write(f"A soma dos pesos do portfolio é: {stocks_weights['Pesos'].sum() * 100:.2f} %")
    st.write(f"O tracking error para {K} ativos no ano de {ano} é de {tracking_error * 100:.4f}%")



