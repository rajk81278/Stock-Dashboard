## streamlit run stock_dashboard.py

import streamlit as st, pandas as pd, numpy as np, yfinance as yf
import plotly.express as px
import datetime
import nsepython as nsepy
import pandas_ta as ta
import inspect





# Inject custom CSS for black theme
st.markdown("""
    <style>
        .main {
            background-color: #000000;
            color: white;
        }
        .css-18e3th9 {  /* Sidebar background */
            background-color: #1c1c1c !important;
        }
        .css-1d391kg {  /* Main content background */
            background-color: #000000 !important;
        }
        .css-10trblm {  /* Text elements */
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Stock Dashboard")
ticker = st.sidebar.text_input('Ticker')
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input('End Date')
ticker=ticker+'.NS'


data = yf.download(ticker, start=start_date, end=end_date)

data.columns=[i[0] for i in data.columns]

# print(data)
# st.dataframe(data)
symbol=ticker[:-3]
symbol=symbol.upper()
fig =px.line(data, x=data.index, y=data['Close'], title=symbol)
st.plotly_chart(fig)


# pricing_data, fundaamenta_data, tech_indicator, news= st.tabs(["Pricing Data", "Fundamental Data",'Techinal Analysis', 'Top 10 News'])
pricing_data, fundamental_data, tech_indicator, option_tab = st.tabs(["Pricing Data", "Fundamental Data",'Techinal Analysis', "Options & IV"])


# with pricing_data :
#     st.header('Price Movements')
#     data2=data
#     data2['Daily Return'] = data['Close']/data['Close'].shift(1)
#     data2.dropna(inplace=True)
#     st.write(data2)
#     cumulative_return = data2['Daily Return'].prod()
#     n_days = len(data2)
#     annual_return = (cumulative_return ** (252 / n_days) - 1) * 100
#     st.write('Annual Return is ', round(annual_return, 1), '%')
#     stdev= np.std(data2['Daily Return'])*np.sqrt(252)
#     st.write('Standard Deviation is ', round(stdev*100, 2), '%')
#     st.write('Risk Adj. Return is ', round(annual_return/(stdev*100), 2))

with pricing_data:
    st.header('📊 Price Movements Analysis')

    data2 = data.copy()
    data2['Daily Return'] = data2['Close'] / data2['Close'].shift(1)
    data2.dropna(inplace=True)

    # Show processed data
    st.subheader("Processed Data")
    st.write(data2)

    # Cumulative Return
    data2['Cumulative Return'] = (1 + data2['Daily Return'] - 1).cumprod()

    # Annual Return
    cumulative_return = data2['Daily Return'].prod()
    n_days = len(data2)
    annual_return = (cumulative_return ** (252 / n_days) - 1) * 100
    st.write('📈 Annual Return:', round(annual_return, 2), '%')

    # CAGR
    cagr = (data2['Close'].iloc[-1] / data2['Close'].iloc[0]) ** (252 / n_days) - 1
    st.write('📌 CAGR:', round(cagr * 100, 2), '%')

    # Standard Deviation (Annualized Volatility)
    stdev = np.std(data2['Daily Return']) * np.sqrt(252)
    st.write('📉 Annualized Volatility:', round(stdev * 100, 2), '%')

    # Sharpe Ratio
    rf = 0.05  # 5% risk-free rate
    sharpe_ratio = (annual_return / 100 - rf) / stdev
    st.write('⚖️ Sharpe Ratio:', round(sharpe_ratio, 2))

    # Sortino Ratio
    negative_returns = data2[data2['Daily Return'] < 1]['Daily Return'] - 1
    downside_std = np.std(negative_returns) * np.sqrt(252)
    sortino_ratio = (annual_return / 100 - rf) / downside_std if downside_std != 0 else np.nan
    st.write('📉 Sortino Ratio:', round(sortino_ratio, 2))

    # Max Drawdown
    cumulative = data2['Cumulative Return']
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_drawdown = drawdown.min()
    st.write('📉 Max Drawdown:', round(max_drawdown * 100, 2), '%')

    # Charts Section
    st.subheader("📊 Charts")

    st.markdown("### 🔷 Cumulative Return Over Time")
    st.line_chart(data2['Cumulative Return'], use_container_width=True)

    st.markdown("### 🔻 Drawdown Over Time")
    st.line_chart(drawdown, use_container_width=True)

    data2['Rolling Volatility'] = data2['Daily Return'].rolling(window=21).std() * np.sqrt(252)
    st.markdown("### 📉 21-Day Rolling Volatility (Annualized)")
    st.line_chart(data2['Rolling Volatility'], use_container_width=True)

    # Export CSV
    st.download_button("📥 Download CSV", data2.to_csv(), "price_analysis.csv", "text/csv")

# with fundamental_data :
#     st.write('Fundamental')
#     st.subheader('Balance Sheet')
#     bs=yf.Ticker(ticker)
#     bs = bs.balance_sheet
#     bs=pd.DataFrame(bs)
#     bs.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in bs.columns]
#     st.write(bs)

    
#     st.download_button("Download Balance Sheet", bs.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]} Balance_sheet.csv')




#     st.subheader('Quarterly Balance Sheet')
#     q_bs=yf.Ticker(ticker)
#     q_bs = q_bs.quarterly_balancesheet
#     q_bs=pd.DataFrame(q_bs)
#     q_bs.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in q_bs.columns]
#     st.write(q_bs)
    
#     st.download_button("Download Quarterly Balance Sheet", q_bs.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]} Quarterly Balance Sheet.csv')


#     st.subheader('Income Statement')
#     income_st=yf.Ticker(ticker)
#     income_st = income_st.incomestmt
#     income_st=pd.DataFrame(income_st)
#     income_st.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in income_st.columns]
#     st.write(income_st)
#     st.download_button("Download Income Statement", income_st.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]} Income Statement.csv')


#     st.subheader('Quarterly Income Statement')
#     quter_income_st=yf.Ticker(ticker)
#     quter_income_st = quter_income_st.quarterly_incomestmt
#     quter_income_st=pd.DataFrame(quter_income_st)
#     quter_income_st.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in quter_income_st.columns]
#     st.write(quter_income_st)
#     st.download_button("Download Quarterly Income Statement", quter_income_st.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]} Quarterly Income Statement.csv')


#     st.subheader('Casflow Statement')
#     CashF=yf.Ticker(ticker)
#     CashF = CashF.cashflow
#     CashF=pd.DataFrame(CashF)
#     CashF.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in CashF.columns]
#     st.write(CashF)
#     st.download_button("Download Casflow Statement", CashF.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]} Casflow Statement.csv')


with fundamental_data:
    st.header(f"📚 Fundamental Analysis for {ticker[:-3]}")
    st.markdown("This section includes detailed financial statements, trend charts, and key financial ratios for deeper insight.")

    ticker_obj = yf.Ticker(ticker)

    # Load all financials once
    bs = pd.DataFrame(ticker_obj.balance_sheet)
    q_bs = pd.DataFrame(ticker_obj.quarterly_balancesheet)
    income_st = pd.DataFrame(ticker_obj.incomestmt)
    quter_income_st = pd.DataFrame(ticker_obj.quarterly_incomestmt)
    cashflow = pd.DataFrame(ticker_obj.cashflow)

    # Format column headers
    for df in [bs, q_bs, income_st, quter_income_st, cashflow]:
        df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]

    # Key Financial Ratios
    st.subheader("📌 Key Financial Ratios")
    try:
        current_assets = bs.loc['Total Current Assets'].iloc[0]
        current_liabilities = bs.loc['Total Current Liabilities'].iloc[0]
        total_liabilities = bs.loc['Total Liab'].iloc[0]
        shareholder_equity = bs.loc['Total Stockholder Equity'].iloc[0]
        net_income = income_st.loc['Net Income'].iloc[0]
        total_revenue = income_st.loc['Total Revenue'].iloc[0]

        current_ratio = current_assets / current_liabilities
        debt_to_equity = total_liabilities / shareholder_equity
        net_profit_margin = net_income / total_revenue
        roe = net_income / shareholder_equity

        st.write(f"✅ **Current Ratio**: {current_ratio:.2f}")
        st.write(f"✅ **Debt to Equity**: {debt_to_equity:.2f}")
        st.write(f"✅ **Net Profit Margin**: {net_profit_margin:.2%}")
        st.write(f"✅ **Return on Equity (ROE)**: {roe:.2%}")
    except Exception as e:
        st.warning("⚠️ Some key ratios could not be calculated.")

    # Trend Charts
    st.subheader("📈 Trend: Revenue & Net Income")
    try:
        trends = pd.DataFrame({
            'Revenue': income_st.loc['Total Revenue'],
            'Net Income': income_st.loc['Net Income']
        }).T
        st.line_chart(trends.T)
    except Exception as e:
        st.warning("⚠️ Unable to plot trends due to missing data.")

    # Expandable Financial Statements
    st.subheader("📑 Detailed Financial Statements")
    with st.expander("📊 Balance Sheet"):
        st.write(bs)
        st.download_button("Download Balance Sheet", bs.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]}_Balance_Sheet.csv')

    with st.expander("📆 Quarterly Balance Sheet"):
        st.write(q_bs)
        st.download_button("Download Quarterly Balance Sheet", q_bs.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]}_Quarterly_Balance_Sheet.csv')

    with st.expander("📄 Income Statement"):
        st.write(income_st)
        st.download_button("Download Income Statement", income_st.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]}_Income_Statement.csv')

    with st.expander("📄 Quarterly Income Statement"):
        st.write(quter_income_st)
        st.download_button("Download Quarterly Income Statement", quter_income_st.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]}_Quarterly_Income_Statement.csv')

    with st.expander("💰 Cash Flow Statement"):
        st.write(cashflow)
        st.download_button("Download Cash Flow Statement", cashflow.to_csv().encode('utf-8'), file_name=f'{ticker[:-3]}_Cash_Flow_Statement.csv')


with tech_indicator:
    st.subheader('Technical Analysis Dashboard')
    # df=pd.DataFrame()

    

    # Get all callable functions defined in pandas_ta
    all_funcs = [name for name, obj in inspect.getmembers(ta) if inspect.isfunction(obj)]

    # Filter out internal/private functions (optional)
    indicators = [func for func in all_funcs if not func.startswith('_')]
    ind_list = []
    # Print
    print(f"Total indicators: {len(indicators)}\n")
    for ind in sorted(indicators):
        print(ind)
        ind_list.append(ind)   

    # st.write(ind_list)
    technical_indicator =st.selectbox('Tech Indicator', options=ind_list)
    method =technical_indicator
    proc=getattr(ta, method)(low=data['Low'], close=data['Close'], high =data['High'], open = data['Open'], volume=data['Volume'])
    proc=pd.DataFrame(proc)
    # proc=proc.dropna(inplace=True)
    proc['Close'] = data['Close']
    fig_ind_new =px.line(proc)
    st.plotly_chart(fig_ind_new)
    st.write(proc)

from nselib import capital_market
from nselib import derivatives

with option_tab:
    st.header("📊 Option Chain & IV Analysis")
    st.subheader('NSE Derivatives Market')
    derivatives_data_type = st.selectbox('Select Data to Extract', options=(
        'expiry_dates_future', 
            'future_price_volume_data', 
        'option_price_volume_data'))

    if derivatives_data_type in ['expiry_dates_future', 'expiry_dates_option_index']:
        data = getattr(derivatives, derivatives_data_type)()

    elif derivatives_data_type == 'future_price_volume_data':
        ticker = st.text_input('Ticker', value=ticker[:-3])
        inst_type = st.text_input('Instrument Type', 'FUTSTK')
        period_ = st.text_input('Period', '1M')
        data = derivatives.future_price_volume_data(ticker, inst_type, period=period_)

    elif derivatives_data_type == 'option_price_volume_data':
        ticker = st.text_input('Ticker', value=ticker[:-3])
        inst_type = st.text_input('Instrument Type', 'OPTSTK')
        period_ = st.text_input('Period', '1M')
        data = derivatives.option_price_volume_data(ticker, inst_type, period=period_)


    st.dataframe(data)

