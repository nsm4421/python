import W
import streamlit as st


st.title('해지율 튜토리얼')
input_years = st.number_input(label = '연수', value = 10, min_value = 3)

@st.cache
def make_df(years = input_years):
    df_month, df_annual = W.main(years)
    return df_month, df_annual

df_month, df_annual = make_df()

st.text('연도별 데이터')
st.dataframe(df_annual.transpose())

st.text('월별 데이터')
st.dataframe(df_month.transpose())

st.text('skew 시각화 월별')
st.line_chart(data = df_month['skew'])

st.text('skew해지율 시각화 월별')
st.line_chart(data = df_month['skew해지율'])