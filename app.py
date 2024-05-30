import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re

st.set_page_config(
    page_title="FC Manager",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if "df" not in st.session_state:
    st.session_state.df = None

# 사이드바 헤더
st.sidebar.header("설정")

# 한글 폰트 설정 선택 라디오 버튼
font_option = st.sidebar.radio(
    "OS 선택",
    ('Windows : Malgun Gothic', 'MAC : AppleGothic')
)

# 조사 종류 선택
survey_option = st.sidebar.radio(
    "항목 선택",
    ('데일리 만족도 조사 추이 분석', '정성 평가 감정 분석 [개발 중]')
)

# 데일리 만족도 조사 화면
if survey_option == '데일리 만족도 조사 추이 분석':
    # Streamlit 애플리케이션 제목
    st.title("Daily 만족도 조사 변화 추이 분석")

    # 사용자에게 Google Sheets 링크를 입력받는 텍스트 박스
    sheet_url = st.text_input("[링크가 있는 사용자 : 뷰어] 권한이 부여된 Google Sheets 링크를 입력하세요")
    sheet_url_btn = st.button("Enter")

    # 텍스트 박스에 URL이 입력되었을 때만 실행
    if sheet_url and sheet_url_btn:
        try:
            # Google Sheets 링크에서 스프레드시트 ID 추출
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", sheet_url)
            if not match:
                raise ValueError("유효한 Google Sheets 링크를 입력하세요.")
            sheet_id = match.group(1)

            # Google Sheets 링크를 CSV 형식의 export 링크로 변환
            csv_export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"

            # 데이터 프레임 생성
            st.session_state.df = pd.read_csv(csv_export_url)

        except Exception as e:
            st.error(f"Google Sheets와 연결하는 데 실패했습니다: {e}")

    # 데이터 프레임이 세션 상태에 존재할 때만 실행
    if st.session_state.df is not None:
        df = st.session_state.df

        # 레이아웃 분할
        col1, col2, col3 = st.columns([1, 0.05, 4])  # 왼쪽과 오른쪽 열의 비율을 1:0.05:4로 설정

        with col1:
            # 라디오 버튼과 체크박스 생성
            graph_type = st.radio("그래프 종류 선택:", ["Bar", "Line", "Scatter"])
            
            # X Label 선택
            x_column = st.selectbox("X Label : 날짜가 있는 컬럼 선택", ['Choose an option'] + df.columns.tolist())
            
            # Y Label 선택
            y_columns = st.multiselect("Y Label : 정량 평가 수치가 있는 컬럼", df.columns.tolist())

        with col2:
            # 구분선을 추가하기 위한 HTML/CSS
            st.markdown(
                """
                <style>
                .divider {
                    height: 150vh;
                    width: 2px;
                    background-color: #cccccc;
                    margin: 0 auto;
                }
                </style>
                <div class="divider"></div>
                """,
                unsafe_allow_html=True
            )

        with col3:
            # 그래프 생성
            if x_column and y_columns:
                grouped_df = df.groupby(x_column)[y_columns].mean().reset_index()

                fig = go.Figure()

                for y_column in y_columns:
                    if graph_type == "Bar":
                        fig.add_trace(go.Bar(x=grouped_df[x_column], y=grouped_df[y_column], name=y_column, text=grouped_df[y_column], texttemplate='%{text:.2f}', textposition='outside', textfont=dict(size=16)))
                    elif graph_type == "Line":
                        fig.add_trace(go.Scatter(x=grouped_df[x_column], y=grouped_df[y_column], mode='lines+markers+text', name=y_column, text=grouped_df[y_column], texttemplate='%{text:.2f}', textposition='top center', textfont=dict(size=16)))
                    elif graph_type == "Scatter":
                        fig.add_trace(go.Scatter(x=grouped_df[x_column], y=grouped_df[y_column], mode='markers+text', name=y_column, text=grouped_df[y_column], texttemplate='%{text:.2f}', textposition='top center', textfont=dict(size=16)))

                fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="평균 값",
                    title='데일리 만족도 조사 변동 추이',
                    title_font_size=20,
                    xaxis_tickfont_size=12,
                    yaxis_tickfont_size=20,
                    legend_title_font_size=16,
                    legend_font_size=16,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    width=1000,
                    height=800
                )

                st.plotly_chart(fig)
                # 데이터 프레임 출력
                st.write("Google Sheets 데이터:")
                st.dataframe(df)
