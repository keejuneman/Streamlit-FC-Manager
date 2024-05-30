import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re
import matplotlib.font_manager as fm
import os

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

# 폰트 파일 경로 설정
if font_option == 'Windows : Malgun Gothic':
    font_path = os.path.join(os.path.dirname(__file__), 'Fonts', 'malgun.ttf')  # Windows
else:
    font_path = os.path.join(os.path.dirname(__file__), 'Fonts', 'AppleGothic.ttf')  # Mac

# 폰트 파일 로드
fontprop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = fontprop.get_name()
plt.rcParams['axes.unicode_minus'] = False

# 조사 종류 선택
survey_option = st.sidebar.radio(
    "항목 선택",
    ('데일리 만족도 조사 추이 분석', '정성 평가 감정 분석')
)

# 데일리 만족도 조사 화면
if survey_option == '데일리 만족도 조사 추이 분석':
    # Streamlit 애플리케이션 제목
    st.title("Daily 만족도 조사 변화 추이 분석")

    # 사용자에게 Google Sheets 링크를 입력받는 텍스트 박스
    sheet_url = st.text_input("Google Sheets 링크를 입력하세요:")
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
        col1, col2 = st.columns([1, 4])  # 왼쪽과 오른쪽 열의 비율을 1:4로 설정

        with col1:
            # 라디오 버튼과 체크박스 생성
            x_column = st.radio("X Label을 위한 컬럼 선택 (날짜가 있는 컬럼 선택):", df.columns.tolist())
            y_columns = st.multiselect("Y Label을 위한 컬럼 선택 (정량 평가 수치가 있는 컬럼 선택):", df.columns.tolist())
            
            # 그래프 타입 선택
            graph_type = st.radio("그래프 종류 선택:", ["Bar", "Line", "Scatter"])

            # 데이터 프레임 출력
            st.write("Google Sheets 데이터:")
            st.dataframe(df)

        with col2:
            # 그래프 생성
            if x_column and y_columns:
                grouped_df = df.groupby(x_column)[y_columns].mean().reset_index()
                width = 1000
                height = 800
                if graph_type == "Bar":
                    fig = px.bar(grouped_df, x=x_column, y=y_columns, barmode='group', title='데일리 만족도 조사 변동 추이', width=width, height=height)
                elif graph_type == "Line":
                    fig = px.line(grouped_df, x=x_column, y=y_columns, title='데일리 만족도 조사 변동 추이', width=width, height=height)
                elif graph_type == "Scatter":
                    fig = px.scatter(grouped_df, x=x_column, y=y_columns, title='데일리 만족도 조사 변동 추이', width=width, height=height)

                fig.update_layout(
                    xaxis_title="날짜",
                    yaxis_title="평균 값",
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
                    )
                )

                st.plotly_chart(fig)

