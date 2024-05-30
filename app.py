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

if font_option == 'Windows : Malgun Gothic':
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'AppleGothic'

plt.rcParams['axes.unicode_minus'] = False

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

            # 데이터 프레임 출력
            st.write("Google Sheets 데이터:")
            st.dataframe(df)

        with col2:
            # 그래프 생성
            if x_column and y_columns:
                grouped_df = df.groupby(x_column)[y_columns].mean()

                fig, ax = plt.subplots(figsize=(18, 12))
                grouped_df.plot(kind='bar', ax=ax)

                # 수치 값 텍스트로 표기
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center', fontsize=20, color='black', xytext=(0, 10),
                                textcoords='offset points')

                ax.set_xlabel("날짜", fontsize=20)
                ax.set_ylabel("평균 값", fontsize=20)
                ax.set_title(f"데일리 만족도 조사 변동 추이", fontsize=20)

                # x label 가로 정렬
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)

                # y label 폰트 크기 설정
                ax.tick_params(axis='y', labelsize=20)

                # 범례 폰트 크기 설정
                ax.legend(fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=len(y_columns))

                st.pyplot(fig)


# # 정성 평가 분석 화면
# elif survey_option == '정성 평가 감정 분석':
#     st.title("정성 평가 감정 분석")

#     # 사용자에게 Google Sheets 링크를 입력받는 텍스트 박스
#     sheet_url_qual = st.text_input("Google Sheets 링크를 입력하세요:")
#     sheet_url_btn_qual = st.button("Enter")

#     # 세션 상태 초기화
#     if "df_qual" not in st.session_state:
#         st.session_state.df_qual = None

#     # 텍스트 박스에 URL이 입력되었을 때만 실행
#     if sheet_url_qual and sheet_url_btn_qual:
#         try:
#             # Google Sheets와 연결 생성
#             conn_qual = st.connection("gsheets", type=GSheetsConnection, url=sheet_url_qual)

#             # 데이터 프레임 읽기
#             st.session_state.df_qual = conn_qual.read()

#         except Exception as e:
#             st.error(f"Google Sheets와 연결하는 데 실패했습니다: {e}")

#     # 데이터 프레임이 세션 상태에 존재할 때만 실행
#     if st.session_state.df_qual is not None:
#         df_qual = st.session_state.df_qual

#         # 데이터 프레임 출력
#         st.write("Google Sheets 데이터:")
#         st.dataframe(df_qual)

#         # 감정 분석을 위한 컬럼 선택
#         columns_list = [""] + df_qual.columns.tolist()
#         text_column = st.radio("감정 분석을 위한 컬럼 선택:", columns_list)

#         if text_column:
#             if text_column != "":
#                 # 모든 값을 문자열로 변환
#                 df_qual[text_column] = df_qual[text_column].astype(str)

#                 # BERT 모델과 토크나이저 로드
#                 tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
#                 model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)
                
#                 # GPU 사용 설정
#                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
#                 # 학습된 모델의 가중치 로드 (CPU 사용 시 map_location 설정)
#                 state_dict = torch.load('trained_model.pth', map_location=device)
                
#                 # 새로운 모델의 state_dict를 가져오기
#                 new_state_dict = model.state_dict()

#                 # 학습된 모델의 state_dict를 새 모델로 옮기기 (임베딩 레이어를 제외한 나머지 파라미터만)
#                 for name, param in state_dict.items():
#                     if name in new_state_dict and param.size() == new_state_dict[name].size():
#                         new_state_dict[name].copy_(param)

#                 model.load_state_dict(new_state_dict)
                
#                 model.to(device)

#                 # 데이터셋 정의
#                 class TextDataset(Dataset):
#                     def __init__(self, texts, tokenizer, max_len):
#                         self.texts = texts
#                         self.tokenizer = tokenizer
#                         self.max_len = max_len

#                     def __len__(self):
#                         return len(self.texts)

#                     def __getitem__(self, idx):
#                         text = self.texts[idx]
#                         encoding = self.tokenizer.encode_plus(
#                             text,
#                             add_special_tokens=True,
#                             max_length=self.max_len,
#                             return_token_type_ids=False,
#                             padding='max_length',
#                             return_attention_mask=True,
#                             return_tensors='pt',
#                             truncation=True,
#                         )

#                         return {
#                             'text': text,
#                             'input_ids': encoding['input_ids'].flatten(),
#                             'attention_mask': encoding['attention_mask'].flatten()
#                         }

#                 # 감정 분석 수행 함수
#                 def analyze_sentiment(texts):
#                     dataset = TextDataset(texts, tokenizer, max_len=128)
#                     dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

#                     model.eval()
#                     sentiments = []
#                     with torch.no_grad():
#                         for batch in dataloader:
#                             input_ids = batch['input_ids'].to(device)
#                             attention_mask = batch['attention_mask'].to(device)

#                             outputs = model(input_ids, attention_mask=attention_mask)
#                             logits = outputs.logits
#                             predictions = torch.argmax(logits, dim=1)
#                             sentiments.extend(predictions.cpu().numpy())

#                     return ["긍정" if sentiment == 1 else "부정" for sentiment in sentiments]

#                 df_qual['감정 분석 결과'] = analyze_sentiment(df_qual[text_column].tolist())

#                 # 결과 출력
#                 st.write("감정 분석 결과:")
#                 st.dataframe(df_qual[['감정 분석 결과'] + [text_column]])
