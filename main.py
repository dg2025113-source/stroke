import streamlit as st
import pandas as pd

# ------------------------------
# 페이지 기본 설정
# ------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide"
)

# ------------------------------
# 데이터 불러오기
# ------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

# ------------------------------
# 제목
# ------------------------------
st.title("🧠 뇌졸중 예측 실습실")
st.markdown("### 뇌졸중 관련 데이터를 탐구하고 예측 모델을 만들어보는 실습 공간입니다.")
st.divider()

# ------------------------------
# 데이터 소개
# ------------------------------
st.header("📌 데이터 소개")
st.write(
    """
    이 데이터는 환자의 개인 정보와 건강 상태를 바탕으로
    **뇌졸중(stroke) 발생 여부**를 기록한 자료입니다.
    아래에서 데이터의 전체적인 특징을 확인해 보세요.
    """
)

# ------------------------------
# 큰 숫자 카드 4개
# ------------------------------
total_people = len(df)
total_columns = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = round(stroke_count / total_people * 100, 2)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="전체 사람 수", value=f"{total_people:,} 명")

with col2:
    st.metric(label="열 개수", value=f"{total_columns} 개")

with col3:
    st.metric(label="뇌졸중(stroke=1) 인원", value=f"{stroke_count:,} 명")

with col4:
    st.metric(label="뇌졸중 비율", value=f"{stroke_ratio} %")

st.divider()

# ------------------------------
# 열 설명 표
# ------------------------------
st.header("📋 열(컬럼) 설명")

column_names = [
    "id", "gender", "age", "hypertension", "heart_disease",
    "ever_married", "work_type", "Residence_type",
    "avg_glucose_level", "bmi", "smoking_status", "stroke"
]

# 값의 종류 요약 함수
def get_value_summary(col):
    if df[col].dtype == "object" or df[col].nunique() < 10:
        unique_vals = df[col].dropna().unique()
        unique_vals = sorted([str(v) for v in unique_vals])
        return ", ".join(unique_vals)
    else:
        return f"연속형 숫자 (최소 {df[col].min()} ~ 최대 {df[col].max()})"

value_summaries = [get_value_summary(col) for col in column_names]
missing_counts = [int(df[col].isnull().sum()) for col in column_names]

column_info = pd.DataFrame({
    "열 이름": column_names,
    "우리말 뜻": [""] * len(column_names),  # 직접 채워 넣을 빈 칸
    "값의 종류": value_summaries,
    "빈 값 개수": missing_counts
})

st.dataframe(column_info, use_container_width=True, hide_index=True)
st.caption("💡 '우리말 뜻' 칸은 교재를 참고해서 직접 채워 넣어보세요.")

st.divider()

# ------------------------------
# 데이터 미리보기 (처음 5줄)
# ------------------------------
st.header("🔍 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(5), use_container_width=True)

st.divider()

# ------------------------------
# 데이터 출처 (직접 작성)
# ------------------------------
st.header("📚 데이터 출처")
st.info("여기에 교재에 나온 데이터 출처 내용을 직접 적어보세요.")

st.text_area(
    label="출처 작성란",
    placeholder="예: 이 데이터는 ○○○에서 제공하며...",
    height=150
)
