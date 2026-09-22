import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# ------------------------------
# 페이지 기본 설정
# ------------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🤖",
    layout="wide"
)

RANDOM_STATE = 42

# ------------------------------
# 데이터 불러오기
# ------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"
    df = pd.read_csv(url, encoding="utf-8")
    return df

df = load_data()

st.title("🤖 분류 모델 만들기")
st.markdown("나이, 혈당, 체질량지수, 고혈압, 심장병 정보로 뇌졸중을 예측하는 모델을 만들어봅니다.")
st.divider()

# ------------------------------
# 열 이름 <-> 우리말 이름 매핑
# ------------------------------
col_kor = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병"
}
kor_col = {v: k for k, v in col_kor.items()}

all_features = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
default_features = ["age", "avg_glucose_level", "hypertension", "heart_disease"]  # bmi 제외

# ------------------------------
# 1. 속성 선택
# ------------------------------
st.header("1️⃣ 입력으로 사용할 속성 선택")

selected_kor = st.multiselect(
    "모델의 입력으로 사용할 속성을 골라주세요.",
    options=[col_kor[c] for c in all_features],
    default=[col_kor[c] for c in default_features]
)

selected_features = [kor_col[k] for k in selected_kor]

if len(selected_features) < 2:
    st.warning("⚠️ 속성을 두 개 이상 골라주세요.")
    st.stop()

st.divider()

# ------------------------------
# 2. 데이터 준비: 정렬, 10명씩 묶어 테스트 분리
# ------------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)
df_sorted["group_pos"] = np.arange(len(df_sorted)) % 10

test_mask = df_sorted["group_pos"] < 3
train_df = df_sorted[~test_mask].copy()
test_df = df_sorted[test_mask].copy()

# bmi 결측치 처리: 선택된 경우에만, 학습용 중앙값으로
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

# ------------------------------
# 3. 학습용 데이터 크기 맞추기 (언더샘플링)
# ------------------------------
train_pos = train_df[train_df["stroke"] == 1]
train_neg = train_df[train_df["stroke"] == 0]

min_count = min(len(train_pos), len(train_neg))

train_pos_sampled = train_pos.sample(n=min_count, random_state=RANDOM_STATE)
train_neg_sampled = train_neg.sample(n=min_count, random_state=RANDOM_STATE)

train_balanced = pd.concat([train_pos_sampled, train_neg_sampled]).sample(
    frac=1, random_state=RANDOM_STATE
).reset_index(drop=True)

X_train = train_balanced[selected_features]
y_train = train_balanced["stroke"]

X_test = test_df[selected_features]
y_test = test_df["stroke"]

st.info(f"📌 학습용 데이터(크기 맞춘 뒤): {len(train_balanced)}명 / 테스트용 데이터: {len(test_df)}명")

st.divider()

# ------------------------------
# 4. 모델 학습
# ------------------------------
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=RANDOM_STATE
)
tree_model.fit(X_train, y_train)

# 다수결(더미) 모델: 훈련용에서 많은 쪽으로만 답
majority_class = y_train.value_counts().idxmax()
train_pred_dummy = np.full(len(y_train), majority_class)
test_pred_dummy = np.full(len(y_test), majority_class)

# ------------------------------
# 5. 정확도 계산
# ------------------------------
def get_accuracies(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    return train_acc, test_acc

log_train_acc, log_test_acc = get_accuracies(log_model, X_train, y_train, X_test, y_test)
tree_train_acc, tree_test_acc = get_accuracies(tree_model, X_train, y_train, X_test, y_test)

dummy_train_acc = accuracy_score(y_train, train_pred_dummy)
dummy_test_acc = accuracy_score(y_test, test_pred_dummy)

# ------------------------------
# 6. 정확도 카드
# ------------------------------
st.header("2️⃣ 모델 정확도 비교")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="로지스틱 회귀 (확률로 답하는 모델)",
        value=f"{log_test_acc*100:.1f}%"
    )
    st.caption(f"훈련 정확도: {log_train_acc*100:.1f}%  ｜  테스트 정확도: {log_test_acc*100:.1f}%")

with col2:
    st.metric(
        label="의사결정트리 (질문으로 답하는 모델)",
        value=f"{tree_test_acc*100:.1f}%"
    )
    st.caption(f"훈련 정확도: {tree_train_acc*100:.1f}%  ｜  테스트 정확도: {tree_test_acc*100:.1f}%")

with col3:
    st.metric(
        label="다수결 모델 (입력을 보지 않는 모델)",
        value=f"{dummy_test_acc*100:.1f}%"
    )
    st.caption(f"훈련 정확도: {dummy_train_acc*100:.1f}%  ｜  테스트 정확도: {dummy_test_acc*100:.1f}%")

st.divider()

# ------------------------------
# 7. 산점도 + 결정 경계
# ------------------------------
st.header("3️⃣ 산점도로 보는 결정 경계")

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 골라야 그림을 그릴 수 있습니다.")
    st.stop()

axis_kor_options = [col_kor[c] for c in selected_features]

col_a, col_b = st.columns(2)
with col_a:
    x_kor = st.selectbox("가로축으로 사용할 속성", options=axis_kor_options, index=0)
with col_b:
    remaining = [k for k in axis_kor_options if k != x_kor]
    y_kor = st.selectbox("세로축으로 사용할 속성", options=remaining, index=0)

x_col = kor_col[x_kor]
y_col = kor_col[y_kor]

other_features = [f for f in selected_features if f not in [x_col, y_col]]

# 두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정
fixed_values = {}
for f in other_features:
    fixed_values[f] = X_test[f].median()

if fixed_values:
    fixed_text = ", ".join([f"{col_kor[f]} = {v:.2f}" for f, v in fixed_values.items()])
    st.caption(f"📌 그림에 나타나지 않는 속성은 다음 값으로 고정했습니다: {fixed_text}")

# 격자 생성 (결정 경계용)
x_min, x_max = X_test[x_col].min(), X_test[x_col].max()
y_min, y_max = X_test[y_col].min(), X_test[y_col].max()

x_range = np.linspace(x_min, x_max, 200)
y_range = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(x_range, y_range)

grid_df = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})
for f, v in fixed_values.items():
    grid_df[f] = v

grid_df = grid_df[selected_features]  # 학습 시 사용한 컬럼 순서 맞추기

# 트리 모델의 영역(배경색)
tree_grid_pred = tree_model.predict(grid_df)
tree_grid_pred = tree_grid_pred.reshape(xx.shape)

fig = go.Figure()

# 트리 결정 영역 배경
fig.add_trace(go.Contour(
    x=x_range, y=y_range, z=tree_grid_pred,
    showscale=False,
    colorscale=[[0, "rgba(99,110,250,0.15)"], [1, "rgba(239,85,59,0.15)"]],
    contours=dict(coloring="fill"),
    line=dict(width=0),
    name="트리 영역",
    hoverinfo="skip"
))

# 테스트 데이터 산점도
test_plot_df = test_df.copy()
test_plot_df["stroke_label"] = test_plot_df["stroke"].map({0: "정상", 1: "뇌졸중"})

for label, color in [("정상", "blue"), ("뇌졸중", "red")]:
    subset = test_plot_df[test_plot_df["stroke_label"] == label]
    fig.add_trace(go.Scatter(
        x=subset[x_col], y=subset[y_col],
        mode="markers",
        name=label,
        marker=dict(color=color, size=6, opacity=0.6)
    ))

# 로지스틱 회귀 0.5 기준선 (두 축 좌표계에서 직선 계산)
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

feature_index = {f: i for i, f in enumerate(selected_features)}
x_idx = feature_index[x_col]
y_idx = feature_index[y_col]

# 다른 속성으로 인한 상수항 계산
other_sum = intercept
for f in other_features:
    other_sum += coef[feature_index[f]] * fixed_values[f]

coef_x = coef[x_idx]
coef_y = coef[y_idx]

line_drawn = False
if abs(coef_y) > 1e-10:
    # y = -(coef_x * x + other_sum) / coef_y
    line_x = np.linspace(x_min, x_max, 100)
    line_y = -(coef_x * line_x + other_sum) / coef_y

    # 그래프 범위 안에 있는지 확인
    in_range_mask = (line_y >= y_min) & (line_y <= y_max)
    if in_range_mask.any():
        fig.add_trace(go.Scatter(
            x=line_x[in_range_mask], y=line_y[in_range_mask],
            mode="lines",
            name="로지스틱 회귀 경계선(0.5)",
            line=dict(color="black", width=2, dash="dash")
        ))
        line_drawn = True

if not line_drawn:
    st.warning("⚠️ 로지스틱 회귀의 0.5 경계선이 그림 범위 밖에 있어 표시되지 않았습니다.")

fig.update_layout(
    title=f"{x_kor} vs {y_kor} 산점도와 결정 경계",
    xaxis_title=x_kor,
    yaxis_title=y_kor,
    legend_title="실제 뇌졸중 여부"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ------------------------------
# 8. 트리 시각화 (graphviz DOT)
# ------------------------------
st.header("4️⃣ 의사결정트리 구조")

tree_ = tree_model.tree_
feature_names = selected_features

def build_dot(tree_, feature_names, kor_names):
    dot_lines = ["digraph Tree {", 'node [shape=box, style="filled", fontname="Malgun Gothic"];']

    def recurse(node_id):
        n_samples = tree_.n_node_samples[node_id]
        value = tree_.value[node_id][0]
        n_pos = int(value[1]) if len(value) > 1 else 0
        ratio = n_pos / n_samples if n_samples > 0 else 0

        is_leaf = tree_.children_left[node_id] == tree_.children_right[node_id]

        if is_leaf:
            pred_class = np.argmax(value)
            label_text = "뇌졸중" if pred_class == 1 else "정상"
            color = "#F4CCCC" if pred_class == 1 else "#CCE5FF"
            label = f"인원 {n_samples}명\\n뇌졸중 {n_pos}명 ({ratio*100:.1f}%)\\n답: {label_text}"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="{color}"];')
        else:
            feature_idx = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_kor = kor_names[feature_names[feature_idx]]
            label = f"{feature_kor} <= {threshold:.2f} ?\\n인원 {n_samples}명\\n뇌졸중 {n_pos}명 ({ratio*100:.1f}%)"
            dot_lines.append(f'{node_id} [label="{label}", fillcolor="#FFFFFF"];')

            left_id = tree_.children_left[node_id]
            right_id = tree_.children_right[node_id]

            recurse(left_id)
            recurse(right_id)

            dot_lines.append(f'{node_id} -> {left_id} [label="예"];')
            dot_lines.append(f'{node_id} -> {right_id} [label="아니요"];')

    recurse(0)
    dot_lines.append("}")
    return "\n".join(dot_lines)

dot_str = build_dot(tree_, feature_names, col_kor)
st.graphviz_chart(dot_str)

st.divider()

# ------------------------------
# 9. 트리 요약 정보
# ------------------------------
st.header("5️⃣ 트리 요약")

leaf_ids = [i for i in range(tree_.node_count) if tree_.children_left[i] == tree_.children_right[i]]
total_leaves = len(leaf_ids)

neg_leaves = 0
for leaf_id in leaf_ids:
    value = tree_.value[leaf_id][0]
    pred_class = np.argmax(value)
    if pred_class == 0:
        neg_leaves += 1

used_features_idx = set(tree_.feature[tree_.feature >= 0])
used_features = [feature_names[i] for i in used_features_idx]
used_features_kor = [col_kor[f] for f in used_features]

st.write(f"- 답을 내는 마디(잎)는 모두 **{total_leaves}칸**이고, 그중 **{neg_leaves}칸**이 '정상(아님)'이라고 답합니다.")
st.write(f"- 고른 속성 가운데 이 나무가 실제로 물어본 속성은: **{', '.join(used_features_kor)}** 입니다.")
