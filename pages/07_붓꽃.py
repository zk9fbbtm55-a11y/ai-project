
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier

# 페이지 설정
st.set_page_config(
    page_title="붓꽃 품종 예측 프로그램",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 붓꽃 품종 예측 프로그램")
st.write(
    "꽃잎의 길이와 너비를 입력하면 "
    "머신러닝 알고리즘이 붓꽃의 품종을 예측합니다."
)

# ---------------------------------
# 1. 데이터 불러오기
# ---------------------------------

@st.cache_data
def load_data():
    try:
        data = pd.read_csv("iris.csv", encoding="cp949")
    except UnicodeDecodeError:
        data = pd.read_csv("iris.csv", encoding="utf-8-sig")

    data.columns = data.columns.str.strip()
    return data

try:
    df = load_data()

    required_columns = [
        "꽃잎 길이",
        "꽃잎 너비",
        "품종"
    ]

    if not all(col in df.columns for col in required_columns):
        st.error("CSV 파일의 컬럼명을 확인해 주세요.")
        st.stop()

    df = df.dropna(subset=required_columns)

    # ---------------------------------
    # 2. 학습 데이터 설정
    # ---------------------------------

    X = df[["꽃잎 길이", "꽃잎 너비"]]
    y = df["품종"]

    # 랜덤 포레스트 모델
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    # 실제 데이터의 입력 범위
    min_length = float(X["꽃잎 길이"].min())
    max_length = float(X["꽃잎 길이"].max())

    min_width = float(X["꽃잎 너비"].min())
    max_width = float(X["꽃잎 너비"].max())

    # ---------------------------------
    # 3. 사용자 입력
    # ---------------------------------

    st.subheader("✏️ 꽃잎 정보 입력")

    st.info(
        f"""
        입력 가능한 범위

        - 꽃잎 길이: {min_length} ~ {max_length} cm
        - 꽃잎 너비: {min_width} ~ {max_width} cm

        ※ Iris 데이터의 실제 측정 범위를 기준으로 합니다.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        petal_length = st.number_input(
            "꽃잎 길이 (cm)",
            min_value=min_length,
            max_value=max_length,
            value=3.0,
            step=0.1,
            format="%.1f"
        )

    with col2:
        petal_width = st.number_input(
            "꽃잎 너비 (cm)",
            min_value=min_width,
            max_value=max_width,
            value=1.0,
            step=0.1,
            format="%.1f"
        )

    # ---------------------------------
    # 4. 예측 버튼
    # ---------------------------------

    if st.button("🌸 품종 예측하기", use_container_width=True):

        input_data = pd.DataFrame(
            [[petal_length, petal_width]],
            columns=["꽃잎 길이", "꽃잎 너비"]
        )

        # 품종 예측
        prediction = model.predict(input_data)[0]

        # 품종별 예측 확률
        probabilities = model.predict_proba(input_data)[0]

        st.divider()

        st.subheader("🔍 예측 결과")

        st.success(f"예측된 붓꽃 품종: {prediction}")

        st.write(
            f"입력한 꽃잎 길이: **{petal_length} cm**"
        )
        st.write(
            f"입력한 꽃잎 너비: **{petal_width} cm**"
        )

        # ---------------------------------
        # 5. 품종별 확률
        # ---------------------------------

        st.subheader("📊 품종별 예측 확률")

        probability_df = pd.DataFrame({
            "품종": model.classes_,
            "예측 확률": probabilities
        })

        probability_df["예측 확률"] *= 100

        st.bar_chart(
            probability_df.set_index("품종"),
            y="예측 확률"
        )

        for species, probability in zip(
            model.classes_, probabilities
        ):
            st.write(
                f"**{species}** : {probability * 100:.2f}%"
            )

        # ---------------------------------
        # 6. 분류 영역 시각화
        # ---------------------------------

        st.divider()
        st.subheader("🎨 품종 분류 영역 시각화")

        # 그래프 영역 생성
        x_min = min_length - 0.2
        x_max = max_length + 0.2
        y_min = min_width - 0.2
        y_max = max_width + 0.2

        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.02),
            np.arange(y_min, y_max, 0.02)
        )

        grid = pd.DataFrame({
            "꽃잎 길이": xx.ravel(),
            "꽃잎 너비": yy.ravel()
        })

        # 각 영역의 품종 예측
        Z = model.predict(grid)

        # 품종을 숫자로 변환
        class_map = {
            species: i
            for i, species in enumerate(model.classes_)
        }

        Z = np.array([
            class_map[value] for value in Z
        ]).reshape(xx.shape)

        colors = [
            "#F7B7B2",
            "#B8D9F5",
            "#C7E9C0"
        ]

        cmap = ListedColormap(colors[:len(model.classes_)])

        fig, ax = plt.subplots(figsize=(10, 7))

        # 배경 분류 영역
        ax.contourf(
            xx,
            yy,
            Z,
            alpha=0.45,
            cmap=cmap,
            levels=np.arange(len(model.classes_) + 1) - 0.5
        )

        # 실제 데이터 표시
        for i, species in enumerate(model.classes_):
            subset = df[df["품종"] == species]

            ax.scatter(
                subset["꽃잎 길이"],
                subset["꽃잎 너비"],
                label=species,
                s=45,
                alpha=0.8,
                edgecolors="black",
                linewidths=0.4
            )

        # 사용자 입력값: 초록색 별
        ax.scatter(
            petal_length,
            petal_width,
            marker="*",
            s=450,
            color="lime",
            edgecolors="black",
            linewidths=1.5,
            label="Your Input",
            zorder=10
        )

        ax.set_xlabel("Petal Length (cm)", fontsize=12)
        ax.set_ylabel("Petal Width (cm)", fontsize=12)

        ax.set_title(
            f"Random Forest Classification: {prediction}",
            fontsize=15
        )

        ax.legend()
        ax.grid(alpha=0.2)

        st.pyplot(fig)
        plt.close(fig)

        # ---------------------------------
        # 7. 분류 이유 설명
        # ---------------------------------

        st.subheader("💡 왜 이 품종으로 분류되었을까?")

        st.write(
            f"""
            사용자가 입력한 꽃잎 길이는 **{petal_length} cm**,
            꽃잎 너비는 **{petal_width} cm**입니다.

            그래프에서 초록색 별이 위치한 곳은
            **{prediction}** 품종으로 분류되는 영역입니다.

            랜덤 포레스트는 여러 결정 트리의 예측을 종합하여
            가장 높은 투표를 받은 품종을 최종 결과로 선택합니다.

            아래 확률 그래프를 통해 각 품종으로 분류될
            예측 확률도 비교할 수 있습니다.
            """
        )

except FileNotFoundError:
    st.error(
        "iris.csv 파일을 찾을 수 없습니다. "
        "app.py와 같은 GitHub 저장소에 iris.csv를 올려 주세요."
    )

except Exception as e:
    st.error(f"오류가 발생했습니다: {e}")

