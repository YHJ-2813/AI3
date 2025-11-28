# streamlit_py
import os, re
from io import BytesIO
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
from fastai.vision.all import *
import gdown

# ======================
# 페이지/스타일
# ======================
st.set_page_config(page_title="Fastai 이미지 분류기", page_icon="🤖", layout="wide")
st.markdown("""
<style>
h1 { color:#1E88E5; text-align:center; font-weight:800; letter-spacing:-0.5px; }
.prediction-box { background:#E3F2FD; border:2px solid #1E88E5; border-radius:12px; padding:22px; text-align:center; margin:16px 0; box-shadow:0 4px 10px rgba(0,0,0,.06);}
.prediction-box h2 { color:#0D47A1; margin:0; font-size:2.0rem; }
.prob-card { background:#fff; border-radius:10px; padding:12px 14px; margin:10px 0; box-shadow:0 2px 6px rgba(0,0,0,.06); }
.prob-bar-bg { background:#ECEFF1; border-radius:6px; width:100%; height:22px; overflow:hidden; }
.prob-bar-fg { background:#4CAF50; height:100%; border-radius:6px; transition:width .5s; }
.prob-bar-fg.highlight { background:#FF6F00; }
.info-grid { display:grid; grid-template-columns:repeat(12,1fr); gap:14px; }
.card { border:1px solid #e3e6ea; border-radius:12px; padding:14px; background:#fff; box-shadow:0 2px 6px rgba(0,0,0,.05); }
.card h4 { margin:0 0 10px; font-size:1.05rem; color:#0D47A1; }
.thumb { width:100%; height:auto; border-radius:10px; display:block; }
.thumb-wrap { position:relative; display:block; }
.play { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:60px; height:60px; border-radius:50%; background:rgba(0,0,0,.55); }
.play:after{ content:''; border-style:solid; border-width:12px 0 12px 20px; border-color:transparent transparent transparent #fff; position:absolute; top:50%; left:50%; transform:translate(-40%,-50%); }
.helper { color:#607D8B; font-size:.9rem; }
.stFileUploader, .stCameraInput { border:2px dashed #1E88E5; border-radius:12px; padding:16px; background:#f5fafe; }
</style>
""", unsafe_allow_html=True)

st.title("이미지 분류기 (Fastai) — 확률 막대 + 라벨별 고정 콘텐츠")

# ======================
# 세션 상태
# ======================
if "img_bytes" not in st.session_state:
    st.session_state.img_bytes = None
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

# ======================
# 모델 로드
# ======================
FILE_ID = st.secrets.get("GDRIVE_FILE_ID", "1F5RhQzxeztcU7kJpPxG72YlRiTf8Ly0X")
MODEL_PATH = st.secrets.get("MODEL_PATH", "model.pkl")

@st.cache_resource
def load_model_from_drive(file_id: str, output_path: str):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    return load_learner(output_path, cpu=True)

with st.spinner("🤖 모델 로드 중..."):
    learner = load_model_from_drive(FILE_ID, MODEL_PATH)
st.success("✅ 모델 로드 완료")

labels = [str(x) for x in learner.dls.vocab]
st.write(f"**분류 가능한 항목:** `{', '.join(labels)}`")
st.markdown("---")

# ======================
# 라벨 이름 매핑: 여기를 채우세요!
# 각 라벨당 최대 3개씩 표시됩니다.
# ======================
CONTENT_BY_LABEL: dict[str, dict[str, list[str]]] = {
    # 예)
    # "짬뽕": {
    #   "texts": ["짬뽕의 특징과 유래", "국물 맛 포인트", "지역별 스타일 차이"],
    #   "images": ["https://.../jjampong1.jpg", "https://.../jjampong2.jpg"],
    #   "videos": ["https://youtu.be/XXXXXXXXXXX"]
    # },

    labels[0] : {"texts" : ["지수함수는 멋있어"],"images" : ["data:image/png;base64,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"],"videos": ["https://youtu.be/zkLE2pBIhjs?si=__WHcExaW8ygToOf","https://youtu.be/FBAgxbQ931Y?si=45oUFJbNxBRfyJ09"]}, 
    labels[1] : {"texts" : ["삼각함수는 멋있어"],"images" : ["data:image/png;base64,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"],"videos": ["https://youtu.be/n8dsNx3GSI8?si=G3qaAFfCqZAG6QCn","https://youtu.be/T4V4vzQsOZ4?si=V39Jpmha_LXsrMym"]},
    labels[2] : {"texts" : ["로그함수는 멋있어"],"images" : ["data:image/png;base64,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"],"videos": ["https://youtu.be/I_H04p9HHcI?si=aq3OyT7TBlDsDaFX","https://youtu.be/6Ht8VZGZO5o?si=SqSj7XPwejQgHLZ7"]},
}

# ======================
# 유틸
# ======================
def load_pil_from_bytes(b: bytes) -> Image.Image:
    pil = Image.open(BytesIO(b))
    pil = ImageOps.exif_transpose(pil)
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return pil

def yt_id_from_url(url: str) -> str | None:
    if not url: return None
    pats = [r"(?:v=|/)([0-9A-Za-z_-]{11})(?:\?|&|/|$)", r"youtu\.be/([0-9A-Za-z_-]{11})"]
    for p in pats:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def yt_thumb(url: str) -> str | None:
    vid = yt_id_from_url(url)
    return f"https://img.youtube.com/vi/{vid}/hqdefault.jpg" if vid else None

def pick_top3(lst):
    return [x for x in lst if isinstance(x, str) and x.strip()][:3]

def get_content_for_label(label: str):
    """라벨명으로 콘텐츠 반환 (texts, images, videos). 없으면 빈 리스트."""
    cfg = CONTENT_BY_LABEL.get(label, {})
    return (
        pick_top3(cfg.get("texts", [])),
        pick_top3(cfg.get("images", [])),
        pick_top3(cfg.get("videos", [])),
    )

# ======================
# 입력(카메라/업로드)
# ======================
tab_cam, tab_file = st.tabs(["📷 카메라로 촬영", "📁 파일 업로드"])
new_bytes = None

with tab_cam:
    cam = st.camera_input("카메라 스냅샷", label_visibility="collapsed")
    if cam is not None:
        new_bytes = cam.getvalue()

with tab_file:
    f = st.file_uploader("이미지를 업로드하세요 (jpg, png, jpeg, webp, tiff)",
                         type=["jpg","png","jpeg","webp","tiff"])
    if f is not None:
        new_bytes = f.getvalue()

if new_bytes:
    st.session_state.img_bytes = new_bytes

# ======================
# 예측 & 레이아웃
# ======================
if st.session_state.img_bytes:
    top_l, top_r = st.columns([1, 1], vertical_alignment="center")

    pil_img = load_pil_from_bytes(st.session_state.img_bytes)
    with top_l:
        st.image(pil_img, caption="입력 이미지", use_container_width=True)

    with st.spinner("🧠 분석 중..."):
        pred, pred_idx, probs = learner.predict(PILImage.create(np.array(pil_img)))
        st.session_state.last_prediction = str(pred)

    with top_r:
        st.markdown(
            f"""
            <div class="prediction-box">
                <span style="font-size:1.0rem;color:#555;">예측 결과:</span>
                <h2>{st.session_state.last_prediction}</h2>
                <div class="helper">오른쪽 패널에서 예측 라벨의 콘텐츠가 표시됩니다.</div>
            </div>
            """, unsafe_allow_html=True
        )

    left, right = st.columns([1,1], vertical_alignment="top")

    # 왼쪽: 확률 막대
    with left:
        st.subheader("상세 예측 확률")
        prob_list = sorted(
            [(labels[i], float(probs[i])) for i in range(len(labels))],
            key=lambda x: x[1], reverse=True
        )
        for lbl, p in prob_list:
            pct = p * 100
            hi = "highlight" if lbl == st.session_state.last_prediction else ""
            st.markdown(
                f"""
                <div class="prob-card">
                  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <strong>{lbl}</strong><span>{pct:.2f}%</span>
                  </div>
                  <div class="prob-bar-bg">
                    <div class="prob-bar-fg {hi}" style="width:{pct:.4f}%;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True
            )

    # 오른쪽: 정보 패널 (예측 라벨 기본, 다른 라벨로 바꿔보기 가능)
    with right:
        st.subheader("라벨별 고정 콘텐츠")
        default_idx = labels.index(st.session_state.last_prediction) if st.session_state.last_prediction in labels else 0
        info_label = st.selectbox("표시할 라벨 선택", options=labels, index=default_idx)

        texts, images, videos = get_content_for_label(info_label)

        if not any([texts, images, videos]):
            st.info(f"라벨 `{info_label}`에 대한 콘텐츠가 아직 없습니다. 코드의 CONTENT_BY_LABEL에 추가하세요.")
        else:
            # 텍스트
            if texts:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for t in texts:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 12;">
                      <h4>텍스트</h4>
                      <div>{t}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 이미지(최대 3, 3열)
            if images:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for url in images[:3]:
                    st.markdown(f"""
                    <div class="card" style="grid-column:span 4;">
                      <h4>이미지</h4>
                      <img src="{url}" class="thumb" />
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # 동영상(유튜브 썸네일)
            if videos:
                st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                for v in videos[:3]:
                    thumb = yt_thumb(v)
                    if thumb:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank" class="thumb-wrap">
                            <img src="{thumb}" class="thumb"/>
                            <div class="play"></div>
                          </a>
                          <div class="helper">{v}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="card" style="grid-column:span 6;">
                          <h4>동영상</h4>
                          <a href="{v}" target="_blank">{v}</a>
                        </div>
                        """, unsafe_allow_html=True)
else:
    st.info("카메라로 촬영하거나 파일을 업로드하면 분석 결과와 라벨별 콘텐츠가 표시됩니다.")

