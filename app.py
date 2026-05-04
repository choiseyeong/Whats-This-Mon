"""
Pokemon Classifier - Streamlit Demo

사용법:
  pip install -r requirements.txt
  streamlit run app.py

실행 전 준비:
  pokemon_classification.ipynb 마지막 셀에서 저장한 best_model.pt를
  app.py와 같은 폴더에 둘 것.
"""

from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).parent / "best_model.pt"


@st.cache_resource
def load_model(model_path: Path):
    if not model_path.exists():
        return None, None, None

    ckpt = torch.load(model_path, map_location="cpu")
    class_names = ckpt["class_names"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt.get("img_size", 224)
    mean = ckpt.get("imagenet_mean", [0.485, 0.456, 0.406])
    std = ckpt.get("imagenet_std", [0.229, 0.224, 0.225])

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    meta = {
        "experiment": ckpt.get("experiment"),
        "experiment_name": ckpt.get("experiment_name"),
        "num_classes": num_classes,
    }
    return model, transform, class_names, meta


def predict_topk(model, transform, class_names, image: Image.Image, k: int = 5):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
    top_probs, top_idx = probs.topk(k)
    return [
        (class_names[i.item()], p.item())
        for p, i in zip(top_probs, top_idx)
    ]


def main():
    st.set_page_config(page_title="Pokemon Classifier", page_icon="⚡", layout="centered")
    st.title("Pokemon Classifier")
    st.caption("Transfer learning으로 학습한 ResNet18 기반 포켓몬 분류기")

    loaded = load_model(MODEL_PATH)
    if loaded[0] is None:
        st.error(
            f"모델 파일을 찾을 수 없습니다: `{MODEL_PATH.name}`\n\n"
            "노트북 마지막 셀을 실행해 `best_model.pt`를 생성한 뒤 이 폴더에 두세요."
        )
        return

    model, transform, class_names, meta = loaded

    with st.sidebar:
        st.header("Model info")
        st.write(f"**Best experiment:** Exp {meta['experiment']}")
        st.write(f"**Config:** {meta['experiment_name']}")
        st.write(f"**# classes:** {meta['num_classes']}")
        topk = st.slider("Top-K", 1, 10, 5)

    uploaded = st.file_uploader(
        "포켓몬 이미지 업로드",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded is None:
        st.info("이미지 파일을 업로드하면 top-K 예측이 표시됩니다.")
        return

    image = Image.open(uploaded)
    st.image(image, caption=uploaded.name, use_container_width=True)

    with st.spinner("예측 중..."):
        preds = predict_topk(model, transform, class_names, image, k=topk)

    st.subheader(f"Top-{topk} predictions")
    for rank, (name, prob) in enumerate(preds, start=1):
        st.write(f"**{rank}. {name}** — {prob*100:.2f}%")
        st.progress(min(max(prob, 0.0), 1.0))


if __name__ == "__main__":
    main()
