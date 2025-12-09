import os
import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

from animal_data import (
    ANIMAL_CATEGORIES,
    ANIMALS_DATA,
    get_animals_by_category,
    get_animal_detail,
)

# --------------------------
# Env
# --------------------------
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")

# --------------------------
# i18n
# --------------------------
LANGS = {
    "English": "en",
    "中文": "zh",
    "한국어": "ko",
}

T = {
    "app_title": {
        "en": "Animal ID & Encyclopedia",
        "zh": "动物识别与百科",
        "ko": "동물 인식 & 백과",
    },
    "nav_home": {"en": "Home", "zh": "首页", "ko": "홈"},
    "nav_pet": {"en": "Pet Identifier", "zh": "宠物识别", "ko": "반려동물 인식"},
    "nav_ency": {"en": "Animal Encyclopedia", "zh": "动物百科", "ko": "동물 백과"},
    "nav_about": {"en": "About", "zh": "关于", "ko": "소개"},

    "home_intro": {
        "en": "Upload a photo to identify pets and explore animals by category.",
        "zh": "上传照片识别宠物，按分类探索动物百科。",
        "ko": "사진을 업로드해 반려동물을 인식하고 분류별 동물을 탐색하세요.",
    },
    "pet_upload": {"en": "Upload a pet photo", "zh": "上传宠物照片", "ko": "반려동물 사진 업로드"},
    "pet_result": {"en": "Identification Result", "zh": "识别结果", "ko": "인식 결과"},
    "pet_tip": {
        "en": "Tip: Clear face/body photos work best.",
        "zh": "提示：宠物正脸或全身清晰照片效果最好。",
        "ko": "팁: 얼굴/전신이 선명한 사진이 가장 좋아요.",
    },
    "no_key_demo": {
        "en": "No API key found. Running in demo mode (no real AI call).",
        "zh": "未检测到 API Key，已进入演示模式（不会真实调用AI）。",
        "ko": "API 키가 없습니다. 데모 모드로 실행됩니다.",
    },
    "ency_pick_cat": {"en": "Choose a category", "zh": "选择分类", "ko": "분류 선택"},
    "ency_animals": {"en": "Animals", "zh": "动物列表", "ko": "동물 목록"},
    "detail": {"en": "Details", "zh": "详情", "ko": "상세"},
    "habitat": {"en": "Habitat", "zh": "栖息地", "ko": "서식지"},
    "facts": {"en": "Fun facts", "zh": "趣味事实", "ko": "재미있는 사실"},
    "about_text": {
        "en": "A lightweight Streamlit app for pet identification and animal knowledge.",
        "zh": "一个轻量级的 Streamlit 宠物识别与动物科普网站。",
        "ko": "반려동물 인식과 동물 지식을 위한 가벼운 Streamlit 앱입니다.",
    },
}

def tr(key, lang):
    return T.get(key, {}).get(lang, T.get(key, {}).get("en", key))

# --------------------------
# UI helpers
# --------------------------
def inject_css():
    st.markdown(
        """
        <style>
        /* Make sidebar a bit cleaner */
        [data-testid="stSidebar"] {
            padding-top: 1rem;
        }
        /* "Bottom-left" language box hack */
        .lang-footer {
            position: fixed;
            bottom: 14px;
            left: 14px;
            width: 220px;
            background: rgba(255,255,255,0.85);
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 10px;
            padding: 8px 10px 0 10px;
            z-index: 9999;
            backdrop-filter: blur(6px);
        }
        /* Improve image rounding */
        img {
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def image_to_data_url(img: Image.Image):
    # Convert PIL image to base64 data URL
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

# --------------------------
# AI (Pet identification)
# --------------------------
def identify_pet_with_qwen(image: Image.Image, lang: str):
    """
    Uses DashScope OpenAI-compatible endpoint.
    Falls back gracefully if missing key or error.
    """
    if not DASHSCOPE_API_KEY:
        return None, "NO_KEY"

    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )

    data_url = image_to_data_url(image)

    # Prompt in English (model usually handles multilingual output too)
    # We'll ask the model to respond in the chosen language.
    prompt = {
        "en": """You are a pet expert. Identify the pet in the photo.
Return:
1) Species/Breed (if confident)
2) Key visual cues
3) Likely age stage (baby/adult/senior)
4) Care tips (3-5 bullets)
5) Safety note if uncertain

If not a pet, say what the main subject is.""",
        "zh": """你是宠物专家。请识别照片中的宠物。
按以下结构输出：
1）物种/品种（有把握再写）
2）关键视觉依据
3）可能年龄阶段（幼年/成年/老年）
4）饲养与护理建议（3-5条）
5）不确定性与安全提示

如果不是宠物，请说明主要内容。""",
        "ko": """당신은 반려동물 전문가입니다. 사진 속 반려동물을 식별하세요.
다음 구조로 답변:
1) 종/품종(확신할 때만)
2) 핵심 시각적 근거
3) 추정 연령 단계(유/성/노)
4) 사육·관리 팁(3-5개)
5) 불확실성 및 안전 안내

반려동물이 아니면 주요 피사체를 설명하세요.""",
    }[lang]

    try:
        completion = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return completion.choices[0].message.content, None
    except Exception as e:
        return f"Error: {e}", "ERROR"

# --------------------------
# Pages
# --------------------------
def page_home(lang):
    st.title(tr("app_title", lang))
    st.write(tr("home_intro", lang))

    # Quick category preview
    cols = st.columns(3)
    items = list(ANIMAL_CATEGORIES.items())
    for i, (cid, cinfo) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"### {cinfo['icon']} {cinfo['name'][lang]}")
            st.caption(cinfo["description"][lang])
            st.caption(f"{len(get_animals_by_category(cid))} {tr('ency_animals', lang)}")

def page_pet_identifier(lang):
    st.header(f"🐾 {tr('nav_pet', lang)}")
    st.caption(tr("pet_tip", lang))

    uploaded = st.file_uploader(
        tr("pet_upload", lang),
        type=["png", "jpg", "jpeg", "webp"],
    )

    if not DASHSCOPE_API_KEY:
        st.warning(tr("no_key_demo", lang))

    if uploaded:
        try:
            img = Image.open(uploaded).convert("RGBA")
        except Exception:
            st.error("Invalid image file.")
            return

        st.image(img, use_container_width=True)

        if st.button(tr("pet_result", lang)):
            with st.spinner("Analyzing..."):
                result, err_flag = identify_pet_with_qwen(img, lang)

            if err_flag == "NO_KEY":
                st.info(tr("no_key_demo", lang))
                st.markdown(
                    {
                        "en": "Demo result: Looks like a pet photo. Add your API key to enable real identification.",
                        "zh": "演示结果：看起来是一张宠物照片。配置 API Key 后可进行真实识别。",
                        "ko": "데모 결과: 반려동물 사진처럼 보입니다. API 키를 설정하면 실제 인식이 가능합니다.",
                    }[lang]
                )
            else:
                st.subheader(tr("pet_result", lang))
                st.markdown(result)

def page_encyclopedia(lang):
    st.header(f"📚 {tr('nav_ency', lang)}")

    # Category selector
    cat_options = list(ANIMAL_CATEGORIES.keys())
    cat_labels = [f"{ANIMAL_CATEGORIES[c]['icon']} {ANIMAL_CATEGORIES[c]['name'][lang]}" for c in cat_options]

    label_to_id = {label: cid for label, cid in zip(cat_labels, cat_options)}

    chosen_label = st.selectbox(tr("ency_pick_cat", lang), cat_labels)
    category_id = label_to_id[chosen_label]
    category_info = ANIMAL_CATEGORIES[category_id]

    st.markdown(f"### {category_info['name'][lang]}")
    st.caption(category_info["description"][lang])

    animals = get_animals_by_category(category_id)

    # Simple grid cards
    animal_ids = list(animals.keys())
    cols = st.columns(3)

    selected_id = None
    for i, aid in enumerate(animal_ids):
        a = animals[aid]
        with cols[i % 3]:
            st.image(a["image"], use_container_width=True)
            st.markdown(f"**{a['name'][lang]}**")
            st.caption(a["scientific_name"])
            if st.button(tr("detail", lang), key=f"detail_{aid}"):
                selected_id = aid

    # Detail area
    if selected_id:
        animal = get_animal_detail(selected_id)
        st.divider()
        st.subheader(animal["name"][lang])
        st.image(animal["image"], width=520)
        st.caption(animal["scientific_name"])
        st.write(animal["summary"][lang])

        st.markdown(f"**{tr('habitat', lang)}**")
        st.write(animal["habitat"][lang])

        st.markdown(f"**{tr('facts', lang)}**")
        for f in animal["facts"][lang]:
            st.write(f"- {f}")

def page_about(lang):
    st.header(tr("nav_about", lang))
    st.write(tr("about_text", lang))
    st.markdown(
        {
            "en": "This project is designed to stay lightweight while keeping core features reliable.",
            "zh": "本项目目标是在保持轻量的前提下，让核心功能稳定可用。",
            "ko": "핵심 기능의 안정성을 유지하면서 가볍게 구성한 프로젝트입니다.",
        }[lang]
    )
    st.markdown(
        """
**Security note**
- Do NOT commit real API keys to GitHub.
- Use `.env` locally and keep it in `.gitignore`.
        """
    )

# --------------------------
# Main
# --------------------------
def main():
    st.set_page_config(
        page_title="Animal App",
        page_icon="🐾",
        layout="wide",
    )

    inject_css()

    # Language selector "bottom-left" attempt
    # We place it in sidebar but wrap with fixed CSS class.
    st.sidebar.markdown('<div class="lang-footer">', unsafe_allow_html=True)
    lang_label = st.sidebar.selectbox(
        "Language",
        list(LANGS.keys()),
        index=0,
        key="lang_select",
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    lang = LANGS[lang_label]

    # Navigation
    nav = st.sidebar.radio(
        "Navigation",
        [
            tr("nav_home", lang),
            tr("nav_pet", lang),
            tr("nav_ency", lang),
            tr("nav_about", lang),
        ],
        label_visibility="collapsed",
    )

    if nav == tr("nav_home", lang):
        page_home(lang)
    elif nav == tr("nav_pet", lang):
        page_pet_identifier(lang)
    elif nav == tr("nav_ency", lang):
        page_encyclopedia(lang)
    else:
        page_about(lang)

if __name__ == "__main__":
    main()
