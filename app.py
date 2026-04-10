import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os
import requests
import base64
import time

st.set_page_config(page_title="Floor Texture Replacer", page_icon="🏠", layout="centered")

TEXTURES = {
    "MKSC-01": "textures/MKSC-01.png",
    "MKSC-03": "textures/MKSC-03.png",
    "MKSC-05": "textures/MKSC-05.png",
    "MKSC-07": "textures/MKSC-07.png",
    "MKSC-09": "textures/MKSC-09.png",
    "MKSC-10": "textures/MKSC-10.png",
    "MKSC-11": "textures/MKSC-11.png",
    "MKSC-12": "textures/MKSC-12.png",
}

MAX_FILE_SIZE_MB = 10
MAX_DIMENSION    = 768   # kelipatan 64 untuk SD
ALLOWED_TYPES    = {"jpg", "jpeg", "png"}

# Model: SD 1.5 inpainting + ControlNet tile
# ControlNet tile membaca texture reference → terapkan ke area inpaint
HF_INPAINT_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"

# ── MODEL SEGMENTASI ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists("best.onnx"):
        st.error("Model best.onnx tidak ditemukan.")
        st.stop()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession("best.onnx", sess_options=opts,
                                providers=["CPUExecutionProvider"])

session    = load_model()
input_name = session.get_inputs()[0].name

# ── HELPERS ──────────────────────────────────────────────────────────────────
def pil_to_b64(img_pil, fmt="PNG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()

def validate_image(f):
    if f is None:
        return None, "File tidak ditemukan."
    ext = f.name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        return None, f"Format tidak didukung: .{ext}."
    if f.size / 1024 / 1024 > MAX_FILE_SIZE_MB:
        return None, f"File terlalu besar. Maksimal {MAX_FILE_SIZE_MB} MB."
    try:
        img = Image.open(f).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_DIMENSION:
            s = MAX_DIMENSION / max(w, h)
            img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        # Pastikan kelipatan 8 untuk SD
        w, h = img.size
        w = (w // 8) * 8
        h = (h // 8) * 8
        img = img.resize((w, h), Image.LANCZOS)
        if w < 64 or h < 64:
            return None, f"Gambar terlalu kecil ({w}x{h})."
        return img, None
    except Exception:
        return None, "File rusak atau bukan gambar valid."

def resize_preview(img_pil, max_side=1280):
    w, h = img_pil.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img_pil = img_pil.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img_pil

# ── MASK DETECTION ───────────────────────────────────────────────────────────
def preprocess_image(img_bgr, imgsz=640):
    img = cv2.resize(img_bgr, (imgsz, imgsz))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

def get_floor_mask(sess, img_bgr, conf_threshold=0.25):
    orig_h, orig_w = img_bgr.shape[:2]
    imgsz = 640
    outputs    = sess.run(None, {input_name: preprocess_image(img_bgr, imgsz)})
    detections = outputs[0][0].transpose(1, 0)
    proto      = outputs[1][0]

    best_det, best_area = None, 0
    for det in detections:
        if float(det[4]) < conf_threshold:
            continue
        area = float(det[2]) * float(det[3])
        if area > best_area:
            best_area, best_det = area, det

    if best_det is None:
        return None

    cx, cy, w, h = [float(best_det[i]) for i in range(4)]
    mask_coef    = best_det[5:37]
    mask_raw     = np.clip(np.einsum('c,chw->hw', mask_coef, proto), -10, 10)
    mask_sig     = 1 / (1 + np.exp(-mask_raw))
    mask_sig     = cv2.GaussianBlur(mask_sig, (7, 7), 0)

    x1 = max(0,   int((cx - w/2) / imgsz * 160))
    y1 = max(0,   int((cy - h/2) / imgsz * 160))
    x2 = min(160, int((cx + w/2) / imgsz * 160))
    y2 = min(160, int((cy + h/2) / imgsz * 160))
    if x2 <= x1 or y2 <= y1:
        return None

    mask_crop = np.zeros((160, 160), dtype=np.float32)
    mask_crop[y1:y2, x1:x2] = mask_sig[y1:y2, x1:x2]
    mask_full = cv2.resize(mask_crop, (imgsz, imgsz))
    mask_orig = cv2.resize(mask_full, (orig_w, orig_h))
    binary    = (mask_orig > 0.65).astype(np.uint8)

    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, ke)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  ko)
    binary = _remove_noise(binary)
    binary = _largest_component(binary)
    binary = (cv2.GaussianBlur(binary.astype(np.float32), (15,15), 0) > 0.5).astype(np.uint8)
    return binary

def _largest_component(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    return (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)

def _remove_noise(mask, min_area=5000):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            out[labels == i] = 1
    return out

# ── TEXTURE → PROMPT ─────────────────────────────────────────────────────────
def texture_to_prompt(texture_pil):
    """
    Analisis warna dominan texture untuk buat prompt yang akurat.
    """
    arr  = np.array(texture_pil.resize((50, 50)))
    mean = arr.mean(axis=(0,1))  # RGB mean
    r, g, b = mean

    # Tentukan tone warna
    brightness = (r + g + b) / 3
    if brightness > 200:
        tone = "very light white"
    elif brightness > 170:
        tone = "light"
    elif brightness > 130:
        tone = "medium"
    elif brightness > 90:
        tone = "dark"
    else:
        tone = "very dark"

    # Tentukan hue
    if r > g and r > b:
        hue = "warm reddish brown"
    elif g > r and g > b:
        hue = "greenish"
    elif b > r and b > g:
        hue = "cool gray"
    elif abs(r-g) < 20 and abs(g-b) < 20:
        hue = "neutral gray"
    else:
        hue = "beige"

    return (
        f"photorealistic {tone} {hue} wood floor planks, "
        f"natural wood grain texture, same perspective and lighting as the room, "
        f"high quality interior photography, realistic floor material, 8k"
    )

# ── HF INPAINTING API ────────────────────────────────────────────────────────
def call_hf_inpainting(hf_token, img_pil, mask_pil, prompt,
                       negative_prompt="", max_retries=3):
    """
    Panggil HF Inference API untuk inpainting.
    Otomatis retry jika model masih loading (503).
    """
    headers = {"Authorization": f"Bearer {hf_token}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt,
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 0.95,
        },
        "image":      pil_to_b64(img_pil),
        "mask_image": pil_to_b64(mask_pil),
    }

    for attempt in range(max_retries):
        response = requests.post(HF_API_URL, headers=headers,
                                 json=payload, timeout=120)

        if response.status_code == 200:
            result = Image.open(io.BytesIO(response.content)).convert("RGB")
            # Resize result ke ukuran input jika berbeda
            if result.size != img_pil.size:
                result = result.resize(img_pil.size, Image.LANCZOS)
            return result, None

        elif response.status_code == 503:
            wait = 30 * (attempt + 1)
            st.warning(f"⏳ Model masih loading... tunggu {wait} detik (attempt {attempt+1}/{max_retries})")
            time.sleep(wait)
            continue

        else:
            return None, f"API error {response.status_code}: {response.text[:300]}"

    return None, "Model tidak respond setelah beberapa percobaan. Coba lagi dalam 1-2 menit."

# ── COMPOSITE: TEMPEL HASIL AI KE FOTO ASLI ──────────────────────────────────
def composite_result(original_pil, result_pil, mask_np, feather=15):
    """
    Tempel area hasil AI ke foto asli menggunakan mask.
    Ini memastikan area di luar lantai tetap persis sama.
    """
    orig = np.array(original_pil).astype(np.float32)
    res  = np.array(result_pil.resize(original_pil.size, Image.LANCZOS)).astype(np.float32)

    # Feather mask dengan blur
    k      = feather * 2 + 1
    mask_f = cv2.GaussianBlur((mask_np * 255).astype(np.uint8), (k, k), 0).astype(np.float32) / 255.0

    # Blend per channel — hindari numpy broadcasting issue
    out = orig.copy()
    for c in range(3):
        out[:,:,c] = np.clip(
            mask_f * res[:,:,c] + (1.0 - mask_f) * orig[:,:,c],
            0, 255
        )

    return Image.fromarray(out.astype(np.uint8))

# ── UI ───────────────────────────────────────────────────────────────────────
st.title("🏠 Floor Texture Replacer")
st.write("Upload foto ruangan, pilih tekstur lantai, lalu lihat hasilnya.")

# Token input
with st.expander("🔑 Setup API Token", expanded=True):
    hf_token = st.text_input("Hugging Face Token", type="password",
                              placeholder="hhf_GbAdbMtEureKxfrcUgYwwhIVSagJhpWYdg")

room_file = st.file_uploader("📷 Upload foto ruangan (JPG/PNG, maks 10 MB)",
                              type=list(ALLOWED_TYPES))

st.subheader("Pilih tekstur lantai")
cols             = st.columns(4)
selected_texture = st.session_state.get("selected_texture", "MKSC-01")

for i, (name, path) in enumerate(TEXTURES.items()):
    with cols[i % 4]:
        if os.path.exists(path):
            st.image(path, caption=name, use_container_width=True)
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state["selected_texture"] = name
            selected_texture = name

st.info(f"Tekstur dipilih: **{selected_texture}**")

with st.expander("⚙️ Pengaturan lanjutan"):
    conf_threshold = st.slider("Sensitivitas deteksi", 0.10, 0.90, 0.25, 0.05)
    custom_prompt  = st.text_area("Custom prompt (kosongkan = auto dari texture)",
                                  value="")
    negative_prompt = st.text_input(
        "Negative prompt",
        value="blurry, low quality, distorted, people, furniture, carpet, rug, tiles"
    )
    feather_radius = st.slider("Kelembutan tepi blend", 0, 40, 15, 5)

if room_file and hf_token:
    room_img, err = validate_image(room_file)
    if err:
        st.error(err); st.stop()

    st.image(resize_preview(room_img), caption="Foto yang diupload", use_container_width=True)
    room_bgr = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2BGR)

    if st.button("🎨 Terapkan Tekstur", type="primary", use_container_width=True):

        # Step 1: Deteksi lantai
        with st.spinner("🔍 Mendeteksi lantai..."):
            mask = get_floor_mask(session, room_bgr, conf_threshold=conf_threshold)

        if mask is None or mask.sum() == 0:
            st.warning("⚠️ Lantai tidak terdeteksi. Coba turunkan sensitivitas deteksi.")
            st.stop()

        floor_pct = mask.sum() / mask.size * 100
        st.success(f"✅ Lantai terdeteksi ({floor_pct:.1f}% area gambar)")

        with st.expander("🔬 Lihat mask deteksi lantai"):
            ov = np.array(room_img).copy()
            ov[mask > 0] = (ov[mask > 0] * 0.5 + np.array([0,255,0]) * 0.5).astype(np.uint8)
            st.image(resize_preview(Image.fromarray(ov)),
                     caption="Area lantai terdeteksi (hijau)", use_container_width=True)

        # Step 2: Load texture dan buat prompt
        tex_path = TEXTURES[selected_texture]
        if not os.path.exists(tex_path):
            st.error(f"File texture {selected_texture} tidak ditemukan.")
            st.stop()

        tex_pil = Image.open(tex_path).convert("RGB")

        if custom_prompt.strip():
            prompt = custom_prompt.strip()
        else:
            prompt = texture_to_prompt(tex_pil)

        st.write(f"📝 **Prompt:** `{prompt}`")

        # Step 3: Siapkan mask (putih = area yang di-generate)
        mask_resized = cv2.resize(mask, (room_img.size[0], room_img.size[1]),
                                  interpolation=cv2.INTER_NEAREST)
        mask_pil = Image.fromarray((mask_resized * 255).astype(np.uint8)).convert("RGB")

        # Step 4: Panggil HF inpainting
        with st.spinner("🤖 AI sedang generate lantai... (30-90 detik pertama kali)"):
            result_pil, err = call_hf_inpainting(
                hf_token, room_img, mask_pil,
                prompt, negative_prompt
            )

        if err:
            st.error(f"❌ {err}")
            st.info("💡 Tips: Jika error 503, model masih loading — tunggu 1 menit lalu coba lagi.")
            st.stop()

        # Step 5: Composite hasil AI ke foto asli dengan mask
        final_pil = composite_result(room_img, result_pil, mask_resized, feather=feather_radius)

        # Tampilkan hasil
        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_preview(room_img), caption="Original", use_container_width=True)
        with col2:
            st.image(resize_preview(final_pil), caption=f"Tekstur {selected_texture} (AI)",
                     use_container_width=True)

        # Juga tampilkan raw AI output untuk perbandingan
        with st.expander("🔬 Raw AI output (sebelum composite)"):
            st.image(resize_preview(result_pil), caption="Output langsung dari AI",
                     use_container_width=True)

        # Download
        buf = io.BytesIO()
        final_pil.save(buf, "JPEG", quality=95)
        st.download_button("⬇️ Download hasil", buf.getvalue(),
                           f"floor_{selected_texture}_ai.jpg", "image/jpeg",
                           use_container_width=True)

elif room_file and not hf_token:
    st.warning("⚠️ Masukkan Hugging Face API token dulu di bagian atas.")
elif not room_file and hf_token:
    st.info("📷 Upload foto ruangan untuk mulai.")