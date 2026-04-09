import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os

# =============================================
# CONFIG
# =============================================
st.set_page_config(
    page_title="Floor Texture Replacer",
    page_icon="🏠",
    layout="centered"
)

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

MAX_FILE_SIZE_MB  = 10
MAX_DIMENSION     = 2048
ALLOWED_TYPES     = {"jpg", "jpeg", "png"}
TEXTURE_TILE_SIZE = 300

# =============================================
# MODEL
# =============================================
@st.cache_resource
def load_model():
    if not os.path.exists("best.onnx"):
        st.error("Model best.onnx tidak ditemukan.")
        st.stop()
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    return ort.InferenceSession(
        "best.onnx",
        sess_options=sess_options,
        providers=["CPUExecutionProvider"]
    )

session    = load_model()
input_name = session.get_inputs()[0].name

# =============================================
# VALIDATION
# =============================================
def validate_image(uploaded_file):
    if uploaded_file is None:
        return None, "File tidak ditemukan."
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        return None, f"Format tidak didukung: .{ext}. Gunakan JPG atau PNG."
    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"Ukuran file terlalu besar ({size_mb:.1f} MB). Maksimal {MAX_FILE_SIZE_MB} MB."
    try:
        img  = Image.open(uploaded_file).convert("RGB")
        w, h = img.size
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            scale = MAX_DIMENSION / max(w, h)
            img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        w, h = img.size
        if w < 64 or h < 64:
            return None, f"Gambar terlalu kecil ({w}x{h}). Minimal 64x64px."
        return img, None
    except Exception:
        return None, "File rusak atau bukan gambar yang valid."

def validate_texture(name):
    if name not in TEXTURES:
        return None, "Tekstur tidak valid."
    path = TEXTURES[name]
    if not os.path.exists(path):
        return None, f"File tekstur {name} tidak ditemukan di server."
    texture = cv2.imread(path)
    if texture is None:
        return None, f"File tekstur {name} tidak bisa dibaca."
    return texture, None

# =============================================
# PREPROCESSING
# =============================================
def preprocess_image(img_bgr, imgsz=640):
    img_resized    = cv2.resize(img_bgr, (imgsz, imgsz))
    img_rgb        = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm       = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)

# =============================================
# MASK DETECTION
# =============================================
def get_floor_mask(session, img_bgr, conf_threshold=0.25):
    orig_h, orig_w = img_bgr.shape[:2]
    imgsz = 640

    inp     = preprocess_image(img_bgr, imgsz)
    outputs = session.run(None, {input_name: inp})

    detections = outputs[0][0].transpose(1, 0)
    proto      = outputs[1][0]

    best_det  = None
    best_area = 0

    for det in detections:
        cx, cy, w, h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        cls_score    = float(det[4])
        if cls_score < conf_threshold:
            continue
        area = w * h
        if area > best_area:
            best_area = area
            best_det  = det

    if best_det is None:
        return None

    mask_combined = np.zeros((imgsz, imgsz), dtype=np.float32)
    det           = best_det
    cx, cy, w, h  = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    mask_coef     = det[5:37]

    mask_raw = np.einsum('c,chw->hw', mask_coef, proto)
    mask_raw = np.clip(mask_raw, -10, 10)
    mask_sig = 1 / (1 + np.exp(-mask_raw))
    mask_sig = cv2.GaussianBlur(mask_sig, (7, 7), 0)

    x1 = max(0,   int((cx - w / 2) / imgsz * 160))
    y1 = max(0,   int((cy - h / 2) / imgsz * 160))
    x2 = min(160, int((cx + w / 2) / imgsz * 160))
    y2 = min(160, int((cy + h / 2) / imgsz * 160))

    if x2 <= x1 or y2 <= y1:
        return None

    mask_crop               = np.zeros((160, 160), dtype=np.float32)
    mask_crop[y1:y2, x1:x2] = mask_sig[y1:y2, x1:x2]

    mask_full     = cv2.resize(mask_crop, (imgsz, imgsz))
    mask_combined = np.maximum(mask_combined, mask_full)

    mask_orig   = cv2.resize(mask_combined, (orig_w, orig_h))
    binary_mask = (mask_orig > 0.65).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    binary_mask  = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    binary_mask  = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel_open)
    binary_mask  = remove_small_noise(binary_mask, min_area=8000)
    binary_mask  = keep_largest_component(binary_mask)
    binary_mask  = cv2.GaussianBlur(binary_mask.astype(np.float32), (15, 15), 0)
    binary_mask  = (binary_mask > 0.5).astype(np.uint8)

    return binary_mask

def keep_largest_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

def remove_small_noise(mask, min_area=5000):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    new_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            new_mask[labels == i] = 1
    return new_mask

# =============================================
# TEXTURE TILING
# =============================================
def tile_texture(texture_bgr, target_w, target_h, tile_size=TEXTURE_TILE_SIZE):
    interp   = cv2.INTER_AREA if tile_size < texture_bgr.shape[1] else cv2.INTER_LANCZOS4
    tex_tile = cv2.resize(texture_bgr, (tile_size, tile_size), interpolation=interp)
    tiles_x  = -(-target_w // tile_size)
    tiles_y  = -(-target_h // tile_size)
    tiled    = np.tile(tex_tile, (tiles_y, tiles_x, 1))
    return tiled[:target_h, :target_w]

# =============================================
# FLOOR QUAD
# =============================================
def order_points(pts):
    rect    = np.zeros((4, 2), dtype=np.float32)
    s       = pts.sum(axis=1)
    diff    = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def get_floor_quad(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    hull    = cv2.convexHull(largest).reshape(-1, 2).astype(np.float32)
    if len(hull) < 4:
        rect = cv2.minAreaRect(largest)
        box  = cv2.boxPoints(rect).astype(np.float32)
        return order_points(box)
    tl = hull[np.argmin(hull[:, 0] + hull[:, 1])]
    tr = hull[np.argmin(-hull[:, 0] + hull[:, 1])]
    br = hull[np.argmax(hull[:, 0] + hull[:, 1])]
    bl = hull[np.argmax(-hull[:, 0] + hull[:, 1])]
    return order_points(np.array([tl, tr, br, bl], dtype=np.float32))

# =============================================
# AMBIENT OCCLUSION
# =============================================
def apply_ambient_occlusion(img_bgr, mask, intensity=0.30):
    mask_255 = (mask > 0).astype(np.uint8) * 255
    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        return img_bgr

    h_floor     = y_idx.max() - y_idx.min()
    w_floor     = x_idx.max() - x_idx.min()
    kernel_size = max(int(min(w_floor, h_floor) * 0.04), 11)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_eroded = cv2.erode(mask_255, kernel)
    edge_area   = cv2.subtract(mask_255, mask_eroded).astype(np.float32)

    blur_size = kernel_size * 2 - 1
    if blur_size % 2 == 0:
        blur_size += 1
    shadow_map = cv2.GaussianBlur(edge_area, (blur_size, blur_size), 0)
    shadow_max = shadow_map.max()
    if shadow_max < 0.01:
        return img_bgr

    shadow_map    = (shadow_map / shadow_max) * (edge_area > 0).astype(np.float32)
    shadow_3ch    = np.stack([shadow_map] * 3, axis=-1)
    darken_factor = np.clip(1.0 - intensity * shadow_3ch, 0.0, 1.0)
    result        = img_bgr.astype(np.float32) * darken_factor
    return np.clip(result, 0, 255).astype(np.uint8)

# =============================================
# CORE: APPLY TEXTURE
# Pendekatan: warp lantai ke flat space → tempel texture → warp balik
# Ini memastikan perspektif texture mengikuti lantai dengan benar
# =============================================
def apply_texture_perspective(img_bgr, mask, texture_bgr,
                              tile_size=TEXTURE_TILE_SIZE, feather_radius=15):
    img_bgr = img_bgr.copy()
    mask    = (mask > 0).astype(np.uint8)
    orig_h, orig_w = img_bgr.shape[:2]

    # Dapatkan 4 titik sudut lantai
    dst_pts = get_floor_quad(mask)
    if dst_pts is None:
        st.warning("Quad lantai tidak ditemukan.")
        return img_bgr

    # Hitung ukuran flat space dari panjang sisi quad
    w_top   = np.linalg.norm(dst_pts[1] - dst_pts[0])
    w_bot   = np.linalg.norm(dst_pts[2] - dst_pts[3])
    h_left  = np.linalg.norm(dst_pts[3] - dst_pts[0])
    h_right = np.linalg.norm(dst_pts[2] - dst_pts[1])
    flat_w  = max(int(max(w_top, w_bot)), 1)
    flat_h  = max(int(max(h_left, h_right)), 1)

    flat_pts = np.array([
        [0,          0         ],
        [flat_w - 1, 0         ],
        [flat_w - 1, flat_h - 1],
        [0,          flat_h - 1],
    ], dtype=np.float32)

    # Warp lantai asli ke flat space untuk ambil statistik warna
    M_to_flat  = cv2.getPerspectiveTransform(dst_pts, flat_pts)
    floor_flat = cv2.warpPerspective(img_bgr, M_to_flat, (flat_w, flat_h))

    # Tile texture di flat space
    texture_flat = tile_texture(texture_bgr, flat_w, flat_h, tile_size)

    # Transfer lighting di flat space (lebih akurat karena tidak ada distorsi)
    floor_lab = cv2.cvtColor(floor_flat, cv2.COLOR_BGR2LAB).astype(np.float32)
    tex_lab   = cv2.cvtColor(texture_flat, cv2.COLOR_BGR2LAB).astype(np.float32)

    orig_mean = floor_lab[:, :, 0].mean()
    orig_std  = floor_lab[:, :, 0].std() + 1e-6
    tex_mean  = tex_lab[:, :, 0].mean()
    tex_std   = tex_lab[:, :, 0].std() + 1e-6

    # Sesuaikan brightness, pertahankan sedikit warna asli texture
    tex_lab[:, :, 0] = np.clip(
        (tex_lab[:, :, 0] - tex_mean) / tex_std * orig_std + orig_mean, 0, 255
    )
    # Blend chroma: 70% texture asli + 30% warna lantai
    tex_lab[:, :, 1] = tex_lab[:, :, 1] * 0.7 + floor_lab[:, :, 1] * 0.3
    tex_lab[:, :, 2] = tex_lab[:, :, 2] * 0.7 + floor_lab[:, :, 2] * 0.3

    texture_lit_flat = cv2.cvtColor(
        np.clip(tex_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    # Warp texture yang sudah di-lit balik ke perspektif asli
    M_to_persp     = cv2.getPerspectiveTransform(flat_pts, dst_pts)
    texture_warped = cv2.warpPerspective(texture_lit_flat, M_to_persp, (orig_w, orig_h))

    # Tempel texture hanya di area mask
    result           = img_bgr.copy()
    result[mask > 0] = texture_warped[mask > 0]

    # Feather tepi mask agar transisi halus
    if feather_radius > 0:
        k       = feather_radius * 2 + 1
        mask_f  = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
        mask_f  = np.clip(mask_f, 0, 1)
        alpha_3 = np.stack([mask_f] * 3, axis=-1)
        result  = np.clip(
            alpha_3 * result.astype(np.float32) + (1 - alpha_3) * img_bgr.astype(np.float32),
            0, 255
        ).astype(np.uint8)

    # Ambient occlusion di tepi lantai
    result = apply_ambient_occlusion(result, mask)

    return result

# =============================================
# HELPER
# =============================================
def resize_for_preview(img_pil, max_side=1280):
    w, h = img_pil.size
    if max(w, h) > max_side:
        scale   = max_side / max(w, h)
        img_pil = img_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img_pil

# =============================================
# UI
# =============================================
st.title("🏠 Floor Texture Replacer")
st.write("Upload foto ruangan, pilih tekstur lantai, lalu lihat hasilnya.")

room_file = st.file_uploader(
    "📷 Upload foto ruangan (JPG/PNG, maks 10 MB)",
    type=list(ALLOWED_TYPES)
)

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
    conf_threshold = st.slider(
        "Sensitivitas deteksi", 0.10, 0.90, 0.25, 0.05,
        help="Turunkan jika lantai tidak terdeteksi."
    )
    tile_size = st.slider(
        "Ukuran tile tekstur (px)", 100, 600, TEXTURE_TILE_SIZE, 50,
        help="Kecil = serat lebih rapat. Besar = serat lebih kasar."
    )
    feather_radius = st.slider(
        "Kelembutan tepi", 0, 40, 15, 5,
        help="Semakin besar = tepi texture lebih halus/menyatu."
    )

if room_file:
    room_img, err = validate_image(room_file)
    if err:
        st.error(err)
        st.stop()

    texture_bgr, err = validate_texture(selected_texture)
    if err:
        st.error(err)
        st.stop()

    st.image(resize_for_preview(room_img), caption="Foto yang diupload", use_container_width=True)
    room_bgr = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2BGR)

    if st.button("🎨 Terapkan Tekstur", type="primary", use_container_width=True):

        with st.spinner("🔍 Mendeteksi lantai..."):
            mask = get_floor_mask(session, room_bgr, conf_threshold=conf_threshold)

        if mask is None or mask.sum() == 0:
            st.warning(
                "⚠️ Lantai tidak terdeteksi. Coba:\n"
                "- Turunkan sensitivitas deteksi\n"
                "- Gunakan foto dengan lantai yang lebih terlihat"
            )
            st.stop()

        floor_pct = mask.sum() / mask.size * 100
        st.success(f"✅ Lantai terdeteksi ({floor_pct:.1f}% area gambar)")

        with st.spinner("🖌️ Menerapkan tekstur..."):
            result_bgr = apply_texture_perspective(
                room_bgr, mask, texture_bgr,
                tile_size=tile_size,
                feather_radius=feather_radius
            )
            result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            result_pil = resize_for_preview(Image.fromarray(result_rgb))

        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_for_preview(room_img), caption="Original", use_container_width=True)
        with col2:
            st.image(result_pil, caption=f"Tekstur {selected_texture}", use_container_width=True)

        buf         = io.BytesIO()
        full_result = Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
        full_result.save(buf, format="JPEG", quality=95)
        st.download_button(
            label="⬇️ Download hasil (resolusi penuh)",
            data=buf.getvalue(),
            file_name=f"floor_{selected_texture}.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

        with st.expander("🔬 Lihat mask deteksi lantai"):
            overlay_arr           = np.array(room_img).copy()
            overlay_arr[mask > 0] = (
                overlay_arr[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
            ).astype(np.uint8)
            st.image(resize_for_preview(Image.fromarray(overlay_arr)),
                     caption="Area lantai yang terdeteksi (hijau)",
                     use_container_width=True)