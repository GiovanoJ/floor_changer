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
    proto      = outputs[1][0]  # (32, 160, 160)

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

    det              = best_det
    cx, cy, w, h     = float(det[0]), float(det[1]), float(det[2]), float(det[3])
    mask_coef        = det[5:37]

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

    mask_crop              = np.zeros((160, 160), dtype=np.float32)
    mask_crop[y1:y2, x1:x2] = mask_sig[y1:y2, x1:x2]

    mask_full     = cv2.resize(mask_crop, (imgsz, imgsz))
    mask_combined = np.maximum(mask_combined, mask_full)

    mask_orig   = cv2.resize(mask_combined, (orig_w, orig_h))
    binary_mask = (mask_orig > 0.65).astype(np.uint8)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_close)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN,  kernel_open)
    binary_mask = remove_small_noise(binary_mask, min_area=8000)
    binary_mask = keep_largest_component(binary_mask)

    binary_mask = cv2.GaussianBlur(binary_mask.astype(np.float32), (15, 15), 0)
    binary_mask = (binary_mask > 0.5).astype(np.uint8)

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
# PERSPECTIVE UTILS
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

    pts = np.array([tl, tr, br, bl], dtype=np.float32)
    return order_points(pts)

# =============================================
# LIGHTING TRANSFER
# FIX: gunakan img_bgr asli (bukan original_floor yang sudah di-mask hitam)
# =============================================
def transfer_lighting(img_bgr, texture_warped_bgr, mask):
    """
    Sesuaikan kecerahan texture dengan kecerahan lantai asli.
    img_bgr    : gambar ASLI (bukan yang sudah di-mask)
    mask       : binary mask lantai (0/1)
    """
    floor_area = mask > 0
    if floor_area.sum() == 0:
        return texture_warped_bgr

    # Ambil statistik LAB dari gambar asli di area lantai
    orig_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    tex_lab  = cv2.cvtColor(texture_warped_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    orig_l = orig_lab[:, :, 0]
    tex_l  = tex_lab[:, :, 0]

    # Statistik hanya dari area lantai
    orig_mean = orig_l[floor_area].mean()
    orig_std  = orig_l[floor_area].std() + 1e-6
    tex_mean  = tex_l[floor_area].mean()
    tex_std   = tex_l[floor_area].std() + 1e-6

    # Sesuaikan brightness texture ke brightness lantai asli
    tex_l_adj = (tex_l - tex_mean) / tex_std * orig_std + orig_mean
    tex_l_adj = np.clip(tex_l_adj, 0, 255)

    # Blend: 70% adjusted + 30% original texture brightness
    blend_l = 0.7 * tex_l_adj + 0.3 * tex_l
    blend_l = np.clip(blend_l, 0, 255)

    tex_lab[:, :, 0] = blend_l
    return cv2.cvtColor(tex_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

# =============================================
# AMBIENT OCCLUSION
# =============================================
def apply_ambient_occlusion(img_bgr, mask):
    mask_255 = (mask > 0).astype(np.uint8) * 255

    y_idx, x_idx = np.where(mask > 0)
    if len(y_idx) == 0:
        return img_bgr

    h_floor     = y_idx.max() - y_idx.min()
    w_floor     = x_idx.max() - x_idx.min()
    kernel_size = max(int(min(w_floor, h_floor) * 0.05), 11)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_eroded = cv2.erode(mask_255, kernel)

    # Edge area: hanya piksel di pinggir mask
    edge_area  = cv2.subtract(mask_255, mask_eroded).astype(np.float32)

    blur_size  = kernel_size * 2 - 1
    if blur_size % 2 == 0:
        blur_size += 1
    shadow_map = cv2.GaussianBlur(edge_area, (blur_size, blur_size), 0)

    shadow_max = shadow_map.max()
    if shadow_max < 0.01:
        return img_bgr

    shadow_map = shadow_map / shadow_max  # 0.0–1.0

    # Shadow hanya di area edge
    edge_mask  = (edge_area > 0).astype(np.float32)
    shadow_map = shadow_map * edge_mask

    shadow_3ch    = np.stack([shadow_map] * 3, axis=-1)
    darken_factor = np.clip(1.0 - (0.35 * shadow_3ch), 0.0, 1.0)

    result = img_bgr.astype(np.float32) * darken_factor
    return np.clip(result, 0, 255).astype(np.uint8)

# =============================================
# FEATHERED EDGE BLENDING
# =============================================
def create_feathered_mask(mask, blur_radius=21, power=1.5):
    """
    Buat alpha mask dengan tepi yang halus menggunakan erode + blur.
    Nilai tengah lantai = 1.0, tepi fade ke 0.
    """
    binary = (mask > 0).astype(np.float32)

    # Erode dulu agar blur tidak "memakan" area tengah
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (blur_radius, blur_radius))
    eroded  = cv2.erode(binary, kernel)

    # Blur dari eroded mask — nilai tengah tetap tinggi, tepi fade
    k       = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(eroded, (k, k), 0)

    if blurred.max() < 0.01:
        return binary  # fallback: hard mask

    blurred   = blurred / blurred.max()  # normalize 0–1
    feathered = np.power(np.clip(blurred, 0, 1), power)
    return feathered.astype(np.float32)

# =============================================
# CORE: APPLY TEXTURE
# =============================================
def apply_texture_perspective(img_bgr, mask, texture_bgr,
                              tile_size=TEXTURE_TILE_SIZE, feather_power=1.5):

    # Selalu copy agar tidak memodifikasi array asli
    img_bgr = img_bgr.copy()
    mask    = (mask > 0).astype(np.uint8)

    orig_h, orig_w = img_bgr.shape[:2]

    # --- Floor quad ---
    dst_pts = get_floor_quad(mask)
    if dst_pts is None:
        st.warning("Quad lantai tidak ditemukan")
        return img_bgr

    max_w = max(int(max(
        np.linalg.norm(dst_pts[1] - dst_pts[0]),
        np.linalg.norm(dst_pts[2] - dst_pts[3])
    )), 1)
    max_h = max(int(max(
        np.linalg.norm(dst_pts[3] - dst_pts[0]),
        np.linalg.norm(dst_pts[2] - dst_pts[1])
    )), 1)

    src_pts = np.array([
        [0,         0        ],
        [max_w - 1, 0        ],
        [max_w - 1, max_h - 1],
        [0,         max_h - 1],
    ], dtype=np.float32)

    # --- Tile & warp texture ---
    texture_tiled = tile_texture(texture_bgr, orig_w, orig_h, tile_size)

    # Warp gambar asli KE flat space, lalu overlay texture, lalu warp balik
    # Ini lebih reliable daripada warp texture ke perspective
    M_fwd = cv2.getPerspectiveTransform(dst_pts, src_pts)  # lantai → flat
    M_inv = cv2.getPerspectiveTransform(src_pts, dst_pts)  # flat → lantai

    # Tile texture di flat space sesuai ukuran quad
    texture_flat  = tile_texture(texture_bgr, max_w, max_h, tile_size)

    # Warp texture flat KE perspective lantai
    texture_warped = cv2.warpPerspective(texture_flat, M_inv, (orig_w, orig_h))

    # Verifikasi
    st.write(f"texture_warped di mask: {texture_warped[mask > 0][:3]}")
    st.write(f"dst_pts: {dst_pts}")

    h, w = img_bgr.shape[:2]
    st.write(f"image size: {w}x{h}")
    st.write(f"dst_pts: {dst_pts}")
    st.write(f"dst_pts in bounds: x={dst_pts[:,0].min():.0f}–{dst_pts[:,0].max():.0f}, y={dst_pts[:,1].min():.0f}–{dst_pts[:,1].max():.0f}")

    if texture_warped.max() == 0:
        st.warning("Texture warp gagal, mengembalikan gambar asli.")
        return img_bgr

    # --- Lighting transfer ---
    # FIX: kirim img_bgr ASLI (bukan original_floor yang hitam di luar mask)
    texture_lit = transfer_lighting(img_bgr, texture_warped, mask)

    # --- Alpha blending ---
    alpha     = create_feathered_mask(mask, blur_radius=21, power=feather_power)
    alpha     = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    alpha_3ch = np.stack([alpha] * 3, axis=-1)

    img_f     = img_bgr.astype(np.float32)
    tex_f     = np.clip(texture_lit.astype(np.float32), 0.0, 255.0)

    # Blend hanya di area mask, luar mask pakai gambar asli
    blended = alpha_3ch * tex_f + (1.0 - alpha_3ch) * img_f
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # Pastikan area di luar mask benar-benar gambar asli
    result             = img_bgr.copy()
    result[mask > 0]   = blended[mask > 0]

    st.write(f"det(M_inv): {np.linalg.det(M_inv):.6f}")
    st.write(f"texture_flat shape: {texture_flat.shape}, min/max: {texture_flat.min()}, {texture_flat.max()}")

    # Test warp dengan rectangle sederhana — bypass order_points
    dst_simple = np.array([
        [226.,  940.],   # tl
        [2029., 940.],   # tr  
        [2029., 1364.],  # br
        [226.,  1364.],  # bl
    ], dtype=np.float32)
    src_simple = np.array([
        [0.,        0.       ],
        [max_w-1.,  0.       ],
        [max_w-1.,  max_h-1. ],
        [0.,        max_h-1. ],
    ], dtype=np.float32)
    M_test = cv2.getPerspectiveTransform(src_simple, dst_simple)
    test_warped = cv2.warpPerspective(texture_flat, M_test, (orig_w, orig_h))
    st.write(f"test_warped di mask: {test_warped[mask > 0][:3]}")
    st.image(test_warped, caption="test warp", channels="BGR")

    # --- Ambient occlusion ---
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
        "Sensitivitas deteksi",
        0.10, 0.90, 0.25, 0.05,
        help="Turunkan jika lantai tidak terdeteksi. Naikkan jika objek lain ikut terdeteksi."
    )
    tile_size = st.slider(
        "Ukuran tile tekstur (px)",
        100, 600, TEXTURE_TILE_SIZE, 50,
        help="Semakin kecil = serat kayu lebih halus/rapat. Semakin besar = serat lebih kasar."
    )
    feather_power = st.slider(
        "Ketajaman tepi blending",
        0.5, 3.0, 1.5, 0.1,
        help="Nilai lebih tinggi = tepi tekstur lebih tajam. Default 1.5."
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
                "- Gunakan foto dengan lantai yang lebih terlihat\n"
                "- Pastikan foto tidak terlalu gelap atau terlalu dari atas"
            )
            st.stop()

        floor_pct = mask.sum() / mask.size * 100
        st.success(f"✅ Lantai terdeteksi ({floor_pct:.1f}% area gambar)")

        with st.spinner("🖌️ Menerapkan tekstur..."):
            result_bgr = apply_texture_perspective(
                room_bgr, mask, texture_bgr,
                tile_size=tile_size,
                feather_power=feather_power
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
                     caption="Area lantai yang terdeteksi (hijau)", use_container_width=True)