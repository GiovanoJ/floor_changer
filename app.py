import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os

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

MAX_FILE_SIZE_MB  = 10
MAX_DIMENSION     = 2048
ALLOWED_TYPES     = {"jpg", "jpeg", "png"}
TEXTURE_TILE_SIZE = 300

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
        w, h = img.size
        if w < 64 or h < 64:
            return None, f"Gambar terlalu kecil ({w}x{h})."
        return img, None
    except Exception:
        return None, "File rusak atau bukan gambar valid."

def validate_texture(name):
    if name not in TEXTURES:
        return None, "Tekstur tidak valid."
    path = TEXTURES[name]
    if not os.path.exists(path):
        return None, f"File tekstur {name} tidak ditemukan."
    t = cv2.imread(path)
    if t is None:
        return None, f"File tekstur {name} tidak bisa dibaca."
    return t, None

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

def tile_texture(tex_bgr, target_w, target_h, tile_size):
    interp = cv2.INTER_AREA if tile_size < tex_bgr.shape[1] else cv2.INTER_LANCZOS4
    tile   = cv2.resize(tex_bgr, (tile_size, tile_size), interpolation=interp)
    nx     = -(-target_w // tile_size)
    ny     = -(-target_h // tile_size)
    return np.tile(tile, (ny, nx, 1))[:target_h, :target_w]

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
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
        return order_points(cv2.boxPoints(rect).astype(np.float32))
    tl = hull[np.argmin(hull[:,0] + hull[:,1])]
    tr = hull[np.argmin(-hull[:,0] + hull[:,1])]
    br = hull[np.argmax(hull[:,0] + hull[:,1])]
    bl = hull[np.argmax(-hull[:,0] + hull[:,1])]
    return order_points(np.array([tl, tr, br, bl], dtype=np.float32))

def apply_texture(img_bgr, mask, tex_bgr, tile_size=TEXTURE_TILE_SIZE, feather_radius=15):
    img  = img_bgr.copy()
    mask = (mask > 0).astype(np.uint8)
    H, W = img.shape[:2]
    area = mask > 0

    ys, xs = np.where(area)
    if len(ys) == 0:
        return img

    # ── Dapatkan quad lantai untuk perspective warp ──────────────────────────
    quad = get_floor_quad(mask)
    if quad is None:
        return img

    # tl, tr, br, bl
    tl, tr, br, bl = quad

    # ── Ukuran flat space ────────────────────────────────────────────────────
    flat_w = max(int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))), 1)
    flat_h = max(int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))), 1)

    flat_pts = np.array([
        [0,          0         ],
        [flat_w - 1, 0         ],
        [flat_w - 1, flat_h - 1],
        [0,          flat_h - 1],
    ], dtype=np.float32)

    # ── Warp lantai asli ke flat space ───────────────────────────────────────
    M_to_flat  = cv2.getPerspectiveTransform(quad, flat_pts)
    floor_flat = cv2.warpPerspective(img, M_to_flat, (flat_w, flat_h))

    # ── Tile texture di flat space ───────────────────────────────────────────
    tex_flat = tile_texture(tex_bgr, flat_w, flat_h, tile_size)

    # ── Transfer lighting di flat space ─────────────────────────────────────
    # Hitung statistik dari seluruh floor_flat (sudah di-crop, tidak ada area hitam)
    fl  = cv2.cvtColor(floor_flat, cv2.COLOR_BGR2LAB).astype(np.float32)
    tl_ = cv2.cvtColor(tex_flat,   cv2.COLOR_BGR2LAB).astype(np.float32)

    # Sesuaikan hanya L channel
    fl_L_mean = fl[:,:,0].mean()
    fl_L_std  = fl[:,:,0].std() + 1e-6
    tl_L_mean = tl_[:,:,0].mean()
    tl_L_std  = tl_[:,:,0].std() + 1e-6

    tl_[:,:,0] = np.clip(
        (tl_[:,:,0] - tl_L_mean) / tl_L_std * fl_L_std + fl_L_mean,
        0, 255
    )
    # Chroma: pertahankan texture asli sepenuhnya
    tex_lit_flat = cv2.cvtColor(tl_.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Warp texture kembali ke perspektif ──────────────────────────────────
    M_to_persp     = cv2.getPerspectiveTransform(flat_pts, quad)
    texture_warped = cv2.warpPerspective(tex_lit_flat, M_to_persp, (W, H))

    # Cek apakah warp berhasil — jika tidak, fallback ke flat mapping
    ys_mid, xs_mid = ys[len(ys)//2], xs[len(xs)//2]
    if tuple(texture_warped[ys_mid, xs_mid]) == (0, 0, 0):
        # Fallback: remap flat langsung
        tex_full = tile_texture(tex_bgr, W, H, tile_size)
        y_min, y_max = int(ys.min()), int(ys.max())
        x_min, x_max = int(xs.min()), int(xs.max())
        fh = max(y_max - y_min, 1)
        fw = max(x_max - x_min, 1)
        yy, xx   = np.mgrid[0:H, 0:W].astype(np.float32)
        map_x    = np.clip((xx - x_min) / fw, 0, 1) * (W - 1)
        map_y    = np.clip((yy - y_min) / fh, 0, 1) * (H - 1)
        tex_full_warped = cv2.remap(tex_full,
                                    map_x.astype(np.float32),
                                    map_y.astype(np.float32),
                                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Transfer lighting untuk fallback
        fl2  = cv2.cvtColor(img,            cv2.COLOR_BGR2LAB).astype(np.float32)
        tl2  = cv2.cvtColor(tex_full_warped, cv2.COLOR_BGR2LAB).astype(np.float32)
        o_m  = fl2[:,:,0][area].mean()
        o_s  = fl2[:,:,0][area].std() + 1e-6
        t_m  = tl2[:,:,0][area].mean()
        t_s  = tl2[:,:,0][area].std() + 1e-6
        tl2[:,:,0] = np.clip((tl2[:,:,0] - t_m) / t_s * o_s + o_m, 0, 255)
        texture_warped = cv2.cvtColor(tl2.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── Paste texture ke area mask ───────────────────────────────────────────
    result         = img.copy()
    result[area]   = texture_warped[area]

    # ── Feather tepi dengan cv2.seamlessClone jika memungkinkan ─────────────
    # Gunakan addWeighted per-channel — hindari numpy broadcasting
    if feather_radius > 0:
        k       = feather_radius * 2 + 1
        # Buat alpha mask: blur binary mask → float 0-255
        alpha   = cv2.GaussianBlur((mask * 255).astype(np.uint8), (k, k), 0)
        # Blend dengan cv2.addWeighted per channel
        for c in range(3):
            a        = alpha.astype(np.float32) / 255.0
            result[:,:,c] = np.clip(
                a * result[:,:,c].astype(np.float32) +
                (1.0 - a) * img[:,:,c].astype(np.float32),
                0, 255
            ).astype(np.uint8)

    # ── Ambient occlusion ────────────────────────────────────────────────────
    result = _ambient_occlusion(result, mask)
    return result

def _ambient_occlusion(img_bgr, mask, intensity=0.28):
    m255 = (mask > 0).astype(np.uint8) * 255
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return img_bgr
    fh = int(ys.max() - ys.min())
    fw = int(xs.max() - xs.min())
    ks = max(int(min(fh, fw) * 0.04), 11)
    if ks % 2 == 0:
        ks += 1
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    eroded  = cv2.erode(m255, kernel)
    edge    = cv2.subtract(m255, eroded).astype(np.float32)
    bs = ks * 2 - 1
    if bs % 2 == 0:
        bs += 1
    smap = cv2.GaussianBlur(edge, (bs, bs), 0)
    smax = smap.max()
    if smax < 0.01:
        return img_bgr
    smap = (smap / smax) * (edge > 0).astype(np.float32)
    # Gunakan per-channel untuk hindari numpy broadcasting issue
    result = img_bgr.copy().astype(np.float32)
    for c in range(3):
        result[:,:,c] = np.clip(result[:,:,c] * (1.0 - intensity * smap), 0, 255)
    return result.astype(np.uint8)

def resize_preview(img_pil, max_side=1280):
    w, h = img_pil.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img_pil = img_pil.resize((int(w*s), int(h*s)), Image.LANCZOS)
    return img_pil

st.title("🏠 Floor Texture Replacer")
st.write("Upload foto ruangan, pilih tekstur lantai, lalu lihat hasilnya.")

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
    conf_threshold = st.slider("Sensitivitas deteksi", 0.10, 0.90, 0.25, 0.05,
        help="Turunkan jika lantai tidak terdeteksi.")
    tile_size = st.slider("Ukuran tile tekstur (px)", 100, 600, TEXTURE_TILE_SIZE, 50,
        help="Kecil = serat lebih rapat. Besar = serat lebih kasar.")
    feather_radius = st.slider("Kelembutan tepi", 0, 40, 15, 5,
        help="Semakin besar = tepi texture lebih halus.")

if room_file:
    room_img, err = validate_image(room_file)
    if err:
        st.error(err); st.stop()

    tex_bgr, err = validate_texture(selected_texture)
    if err:
        st.error(err); st.stop()

    st.image(resize_preview(room_img), caption="Foto yang diupload", use_container_width=True)
    room_bgr = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2BGR)

    if st.button("🎨 Terapkan Tekstur", type="primary", use_container_width=True):

        with st.spinner("🔍 Mendeteksi lantai..."):
            mask = get_floor_mask(session, room_bgr, conf_threshold=conf_threshold)

        if mask is None or mask.sum() == 0:
            st.warning("⚠️ Lantai tidak terdeteksi. Coba turunkan sensitivitas deteksi.")
            st.stop()

        st.success(f"✅ Lantai terdeteksi ({mask.sum()/mask.size*100:.1f}% area gambar)")

        with st.spinner("🖌️ Menerapkan tekstur..."):
            result_bgr = apply_texture(room_bgr, mask, tex_bgr,
                                       tile_size=tile_size,
                                       feather_radius=feather_radius)
            result_pil = resize_preview(Image.fromarray(
                cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)))

        col1, col2 = st.columns(2)
        with col1:
            st.image(resize_preview(room_img), caption="Original", use_container_width=True)
        with col2:
            st.image(result_pil, caption=f"Tekstur {selected_texture}", use_container_width=True)

        buf = io.BytesIO()
        Image.fromarray(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=95)
        st.download_button("⬇️ Download hasil (resolusi penuh)", buf.getvalue(),
                           f"floor_{selected_texture}.jpg", "image/jpeg",
                           use_container_width=True)

        with st.expander("🔬 Lihat mask deteksi lantai"):
            ov = np.array(room_img).copy()
            ov[mask > 0] = (ov[mask > 0] * 0.5 + np.array([0,255,0]) * 0.5).astype(np.uint8)
            st.image(resize_preview(Image.fromarray(ov)),
                     caption="Area lantai terdeteksi (hijau)", use_container_width=True)