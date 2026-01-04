import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="MorphLab - Image Processing",
    page_icon="",
    layout="wide", # Menggunakan layout lebar
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS FOR UI POLISH ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa; 
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .css-1aumxhk {
        padding: 1rem;
    }
    /* Kotak penjelasan */
    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #e8f4f8;
        border-left: 5px solid #3498db;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def process_image(image, operation, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'Erosi (Erosion)':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'Dilasi (Dilation)':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'Opening':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'Closing':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

# --- 4. MAIN APP ---
def main():
    # Header
    st.title(" MorphLab Analisis Morfologi Citra")
    st.markdown("<div style='text-align: center; color: gray;'>Aplikasi Web Pengolahan Citra Digital (Erosi, Dilasi, Opening, Closing)</div>", unsafe_allow_html=True)
    st.markdown("---")

    # --- SIDEBAR (CONTROLS) ---
    with st.sidebar:
        st.header(" Panel Kontrol")
        
        # Upload Section
        st.subheader("1. Input Gambar")
        uploaded_file = st.file_uploader("Upload file gambar", type=['jpg', 'png', 'jpeg'])
        
        # Settings Section
        st.subheader("2. Pengaturan")
        # Preprocessing Toggle
        use_binary = st.checkbox("Ubah ke Hitam-Putih (Binary)", value=True, help="Operasi morfologi bekerja paling baik pada citra biner (hitam putih).")
        
        operation = st.selectbox(
            'Pilih Operasi:',
            ('Erosi', 'Dilasi', 'Opening', 'Closing')
        )

        kernel_size = st.slider("Ukuran Kernel (Kekuatan)", 3, 25, 5, step=2)
        st.caption(f"Kernel Matrix: {kernel_size} x {kernel_size} pixel")

        st.info("💡 **Tips:** Gunakan 'Opening' untuk menghilangkan bintik putih kecil, dan 'Closing' untuk menutup lubang hitam kecil.")

    # --- MAIN AREA ---
    if uploaded_file is not None:
        # Load Image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_image = cv2.imdecode(file_bytes, 1)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Preprocessing (UX Improvement: Auto Binerisasi)
        if use_binary:
            gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
            # Otsu's Thresholding untuk hasil biner terbaik
            _, img_to_process = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            display_mode = "Citra Biner (Pre-processed)"
        else:
            img_to_process = original_image
            display_mode = "Citra Asli (RGB)"

        # Processing
        with st.spinner('Sedang memproses citra...'):
            result_image = process_image(img_to_process, operation, kernel_size)

        # --- LAYOUT DUA KOLOM (Side-by-Side Comparison) ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(" Sebelum ")
            st.image(img_to_process, caption=display_mode, use_container_width=True)

        with col2:
            st.subheader(f" Sesudah ({operation})")
            st.image(result_image, caption=f"Hasil {operation} (Kernel {kernel_size})", use_container_width=True)

        # --- EXPLANATION SECTION ---
        st.markdown("---")
        with st.expander(" Penjelasan Teknis (Apa yang terjadi?)", expanded=True):
            if operation == 'Erosi':
                st.markdown("""
                **Erosi** mengikis batas objek depan (biasanya piksel putih). 
                * **Efek:** Objek menjadi lebih kecil/tipis.
                * **Kegunaan:** Menghilangkan noise (bintik putih kecil) di latar belakang.
                """)
            elif operation == 'Dilasi':
                st.markdown("""
                **Dilasi** menambahkan piksel ke batas objek pada gambar.
                * **Efek:** Objek menjadi lebih besar/tebal.
                * **Kegunaan:** Menyambung bagian objek yang terputus.
                """)
            elif operation == 'Opening':
                st.markdown("""
                **Opening** adalah Erosi diikuti oleh Dilasi.
               
                * **Kegunaan:** Sangat ampuh menghilangkan noise di luar objek tanpa mengubah ukuran objek asli secara signifikan.
                """)
            elif operation == 'Closing':
                st.markdown("""
                **Closing** adalah Dilasi diikuti oleh Erosi.
                
                * **Kegunaan:** Menutup lubang-lubang kecil di dalam objek (foreground).
                """)

    else:
        # --- EMPTY STATE (Tampilan awal yang bersih) ---
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2> Selamat Datang!</h2>
            <p>Silakan upload gambar melalui panel di sebelah kiri untuk memulai analisis.</p>
            <p style="font-size: 50px;"></p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()