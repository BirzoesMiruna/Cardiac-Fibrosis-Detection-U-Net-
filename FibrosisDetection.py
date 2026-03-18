import os
import cv2
import pydicom
import numpy as np
from tkinter import filedialog, Tk
import matplotlib.pyplot as plt

def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selectează o imagine DICOM",
        filetypes=[("DICOM files", "*.dcm")]
    )
    return file_path

def normalize_hu(image, intercept, slope):
    return np.clip(image * slope + intercept, -1000, 1000)

def detect_fibrosis_ct(image):
    # Normalizez imaginea la 0-255 pentru afisare
    image_uint8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    # 1️⃣ Selectare manuală a zonei inimii (ROI)
    r = cv2.selectROI("Selecteaza inima si apasa ENTER", image_rgb, False, False)
    cv2.destroyWindow("Selecteaza inima si apasa ENTER")
    x, y, w, h = r
    heart_roi = image[y:y+h, x:x+w]

    # 2️⃣ Afișează histograma ROI
    plt.figure(figsize=(6,4))
    plt.hist(heart_roi.ravel(), bins=100, color='gray')
    plt.title("Histograma intensității ROI")
    plt.xlabel("Intensitate")
    plt.ylabel("Număr de pixeli")
    plt.show()

    # 3️⃣ Praguri ajustabile (ALEGE pe baza histogramei ROI!)
    min_threshold = 100   # ajustează!
    max_threshold = 300   # ajustează!

    fibrosis_mask_roi = np.logical_and(heart_roi >= min_threshold, heart_roi <= max_threshold)

    # 4️⃣ Eliminare zgomot cu morfologie
    mask_roi_uint8 = fibrosis_mask_roi.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_roi_cleaned = cv2.morphologyEx(mask_roi_uint8, cv2.MORPH_OPEN, kernel)

    # 5️⃣ Eliminare componente foarte mici
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_roi_cleaned, connectivity=8)
    min_area = 50  # ajustează dacă vrei
    final_mask_roi = np.zeros_like(mask_roi_cleaned)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask_roi[labels == i] = 255

    # 6️⃣ ROI colorat mov unde e fibroză
    heart_roi_rgb = cv2.cvtColor(
        cv2.normalize(heart_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLOR_GRAY2BGR
    )
    heart_roi_rgb[final_mask_roi > 0] = [128, 0, 128]

    # 7️⃣ Imagine completă cu ROI colorat
    result_image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)
    result_image_rgb[y:y+h, x:x+w] = heart_roi_rgb

    # 8️⃣ Mască finală pe imagine completă
    mask_full = np.zeros(image.shape, dtype=np.uint8)
    mask_full[y:y+h, x:x+w] = (final_mask_roi > 0).astype(np.uint8)

    return result_image_rgb, mask_full

def save_results(original_image, mask, original_path):
    images_dir = r"C:\Users\..."
    masks_dir = r"C:\Users\..."
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    filename = os.path.basename(original_path).replace(".", "_")

    # Salvează imaginea originală și masca
    original_uint8 = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(images_dir, f"{filename}_original.png"), original_uint8)
    cv2.imwrite(os.path.join(masks_dir, f"{filename}_mask.png"), mask * 255)
    print(" Imagine originală salvată și mască salvată!")
    print(f"Dimensiune imagine: {original_image.shape}")
    print(f"Dimensiune mască: {mask.shape}")
    assert original_image.shape == mask.shape, " Dimensiunile NU se potrivesc!"

def calculate_noise(mask):
    percent_white = (np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])) * 100
    print(f" Procent pixeli pozitivi: {percent_white:.2f}%")
    if percent_white < 1:
        print(" Masca poate conține mult zgomot (sub 1%).")
    elif percent_white > 50:
        print(" Masca e foarte mare (>50%).")
    else:
        print(" Masca pare rezonabilă.")

def display_images(original_image, processed_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Imagine Originală')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    plt.title('Imagine ROI cu Fibroză evidențiată')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    file_path = select_image()
    if not file_path:
        print("Nu a fost selectată nici o imagine.")
        return

    try:
        dicom_data = pydicom.dcmread(file_path)
        image = dicom_data.pixel_array.astype(np.float32)
        intercept = getattr(dicom_data, 'RescaleIntercept', 0)
        slope = getattr(dicom_data, 'RescaleSlope', 1)
        image = normalize_hu(image, intercept, slope)

        processed_image, mask_full = detect_fibrosis_ct(image)
        display_images(image, processed_image)
        save_results(image, mask_full, file_path)
        calculate_noise(mask_full)

    except Exception as e:
        print(f" Eroare la procesare: {e}")

if __name__ == "__main__":
    main()
