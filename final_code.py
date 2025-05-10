import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy import ndimage


# Global variables
file_path = None
noise_type = None

def select_image_and_noise():
    global file_path, noise_type
    root = tk.Tk()
    root.title("Select The Photo And The Noise Type")
    root.geometry("400x300")

    # --- وظائف الأزرار ---
    def open_file():
        global file_path
        file_path = filedialog.askopenfilename(title=" Select The Photo ",
                                               filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            label_file.config(text=" Selected Successfuly ✅")

    def choose_periodic():
        global noise_type
        noise_type = "periodic"
        root.destroy()

    def choose_salt():
        global noise_type
        noise_type = "salt"
        root.destroy()

    label = tk.Label(root, text="Select Photo :", font=("Arial", 14))
    label.pack(pady=10)

    btn_open = tk.Button(root, text="Select Photo", width=20, height=2, command=open_file)
    btn_open.pack(pady=5)

    label_file = tk.Label(root, text="Not Selected Yet ⌛", fg="red")
    label_file.pack(pady=5)

    label_noise = tk.Label(root, text="Select Noise Type", font=("Arial", 14))
    label_noise.pack(pady=10)

    btn_periodic = tk.Button(root, text="Periodic Noise", width=20, height=2, command=choose_periodic)
    btn_periodic.pack(pady=5)

    btn_salt = tk.Button(root, text="Random Noise", width=20, height=2, command=choose_salt)
    btn_salt.pack(pady=5)

    root.mainloop()


def remove_all_periodic_noise(fft_shifted):
    rows, cols = fft_shifted.shape
    crow, ccol = rows // 2, cols // 2

    # حساب طيف المقدار
    magnitude_spectrum = np.log(1 + np.abs(fft_shifted)) # 0

    # تنعيم الطيف باستخدام مرشح جاوسي
    smoothed = ndimage.gaussian_filter(magnitude_spectrum, sigma=3)

    # حساب الفرق بين الطيف الأصلي والمنعم
    diff = magnitude_spectrum - smoothed # Noise --> DIFF 

    # تحديد عتبة ديناميكية
    threshold = np.mean(diff) + 2.5 * np.std(diff) 

    # إنشاء قناع للقمم
    peaks = diff > threshold # Noise --> True 1 - False 0 

    # استبعاد المنطقة المركزية
    center_radius = 15
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    center_mask = x * x + y * y <= center_radius * center_radius # x^2 + y^2 <= r^2 
    peaks[center_mask] = False


    # توسيع القمم لضمان تغطية كاملة لمناطق التشويش
    peaks = ndimage.binary_dilation(peaks, iterations=2)

    # إنشاء قناع للتصفية
    filter_mask = np.ones_like(fft_shifted, dtype=np.complex64)
    filter_mask[peaks] = 0
    # ARR[0] ARR[I] 

    # تطبيق القناع
    return fft_shifted * filter_mask


def create_frequency_filters(img):
    rows, cols = img.shape
    rm, clm = rows // 2, cols // 2
    x, y = np.meshgrid(np.linspace(-clm, clm - 1, cols), np.linspace(-rm, rm - 1, rows))
    z = np.sqrt(x**2 + y**2)
    # Space Between (0,0) and Pixels
    radius = 15
    cL = z < radius 
    cH = ~cL # Not ~ 
    return cL, cH


def main():
    global file_path, noise_type

    select_image_and_noise()

    if not file_path:
        print("Choose The Photo First")
        return

    if noise_type not in ["periodic", "salt"]:
        print("Error : Choose From The Specific Noise Types")
        return

    print(f"The Image : {file_path}")
    print(f"Noise Type : {noise_type}")

    color_img = cv2.imread(file_path)
    if color_img is None:
        print(f"Error While Uploading Photo From : {file_path}")
        return

    img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    ft2 = np.fft.fft2(img)
    ft = np.fft.fftshift(ft2)
    Fmag = np.log(1 + np.abs(ft))

    cL, cH = create_frequency_filters(img)
    l_ft = ft * cL
    h_ft = ft * cH

    low_filtered_image = np.fft.ifft2(np.fft.ifftshift(l_ft))
    high_filtered_image = np.fft.ifft2(np.fft.ifftshift(h_ft))
    low_f = np.abs(low_filtered_image).astype(np.uint8)
    high_f = np.abs(high_filtered_image).astype(np.uint8)

    plt.figure(figsize=(20, 10))

    plt.subplot(2, 3, 1)
    plt.title('Original Image (Grayscale)')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    # Display low-pass filter mask
    plt.subplot(2, 3, 2)
    plt.title("Low-frequency filter (Mask)")
    plt.imshow(cL, cmap='gray')
    plt.axis('off')

    # Display high-pass filter mask
    plt.subplot(2, 3, 3)
    plt.title("High-frequency filter (Mask)")
    plt.imshow(cH, cmap='gray')
    plt.axis('off')

    # Process based on noise type
    if noise_type == "periodic":
        # Create a copy of the FFT for manipulation
        filtered_ft = ft.copy()

        # Display original frequency spectrum
        plt.subplot(2, 3, 4)
        plt.title('Original Frequency Spectrum')
        plt.imshow(np.log(1 + np.abs(ft)), cmap='gray')
        plt.axis('off')

        # Apply automatic periodic noise removal
        filtered_ft = remove_all_periodic_noise(filtered_ft)

        # Inverse FFT to get denoised image
        denoised = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered_ft))).astype(np.uint8)

        # Display the filtered spectrum (shows where we removed noise)
        plt.subplot(2, 3, 5)
        plt.title('Filtered Frequency Spectrum')
        plt.imshow(np.log(1 + np.abs(filtered_ft)), cmap='gray')
        plt.axis('off')

        # Display the result after noise removal
        plt.subplot(2, 3, 6)
        plt.title('Periodic Noise Removed')
        plt.imshow(denoised, cmap='gray')
        plt.axis('off')
        # Display low-frequency image
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title("Low-frequency image")
        plt.imshow(low_f, cmap='gray')
        plt.axis('off')

        # Display high-frequency image
        plt.subplot(1, 2, 2)
        plt.title("High-frequency image")
        plt.imshow(high_f, cmap='gray')
        plt.axis('off')

        plt.tight_layout()


    elif noise_type == "salt":
        # Apply median filter to the color image first (like in your example)
        median_denoised_img = cv2.medianBlur(color_img, 5)
        median_denoised_gray = cv2.cvtColor(median_denoised_img, cv2.COLOR_BGR2GRAY)

        # Display median denoised image
        plt.subplot(2, 3, 4)
        plt.title('Median Blur')
        plt.imshow(median_denoised_gray, cmap='gray')
        plt.axis('off')

        # Display low-frequency image
        plt.subplot(2, 3, 5)
        plt.title("Low-frequency image")
        plt.imshow(low_f, cmap='gray')
        plt.axis('off')

        # Display high-frequency image
        plt.subplot(2, 3, 6)
        plt.title("High-frequency image")
        plt.imshow(high_f, cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()