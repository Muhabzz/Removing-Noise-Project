# Image Denoising using Fast Fourier Transform (FFT)

This project aims to **remove periodic noise** from grayscale images using the **Fast Fourier Transform (FFT)** technique. 

---

## 🧠 Project Idea

Images captured through sensors or scanned images may contain **periodic noise** that repeats at regular intervals, affecting image quality. This project utilizes FFT to identify and filter out these noise components.

---

## 🛠️ Tools & Libraries

- **Python 3**
- **OpenCV** (`cv2`)
- **NumPy**
- **Matplotlib** (for visualization)

---

## ⚙️ How It Works

1. **Load Image**: The image is loaded and converted to grayscale.
2. **FFT Transform**: We apply `np.fft.fft2()` to transform the image into the frequency domain.
3. **Notch Filter**: High-frequency periodic noise is identified and filtered out.
4. **Inverse FFT**: The image is reconstructed using `np.fft.ifft2()`.

---

## ▶️ How to Run

1. Install dependencies:
    ```bash
    pip install opencv-python numpy matplotlib
    ```

2. Run the script:
    ```bash
    python remove_noise.py
    ```

3. The script will display:
   - Original image
   - FFT spectrum before and after filtering
   - Cleaned image

---

## 📸 Example Results

Before | After
--- | ---
![Original]([images/original.png](https://github.com/Muhabzz/Removing-Noise-Project/blob/master/Sampels/Periodic/Vertical.png)) | ![Cleaned]([images/cleaned.png](https://github.com/Muhabzz/Removing-Noise-Project/blob/master/Output/Screenshot%202025-05-05%20201233.png))

---

## ✍️ Author

Developed by **Mohab** as part of the Math 3 Final Project — Computer Science Faculty.
