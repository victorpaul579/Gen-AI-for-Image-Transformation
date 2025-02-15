import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_processing_pipeline(image_path):
    """
    Complete image processing pipeline with visualization of each step
    """
    # Read the image
    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Could not read the image")
    
    # Create figure for all steps
    plt.figure(figsize=(20, 15))
    current_plot = 1
    
    def show_step(img, title):
        nonlocal current_plot
        plt.subplot(3, 3, current_plot)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis('off')
        current_plot += 1
        return img
    
    # 1. Original Image
    show_step(original, 'Original Image')
    
    # 2. Apply CLAHE
    lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    clahe_img = cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)
    show_step(clahe_img, 'After CLAHE')
    
    # 3. Denoise
    denoised = cv2.fastNlMeansDenoisingColored(clahe_img, None, 10, 10, 7, 21)
    show_step(denoised, 'After Denoising')
    
    # 4. Sharpen
    gaussian_blur = cv2.GaussianBlur(denoised, (0, 0), 2.0)
    sharpened = cv2.addWeighted(denoised, 1.5, gaussian_blur, -0.5, 0)
    show_step(sharpened, 'After Sharpening')
    
    # 5. Color Balance
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    a = cv2.add(a, -3)  # Reduce magenta tint
    b = cv2.add(b, 3)   # Add warmth
    color_balanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    show_step(color_balanced, 'After Color Balance')
    
    # 6. Edge Enhancement
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    edge_enhanced = cv2.filter2D(color_balanced, -1, kernel)
    show_step(edge_enhanced, 'After Edge Enhancement')
    
    # 7. Gamma Correction
    gamma = 1.2
    gamma_corrected = np.power(edge_enhanced/255.0, gamma)
    gamma_corrected = np.uint8(gamma_corrected * 255)
    final_image = show_step(gamma_corrected, 'After Gamma Correction')
    
    # 8. Show histograms of original and final image
    def plot_histogram(img, subplot_pos, title):
        plt.subplot(3, 3, subplot_pos)
        colors = ('b', 'g', 'r')
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(hist, color=color)
        plt.title(title)
        plt.xlim([0, 256])
    
    # Original histogram
    plot_histogram(original, 8, 'Original Histogram')
    
    # Final histogram
    plot_histogram(final_image, 9, 'Final Histogram')
    
    plt.tight_layout()
    plt.show()
    
    # Save final image
    cv2.imwrite('enhanced_image.jpg', final_image)
    
    return final_image

final_image = create_processing_pipeline('saved/00070.png')
