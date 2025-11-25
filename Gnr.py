import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_entropy(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    prob_dist = hist / hist.sum()
    
    prob_dist = prob_dist[prob_dist > 0]
    
    entropy = -np.sum(prob_dist * np.log2(prob_dist))
    return entropy

def calculate_ambe(image_original, image_enhanced):
    mean_orig = np.mean(image_original)
    mean_enh = np.mean(image_enhanced)
    return abs(mean_orig - mean_enh)

def analyze_and_report(image_path=None):
    original_img = cv2.imread(image_path, 0)
    print(f"Processing: {image_path}")

    he_img = cv2.equalizeHist(original_img)

    clahe = cv2.createCLAHE(clipLimit=1000.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(original_img)

    cv2.imwrite("enhanced_he_image.png", he_img)
    cv2.imwrite("enhanced_ahe_image.png", clahe_img)

    ent_orig = calculate_entropy(original_img)
    ent_he = calculate_entropy(he_img)
    ent_clahe = calculate_entropy(clahe_img)

    std_orig = np.std(original_img)
    std_he = np.std(he_img)
    std_clahe = np.std(clahe_img)
    
    ambe_he = calculate_ambe(original_img, he_img)
    ambe_clahe = calculate_ambe(original_img, clahe_img)
    
    print("\n" + "="*50)
    print(f"{'METRIC':<20} | {'ORIGINAL':<10} | {'STD HE':<10} | {'AHE':<10}")
    print("-" * 58)
    print(f"{'Entropy (Detail)':<20} | {ent_orig:.4f}     | {ent_he:.4f}     | {ent_clahe:.4f}")
    print(f"{'Std Dev (Contrast)':<20} | {std_orig:.4f}    | {std_he:.4f}    | {std_clahe:.4f}")
    print(f"{'AMBE (Brightness Err)':<20} | {'N/A':<10} | {ambe_he:.4f}     | {ambe_clahe:.4f}")
    print("="*50 + "\n")
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    plt.subplots_adjust(top=0.85, hspace=0.3)
    
    fig.suptitle(f"Quantitative Comparison\nOriginal Entropy: {ent_orig:.2f} | HE Entropy: {ent_he:.2f} | AHE Entropy: {ent_clahe:.2f}", fontsize=14)
    
    axs[0, 0].imshow(original_img, cmap='gray')
    axs[0, 0].set_title("Original")
    
    axs[0, 1].imshow(he_img, cmap='gray')
    axs[0, 1].set_title(f"Standard HE\nAMBE: {ambe_he:.2f} (Lower is better)")
    
    axs[0, 2].imshow(clahe_img, cmap='gray')
    axs[0, 2].set_title(f"AHE\nAMBE: {ambe_clahe:.2f}")
    
    colors = ('black', 'blue', 'green')
    titles = ('Original Hist', 'Standard HE Hist', 'AHE Hist')
    images = (original_img, he_img, clahe_img)

    for i in range(3):
        hist = cv2.calcHist([images[i]], [0], None, [256], [0, 256])
        axs[1, i].plot(hist, color=colors[i])
        axs[1, i].set_title(titles[i])
        axs[1, i].set_xlim([0, 256])
        axs[1, i].grid(alpha=0.3)

        plt.figure()
        plt.plot(hist, color=colors[i])
        plt.title(titles[i])
        plt.xlim([0, 256])
        plt.grid(alpha=0.3)
        plt.savefig(f"graph_{titles[i].replace(' ', '_')}.png")
        plt.close()

    for ax in axs[0]: ax.axis('off')
        
    plt.show()

if __name__ == "__main__":
    image_path = "remotesensing-09-00025-g001a_1.png" 

    analyze_and_report(image_path)
