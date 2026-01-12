# ArtGuardia

**ArtGuardia** is a desktop application designed to protect digitized paintings from unauthorized replication by Generative AI models, specifically **Stable Diffusion (v1.5)**. By applying adversarial noise within the **latent space** rather than just the pixel level, ArtGuardia creates "adversarial paintings" that look normal to humans but confuse AI replication tools.

## Problem Statement
Traditional protection (pixel-space perturbations) is often ignored by advanced AI models that convert images into latent representations. ArtGuardia addresses this by targeting the **Latent Representation** directly using the **Projected Gradient Descent (PGD)** algorithm.

## Key Features
- **Latent Space Protection:** Modifies hidden features that AI models use to "understand" images.
- **Adjustable Intensity:** Users can choose between subtle protection (Default) and robust security (Maximum).
- **Integrated Preview:** Side-by-side comparison of the original and protected artwork.
- **Agile Development:** Built using Python and PyQt6 for a smooth, professional user interface.

## 📊 Performance & Metrics
The effectiveness of the perturbations is validated using three industry-standard metrics:
* **SSIM (Structural Similarity Index):** Measures structural integrity.
* **LPIPS:** Evaluates perceptual differences using deep learning.
* **FID (Fréchet Inception Distance):** Measures how much the AI's "understanding" of the image has been diverted.

| Protection Level | Success Rate | Generation Time |
| :--- | :--- | :--- |
| **Default** | 58.48% | ~18-20 Minutes |
| **Maximum** | 79.73% | ~25-30 Minutes |


## Technical Stack
- **Framework:** Python, PyQt6.
- **AI Libraries:** PyTorch, Diffusers, Torchvision .
- **Model:** RunwayML/Stable Diffusion v1.5.
- **Algorithm:** Projected Gradient Descent (PGD).

## How to Run (On Linux)
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/yourusername/ArtGuardia.git](https://github.com/yourusername/ArtGuardia.git)
   cd ArtGuardia
2. **Install Dependencies:**
   ```bash
   pip install torch torchvision diffusers PyQt6 lpips pillow numpy
3.**Run ArtGuardia:**
   ```bash
  python3 main.py
