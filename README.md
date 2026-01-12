# ArtGuardia: Protecting Art via Latent Space Perturbations

[cite_start]**ArtGuardia** is a desktop application designed to protect digitized paintings from unauthorized replication by Generative AI models, specifically **Stable Diffusion (v1.5)**[cite: 1, 2, 92]. [cite_start]By applying adversarial noise within the **latent space** rather than just the pixel level, ArtGuardia creates "adversarial paintings" that look normal to humans but confuse AI replication tools[cite: 20, 67, 187].

## 🖼️ Problem Statement
[cite_start]Traditional protection (pixel-space perturbations) is often ignored by advanced AI models that convert images into latent representations[cite: 18, 19, 65]. [cite_start]ArtGuardia addresses this by targeting the **Latent Representation** directly using the **Projected Gradient Descent (PGD)** algorithm[cite: 20, 21, 69].

## ✨ Key Features
- [cite_start]**Latent Space Protection:** Modifies hidden features that AI models use to "understand" images[cite: 67, 68].
- [cite_start]**Adjustable Intensity:** Users can choose between subtle protection (Default) and robust security (Maximum)[cite: 85, 87].
- [cite_start]**Integrated Preview:** Side-by-side comparison of the original and protected artwork[cite: 140].
- [cite_start]**Agile Development:** Built using Python and PyQt6 for a smooth, professional user interface[cite: 74, 133].

## 📊 Performance & Metrics
[cite_start]The effectiveness of the perturbations is validated using three industry-standard metrics[cite: 150, 190]:
* [cite_start]**SSIM (Structural Similarity Index):** Measures structural integrity[cite: 151].
* [cite_start]**LPIPS:** Evaluates perceptual differences using deep learning[cite: 152].
* [cite_start]**FID (Fréchet Inception Distance):** Measures how much the AI's "understanding" of the image has been diverted[cite: 153].

| Protection Level | Success Rate | Generation Time |
| :--- | :--- | :--- |
| **Default** | 58.48% | ~18-20 Minutes |
| **Maximum** | 79.73% | ~25-30 Minutes |
[cite_start][cite: 145, 146, 177]

## 🛠️ Technical Stack
- [cite_start]**Framework:** Python, PyQt6[cite: 133].
- **AI Libraries:** PyTorch, Diffusers, Torchvision .
- [cite_start]**Model:** RunwayML/Stable Diffusion v1.5[cite: 92].
- [cite_start]**Algorithm:** Projected Gradient Descent (PGD)[cite: 21, 69].

## 🚀 How to Run (On Linux)
1. **Clone the repo:**
   ```bash
   git clone [https://github.com/yourusername/ArtGuardia.git](https://github.com/yourusername/ArtGuardia.git)
   cd ArtGuardia
