import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

import logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="artguardia_debug.log",
    filemode="w",
)
logger = logging.getLogger(__name__)

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog,
    QSlider, QHBoxLayout, QGridLayout, QMessageBox, QFrame, QSpacerItem, 
    QSizePolicy, QDialog, QProgressBar
)
from PyQt6.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt6.QtCore import Qt, QTimer
import torch
import torch.nn.functional as F
from torchvision import transforms
from diffusers import AutoencoderKL, StableDiffusionPipeline
from PIL import Image, ImageFilter
import numpy as np
import sys
import os
import io
from lpips import LPIPS 

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Loading ArtGuardia...")
        self.setFixedSize(400, 200)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)  # Remove window borders

        layout = QVBoxLayout()
        self.label = QLabel("Loading ArtGuardia...\nPlease wait.")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.setLayout(layout)

        # Simulate loading process
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.progress = 0
        self.timer.start(50)  # Update every 50ms

    def update_progress(self):
        self.progress += 5
        self.progress_bar.setValue(self.progress)
        if self.progress >= 100:
            self.timer.stop()
            self.close()
            self.launch_main_app()

    def launch_main_app(self):
        self.main_window = ArtGuardiaApp()
        self.main_window.show()

class ProgressDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Processing")
        self.setFixedSize(300, 100)
        layout = QVBoxLayout()
        self.label = QLabel("Generating adversarial image...")
        layout.addWidget(self.label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)

    def update_progress(self, value):
        self.progress_bar.setValue(value)

class ArtGuardiaApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device: {self.device}")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to(self.device).eval()
        self.lpips = LPIPS(net='alex').to(self.device).eval()
        self.image_path = None
        self.generated_image = None
        self.original_size = None
        self.perturbed_latents = None

    def initUI(self):
        self.setWindowTitle("ArtGuardia")
        self.setFixedSize(900, 600)

        # Set custom window icon
        icon_path = r"C:\\Users\\astre\\OneDrive\\Documents\\Capstone\\backend\\ArtGuardia_Logo.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            logger.debug("Window icon set successfully!")
        else:
            logger.error(f"Icon image not found at {icon_path}")

        # Background image path
        bg_image_path = r"C:\\Users\\astre\\OneDrive\\Documents\\Capstone\\backend\\ArtGuardia_BG.PNG"

        if not os.path.exists(bg_image_path):
            logger.error(f"Background image not found at {bg_image_path}")
        else:
            logger.debug("Background image found!")
            self.bg_label = QLabel(self)
            self.bg_label.setGeometry(0, 0, 900, 600)
            pixmap = QPixmap(bg_image_path)
            if not pixmap.isNull():
                logger.debug("Background image loaded successfully!")
                self.bg_label.setPixmap(pixmap)
                self.bg_label.setScaledContents(True)
                self.bg_label.lower()

        layout = QVBoxLayout()

        # Set font to Quicksand for the entire layout
        font = QFont("Quicksand", 10)
        self.setFont(font)

        # Image display area
        image_layout = QHBoxLayout()

        image_box_color = "#3C3C3C"

        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setFixedSize(400, 400)
        self.original_image_label.setStyleSheet(f"background-color: {image_box_color}; border-radius: 10px; color: white;")
        image_layout.addWidget(self.original_image_label)

        self.perturbed_image_label = QLabel("Perturbed Image")
        self.perturbed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.perturbed_image_label.setFixedSize(400, 400)
        self.perturbed_image_label.setStyleSheet(f"background-color: {image_box_color}; border-radius: 10px; color: white;")
        image_layout.addWidget(self.perturbed_image_label)

        layout.addLayout(image_layout)

        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Slider + Buttons Layout
        slider_buttons_layout = QVBoxLayout()
        slider_buttons_layout.setContentsMargins(0, 5, 0, 0)

        # Slider Box
        slider_box = QVBoxLayout()
        slider_box.setSpacing(2)
        slider_box.setContentsMargins(0, 0, 0, 0)

        slider_frame = QFrame()
        slider_frame.setStyleSheet("background-color: #2D2D2D; border-radius: 10px; padding: 10px;")
        slider_frame.setLayout(slider_box)

        self.slider_label = QLabel("Perturbation Intensity Level")
        self.slider_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_label.setStyleSheet("color: white;")
        slider_box.addWidget(self.slider_label)

        # Slider
        slider_layout = QHBoxLayout()
        self.intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self.intensity_slider.setMinimum(1)
        self.intensity_slider.setMaximum(20)
        self.intensity_slider.setValue(5)
        self.intensity_slider.valueChanged.connect(self.updateSliderValue)
        self.updateSliderColor()
        slider_layout.addWidget(self.intensity_slider)

        self.slider_value_label = QLabel("5")
        self.slider_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_value_label.setStyleSheet("color: white;")
        slider_layout.addWidget(self.slider_value_label)

        slider_box.addLayout(slider_layout)
        slider_buttons_layout.addWidget(slider_frame)

        slider_buttons_layout.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # Buttons - Modified layout with merged button
        button_grid = QGridLayout()

        self.upload_button = QPushButton("Upload Image")
        self.upload_button.clicked.connect(self.uploadImage)
        button_grid.addWidget(self.upload_button, 0, 0)

        # Save Image button
        self.save_button = QPushButton("Save Image")
        self.save_button.clicked.connect(self.saveImage)
        button_grid.addWidget(self.save_button, 0, 1)

        # New merged button in bottom left (1,0) that performs both operations
        self.generate_button = QPushButton("Generate Adversarial Image")
        self.generate_button.clicked.connect(self.generate_and_bypass)
        button_grid.addWidget(self.generate_button, 1, 0, 1, 2) 

        slider_buttons_layout.addLayout(button_grid)
        layout.addLayout(slider_buttons_layout)

        self.setLayout(layout)

    def generate_and_bypass(self):
        """New method that combines both generation and bypass operations"""
        self.generateImage()
        if self.perturbed_latents is not None:
            self.bypass_noise_scheduler()

    def updateSliderValue(self):
        self.slider_value_label.setText(str(self.intensity_slider.value()))
        self.updateSliderColor()

    def updateSliderColor(self):
        gradient = """
            QSlider::groove:horizontal {
                border: 1px solid #999;
                height: 8px;
                background: qlineargradient(
                    x1: 0, x2: 1,
                    stop: 0 #A075FE,
                    stop: 0.2 #FC72EA,
                    stop: 0.4 #FF7682,
                    stop: 0.6 #FEBB73,
                    stop: 0.8 #B7F375,
                    stop: 1.0 #71FEE4
                );
                border-radius: 4px;
            }

            QSlider::sub-page:horizontal {
                background: transparent;
            }

            QSlider::add-page:horizontal {
                background: rgb(24, 24, 24);
                border-radius: 4px;
            }

            QSlider::handle:horizontal {
                background: white;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -3px 0;
                border-radius: 7px;
            }
        """
        self.intensity_slider.setStyleSheet(gradient)

    def uploadImage(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            self.original_size = image.size
            pixmap = QPixmap(file_path).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)
            self.original_image_label.setPixmap(pixmap)

    def generateImage(self):
        if not self.image_path:
            QMessageBox.warning(self, "Error", "Please upload an image first!")
            logger.warning("No image uploaded. Please upload an image first.")
            return

        progress_dialog = ProgressDialog()
        progress_dialog.show()
        QApplication.processEvents()

        try:
            logger.info("Loading and processing image...")
            image = Image.open(self.image_path).convert("RGB")
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ])
            input_tensor = transform(image).unsqueeze(0).to(self.device) * 2 - 1

            with torch.no_grad():
                logger.debug("Encoding image to latent space...")
                latents = self.vae.encode(input_tensor).latent_dist.sample()

            epsilon = self.intensity_slider.value() / 5 
            alpha = epsilon / 5  
            num_steps = 20  

            # For logging the perturbation parameters
            logger.info(f"Perturbation Parameters: epsilon={epsilon}, alpha={alpha}, num_steps={num_steps}")

            perturbed_latents = latents.clone().detach()
            for step in range(num_steps):
                progress_dialog.update_progress(int((step + 1) / num_steps * 100))
                QApplication.processEvents()
                perturbed_latents.requires_grad = True
                reconstructed = self.vae.decode(perturbed_latents).sample

                # Misalign latent distribution

                mu, log_var = self.vae.encode(reconstructed).latent_dist.mean, self.vae.encode(reconstructed).latent_dist.logvar
                adv_loss = F.kl_div(mu, log_var.exp(), reduction='batchmean')

                # For preserving visual quality

                perceptual_loss = self.lpips(input_tensor, reconstructed).mean()

                # Total loss
                loss = adv_loss + 0.5 * perceptual_loss  

                # Log losses
                logger.debug(f"Step {step + 1}/{num_steps}: adv_loss={adv_loss.item()}, perceptual_loss={perceptual_loss.item()}, total_loss={loss.item()}")

                self.vae.zero_grad()
                loss.backward()

                # Projected Gradient Descent (PGD)
                perturbation = perturbed_latents.grad.sign() * alpha
                perturbed_latents = perturbed_latents + perturbation
                perturbed_latents = torch.clamp(perturbed_latents, latents - epsilon, latents + epsilon).detach()

                # Log perturbation values
                logger.debug(f"Step {step + 1}/{num_steps}: perturbation values (min={perturbation.min().item()}, max={perturbation.max().item()}, mean={perturbation.mean().item()})")

            with torch.no_grad():
                logger.debug("Decoding adversarial image...")
                adversarial_image = self.vae.decode(perturbed_latents).sample
            adversarial_image = (adversarial_image.clamp(-1, 1) + 1) / 2

            adversarial_image = adversarial_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            adv_image_pil = Image.fromarray((adversarial_image * 255).astype(np.uint8))
            adv_image_pil = adv_image_pil.resize(self.original_size, Image.BICUBIC)
            adv_image_pil = adv_image_pil.filter(ImageFilter.GaussianBlur(radius=0.3)) 

            # Set the generated image
            self.generated_image = adv_image_pil

            qimage = self.pil2pixmap(adv_image_pil)
            self.perturbed_image_label.setPixmap(qimage)
            logger.info("Adversarial image generated successfully.")

            
            self.perturbed_latents = perturbed_latents
            logger.info("Latent representation saved for bypassing the noise scheduler.")

        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        finally:
            progress_dialog.accept()

    def bypass_noise_scheduler(self):
        if self.perturbed_latents is None:
            QMessageBox.warning(self, "Error", "Generate an image first!")
            return

        try:
            logger.info("Bypassing noise scheduler...")

            # Load the pipeline with the correct data type
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype
            ).to(self.device)

            logger.debug("Pipeline components loaded successfully.")

            # Convert latent tensor to correct type
            perturbed_latents = self.perturbed_latents.to(dtype)

            # Ensure the latent representation has the correct shape
            if perturbed_latents.shape != (1, 4, 64, 64):
                logger.error(f"Incorrect latent shape: {perturbed_latents.shape}")
                QMessageBox.critical(self, "Error", "Latent representation has incorrect shape.")
                return

            # Debug: Print the shape and data type of the latent representation
            logger.debug(f"Latent representation shape: {perturbed_latents.shape}")
            logger.debug(f"Latent representation data type: {perturbed_latents.dtype}")

           
            with torch.no_grad():
                logger.debug("Decoding perturbed latents without noise scheduling...")
                image = pipe.vae.decode(perturbed_latents).sample

            # Convert to PIL image
            image = (image / 2 + 0.5).clamp(0, 1)  
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0] 
            image = (image * 255).astype(np.uint8)  
            image_pil = Image.fromarray(image)  

            # Display the image
            qimage = self.pil2pixmap(image_pil)
            self.perturbed_image_label.setPixmap(qimage)
            logger.info("Noise scheduler bypassed successfully.")

        except Exception as e:
            logger.error(f"Error bypassing noise scheduler: {e}")
            QMessageBox.critical(self, "Error", f"{e}")

    def saveImage(self):
        if self.generated_image:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png)")
            if save_path:
                self.generated_image.save(save_path, format="PNG", quality=100)

    def pil2pixmap(self, img):
        img = img.convert("RGB")
        data = io.BytesIO()
        img.save(data, format="PNG", quality=100)
        data.seek(0)
        qimage = QImage.fromData(data.read())
        return QPixmap.fromImage(qimage).scaled(400, 400, Qt.AspectRatioMode.KeepAspectRatio)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = SplashScreen()
    splash.show()
    sys.exit(app.exec())