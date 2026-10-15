import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class ViTExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading ViT-Base on device: {self.device}...")
        
        # Tải trọng số chuẩn của ViT
        weights = models.ViT_B_16_Weights.DEFAULT
        self.vit = models.vit_b_16(weights=weights)
        
        # Cắt bỏ lớp phân loại cuối, giữ lại 768 chiều vector
        self.vit.heads = torch.nn.Identity()
        self.vit.to(self.device).eval()

        # Tiền xử lý chuẩn: Giữ tỷ lệ, cắt tâm ảnh 224x224
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_vector(self, pil_image):
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.vit(img_tensor).squeeze().cpu().numpy().tolist()
        return vector

ai_extractor = ViTExtractor()