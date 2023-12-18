from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import torch

class VGGHandler(BaseHandler):
    def initialize(self, context):
        self.model = torch.load("vgg_cnn.pt", map_location=torch.device('cpu'))
        self.model.eval()
        self.transforms = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    def preprocess(self, data):
        img = Image.open(data).convert('RGB')
        img = self.transforms(img)
        return img

    def inference(self, img):
        with torch.no_grad():
            output = self.model(img.unsqueeze(0))
        return output.numpy()

    def postprocess(self, inference_output):
        return inference_output.tolist()
