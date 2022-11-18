import torch
import torchvision
import cv2
import numpy
from PIL import Image,ImageGrab
from torchvision import transforms
from models.fast_scnn import get_fast_scnn
# An instance of your model.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = get_fast_scnn('citys', pretrained=True, root='./weights', map_cpu=False).to(device)
print('Finished loading model!')
model.eval()

# An example input you would normally provide to your model's forward() method.
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
cap = cv2.VideoCapture(0)
success,img = cap.read()
image = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
image = transform(image).unsqueeze(0).to(device)


# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, image)
traced_script_module.save("model_scnn.pt")