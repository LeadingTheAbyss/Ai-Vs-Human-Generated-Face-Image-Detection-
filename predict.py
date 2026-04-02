import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

model_path = "trained_resnet18_binary.pth", img_path = "detec.png", sz = 224, cutoff = 0.6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((sz, sz)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

model = models.resnet18 (weights = None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load(model_path, map_location = device))
model = model.to(device)
model.eval()

img = Image.open(img_path).convert("RGB"), x = transform(img).unsqueeze(0).to(device)
act = None, grad = None

def f_hook(module, inp, out):
    global act
    act = out

def b_hook(module, grad_inp, grad_out):
    global grad
    grad = grad_out[0]

layer = model.layer4[-1]
fhook = layer.register_forward_hook(f_hook), bhook = layer.register_full_backward_hook(b_hook)

p = torch.sigmoid(model(x)).item()

if p >= cutoff:
    pred = "AI"
    confidence = p
    target = model(x)
else:
    pred = "REAL"
    confidence = 1 - p
    target = -model(x)

model.zero_grad()
target.backward()

wt = torch.mean(grad, dim = [0, 2, 3]), act = act.detach().squeeze(0)

for i in range(act.shape[0]):
    act[i] *= wt[i]

cam = torch.mean(act, dim = 0).cpu().numpy()
cam = np.maximum(cam, 0)

if np.max(cam) != 0:
    cam /= np.max(cam)

cam = cv2.resize(cam, (sz, sz))
background = cv2.cvtColor(np.array(img.resize((sz, sz))), cv2.COLOR_RGB2BGR)
overlay = cv2.addWeighted(background, 0.6, cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), 0.4, 0)

cv2.imwrite("gradcam_heatmap.jpg", cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET))
cv2.imwrite("gradcam_overlay.jpg", overlay)

hi = np.mean(cam > 0.6) * 100
vhi = np.mean(cam > 0.8) * 100

if pred == "AI":
    if vhi > 15:
        reason = "Model has focused mainly on a concentrated set of regions which suggests suspicious local artifacts."
    elif hi > 30:
        reason = "Model used several medium to high attention regions from the image."
    else:
        reason = "Model's attention is spread out so therefore the AI signal is weaker and less localized."
else:
    if vhi > 15:
        reason = "Model strongly relies on specific realistic looking regions."
    elif hi > 30:
        reason = "Model found multiple regions which are supportive and natural looking."
    else:
        reason = "Model's decision is based on distributed evidence rather than one strong suspicious area."
    
fhook.remove()
bhook.remove()