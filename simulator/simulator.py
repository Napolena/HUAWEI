from segmentation.DDRNet_23_slim import *
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def probility_switch(v):
    x = np.random.random()
    return x<v

device = "cuda:0"
model = DualResNet(BasicBlock, [2, 2, 2, 2], num_classes=19, planes=32, spp_planes=128, head_planes=64,)
img_path = "aachen_000003_000019_leftImg8bit.png"

pretrained_state  = torch.load("best_val_smaller.pth",map_location="cpu")
model_dict = model.state_dict()
pretrained_state = {k[6:]: v for k, v in pretrained_state.items()
                    if k[6:] in model_dict.keys()}
model.load_state_dict(pretrained_state, strict = True)

model.eval()
model.to(device)

img = Image.open(img_path)
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225])])
x = transforms(img)
x = x.unsqueeze(0)
x = x.to(device)

with torch.no_grad():
    y = model(x)

pred = torch.nn.Upsample(scale_factor=8,mode="bilinear")(y)
preds = torch.argmax(pred[0],dim=0)

colors = np.array([  (153,153,153), (153,153,153), (153,153,153), (250,170,30), (220,220,0), (107,142,35), (152,251,152), ( 70,130,180), (220,20,60), (255,0,0), (  0,0,142), (  0,0,70), (  0,60,100), (  0,0,90), (  0,0,110), (  0,80,100), (  0,0,230), (119,11,32), (  0,0,142)])

print(colors.__len__())
preds = preds.cpu().numpy()
preds = preds.astype(np.int8)
vis = colors[preds].astype(np.uint8)
# vis = np.moveaxis(vis,[0])
origin_img = np.array(img)

# now = 0.5*origin_img + 0.5*vis

# now = now.reshape((3,1024,2048)).astype(np.float32)

# cv2.imwrite("temp.png",vis)
Image.fromarray(vis).save("temp.png")

gray_img = cv2.cvtColor(vis,cv2.COLOR_BGR2GRAY)
Image.fromarray(gray_img).save("gray.png")

# gray_img = cv2.imread("gray.png",flags=cv2.IMREAD_GRAYSCALE)
gray_img = gray_img / 255

temp = np.pad(gray_img,((1,1),(1,1)),mode="constant")

kernel = np.array([[0,1/8,0],[1/8,1/2,1/8],[0,1/8,0]]) 

filtered_image = cv2.filter2D(temp, -1, kernel)
final_img = filtered_image[1:-1,1:-1]

is_on = np.zeros((1024,2048),dtype=bool)
for i,x in enumerate(final_img):
    for j,y in enumerate(x):
        tt = probility_switch(j)
        is_on[i,j] = tt

ff = is_on * final_img
ff = ff.astype(np.float32)
ff = abs(1-ff)

image = cv2.imread(img_path)

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
brightness_channel = hsv_img[:, :, 2]
new_brightness_channel = brightness_channel *ff  
 

hsv_img[:, :, 2] = new_brightness_channel
 
output_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

cv2.imwrite("output.png",output_image)