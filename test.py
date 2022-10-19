
from model.mobilenetv3 import MobileNetV3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import time
from PIL import Image
def pre_process(img):
    data_transforms=transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return data_transforms(img)
if __name__ == "__main__":
    model =MobileNetV3()
    lin=model.classifier
    classnames=["art","bee"]
    new_lin=nn.Sequential(
            nn.Dropout(p=0.8),    # refer to paper section 6
            nn.Linear(1280,1000),
            nn.Linear(1000,2)
        )
    model.classifier=new_lin
    model.load_state_dict(torch.load("output/best_weights.pth",map_location="cpu"),strict=False)
    model.eval()
    img=Image.open("/home/cuong/Desktop/Image_classify/bee.jpg")
    x=pre_process(img)
    t1=time.time()
    outputs=model(x.unsqueeze(0))
    print("Time pred : ",(time.time()-t1))
    _, preds = torch.max(outputs, 1)
    print("model pred : ",classnames[preds])