import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


def get_model_instance_segmentation(num_classes):
  
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def fasterrcnn():

    print('start loading model')
    model = get_model_instance_segmentation(4)
    print('finished loading model')

    if torch.cuda.is_available():
        device = torch.device('cuda')
        model.to(device)
        model.load_state_dict(torch.load('fasterrcnn_05.pt'))
    else:
        device = torch.device('cpu')
        model.to(device)
        model.load_state_dict(torch.load('fasterrcnn_05.pt', map_location=device))

    print('finished loading weight')

    model.eval()
    
    return model 


