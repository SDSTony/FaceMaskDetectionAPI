import io
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torchvision
import torch
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
from flask import Flask, jsonify, request, send_file, make_response



#send file을 해줘야하는 것 같음

app = Flask(__name__)

def get_model_instance_segmentation(num_classes):
  
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

model = get_model_instance_segmentation(4)

if torch.cuda.is_available():
    device = torch.device('cuda')
    model.to(device)
    model.load_state_dict(torch.load('fasterrcnn_05.pt'))
else:
    device = torch.device('cpu')
    model.to(device)
    model.load_state_dict(torch.load('fasterrcnn_05.pt', map_location=device))

model.eval()

def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    fig,ax = plt.subplots(1)
    ax.imshow(img)
    
    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 1 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 2 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        ax.add_patch(rect)

    #plt.show()
    return fig, ax


# change 1 image to tensor
def transform_image(image_bytes):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0)


def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : 
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]

    return preds


@app.route('/predict', methods=['POST']) #ainize API 창에 보이는 명령어 이름 기입, methods only accepts POST 
def predict():
    if request.method == 'POST':
        # get the file from the request
        file = request.files['file']
        # convert to bytes
        img_bytes = file.read()
        tensor = transform_image(image_bytes=img_bytes)
        tensor = tensor.to(device)
        output = make_prediction(model, tensor, 0.5)
        _fig, _ax = plot_image_from_output(tensor[0], output[0])
        
        canvas = FigureCanvas(_fig)
        output = io.BytesIO()
        canvas.print_png(output)
        response = make_response(output.getvalue())
        response.mimetype = 'image/png'
        return response


if __name__ == '__main__':
    app.run(port=80, host='0.0.0.0')