# Face Mask Detection API

[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.ai/SDSTony/FaceMaskDetectionApi)



## Demo

![demo](https://raw.githubusercontent.com/SDSTony/FaceMaskDetectionApi/master/demo_img.PNG)

Detects whether a face mask is worn with green and red bounding boxes. Green if mask worn, red if not worn. 

## How to request to API using cURL

1. Download any image on Google that has mask on or no mask. You can also use the sample images in 'imgs' directory on [this repo](https://github.com/SDSTony/FaceMaskDetectionApi). Name the image as 'input_image.png'. File extension can be either 'jpg', 'jpeg', 'png', or 'jfif'.

2. Open up cmd prompt and navigate to the directory that has the image you have acquired at step 1
3. Try the code below to save the detected result as 'result.jpg'. It will send 'input_image.png' file to the API and save the detected result to 'result.jpg'. Face with mask on will have green bounding box while face with no mask on will have red bounding box. 

```
curl -X POST "https://master-face-mask-detection-api-sds-tony.endpoint.ainize.ai/predict" -F "file=@input_image.png;type=image/png" --output result.jpg
```



## References

[PseudoLab Tutorial-Book Object Detection](https://pseudo-lab.github.io/Tutorial-Book/chapters/object_detection/Ch1%20Object%20Detection.html)