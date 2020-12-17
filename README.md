# Face Mask Detection API

[![Run on Ainize](https://ainize.ai/static/images/run_on_ainize_button.svg)](https://ainize.ai/SDSTony/FaceMaskDetectionApi)

## How to request to API using cURL

1. Download any image on Google that has mask on or no mask. You can also use the sample images in 'imgs' directory on [this repo](https://github.com/SDSTony/FaceMaskDetectionApi)

2. Open up cmd prompt and navigate to the directory with the images that you have acquired at step 1
3. Try the code below to save the detected results to 'result.jpg'. It will send 'mask1.png' file to the API and save the detect results to 'result.jpg'

```
curl -X POST "https://master-face-mask-detection-api-sds-tony.endpoint.ainize.ai/predict" -F "file=@mask1.png;type=image/png" --output result.jpg
```



## References

[PseudoLab Tutorial-Book Object Detection](https://pseudo-lab.github.io/Tutorial-Book/chapters/object_detection/Ch1%20Object%20Detection.html)