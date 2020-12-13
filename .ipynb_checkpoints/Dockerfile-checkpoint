FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 libgtk2.0-dev

COPY requirements.txt .


RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD python app.py
