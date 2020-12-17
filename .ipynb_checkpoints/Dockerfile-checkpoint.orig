FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN pip install --upgrade pip

COPY requirements.txt .


RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD python -u app_face.py
