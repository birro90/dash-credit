# syntax=docker/dockerfile:1

FROM python:3.7-slim-buster

WORKDIR /app

#ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

COPY requirements.txt requirements.txt
#COPY P7.py app.py 
RUN pip3 install -r requirements.txt
#EXPOSE  7878

COPY P7.py app.py 
COPY . .

#CMD [ "python3", "-m" , "flask", "run", "--host=51.158.147.66", "--port=7878"]
#CMD [ "python3", "-m" , "flask", "run"]

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]
