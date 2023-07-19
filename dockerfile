FROM python:3.7.3-stretch
RUN mkdir app
COPY ./requirements.txt /app/requirements.txt
WORKDIR app
RUN pip3 install -r requirements.txt
COPY . /app
CMD ["python3","server.py"]