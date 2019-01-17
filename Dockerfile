FROM ubuntu:14.04
MAINTAINER Shashvat Kedia <sk261@snu.edu.in>

RUN mkdir -p /quora_classifier
WORKDIR /quora_classifier
COPY . .

RUN apt-get install -y python python-setuptools python-pip
RUN pip install -r requirements.txt

CMD python lstm_model.py