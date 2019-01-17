FROM ubuntu:14.04
MAINTAINER Shashvat Kedia <sk261@snu.edu.in>

RUN mkdir -p /quora_classifier
WORKDIR /quora_classifier
COPY . .

RUN apt-get update 
RUN apt-get install -y python3.6 
