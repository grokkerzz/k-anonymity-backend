FROM python:3.6

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app
RUN pip install -r requirements.txt

ENTRYPOINT ["flask"]
CMD ["run", "--host=0.0.0.0"]
