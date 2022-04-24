FROM python:3.8.13
ENV PYTHONUNBUFFERED 1
RUN mkdir /voiceapi
RUN apt-get update \
&& apt-get upgrade -y \
&& apt-get install -y \
&& apt-get -y install apt-utils gcc libpq-dev libsndfile-dev
WORKDIR /voiceapi
COPY requirements.txt /voiceapi/
RUN pip install -r requirements.txt
COPY . /voiceapi/

ENTRYPOINT ["python", "manage.py"]
CMD ["runserver", "0.0.0.0:8800"]