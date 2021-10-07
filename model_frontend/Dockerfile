FROM python:3.7.7-buster

SHELL ["/bin/bash", "-c"]

RUN python -m pip install flask WTForms requests

WORKDIR /workarea

COPY ./app/ ./

EXPOSE 5000

# ENV FLASK_APP="main.py"
ENV FLASK_HOST="0.0.0.0"
ENV FLASK_PORT="5000"
ENV FLASK_DEBUG="true"
# ENV FLASK_SECRET_KEY="1234"

ENTRYPOINT ["python", "main.py"]
