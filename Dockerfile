# Specify which base layers (default dependencies) to use
# You may find more base layers at https://hub.docker.com/
FROM python:3.13.2

#
# Creates directory within your Docker image
RUN mkdir -p /app/src/

WORKDIR /app/src
#
# Copies file from your Local system TO path in Docker image
COPY chatbot.py /app/src
COPY requirements.txt /app/src
COPY config.json /app/src
#

#
# Enable permission to execute anything inside the folder app
RUN chgrp -R 65534 /app && \
    chmod -R 777 /app    
    
# Installs dependencies within you Docker image
RUN pip3 install -r /app/src/requirements.txt

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "chatbot.py"]    