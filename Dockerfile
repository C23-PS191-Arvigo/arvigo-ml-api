FROM nginx:alpine

RUN apk update && apk upgrade

RUN apk add --no-cache python3 \
    py3-gunicorn

COPY src/docker .
RUN chmod +x *.sh && \
    mv nginx.conf /etc/nginx/nginx.conf

WORKDIR /opt/ctf/app
COPY src/server .
RUN apk --no-cache add python3-dev py3-pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    apk del python3-dev py3-pip

ENV PRODUCTION=true
ENV GUNICORN_WSGI_MODULE=main:app
ENV GUNICORN_NAME=ctf GUNICORN_SOCKFILE=/run/gunicorn.sock
ENV GUNICORN_USER=nobody GUNICORN_GROUP=nobody
RUN chmod 744 /tmp

CMD ["/entrypoint.sh"]