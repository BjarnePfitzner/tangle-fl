FROM node:latest

WORKDIR /app

RUN apt-get update \
    && apt-get install -y \
     python3-pip \
     dnsutils

COPY package.json requirements.txt ./

RUN npm install \
    && pip3 install -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh
RUN npm run build

EXPOSE 9000

CMD ["./entrypoint.sh"]
