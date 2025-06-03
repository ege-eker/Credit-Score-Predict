FROM oven/bun:1.2.13-alpine


WORKDIR /app

COPY bun.lock .
COPY package.json .

RUN bun install

COPY . .

RUN bun next build

ENV NODE_ENV=production
EXPOSE 3000

CMD ["bun", "next", "start"]