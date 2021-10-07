#!/bin/bash

RED='\033[1;31m'
NC='\033[0m'

SCORING_ENDPOINT=""
read -rp "REST endpoint: " SCORING_ENDPOINT

if [ -z ${SCORING_ENDPOINT} ]; then
  echo -e "${RED}please provide REST endpoint${NC}"
  exit
fi

SWAGGER_URI=""
read -rp "Swagger URI: " SWAGGER_URI

if [ -z ${SWAGGER_URI} ]; then
  echo -e "${RED}please provide Swagger URI${NC}"
  exit
fi

curl --silent "$SWAGGER_URI" | jq "." - > ./app/swagger.json


export RG_NAME="$(az group list | jq -r '.[0].name' -)"
export ACR_LOGIN_SERVER="$(az acr list | jq -r '.[0].loginServer' -)"
export ACR_NAME="$(az acr list | jq -r '.[0].name' -)"
export IMAGE_NAME="ai-in-a-day-demo"
export IMAGE_TAG="latest"
export CONTAINER_GROUP_NAME="ai-in-a-day-demo"
export RAND_STRING="$(date +%s | sha256sum | base64 | head -c 32 | tr '[:upper:]' '[:lower:]')"
export ACR_USERNAME="$(az acr credential show --name $ACR_NAME --resource-group $RG_NAME | jq -r '.username' -)"
export ACR_PASSWORD="$(az acr credential show --name $ACR_NAME --resource-group $RG_NAME | jq -r '.passwords[0].value' -)"



az acr build --registry "$ACR_NAME" --image "${IMAGE_NAME}:${IMAGE_TAG}" .

az container create \
  --resource-group "$RG_NAME" \
  --name "$CONTAINER_GROUP_NAME" \
  --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
  --registry-username "$ACR_USERNAME" \
  --registry-password "$ACR_PASSWORD" \
  --dns-name-label "$RAND_STRING" \
  --ports 5000 \
  --environment-variables SCORING_ENDPOINT=$SCORING_ENDPOINT

