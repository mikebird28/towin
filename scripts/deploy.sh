#!/bin/sh

#引数の確認
[ -z "$1" ] && { echo "Error : Need do specify target file"; exit 1; }

#環境変数の確認
[ -z "$CLOUDML_HOST_REGION" ] && { echo "Error : Need to set CLOUDML_HOST_REGION"; exit 1; }
[ -z "$CLOUDML_PROJECT_ID" ] && { echo "Error : Need to set CLOUDML_PROJECT_ID";  exit 1; }
[ -z "$CLOUDML_IMAGE_NAME" ] && { echo "Error : Need to set CLOUDML_IMAGE_NAME"; exit 1; }

REPOGITRY_NAME=$CLOUDML_HOST_REGION/$CLOUDML_PROJECT_ID/$CLOUDML_IMAGE_NAME

echo "Step.1 : Build image"
docker build . -t $REPOGITRY_NAME --build-arg SCRIPT_FILE="$1" || { echo "Failed to build image"; exit 1; }

echo "\nStep.2 : Push image"
docker push $REPOGITRY_NAME || { echo "Failed to push image";  exit 1; }

echo "\nSuccess to upload $REPOGITRY_NAME"

