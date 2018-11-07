#!/bin/sh

#引数の確認
[ -z "$1" ] && { echo "Error : you should specify instance name"; exit 1; }

gcloud compute --project "$CLOUDML_PROJECT_ID" ssh --zone "us-east1-b" "$1" \
    --command 'docker logs -f `docker ps -l --format "{{.ID}}"`'
