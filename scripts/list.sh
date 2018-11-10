#!/bin/sh
gcloud container images list --repository=$CLOUDML_HOST_REGION/$CLOUDML_PROJECT_ID
