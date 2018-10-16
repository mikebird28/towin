
TAG="lgb-container"
HOST_REGION="asia.gcr.io"
PROJECT_ID="ml-env-217713"
IMAGE_NAME="lgb_container"

gcloud container images delete $HOST_REGION/$PROJECT_ID/$IMAGE_NAME --force-delete-tags
