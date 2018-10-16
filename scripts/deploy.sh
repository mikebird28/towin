
TAG="lgb-container"
HOST_REGION="asia.gcr.io"
PROJECT_ID="ml-env-217713"
IMAGE_NAME="lgb_container"

#https://cloud.google.com/container-registry/docs/pushing-and-pulling
echo "Step.1 : Build image"
docker build . -t $IMAGE_NAME

echo "\nStep.2 : Push image"
docker tag $IMAGE_NAME $HOST_REGION/$PROJECT_ID/$IMAGE_NAME
docker push $HOST_REGION/$PROJECT_ID/$IMAGE_NAME

#gcloud beta compute instances create-with-container [INSTANCE_NAME] \
#     --container-image [DOCKER_IMAGE]
"
gcloud beta compute --project=ml-env-217713 instances create-with-container instance-1 \
    --zone=us-east1-b \
    --machine-type=custom-1-12288-ext \
    --subnet=default \
    --network-tier=PREMIUM \
    --metadata=google-logging-enabled=true \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --service-account=665832082088-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append 
    --image=cos-stable-69-10895-71-0 
    --image-project=cos-cloud 
    --boot-disk-size=10GB 
    --boot-disk-type=pd-standard 
    --boot-disk-device-name=instance-1 
    --container-image=asia.gcr.io/ml-env-217713/lgb_container 
    --container-restart-policy=always 
    --labels=container-vm=cos-stable-69-10895-71-0
"
 


