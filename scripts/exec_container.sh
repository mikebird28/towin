#!/bin/sh

[ -z "$1" ] && { echo "Error : you should specify instance name."; exit 1; }
[ -z "$2" ] && { echo "Error : you should specify container image."; exit 1; }

INSTANCE_NAME=$1
CONTAINER_IMAGE=$2

#Google Compute Engineでイメージを実行
#引数の詳細は https://cloud.google.com/sdk/gcloud/reference/beta/compute/instances/create-with-container 
# --preemptible ： プリエンプティブモードで実行
# --machine-type : 使用するインスタンスのCPUやメモリなどの設定
#                　"gcloud compute machine-types list"で使用可能な引数の一覧が取得可能
#                  今回は独自設定のものを使用（4コアCPU、メモリ20GB）
# --boot-disk-size=50GB : ディスクサイズ 今回は50GB

gcloud beta compute instances create-with-container ${INSTANCE_NAME} \
    --container-image ${CONTAINER_IMAGE} \
    --preemptible \
    --zone=us-east1-b \
    --machine-type=custom-4-20480 \
    --subnet=default \
    --network-tier=PREMIUM \
    --metadata=google-logging-enabled=true \
    --no-restart-on-failure \
    --maintenance-policy=TERMINATE \
    --preemptible \
    --image=cos-stable-70-11021-62-0 \
    --image-project=cos-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-standard\
    --boot-disk-device-name=${INSTANCE_NAME} \
