#!/bin/bash
set -ex

PYTHON_VERSION=$1
CUDA_VERSION=$2

if [ ! -z $CUSTOM_PYTHON_VERSION ]; then
    PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
fi

if [ ! -z $CUSTOM_CUDA_VERSION ]; then
    CUDA_VERSION=$CUSTOM_CUDA_VERSION
fi

if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.10"
fi

if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="12.6"
fi

ROOT_PATH=$(pwd)
OUTPUT_PATH=$ROOT_PATH/output
mkdir -p $OUTPUT_PATH
cd sgl-kernel

# 获取当前分支名，并将特殊字符转换为下划线
BUILD_TIME=$(date +%Y%m%d%H%M)
BRANCH_NAME=$(git rev-parse --abbrev-ref HEAD)
echo "BRANCH_NAME: $BRANCH_NAME"

# 如果分支是以 release_ 或 release/ 开头，则将 release_ 或 release/ 替换为空
if [[ $BRANCH_NAME =~ ^release[\/_] ]]; then
   echo "release branch"
   BRANCH_NAME=${BRANCH_NAME#release}
   BRANCH_NAME=${BRANCH_NAME#/}
   BRANCH_NAME=${BRANCH_NAME#_}
   # 如果分支里还有 / ，则将 / 替换为 .
   BRANCH_NAME=${BRANCH_NAME//\//.}
   if [[ ! -z $BRANCH_NAME ]]; then
       BRANCH_NAME=.${BRANCH_NAME}
   fi
   VERSION_SUFFIX=+byted${BRANCH_NAME}.${BUILD_TIME}
elif [[ $BRANCH_NAME == ep_main ]]; then
   VERSION_SUFFIX=+iaas.dev.${BUILD_TIME}
else
   echo "not release branch"
   VERSION_SUFFIX=+byted.${BUILD_TIME}
fi

CLEAR_BRANCH_NAME=${BRANCH_NAME//\//.}
CACHE_TAR_NAME=$PYTHON_VERSION-$CUDA_VERSION-$CLEAR_BRANCH_NAME.tar
echo "BRANCH_NAME: $BRANCH_NAME"
echo "CLEAR_BRANCH_NAME: $CLEAR_BRANCH_NAME"
echo "CACHE_TAR_NAME: $CACHE_TAR_NAME"

echo "VERSION_SUFFIX: $VERSION_SUFFIX"

ENABLE_SM90A=$(( ${CUDA_VERSION%.*} >= 12 ? ON : OFF ))

VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)
# 移除可能存在的 byted 后缀，获取基础版本号
BASE_VERSION=$(echo $VERSION | sed 's/+byted.*$//')
echo "Building sglang-python version $BASE_VERSION$VERSION_SUFFIX"
sed -i "s/^version = .*$/version = \"$BASE_VERSION$VERSION_SUFFIX\"/" pyproject.toml

# 如果设置了 CUSTOM_CACHE_TOS_AK，则尝试从 tos 里下载 cache
if [ -z "$CUSTOM_CACHE_TOS_AK" ]; then
    echo "not set CUSTOM_CACHE_TOS_AK, skip download cache"
else
    # 安装 toscli 工具
    https_proxy= curl -s https://luban-source.byted.org/repository/scm/toutiao.tos.toscli_1.0.0.20.tar.gz -o - | tar xz -C . ./toscli && chmod +x ./toscli && mv ./toscli /usr/bin/
    # 从 tos 里下载 cache
    toscli -timeout 30m -bucket iaas-servingkit -accessKey $CUSTOM_CACHE_TOS_AK get -filename /tmp/$CACHE_TAR_NAME cache/sgl-kernel/$CACHE_TAR_NAME || true
    if [ -f "/tmp/$CACHE_TAR_NAME" ]; then
        tar -xvf /tmp/$CACHE_TAR_NAME -C /
    fi
fi




proxy_args=""
if [ ! -z "$http_proxy" ]; then
    proxy_args="$proxy_args -e http_proxy=$http_proxy"
fi
if [ ! -z "$https_proxy" ]; then
    proxy_args="$proxy_args -e https_proxy=$https_proxy"
fi

# 如果设置了 pip 缓存参数，则删除 --no-cache-dir 参数，并映射 ~/.cache 到容器内
if [ "$CUSTOM_CACHE_PIP" == "true" ]; then
    cache_args="-v ~/.cache:/root/.cache"
    sed -i 's|--no-cache-dir||g' build.sh  # 删除 pip 缓存参数
fi

sed -i "s|docker run --rm|docker run --rm --network=host $proxy_args $cache_args|" build.sh
sed -i "s|pytorch/manylinux|hub.byted.org/iaas/manylinux|g" build.sh
sed -i 's|ARCH=$(uname -i)|ARCH=x86_64|g' build.sh  # DinD 可能不支持 ARCH=$(uname -i)

# 如果是 SCM 构建，则准备 docker 环境
if [[ "${SCM_BUILD}" == "True" ]]; then
    source /root/start_dockerd.sh
fi

source build.sh $PYTHON_VERSION $CUDA_VERSION

# 产物放到 output 目录下
cp -r $ROOT_PATH/sgl-kernel/dist/* $OUTPUT_PATH/

TOS_UTIL_URL=https://tos-tools.dualstack.cn-beijing.tos.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi

if [ -z "$CUSTOM_TOS_AK" ] && [ -z "$CUSTOM_TOS_SK" ]; then
    echo "CUSTOM_TOS_AK and CUSTOM_TOS_SK are not set, skip uploading to tos"
else
    # 安装 tosutil
    wget $TOS_UTIL_URL -O /usr/bin/tosutil && chmod +x /usr/bin/tosutil
    # 上传制品到 tos
    for wheel_file in $(find $OUTPUT_PATH -name "*.whl"); do
        echo "uploading $wheel_file to tos..."
        tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/sgl-kernel/$(basename $wheel_file) -re cn-beijing -e dualstack.cn-beijing.tos.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi


# 保存 cache 到 tos
if [ -z "$CUSTOM_CACHE_TOS_AK" ]; then
    echo "not set CUSTOM_CACHE_TOS_AK, skip upload cache"
else
    # 先删除旧的
    toscli -timeout 10m -bucket iaas-servingkit -accessKey $CUSTOM_CACHE_TOS_AK del cache/sgl-kernel/$CACHE_TAR_NAME || true
    tar -cvf $CACHE_TAR_NAME ~/.cache && toscli -timeout 30m -bucket iaas-servingkit -accessKey $CUSTOM_CACHE_TOS_AK put -verbose -prefix cache/sgl-kernel/ -name $CACHE_TAR_NAME $CACHE_TAR_NAME
fi
