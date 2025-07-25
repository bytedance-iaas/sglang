#!/bin/bash
set -ex

PYTHON_VERSION=$1

if [ ! -z $CUSTOM_PYTHON_VERSION ]; then
    PYTHON_VERSION=$CUSTOM_PYTHON_VERSION
fi

if [ -z "$PYTHON_VERSION" ]; then
    PYTHON_VERSION="3.10"
fi

ROOT_PATH=$(pwd)
OUTPUT_PATH=$ROOT_PATH/output
mkdir -p $OUTPUT_PATH
cd sgl-router

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

echo "VERSION_SUFFIX: $VERSION_SUFFIX"

VERSION=$(sed -n 's/^version = "\([^"]*\)"/\1/p' pyproject.toml)
# 移除可能存在的 byted 后缀，获取基础版本号
BASE_VERSION=$(echo $VERSION | sed 's/+byted.*$//')
echo "Building sglang-python version $BASE_VERSION$VERSION_SUFFIX"
sed -i "s/^version = .*$/version = \"$BASE_VERSION$VERSION_SUFFIX\"/" pyproject.toml

proxy_args=""
if [ ! -z "$http_proxy" ]; then
    proxy_args="$proxy_args -e http_proxy=$http_proxy"
fi
if [ ! -z "$https_proxy" ]; then
    proxy_args="$proxy_args -e https_proxy=$https_proxy"
fi

# 开始编译 sglang-router wheel包
source build.sh $PYTHON_VERSION 

# 产物放到 output 目录下
cp -r $ROOT_PATH/sgl-router/dist/* $OUTPUT_PATH/

TOS_UTIL_URL=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi

if [ -z "$CUSTOM_TOS_AK" ] && [ -z "$CUSTOM_TOS_SK" ]; then
    echo "CUSTOM_TOS_AK and CUSTOM_TOS_SK are not set, skip uploading to tos"
else
    # 上传制品到 tos
    wget $TOS_UTIL_URL -O tosutil && chmod +x tosutil
    for wheel_file in $(find $OUTPUT_PATH -name "*.whl"); do
        echo "uploading $wheel_file to tos..."
        ./tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/sglang-router/$(basename $wheel_file) -re cn-beijing -e tos-cn-beijing.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi
