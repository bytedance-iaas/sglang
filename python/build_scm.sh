#!/bin/bash
set -e

ROOT_DIR=$(pwd)
PYTHON=python3
PIP="$PYTHON -m pip"
BUILD_TIME=$(date +%Y%m%d%H%M)
VERSION_SUFFIX=+byted.${BUILD_TIME}

TOS_UTIL_URL=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil
if [ ! -z "$CUSTOM_TOS_UTIL_URL" ]; then
    TOS_UTIL_URL=$CUSTOM_TOS_UTIL_URL
fi

cd ./python

VERSION=$(grep -oP '(?<=version = \")[^\"]+' pyproject.toml)
echo "Building sglang-python version $VERSION$VERSION_SUFFIX"

pyproject_bk=pyproject.toml.bk
cp pyproject.toml $pyproject_bk

sed -i "s/^version = .*$/version = \"$VERSION$VERSION_SUFFIX\"/" pyproject.toml
$PIP install build
$PYTHON -m build

OUTPUT_DIR=$ROOT_DIR/output
mkdir -p $OUTPUT_DIR
mv dist/* $OUTPUT_DIR/
mv $pyproject_bk pyproject.toml

if [ -z "$CUSTOM_TOS_AK" ] || [ -z "$CUSTOM_TOS_SK" ]; then
    echo "CUSTOM_TOS_AK or CUSTOM_TOS_SK is not set, skip uploading to tos"
else
    # 上传制品到 tos
    wget $TOS_UTIL_URL -O tosutil && chmod +x tosutil
    for wheel_file in $(find $OUTPUT_DIR -name "*.whl"); do
        echo "Uploading to tos"
        # 上传制品到 tos
        ./tosutil cp $wheel_file tos://${CUSTOM_TOS_BUCKET}/packages/sglang/$(basename $wheel_file) -re cn-beijing -e tos-cn-beijing.volces.com -i $CUSTOM_TOS_AK -k $CUSTOM_TOS_SK
    done
fi
