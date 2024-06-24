#!/bin/bash
WORK_DIR="work_dirs/selfsup/byol/byol_resnet50_1xb32-300e_OAI/"
#pre_ifs="$IFS"
IFS="/"
array=($WORK_DIR)
echo ${array}
downstream=${array[0]}
downstream+="/"
downstream+=${array[3]}
downstream+="/"
IFS=" "
echo ${downstream}

