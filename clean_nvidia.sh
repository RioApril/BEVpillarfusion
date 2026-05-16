#!/bin/bash
# 清除所有占用 NVIDIA 设备的进程

# 获取进程 ID 列表（跳过标题行，只取第二列的数字，并去重）
pids=$(sudo fuser -v /dev/nvidia* 2>&1 | awk 'NR>2 && $2 ~ /^[0-9]+$/ {print $2}' | sort -u)

if [ -z "$pids" ]; then
    echo "未发现占用 NVIDIA 设备的进程。"
    exit 0
fi

echo "以下进程将被终止："
echo "$pids"
echo

# 可选：取消注释以进行确认
# read -p "确认终止？(y/n) " confirm
# if [ "$confirm" != "y" ]; then
#     echo "操作已取消。"
#     exit 0
# fi

sudo kill -9 $pids
echo "进程已清理。"
