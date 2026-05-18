#!/bin/bash
# ============================================================
# 热力图实验：四卡并行启动脚本
# 将 35 个组合拆成 4 份，分别在 4 张 GPU 上并行运行
#
# 用法：
#   bash run_utility_bert_heatmap_parallel.sh [eps_target]
#
# 默认 eps_target=10，使用 GPU 0, 1, 2, 3
# 可通过修改下方 GPU_LIST 变量更换 GPU 编号
# ============================================================

EPS_TARGET=${1:-10}

# ---- 四张 GPU 的编号（按需修改）----
GPU_LIST=(0 2 3 5)

# ---- eps_low 分组（5 个 eps_low 值分成 4 组，尽量均匀）----
# 每个 eps_low 对应 7 个 eps_high，共 7 个组合
# GPU 0: eps_low = 0       (7 个组合 × 5 runs = 35 次)
# GPU 1: eps_low = 2       (7 个组合 × 5 runs = 35 次)
# GPU 2: eps_low = 4       (7 个组合 × 5 runs = 35 次)
# GPU 3: eps_low = 6, 8    (14 个组合 × 5 runs = 70 次)
GROUP_0="0"
GROUP_1="2"
GROUP_2="4"
GROUP_3="6 8"

echo "============================================"
echo " 热力图实验：四卡并行"
echo " EPS_TARGET: $EPS_TARGET"
echo " GPU 分配:"
echo "   GPU ${GPU_LIST[0]}: eps_low = $GROUP_0  (7 组合)"
echo "   GPU ${GPU_LIST[1]}: eps_low = $GROUP_1  (7 组合)"
echo "   GPU ${GPU_LIST[2]}: eps_low = $GROUP_2  (7 组合)"
echo "   GPU ${GPU_LIST[3]}: eps_low = $GROUP_3  (14 组合)"
echo "============================================"
echo ""

# 启动四个后台任务
CUDA_VISIBLE_DEVICES=${GPU_LIST[0]} bash run_utility_bert_heatmap.sh $EPS_TARGET "$GROUP_0" > log_heatmap_gpu${GPU_LIST[0]}.log 2>&1 &
PID0=$!
echo "GPU ${GPU_LIST[0]} 已启动 (PID=$PID0), eps_low=[$GROUP_0], 日志: log_heatmap_gpu${GPU_LIST[0]}.log"

CUDA_VISIBLE_DEVICES=${GPU_LIST[1]} bash run_utility_bert_heatmap.sh $EPS_TARGET "$GROUP_1" > log_heatmap_gpu${GPU_LIST[1]}.log 2>&1 &
PID1=$!
echo "GPU ${GPU_LIST[1]} 已启动 (PID=$PID1), eps_low=[$GROUP_1], 日志: log_heatmap_gpu${GPU_LIST[1]}.log"

CUDA_VISIBLE_DEVICES=${GPU_LIST[2]} bash run_utility_bert_heatmap.sh $EPS_TARGET "$GROUP_2" > log_heatmap_gpu${GPU_LIST[2]}.log 2>&1 &
PID2=$!
echo "GPU ${GPU_LIST[2]} 已启动 (PID=$PID2), eps_low=[$GROUP_2], 日志: log_heatmap_gpu${GPU_LIST[2]}.log"

CUDA_VISIBLE_DEVICES=${GPU_LIST[3]} bash run_utility_bert_heatmap.sh $EPS_TARGET "$GROUP_3" > log_heatmap_gpu${GPU_LIST[3]}.log 2>&1 &
PID3=$!
echo "GPU ${GPU_LIST[3]} 已启动 (PID=$PID3), eps_low=[$GROUP_3], 日志: log_heatmap_gpu${GPU_LIST[3]}.log"

echo ""
echo "============================================"
echo " 所有任务已在后台启动"
echo " 查看进度: tail -f log_heatmap_gpu*.log"
echo " 等待完成: wait $PID0 $PID1 $PID2 $PID3"
echo "============================================"

# 等待所有任务完成
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "============================================"
echo " 所有 GPU 任务已完成！"
echo "============================================"
