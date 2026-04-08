# ==================== 配置参数 ====================
# IMG_DIRS="${IMG_DIRS:-/home/leo/dataset/drawings/ab_af_c_c_d_anno/dota/images/}"
# ANN_DIRS="${ANN_DIRS:-/home/leo/dataset/drawings/ab_af_c_c_d_anno/dota/labels/}"
# SAVE_DIR="${SAVE_DIR:-/home/leo/dataset/drawings/ab_af_c_c_d_anno/dota/split_output}"  # 输出目录
BASE_JSON="${BASE_JSON:-/home/leo/code/python_code/obb_ai4rs/tools/data/dota/split/split_configs/ss_trainval.json}"  # 输出目录
# =================================================

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 运行 Python 脚本
python "$SCRIPT_DIR/img_split.py" \
    --base-json "$BASE_JSON"