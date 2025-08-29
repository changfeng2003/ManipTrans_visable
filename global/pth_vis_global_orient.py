import torch
import os

def view_single_global_orient(pth_file_path):
    """
    查看单个.pth文件中 pred_mano_params['global_orient'] 的数据
    """
    # 检查文件是否存在
    if not os.path.exists(pth_file_path):
        print(f"错误：文件不存在 -> {pth_file_path}")
        return

    try:
        # 加载.pth文件（用CPU加载，避免设备冲突）
        data = torch.load(pth_file_path, map_location=torch.device('cpu'))

        # 1. 先检查是否有 pred_mano_params 字段
        if 'pred_mano_params' not in data:
            print("文件中未找到 'pred_mano_params' 字段")
            return

        pred_mano = data['pred_mano_params']
        # 2. 检查 pred_mano_params 是否为字典（符合MANO模型参数的常规格式）
        if not isinstance(pred_mano, dict):
            print(f"'pred_mano_params' 不是字典类型，实际类型：{type(pred_mano)}")
            return

        # 3. 检查是否有 global_orient 字段
        if 'global_orient' not in pred_mano:
            print("pred_mano_params 中未找到 'global_orient' 字段")
            print(f"pred_mano_params 包含的字段：{list(pred_mano.keys())}")  # 显示现有字段
            return

        # 4. 提取并打印 global_orient 数据
        global_orient = pred_mano['global_orient']
        print(f"=== {os.path.basename(pth_file_path)} 的 global_orient 数据 ===")
        print(f"数据类型：{type(global_orient)}")
        if isinstance(global_orient, torch.Tensor):
            print(f"张量形状：{global_orient.shape}")  # MANO的global_orient通常是(1,3)或(3,)（3维旋转向量）
            print(f"数据类型（dtype）：{global_orient.dtype}")
            print(f"具体数值：\n{global_orient}")  # 打印完整数据
            # 若需转为Python列表（更易读），可添加：
            # print(f"转为列表：{global_orient.squeeze().tolist()}")
        else:
            print(f"具体数值：{global_orient}")

    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

# --------------------------
# 使用示例
# --------------------------
if __name__ == "__main__":
    # 替换为你的.pth文件路径（单个文件）
    single_pth_path = "/home/azh/桌面/code/ManipTrans/data/HoldBowl/HoldBowl_250826/0061.pth"
    view_single_global_orient(single_pth_path)
