import torch
import os

def print_all_right_bool(data_dir):
    """
    遍历指定目录下所有.pth文件，打印每个文件的right_bool字段（兼容多元素张量）
    """
    if not os.path.exists(data_dir):
        print(f"错误：目录不存在 -> {data_dir}")
        return
    
    pth_files = [f for f in os.listdir(data_dir) if f.endswith('.pth')]
    if not pth_files:
        print(f"提示：在目录 {data_dir} 中未找到.pth文件")
        return
    
    pth_files.sort()
    print(f"找到 {len(pth_files)} 个.pth文件，开始读取right_bool字段...\n")
    
    for idx, filename in enumerate(pth_files, 1):
        file_path = os.path.join(data_dir, filename)
        try:
            data = torch.load(file_path, map_location=torch.device('cpu'))
            
            if 'right_bool' in data:
                right_bool = data['right_bool']
                # 检查是否为张量
                if isinstance(right_bool, torch.Tensor):
                    # 显示张量形状和内容
                    print(f"[{idx}] 文件：{filename} -> right_bool（形状：{right_bool.shape}） = {right_bool.numpy()}")
                else:
                    # 非张量类型直接显示
                    print(f"[{idx}] 文件：{filename} -> right_bool = {right_bool}")
            else:
                print(f"[{idx}] 文件：{filename} -> 未找到right_bool字段")
        
        except Exception as e:
            print(f"[{idx}] 文件：{filename} -> 处理出错：{str(e)}")
    
    print("\n所有文件处理完成")

if __name__ == "__main__":
    data_dir = '/home/azh/桌面/code/ManipTrans/data/HoldBowl/HoldBowl_250825'
    print_all_right_bool(data_dir)
    