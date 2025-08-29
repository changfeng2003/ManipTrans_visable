import torch
import os
from pprint import pprint

def view_pth_file(data_dir):
    # 检查目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误：目录不存在 - {data_dir}")
        return
    
    # 获取目录中所有.pth文件
    pth_files = [f for f in os.listdir(data_dir) if f.endswith('.pth')]
    
    if not pth_files:
        print(f"在目录 {data_dir} 中未找到.pth文件")
        return
    
    # 取第一个.pth文件
    target_file = os.path.join(data_dir, pth_files[0])
    print(f"正在查看文件：{target_file}\n")
    
    try:
        # 加载.pth文件
        data = torch.load(target_file, map_location=torch.device('cpu'))
        
        # 显示文件中包含的所有字段
        print("文件包含以下字段：")
        pprint(list(data.keys()))
        print("\n" + "="*50 + "\n")
        
        # 详细展示每个字段的信息
        for key, value in data.items():
            print(f"字段名: {key}")
            print(f"数据类型: {type(value)}")
            
            # 根据数据类型显示不同信息
            if isinstance(value, torch.Tensor):
                print(f"张量形状: {value.shape}")
                print(f"数据类型: {value.dtype}")
                print(f"示例数据: {value[:2] if value.numel() > 2 else value}")  # 显示前2个元素
            elif isinstance(value, (list, tuple)):
                print(f"长度: {len(value)}")
                print(f"前2个元素: {value[:2] if len(value) > 2 else value}")
            elif isinstance(value, dict):
                print(f"包含键值对数量: {len(value)}")
                print(f"前2个键: {list(value.keys())[:2] if len(value) > 2 else list(value.keys())}")
            else:
                print(f"值: {value}")
            
            print("\n" + "-"*50 + "\n")
            
    except Exception as e:
        print(f"处理文件时出错：{str(e)}")

if __name__ == "__main__":
    # 你的.pth文件所在目录
    data_dir = '/home/azh/桌面/code/ManipTrans/data/HoldPaperCup2MT/HANDS_with_OBJ2_250822'
    view_pth_file(data_dir)
    