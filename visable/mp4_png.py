import cv2
import os
from tqdm import tqdm

def video_to_frames(video_path, output_dir='frames', frame_format='png', interval=1):
    """
    将MP4视频拆分成帧并保存为图片
    
    参数:
        video_path (str): 输入视频文件的路径
        output_dir (str): 帧图片保存的目录，默认是'frames'
        frame_format (str): 保存的图片格式，支持'jpg'或'png'，默认是'jpg'
        interval (int): 帧间隔，1表示保存所有帧，2表示每2帧保存1帧，默认是1
    """
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件不存在 - {video_path}")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print(f"错误：无法打开视频文件 - {video_path}")
        return
    
    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print(f"图片格式: {frame_format}")
    print(f"帧间隔: {interval}")
    
    # 遍历视频帧
    frame_count = 0
    saved_count = 0
    
    # 使用tqdm显示进度条
    with tqdm(total=total_frames, desc="处理进度") as pbar:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            
            # 如果读取失败，说明已到视频末尾
            if not ret:
                break
            
            # 按照间隔保存帧
            if frame_count % interval == 0:
                # 生成保存的文件名，格式为：frame_00000.jpg
                frame_filename = f"frame_{saved_count:05d}.{frame_format}"
                frame_path = os.path.join(output_dir, frame_filename)
                
                # 保存帧
                cv2.imwrite(frame_path, frame)
                saved_count += 1
            
            frame_count += 1
            pbar.update(1)
    
    # 释放资源
    cap.release()
    print(f"\n处理完成，共保存 {saved_count} 帧图片")

if __name__ == "__main__":
    # 示例用法
    video_path = "/home/azh/下载/250826_HoldBowl.mp4"  # 替换为你的视频文件路径
    output_dir = "/home/azh/图片/HoldBowlLH"  # 帧图片保存目录
    frame_format = "png"  # 保存格式：jpg或png
    interval = 1  # 保存所有帧，若改为5则每5帧保存1帧
    
    video_to_frames(
        video_path=video_path,
        output_dir=output_dir,
        frame_format=frame_format,
        interval=interval
    )
    