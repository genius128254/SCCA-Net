import os

def save_image_paths(folder_path, output_file):
    # 打开输出文件，以写入模式
    with open(output_file, 'w') as file:
        # 遍历文件夹内的所有文件
        for root, _, files in os.walk(folder_path):
            for filename in files:
                # 判断文件是否为图像类型
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.mat')):
                    # 获取完整路径，并去掉引号
                    image_path = os.path.join(root, filename)
                    file.write(image_path + '\n')

# 使用示例
folder_path = 'E:\小论文\hsi\HSI/Vegetation/train'  # 替换为你的图像文件夹路径
output_file = 'E:\小论文\参考文献\ZZNet\ZZNet - 修改UBRFC+光谱SFMA +修改cbam/train.txt'   # 替换为输出txt文件路径
save_image_paths(folder_path, output_file)
