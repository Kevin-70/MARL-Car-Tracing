from PIL import Image
import os

# 读取SVG文件
print(os.path.abspath('.'))
svg_image = Image.open('./out2.svg')

# 将SVG图像转换为GIF帧
frames = []
for i in range(10):
    # 创建一个新的图像对象
    new_image = Image.new('RGBA', svg_image.size, (255, 255, 255))

    # 将SVG图像粘贴到新图像对象上
    new_image.paste(svg_image, (0, 0))

    # 添加到帧列表中
    frames.append(new_image)

# 保存为GIF文件
frames[0].save('out2.gif', save_all=True, append_images=frames[1:],
               optimize=False, duration=200, loop=0)
