import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# =========================
# 绘图函数
# n: 格子数量（假设为 n x n）
# forbidden: 禁止进入的格子集合，元素为 (row, col) 元组
# targets: 目标格子集合，元素为 (row, col) 元组
# PI: 策略矩阵，n x n 大小，每个元素为 'up' | 'down' | 'left' | 'right' | None
# V: 价值矩阵，n x n 大小，每个元素为数字
# =========================
def draw_grid_animation(n, forbidden, targets, PI_list, V_list, interval=1000):

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    color_map = {
        "forbidden": "#F0BB1E",
        "target": "#11EDF5",
        "normal": "white"
    }

    # =======================
    # 绘制静态网格、颜色及目标target圆圈
    # =======================
    for x in range(1, n+1): # x 轴
        for y in range(1, n+1): # y 轴
            # 格子左下角坐标和刻度x,y的关系
            rect_x = x - 0.5
            rect_y = n - y + 0.5  # 翻转 y

            # 确定格子颜色
            if(x, y) in forbidden:
                    facecolor = color_map["forbidden"]
            elif (x, y) in targets:
                facecolor = color_map["target"]
            else:
                facecolor = color_map["normal"]

            # 左子图和右子图均画格子
            for k in range(2):
                rect = patches.Rectangle(
                    (rect_x, rect_y), 1, 1,
                    facecolor=facecolor,
                    edgecolor="black",
                    linewidth=1
                )
                ax[k].add_patch(rect)
    
             # 左子图：targets 中的格子画圆圈
            if (x, y) in targets:
                circle = patches.Circle(
                    (x, n - y + 1),
                    radius=0.1,
                    edgecolor="black",
                    facecolor="none",
                    linewidth=2
                )
                ax[0].add_patch(circle)
            
            
    
    # =======================
    # 初始化箭头和价值文本
    # =======================
    arrows = [[None for _ in range(n)] for _ in range(n)]
    values = [[None for _ in range(n)] for _ in range(n)]

    for x in range(1, n+1): # x 轴
        for y in range(1, n+1): # y 轴
            # 转换为PI和V矩阵的行列索引
            i, j = y - 1, x - 1  
            # 初始化箭头占位
            arrows[i][j] = None
            # 初始化价值文本占位
            t = ax[1].text(
                x, n - y + 1,
                "",
                ha="center",
                va="center",
                fontsize=15
            )
            values[i][j] = t
    
    # =======================
    # update函数：更新箭头和价值文本
    # =======================
    def update(frame):
        PI = PI_list[frame]
        V = V_list[frame]

        artists = []

        for x in range(1, n+1): # x 轴
            for y in range(1, n+1): # y 轴
                # 转换为PI和V矩阵的行列索引
                i, j = y - 1, x - 1  
                # 更新箭头
                if arrows[i][j] is not None:
                    arrows[i][j].remove()  # 移除旧箭头
                    arrows[i][j] = None
                

                # 左子图：遍历PI元组矩阵，画箭头
                action = PI[i][j]
                if action in ["up", "down", "left", "right"]:
                    dx, dy = 0, 0
                    if action == "up":
                        dy = 0.4
                    elif action == "down":
                        dy = -0.4
                    elif action == "left":
                        dx = -0.4
                    elif action == "right":
                        dx = 0.4

                    arr = ax[0].arrow(
                        x, n - y + 1,
                        dx, dy,
                        head_width=0.1,
                        head_length=0.1,
                        fc="black",
                        ec="black",
                        length_includes_head=True
                    )
                    arrows[i][j] = arr
                    artists.append(arr)
                
    
                # 右子图：填价值
                values[i][j].set_text(f"{V[i][j]:.1f}")
                artists.append(values[i][j])
        
        return artists

    for k in range(2):
        ax[k].set_xlim(0.5, n+0.5)
        ax[k].set_ylim(0.5, n+0.5)
        # 把x轴上移到顶部
        ax[k].xaxis.set_ticks_position('top')
        # 坐标轴刻度隐藏
        ax[k].xaxis.set_ticks_position('none') 
        ax[k].yaxis.set_ticks_position('none') 
        # 坐标轴标签
        ax[k].set_yticks([i+1 for i in range(n)]) # 控制y轴刻度位置
        ax[k].set_yticklabels([str(n - i) for i in range(n)]) # 控制y轴刻度标签
        ax[k].set_aspect("equal")


    ani = FuncAnimation(
        fig, update,
        frames=len(V_list),
        interval=interval,
        blit=False,
        repeat=False
    )
    plt.show()

    return ani


def draw_grid(n, forbidden, targets, PI, V): 

    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 

    color_map = { 
        "forbidden": "#F0BB1E", 
        "target": "#11EDF5", 
        "normal": "white" 
    } 
    
    for x in range(1, n+1): # x 轴 
        for y in range(1, n+1): # y 轴 
            # 格子左下角坐标和刻度x,y的关系 
            rect_x = x - 0.5 
            rect_y = n - y + 0.5 # 翻转 y 

            # 左子图和右子图均画格子 
            for k in range(2): 
                rect = patches.Rectangle( 
                    (rect_x, rect_y), 1, 1, 
                    facecolor="white", 
                    edgecolor="black", 
                    linewidth=1 
                ) 
                ax[k].add_patch(rect) 
                
                # 上色 
                if(x, y) in forbidden: 
                    facecolor = color_map["forbidden"] 
                elif (x, y) in targets: 
                    facecolor = color_map["target"] 
                else: 
                    facecolor = color_map["normal"] 
                    
                rect = patches.Rectangle( 
                    (rect_x, rect_y), 1, 1, 
                    facecolor=facecolor, 
                    edgecolor="black", 
                    linewidth=1 
                ) 
                ax[k].add_patch(rect) 
                
            i, j = y - 1, x - 1 # 转换为PI和V矩阵的行列索引 
            # 左子图：遍历PI元组矩阵，画箭头 
            action = PI[i][j] 
            if action in ["up", "down", "left", "right"]: 
                dx, dy = 0, 0 
                if action == "up": 
                    dy = 0.4 
                elif action == "down": 
                    dy = -0.4 
                elif action == "left": 
                    dx = -0.4 
                elif action == "right": 
                    dx = 0.4 
                
                ax[0].arrow( 
                    x, n - y + 1, 
                    dx, dy, 
                    head_width=0.1, 
                    head_length=0.1, 
                    fc="black", 
                    ec="black", 
                    length_includes_head=True 
                ) 
                
            # 左子图：targets 中的格子画圆圈 
            if (x, y) in targets: 
                circle = patches.Circle( 
                    (x, n - y + 1), 
                    radius=0.1, 
                    edgecolor="black", 
                    facecolor="none", 
                    linewidth=2 
                )
                ax[0].add_patch(circle) 
                
            # 右子图：填价值
            value = V[i][j] 
            ax[1].text( 
                x, n - y + 1, 
                f"{value:.1f}", 
                ha="center", 
                va="center", 
                fontsize=15 
            ) 
            
    for k in range(2): 
        ax[k].set_xlim(0.5, n+0.5) 
        ax[k].set_ylim(0.5, n+0.5) 
        # 把x轴上移到顶部 
        ax[k].xaxis.set_ticks_position('top') 
        # 坐标轴刻度隐藏 
        ax[k].xaxis.set_ticks_position('none') 
        ax[k].yaxis.set_ticks_position('none') 
        # 坐标轴标签 
        ax[k].set_yticks([i+1 for i in range(n)]) 
        # 控制y轴刻度位置 
        ax[k].set_yticklabels([str(n - i) for i in range(n)]) 
        # 控制y轴刻度标签 
        ax[k].set_aspect("equal") 
        
    plt.show()






forbiddens = {(2, 2), (2, 4), (2, 5), (3, 2), (3, 3), (4, 4)}
targets = {(3, 4)}
PI = [
    ['right', 'right', 'right', 'down', 'down'],
    ['up', 'up', 'right', 'down', 'down'],
    ['up', 'left', 'down', 'right', 'down'],
    ['up', 'right', None, 'left', 'down'],
    ['up', 'right', 'up', 'left', 'left']
]
V = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]
draw_grid(n=5, forbidden=forbiddens, targets=targets, PI=PI, V=V)


# forbiddens = {(2, 2), (2, 4), (2, 5), (3, 2), (3, 3), (4, 4)}
# targets = {(3, 4)}
# n = 5
# PI_list = []
# V_list = []
# actions = ['right', 'down', 'left', 'up']
# for t in range(3):
#     PI = [[actions[(i + j + t) % 4] for j in range(n)] for i in range(n)]
#     V  = np.random.rand(n, n) * (t + 1)
#     PI_list.append(PI)
#     V_list.append(V)

# draw_grid_animation(
#     n=5,
#     forbidden=forbiddens,
#     targets=targets,
#     PI_list=PI_list,
#     V_list=V_list,
#     interval=1000
# )