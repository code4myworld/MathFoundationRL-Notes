import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
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

    bar_bg = Rectangle((0.2, 0.93), 0.6, 0.02,
                   transform=fig.transFigure,
                   facecolor="lightgray")
    bar_fg = Rectangle((0.2, 0.93), 0.0, 0.02,
                    transform=fig.transFigure,
                    facecolor="green")

    fig.patches.extend([bar_bg, bar_fg])


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

        # 更新进度条
        progress = (frame + 1) / len(V_list)
        bar_fg.set_width(0.6 * progress)
        artists.append(bar_fg)

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
                if action in ["up", "down", "left", "right", None]:
                    dx, dy = 0, 0
                    if action == "up":
                        dy = 0.4
                    elif action == "down":
                        dy = -0.4
                    elif action == "left":
                        dx = -0.4
                    elif action == "right":
                        dx = 0.4
                    elif action is None:
                        # 画圆圈表示无动作
                        circle = patches.Circle(
                            (x, n - y + 1),
                            radius=0.1,
                            edgecolor="black",
                            facecolor="none",
                            linewidth=2
                        )
                        ax[0].add_patch(circle)
                        artists.append(circle)

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
            if action in ["up", "down", "left", "right", None]: 
                dx, dy = 0, 0 
                if action == "up": 
                    dy = 0.4 
                elif action == "down": 
                    dy = -0.4 
                elif action == "left": 
                    dx = -0.4 
                elif action == "right": 
                    dx = 0.4 
                elif action is None:
                    # 画圆圈表示无动作
                    circle = patches.Circle( 
                        (x, n - y + 1), 
                        radius=0.1, 
                        edgecolor="black", 
                        facecolor="none", 
                        linewidth=2 
                    )
                    ax[0].add_patch(circle) 

                ax[0].arrow( 
                    x, n - y + 1, 
                    dx, dy, 
                    head_width=0.1, 
                    head_length=0.1, 
                    fc="black", 
                    ec="black", 
                    length_includes_head=True 
                ) 

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


# 根据策略PI构建状态转移概率矩阵P_PI， 
# PI: 策略矩阵，n x n 大小，每个元素为 'up' | 'down' | 'left' | 'right' | None
# P_PI: 共n*n种策略，故状态转移概率矩阵大小为 (n*n) x (n*n)
def get_P_PI_from_PI(PI):
    if len(PI) < 1:
        raise ValueError("PI must be a non-empty matrix")
    
    n = len(PI) ** 2 # 状态数量 n*n
    P_PI = np.zeros((n, n)) 

    for i in range(len(PI)): # 行
        for j in range(len(PI[0])):  # 列
            #第 i * len(PI) + j 个状态， 即P_PI中的第 i * n + j 行
            # col = raw = i * len(PI) + j
            action = PI[i][j]
            next_i, next_j = i, j
            if action is None:
                pass
            elif action == 'right':
                next_j += 1 if next_j + 1 < len(PI[0]) else 0
            elif action == 'left':
                next_j -= 1 if next_j - 1 >= 0 else 0
            elif action == 'up':
                next_i -= 1 if next_i - 1 >= 0 else 0
            elif action == 'down':
                next_i += 1 if next_i + 1 < len(PI) else 0
            else:
                raise ValueError("Invalid action in PI")
            
            col = i * len(PI) + j
            raw = next_i * len(PI) + next_j
            P_PI[col][raw] = 1

    return np.array(P_PI)


# 根据策略PI构建奖励向量r_PI，
# PI: 策略矩阵，n x n 大小，每个元素为 'up' | 'down' | 'left' | 'right' | None
# r_PI: 共n*n种策略，故奖励向量大小为 n*n
# 注意 forbiddens 和 targets 中的坐标是从 (1,1) 开始计数的
def get_r_PI_from_PI(PI, forbiddens, targets):
    if len(PI) < 1:
        raise ValueError("PI must be a non-empty matrix")
    
    n = len(PI) ** 2 # 状态数量 n*n
    r_PI = np.zeros(n) 

    # 根据当前状态和动作，判断下一个状态。
    # if next_state in targets: reward = 1 else reward = 0
    # if next_state in forbiddens: reward = -1 else reward = 0
    # if next_state is out of boundary: reward = -1
    # else reward = 0
    for i in range(len(PI)): # i 代表 y 轴
        for j in range(len(PI[0])): # j 代表 x 轴
            # 第 i * len(PI) + j 个状态， 即r_PI中的第 i * n + j 个元素
            idx = i * len(PI) + j
            action = PI[i][j]
            next_i, next_j = i, j
            if action is None:
                pass
            elif action == 'right':
                next_j += 1        
            elif action == 'left':
                next_j -= 1 
            elif action == 'up':
                next_i -= 1 
            elif action == 'down':
                next_i += 1 
            else:
                raise ValueError("Invalid action in PI")

            next_state = (next_j + 1, next_i + 1) # 转换为从 (1,1) 开始计数的坐标
            if next_state in targets:
                r_PI[idx] = 1
            elif next_state in forbiddens:
                r_PI[idx] = -1
            elif next_i < 0 or next_i >= len(PI) or next_j < 0 or next_j >= len(PI[0]):
                r_PI[idx] = -1
            else:
                r_PI[idx] = 0

    return np.array(r_PI)

# get_P_PI_from_PI(PI=PI)
# get_r_PI_from_PI(PI=PI, forbiddens=forbiddens, targets=targets)


# 给定策略PI, 环境forbidden area, target area, 和 迭代求解次数T, 折扣因子gamma
# 返回状态价值函数v
def iterative_solution_state_value_from_bellman(PI, forbiddens, targets, T, gamma=0.9):
    v = np.zeros(len(PI) * len(PI)) # 初始化状态价值函数向量 v, 等于state数量，即 n*n
    P_PI = get_P_PI_from_PI(PI=PI) # 根据策略PI构建状态转移概率矩阵P_PI
    r_PI = get_r_PI_from_PI(PI=PI, forbiddens=forbiddens, targets=targets) # 根据策略PI构建奖励向量r_PI

    V_list = []
    PI_list = []
    for _ in range(T):
        v = r_PI + gamma * P_PI.dot(v)
        V_list.append(v.reshape(len(PI), len(PI)).copy())
        PI_list.append(PI)

    # for i in range(len(PI)):
    #     for j in range(len(PI)):
    #         print(f"{v[i*len(PI)+j]:.1f}", end=' ')
    #     print()

    # 可视化结果（静态）
    # draw_grid(n=len(PI), forbidden=forbiddens, targets=targets, PI=PI, V=v.reshape(len(PI), len(PI)))

    # 可视化结果（动态）
    draw_grid_animation(
        n=len(PI),
        forbidden=forbiddens,
        targets=targets,
        PI_list=PI_list,
        V_list=V_list,
        interval=100
    )

    return v



forbiddens = {(2, 2), (2, 4), (2, 5), (3, 2), (3, 3), (4, 4)}
targets = {(3, 4)}
# PI = [
#     ['right', 'right', 'right', 'down', 'down'],
#     ['up', 'up', 'right', 'down', 'down'],
#     ['up', 'left', 'down', 'right', 'down'],
#     ['up', 'right', None, 'left', 'down'],
#     ['up', 'right', 'up', 'left', 'left']
# ]

PI = [
    ['right', 'right', 'right', 'right', 'right'],
    ['right', 'right', 'right', 'right', 'right'],
    ['right', 'right', 'right', 'right', 'right'],
    ['right', 'right', 'right', 'right', 'right'],
    ['right', 'right', 'right', 'right', 'right']
]

# PI = [
#     ['right', 'left', 'left', 'up', 'up'],
#     ['down', None, 'right', 'down', 'right'],
#     ['left', 'right', 'down', 'left', None],
#     [None, 'down', 'up', 'up', 'right'],
#     [None, 'right', None, 'right', None]
# ]


iterative_solution_state_value_from_bellman(PI=PI, forbiddens=forbiddens, targets=targets, T=100, gamma=0.9)

# r_PI_case1 = get_r_PI_from_PI(PI, forbiddens, targets)
# for i in range(len(PI)):
#     for j in range(len(PI)):
#         print(r_PI_case1[i*len(PI)+j], end=' ')
#     print()

# P_PI = get_P_PI_from_PI(PI=PI)
# for i in range(len(P_PI)):
#     for j in range(len(P_PI)):
#         print(f"{P_PI[i][j]:.0f}", end=', ')
#     print()

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
