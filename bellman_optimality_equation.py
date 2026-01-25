import numpy as np

ActionSpace = ['up', 'right', 'down', 'left', None]


'''
迭代求解对动作取max的贝尔曼最优公式
input参数：
    n: 网格大小
    T: 迭代次数
    A: 动作空间
gamma: 折扣因子
'''
def iterative_solution_BOE(n,forbiddens, targets, T, A, gamma=0.9):
    v = np.zeros(n*n) # 初始化状态价值函数
    r = np.zeros((len(A),n*n)) # 初始化奖励函数r[a][s]
    p = np.zeros((len(A), n*n, n*n)) # 初始化状态转移概率p[a][s][s'] 

    for i in range(n):
        for j in range(n):
            s = i * n + j # 状态编号
            # 计算奖励函数r[s][a]和状态转移概率p[s][a][s']
            for a in range(len(A)):
                next_i, next_j = i, j
                action = A[a]
                idx = 0 # 用于标记r[idx][s]的值
                if action == 'up':
                    next_i -=1
                    idx = 0 
                elif action == 'right':
                    next_j += 1
                    idx = 1
                elif action == 'down':
                    next_i += 1
                    idx = 2 
                elif action == 'left':
                    next_j -= 1
                    idx = 3
                elif action is None:
                    idx = 4
                else:
                    raise ValueError("Invalid Action")
                             
                next_state = (next_j + 1, next_i + 1) # 转换为从 (1,1) 开始计数的坐标
                if next_state in targets:
                    r[idx][s] = 1
                elif next_state in forbiddens:
                    r[idx][s] = -1
                elif next_i < 0 or next_i >= n or next_j < 0 or next_j >= n:
                    r[idx][s] = -1
                else:
                    r[idx][s] = 0

                # 状态转移概率
                if next_i < 0: next_i = 0
                if next_i >= n: next_i = n - 1
                if next_j < 0: next_j = 0
                if next_j >= n: next_j = n - 1
                s_next = next_i * n + next_j
                p[idx][s][s_next] = 1.0


    # 迭代更新状态价值函数v
    for _ in range(T):
        q = np.zeros((len(A), n*n)) # 初始化动作价值函数q[a][s]
        for a in range(len(A)):
            q[a] = r[a] + gamma * np.dot(p[a], v)
        v = np.max(q, axis=0)

    # 根据最终的状态价值函数v，计算最优策略pi*
    # pi*(s) = argmax_a [ r(s,a) + gamma * sum_{s'} p(s'|s,a) v(s') ]
    pi_star = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            s = i * n + j # 状态编号
            q_s = np.zeros(len(A))
            for a in range(len(A)):
                q_s[a] = r[a][s] + gamma * np.dot(p[a][s], v)
            
            max_a = np.max(q_s)
            best_a = np.random.choice(np.where(q_s == max_a)[0])

            pi_star[i][j] = A[best_a]

    return pi_star, v.reshape((n, n)), r

if __name__ == "__main__":
    forbiddens = {(2, 2), (2, 4), (2, 5), (3, 2), (3, 3), (4, 4)}
    targets = {(3, 4)}
    pi_star, v, r = iterative_solution_BOE(n=5, forbiddens=forbiddens, targets=targets, T=100, A=ActionSpace, gamma=0)

    print(pi_star)
    # print(v)
    # print(r)