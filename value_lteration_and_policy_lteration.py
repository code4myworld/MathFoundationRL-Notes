import numpy as np
import pandas as pd

''' 
值迭代算法，伪代码见上图 Algorithm 4.1
Input:
    P: 状态转移概率和奖励矩阵 p(s', r | s, a)
    v0: 初始状态值v0
    threshold: 收敛阈值
    gamma: 折扣因子
Output:
    pi_list: 最终策略pi迭代序列
    v_list: 状态值v迭代序列
'''
def value_iteration_algorithm(stateSpace, actionSpace, P_r, P_s, threshold=1e-5, gamma=0.9):
    
    # 初始化v
    v = {}
    for s in stateSpace:
        v[s] = 0.0

    # 初始化策略PI
    pi = {}
    for s in stateSpace:
        pi[s] = np.random.choice(actionSpace)
    
    v_change = float('inf') # 初始化值变化量为无穷大
    max_steps = 1000 # 最大迭代步数
    step = 0

    q_sa = pd.DataFrame(0, index=stateSpace, columns=actionSpace) # 初始化q(s,a)
    
    v_list = [] # 状态值迭代序列
    pi_list = [] # 策略迭代序列

    while v_change > threshold or step >= max_steps:
        v_old = v.copy() # 记录上一次的状态值

        for s in stateSpace:  # 对每个状态s进行更新
            for a in actionSpace:  # 对每个动作a计算q(s,a)
                immediate_reward = sum(reward * prob for reward, prob in P_r[s][a].items())
                future_reward = gamma * sum(prob * v[s_next] for s_next, prob in P_s[s][a].items())
                q_sa.loc[s, a] = immediate_reward + future_reward
            # 选择使q(s,a)最大的动作作为当前状态s的最优动作
            v[s] = np.max(q_sa.loc[s, :]) # 更新状态值v_{k+1} = max_a q(s,a)
            pi[s] = actionSpace[np.random.choice(np.where(q_sa.loc[s, :] == v[s])[0])]  # 若有多个最优动作，随机选择一个
                
        v_list.append(v.copy()) # 记录当前状态值v
        pi_list.append(pi.copy()) # 记录当前策略pi
                
        v_change = np.abs(np.array(list(v.values())) - np.array(list(v_old.values()))).max() # 计算值变化量
        step += 1

    return pi_list, v_list

if __name__ == "__main__":
    ''' 
    奖励概率 p(r | s, a) 的表示形式
    P_r = {
        s: { 
            a: {
            r1: prob1,
            r2: prob2,
            ...  
            }   
        }
    }
    '''
    r_boundary = r_forbidden = -1
    r_target = 1

    P_r_ex1 = {
        's1': { 
            'up':    {r_boundary: 1}, # 状态 s1 动作 'up' 后 概率1 获得奖励 -1，
            'right': {r_forbidden: 1},
            'down':  {0: 1},
            'left':  {r_boundary: 1},
            'None':  {0: 1}
        },
        's2': {
            'up':    {r_boundary: 1}, 
            'right': {r_boundary: 1},
            'down':  {r_target: 1},
            'left':  {0: 1},
            'None':  {r_forbidden: 1}
        },
        's3': {
            'up':    {0: 1}, 
            'right': {r_target: 1},
            'down':  {r_boundary: 1},
            'left':  {r_boundary: 1},
            'None':  {0: 1}
        },
        's4': {
            'up':    {r_forbidden: 1}, 
            'right': {r_boundary: 1},
            'down':  {r_boundary: 1},
            'left':  {0: 1},
            'None':  {r_target: 1}
        }
    }

    ''' 
    状态转移概率 p(s' | s, a) 的表示形式 
    P_s = {
        s: { 
            a: {
            s_next1: prob1,
            s_next2: prob2,
            ...  
            }   
        }
    }
    '''
    P_s_ex1 = {
        's1': { 
            'up':    {'s1': 1}, 
            'right': {'s2': 1},
            'down':  {'s3': 1},
            'left':  {'s1': 1},
            'None':  {'s1': 1}
        },
        's2': {
            'up':    {'s2': 1}, 
            'right': {'s2': 1},
            'down':  {'s4': 1},
            'left':  {'s1': 1},
            'None':  {'s2': 1}
        },
        's3': {
            'up':    {'s1': 1}, 
            'right': {'s4': 1},
            'down':  {'s3': 1},
            'left':  {'s3': 1},
            'None':  {'s3': 1}
        },
        's4': {
            'up':    {'s2': 1}, 
            'right': {'s4': 1},
            'down':  {'s4': 1},
            'left':  {'s3': 1},
            'None':  {'s4': 1}
        }
    }

    stateSpace = ['s1', 's2', 's3', 's4'] # 状态空间s1, s2, s3, s4
    actionSpace = ['up', 'right', 'down', 'left', 'None'] # 动作空间：上、右、下、左、保持不动
    # q_sa = pd.DataFrame(0, index=stateSpace, columns=actionSpace)
    # print(q_sa.loc['s1'])
    pi_list, v_list = value_iteration_algorithm(stateSpace, actionSpace, P_r_ex1, P_s_ex1, threshold=1e-5, gamma=0.9)
    print("最终状态值v：", f"{v_list[-1]}")
    print("最终策略pi：", pi_list[-1]) 