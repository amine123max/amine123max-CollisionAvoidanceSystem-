#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUV与ROV碰撞避让策略核心组件
包含动态贝叶斯网络、随机模型预测控制和碰撞避让系统的实现
"""

import numpy as np
import random


class DynamicBayesianNetwork:
    """动态贝叶斯网络类，用于估计ROV的运动状态和意图"""
    
    def __init__(self, state_dim=4, observation_dim=2):
        """
        初始化动态贝叶斯网络
        
        参数:
            state_dim: 状态向量维度 (x, y, vx, vy)
            observation_dim: 观测向量维度 (x, y)
        """
        # 转移概率矩阵的先验
        self.A = np.eye(state_dim)  # 状态转移矩阵
        self.A[0, 2] = 1.0  # x += vx
        self.A[1, 3] = 1.0  # y += vy
        
        # 过程噪声协方差
        self.Q = np.eye(state_dim) * 0.01
        
        # 观测矩阵
        self.C = np.zeros((observation_dim, state_dim))
        self.C[0, 0] = 1.0  # 观测x
        self.C[1, 1] = 1.0  # 观测y
        
        # 观测噪声协方差
        self.R = np.eye(observation_dim) * 0.1
        
        # 状态估计和协方差
        self.x_hat = np.zeros(state_dim)
        self.P = np.eye(state_dim)
        
        # 意图概率 (正常航行, 碰撞风险, 紧急避让)
        self.intent_probs = np.array([0.8, 0.15, 0.05])
        
    def predict(self):
        """预测步骤"""
        # 预测状态
        self.x_hat = self.A @ self.x_hat
        # 预测协方差
        self.P = self.A @ self.P @ self.A.T + self.Q
        
    def update(self, z):
        """
        更新步骤
        
        参数:
            z: 观测向量 [x, y]
        """
        # 计算卡尔曼增益
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ np.linalg.inv(S)
        
        # 更新状态估计
        innovation = z - self.C @ self.x_hat
        self.x_hat = self.x_hat + K @ innovation
        
        # 更新协方差
        self.P = (np.eye(len(self.P)) - K @ self.C) @ self.P
        
    def update_intent(self, auv_state):
        """
        基于AUV和ROV的相对状态更新意图概率
        
        参数:
            auv_state: AUV的状态向量 [x, y, vx, vy]
        """
        # 计算相对距离
        dx = self.x_hat[0] - auv_state[0]
        dy = self.x_hat[1] - auv_state[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        # 计算相对速度
        dvx = self.x_hat[2] - auv_state[2]
        dvy = self.x_hat[3] - auv_state[3]
        rel_velocity = np.sqrt(dvx**2 + dvy**2)
        
        # 计算碰撞风险
        collision_risk = 0.0
        if distance < 50.0:  # 如果距离小于50米
            # 计算相对位置和相对速度的夹角
            if distance > 1e-6:  # 避免除以零
                pos_angle = np.arctan2(dy, dx)
                vel_angle = np.arctan2(dvy, dvx)
                angle_diff = abs(pos_angle - vel_angle)
                # 如果角度差小，说明ROV朝向AUV运动
                if angle_diff < np.pi/4 or angle_diff > 7*np.pi/4:
                    collision_risk = (1.0 - distance/50.0) * 0.8
        
        # 更新意图概率
        if collision_risk > 0.5:
            self.intent_probs = np.array([0.1, 0.6, 0.3])
        elif collision_risk > 0.2:
            self.intent_probs = np.array([0.3, 0.5, 0.2])
        else:
            self.intent_probs = np.array([0.8, 0.15, 0.05])
            
        return self.intent_probs


class StochasticMPC:
    """随机模型预测控制器类，用于AUV避障决策"""
    
    def __init__(self, horizon=10, dt=0.1):
        """
        初始化随机MPC控制器
        
        参数:
            horizon: 预测时域长度
            dt: 时间步长
        """
        self.horizon = horizon
        self.dt = dt
        
        # 控制约束
        self.max_accel = 2.0  # 最大加速度 m/s^2
        self.max_vel = 5.0    # 最大速度 m/s
        
        # 代价函数权重
        self.Q_pos = 1.0      # 位置误差权重
        self.Q_vel = 0.1      # 速度误差权重
        self.R_control = 0.01 # 控制输入权重
        self.Q_obstacle = 10.0 # 避障权重
        
        # 人类意图权重
        self.human_intent_weight = 0.5
        
    def generate_trajectory_samples(self, current_state, n_samples=20):
        """
        生成随机轨迹样本
        
        参数:
            current_state: 当前AUV状态 [x, y, vx, vy]
            n_samples: 样本数量
            
        返回:
            trajectories: 轨迹样本列表
            controls: 对应的控制输入列表
        """
        trajectories = []
        controls = []
        
        for _ in range(n_samples):
            # 初始化轨迹和控制
            traj = [current_state.copy()]
            control_seq = []
            
            # 生成随机控制序列
            for t in range(self.horizon):
                # 随机加速度控制输入
                ax = random.uniform(-self.max_accel, self.max_accel)
                ay = random.uniform(-self.max_accel, self.max_accel)
                control = np.array([ax, ay])
                control_seq.append(control)
                
                # 更新状态 - 使用照片中的动力学方程
                next_state = traj[-1].copy()
                
                # 位置更新: x(t+1) = x(t) + vx*dt
                next_state[0] += next_state[2] * self.dt
                # 位置更新: y(t+1) = y(t) + vy*dt
                next_state[1] += next_state[3] * self.dt
                
                # 速度更新: vx(t+1) = vx(t) + ax*dt
                next_state[2] += control[0] * self.dt
                # 速度更新: vy(t+1) = vy(t) + ay*dt
                next_state[3] += control[1] * self.dt
                
                # 速度限制
                vel_mag = np.sqrt(next_state[2]**2 + next_state[3]**2)
                if vel_mag > self.max_vel:
                    next_state[2] *= self.max_vel / vel_mag
                    next_state[3] *= self.max_vel / vel_mag
                
                traj.append(next_state.copy())
            
            trajectories.append(traj)
            controls.append(control_seq)
            
        return trajectories, controls
    
    def evaluate_cost(self, trajectory, controls, target_state, obstacles, human_intent=None, human_intent_type=None):
        """
        评估轨迹代价
        
        参数:
            trajectory: 轨迹状态序列
            controls: 控制输入序列
            target_state: 目标状态 [x, y, vx, vy]
            obstacles: 障碍物列表，每个元素为 [x, y, vx, vy]
            human_intent: 人类操作员意图向量
            human_intent_type: 人类意图类型概率 [忽视, 激进, 礼貌]
            
        返回:
            cost: 轨迹总代价
        """
        cost = 0.0
        
        # 终端状态代价
        terminal_state = trajectory[-1]
        pos_error = np.sqrt((terminal_state[0] - target_state[0])**2 + 
                            (terminal_state[1] - target_state[1])**2)
        vel_error = np.sqrt((terminal_state[2] - target_state[2])**2 + 
                           (terminal_state[3] - target_state[3])**2)
        
        cost += self.Q_pos * pos_error + self.Q_vel * vel_error
        
        # 控制输入代价
        for u in controls:
            control_mag = np.sqrt(u[0]**2 + u[1]**2)
            cost += self.R_control * control_mag
        
        # 避障代价 - 根据人类意图类型调整
        obstacle_weight = self.Q_obstacle
        if human_intent_type is not None:
            # 根据人类意图类型调整避障权重
            # 忽视 - 增加避障权重，因为人类可能不会避让
            # 激进 - 适度增加避障权重，因为人类可能会采取激进行为
            # 礼貌 - 降低避障权重，因为人类可能会主动避让
            ignore_weight = 2.0  # 忽视情况下的权重倍数
            aggressive_weight = 1.5  # 激进情况下的权重倍数
            polite_weight = 0.8  # 礼貌情况下的权重倍数
            
            # 根据概率加权计算最终避障权重
            obstacle_weight = self.Q_obstacle * (
                human_intent_type[0] * ignore_weight +
                human_intent_type[1] * aggressive_weight +
                human_intent_type[2] * polite_weight
            )
        
        for state in trajectory:
            for obs in obstacles:
                # 计算与障碍物的距离
                dx = state[0] - obs[0]
                dy = state[1] - obs[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                # 如果距离小于安全阈值，增加代价
                safety_threshold = 10.0  # 安全距离阈值
                if distance < safety_threshold:
                    cost += obstacle_weight * (1.0 - distance/safety_threshold)**2
        
        # 人类意图方向代价
        if human_intent is not None:
            # 假设human_intent是一个向量，表示期望的运动方向
            # 计算轨迹的平均运动方向
            avg_direction = np.zeros(2)
            for i in range(1, len(trajectory)):
                dx = trajectory[i][0] - trajectory[i-1][0]
                dy = trajectory[i][1] - trajectory[i-1][1]
                if dx**2 + dy**2 > 1e-6:  # 避免除以零
                    direction = np.array([dx, dy]) / np.sqrt(dx**2 + dy**2)
                    avg_direction += direction
            
            if np.linalg.norm(avg_direction) > 1e-6:
                avg_direction = avg_direction / np.linalg.norm(avg_direction)
                # 计算与人类意图的一致性 (夹角余弦值)
                alignment = np.dot(avg_direction, human_intent)
                
                # 根据人类意图类型调整意图权重
                intent_weight = self.human_intent_weight
                if human_intent_type is not None:
                    # 忽视 - 降低意图权重，因为人类意图不太重要
                    # 激进 - 适度增加意图权重，需要更积极地遵循人类意图
                    # 礼貌 - 增加意图权重，应该更多地尊重人类意图
                    ignore_factor = 0.5
                    aggressive_factor = 1.2
                    polite_factor = 1.5
                    
                    intent_weight = self.human_intent_weight * (
                        human_intent_type[0] * ignore_factor +
                        human_intent_type[1] * aggressive_factor +
                        human_intent_type[2] * polite_factor
                    )
                
                # 一致性越高，代价越低
                cost -= intent_weight * alignment
        
        return cost
    
    def optimize(self, current_state, target_state, obstacles, human_intent=None, human_intent_type=None, n_samples=20):
        """
        优化控制输入
        
        参数:
            current_state: 当前AUV状态 [x, y, vx, vy]
            target_state: 目标状态 [x, y, vx, vy]
            obstacles: 障碍物列表，每个元素为 [x, y, vx, vy]
            human_intent: 人类操作员意图向量
            human_intent_type: 人类意图类型概率 [忽视, 激进, 礼貌]
            n_samples: 采样数量
            
        返回:
            best_control: 最优控制输入
            best_trajectory: 最优轨迹
        """
        # 生成轨迹样本
        trajectories, controls = self.generate_trajectory_samples(current_state, n_samples)
        
        # 评估每个轨迹的代价
        costs = []
        for traj, ctrl in zip(trajectories, controls):
            cost = self.evaluate_cost(traj, ctrl, target_state, obstacles, human_intent, human_intent_type)
            costs.append(cost)
        
        # 选择代价最小的轨迹
        best_idx = np.argmin(costs)
        best_trajectory = trajectories[best_idx]
        best_control = controls[best_idx][0]  # 只返回第一个控制输入
        
        return best_control, best_trajectory


class CollisionAvoidanceSystem:
    """AUV与ROV碰撞避让系统"""
    
    def __init__(self):
        """初始化碰撞避让系统"""
        # 创建动态贝叶斯网络
        self.dbn = DynamicBayesianNetwork()
        
        # 创建随机MPC控制器
        self.mpc = StochasticMPC()
        
        # AUV状态 [x, y, vx, vy]
        self.auv_state = np.array([0.0, 0.0, 0.0, 0.0])
        
        # 目标状态
        self.target_state = np.array([100.0, 100.0, 0.0, 0.0])
        
        # 人类意图方向向量
        self.human_intent = np.array([1.0, 1.0]) / np.sqrt(2)  # 默认为对角线方向
        
        # 人类意图类型概率 [忽视, 激进, 礼貌]
        self.human_intent_type = np.array([0.2, 0.5, 0.3])
        
    def update_human_intent(self, intent_vector):
        """
        更新人类控制意图方向
        
        参数:
            intent_vector: 人类意图向量 [dx, dy]
        """
        norm = np.linalg.norm(intent_vector)
        if norm > 1e-6:  # 避免除以零
            self.human_intent = intent_vector / norm
    
    def update_human_intent_type(self, intent_type):
        """
        更新人类意图类型概率
        
        参数:
            intent_type: 人类意图类型概率 [忽视, 激进, 礼貌]
        """
        # 确保概率和为1
        self.human_intent_type = intent_type / np.sum(intent_type)
    
    def update_auv_state(self, state):
        """
        更新AUV状态
        
        参数:
            state: AUV状态 [x, y, vx, vy]
        """
        self.auv_state = state
    
    def update_target(self, target):
        """
        更新目标状态
        
        参数:
            target: 目标状态 [x, y, vx, vy]
        """
        self.target_state = target
    
    def process_observation(self, rov_obs):
        """
        处理ROV观测并更新状态估计
        
        参数:
            rov_obs: ROV观测 [x, y]
        """
        # 预测
        self.dbn.predict()
        
        # 更新
        self.dbn.update(rov_obs)
        
        # 更新意图
        self.dbn.update_intent(self.auv_state)
    
    def plan_trajectory(self):
        """
        规划避障轨迹
        
        返回:
            control: 控制输入 [ax, ay]
            trajectory: 预测轨迹
        """
        # 将ROV作为障碍物
        obstacles = [self.dbn.x_hat]
        
        # 根据ROV意图调整避障策略
        intent_probs = self.dbn.intent_probs
        
        # 如果碰撞风险高，增加避障权重
        if intent_probs[1] > 0.4 or intent_probs[2] > 0.3:
            self.mpc.Q_obstacle *= 2.0
        else:
            self.mpc.Q_obstacle = 10.0  # 恢复默认值
        
        # 优化控制
        control, trajectory = self.mpc.optimize(
            self.auv_state, 
            self.target_state, 
            obstacles, 
            self.human_intent,
            self.human_intent_type
        )
        
        return control, trajectory
        
    def get_intent_type_description(self):
        """
        获取当前人类意图类型的文字描述
        
        返回:
            description: 意图类型描述
        """
        intent_type_idx = np.argmax(self.human_intent_type)
        if intent_type_idx == 0:
            return "忽视"
        elif intent_type_idx == 1:
            return "激进"
        else:
            return "礼貌"


def simulate_scenario():
    """模拟AUV与ROV碰撞避让场景"""
    # 创建碰撞避让系统
    cas = CollisionAvoidanceSystem()
    
    # 设置初始状态
    auv_state = np.array([0.0, 0.0, 2.0, 2.0])  # AUV从原点出发，向右上方运动
    cas.update_auv_state(auv_state)
    
    # 设置目标
    target = np.array([100.0, 100.0, 0.0, 0.0])
    cas.update_target(target)
    
    # ROV初始状态
    rov_true_state = np.array([50.0, 50.0, -1.0, -1.0])  # ROV从(50,50)出发，向左下方运动
    
    # 模拟参数
    dt = 0.1
    sim_time = 20.0
    steps = int(sim_time / dt)
    
    # 存储轨迹
    auv_trajectory = [auv_state.copy()]
    rov_trajectory = [rov_true_state.copy()]
    
    # 模拟人类意图变化
    # 初始时人类想要直接前往目标
    human_intent = np.array([1.0, 1.0]) / np.sqrt(2)
    cas.update_human_intent(human_intent)
    
    # 模拟场景
    for i in range(steps):
        # 模拟ROV运动
        rov_true_state[0] += rov_true_state[2] * dt
        rov_true_state[1] += rov_true_state[3] * dt
        
        # 生成ROV观测
        rov_obs = rov_true_state[:2] + np.random.normal(0, 0.1, 2)
        
        # 处理观测
        cas.process_observation(rov_obs)
        
        # 在第10秒时，模拟人类意图变化，希望向右绕行
        if i == int(10.0 / dt):
            human_intent = np.array([1.0, 0.0])
            cas.update_human_intent(human_intent)
        
        # 规划轨迹
        control, _ = cas.plan_trajectory()
        
        # 更新AUV状态
        auv_state[2] += control[0] * dt  # vx += ax*dt
        auv_state[3] += control[1] * dt  # vy += ay*dt
        auv_state[0] += auv_state[2] * dt  # x += vx*dt
        auv_state[1] += auv_state[3] * dt  # y += vy*dt
        
        # 更新系统中的AUV状态
        cas.update_auv_state(auv_state.copy())
        
        # 存储轨迹
        auv_trajectory.append(auv_state.copy())
        rov_trajectory.append(rov_true_state.copy())
    
    return np.array(auv_trajectory), np.array(rov_trajectory)


def plot_results(auv_trajectory, rov_trajectory):
    """绘制模拟结果"""
    plt.figure(figsize=(10, 8))
    
    # 绘制AUV轨迹
    plt.plot(auv_trajectory[:, 0], auv_trajectory[:, 1], 'b-', label='AUV轨迹')
    plt.plot(auv_trajectory[0, 0], auv_trajectory[0, 1], 'bo', label='AUV起点')
    plt.plot(auv_trajectory[-1, 0], auv_trajectory[-1, 1], 'b*', label='AUV终点')
    
    # 绘制ROV轨迹
    plt.plot(rov_trajectory[:, 0], rov_trajectory[:, 1], 'r-', label='ROV轨迹')
    plt.plot(rov_trajectory[0, 0], rov_trajectory[0, 1], 'ro', label='ROV起点')
    plt.plot(rov_trajectory[-1, 0], rov_trajectory[-1, 1], 'r*', label='ROV终点')
    
    # 绘制目标点
    plt.plot(100, 100, 'g*', markersize=10, label='目标点')
    
    # 添加网格和图例
    plt.grid(True)
    plt.legend()
    plt.xlabel('X坐标 (m)')
    plt.ylabel('Y坐标 (m)')
    plt.title('AUV与ROV碰撞避让轨迹')
    
    # 计算最小距离
    min_distance = float('inf')
    min_distance_time = 0
    for i in range(len(auv_trajectory)):
        dist = np.sqrt((auv_trajectory[i, 0] - rov_trajectory[i, 0])**2 + 
                       (auv_trajectory[i, 1] - rov_trajectory[i, 1])**2)
        if dist < min_distance:
            min_distance = dist
            min_distance_time = i * 0.1  # dt = 0.1
    
    plt.annotate(f'最小距离: {min_distance:.2f}m @ t={min_distance_time:.1f}s', 
                 xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.savefig('collision_avoidance_simulation.png')
    plt.show()


if __name__ == '__main__':
    print("开始AUV与ROV碰撞避让策略模拟...")
    auv_traj, rov_traj = simulate_scenario()
    plot_results(auv_traj, rov_traj)
    print("模拟完成，结果已保存为collision_avoidance_simulation.png")
