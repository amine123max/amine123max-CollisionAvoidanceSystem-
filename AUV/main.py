#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AUV与ROV碰撞避让策略主程序
结合动态贝叶斯网络和随机模型预测控制(SMPC)
"""

import numpy as np
import argparse
import os
import time
from AUV import DynamicBayesianNetwork, StochasticMPC, CollisionAvoidanceSystem

# 导入键盘控制模块（如果可用）
try:
    from keyboard_control import KeyboardController
    KEYBOARD_CONTROL_AVAILABLE = True
except ImportError:
    KEYBOARD_CONTROL_AVAILABLE = False
    print("警告: 未找到pygame库，键盘控制功能不可用。请安装pygame: pip install pygame")

# 导入模拟场景地图模块（如果可用）
try:
    from simulation_map import SimulationMap
    SIMULATION_MAP_AVAILABLE = True
except ImportError:
    SIMULATION_MAP_AVAILABLE = False
    print("警告: 未找到模拟场景地图模块，地图功能不可用")


def run_simulation(save_dir='results', sim_time=20.0, dt=0.1, 
                   human_intent_change_time=10.0, human_intent_type=None, seed=None,
                   keyboard_control=False, use_map=False, num_obstacles=30,
                   map_width=500, map_height=500):
    """
    运行AUV与ROV碰撞避让模拟
    
    参数:
        save_dir: 结果保存目录
        sim_time: 模拟总时间(秒)
        dt: 时间步长(秒)
        human_intent_change_time: 人类意图改变的时间点(秒)
        human_intent_type: 人类意图类型概率 [忽视, 激进, 礼貌]，如果为None则使用默认值
        seed: 随机数种子，用于复现结果
        keyboard_control: 是否启用键盘控制ROV
        use_map: 是否使用2D模拟场景地图
        num_obstacles: 障碍物数量
        map_width: 地图宽度
        map_height: 地图高度
    """
    # 检查键盘控制可用性
    if keyboard_control and not KEYBOARD_CONTROL_AVAILABLE:
        print("错误: 键盘控制功能不可用，请安装pygame库")
        return None, None, None, None, None
    
    # 检查地图功能可用性
    if use_map and not SIMULATION_MAP_AVAILABLE:
        print("错误: 模拟场景地图功能不可用")
        return None, None, None, None, None
    
    # 设置随机数种子
    if seed is not None:
        np.random.seed(seed)
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    print("开始AUV与ROV碰撞避让策略模拟...")
    
    # 创建碰撞避让系统
    cas = CollisionAvoidanceSystem()
    
    # 创建模拟场景地图（如果启用）
    sim_map = None
    if use_map:
        print(f"创建2D模拟场景地图，障碍物数量: {num_obstacles}")
        sim_map = SimulationMap(
            width=map_width, 
            height=map_height, 
            num_obstacles=num_obstacles,
            seed=seed
        )
    
    # 设置AUV初始状态 - 从左下角出发，向右上方运动
    auv_state = np.array([50.0, 50.0, 5.0, 5.0])  # AUV从(50,50)出发，速度为(5,5)
    cas.update_auv_state(auv_state)
    
    # 设置目标位置 - 在右上角
    target = np.array([450.0, 450.0, 0.0, 0.0])  # 目标位于地图右上角
    cas.update_target(target)
    
    # 设置ROV初始状态 - 从右下角出发，向左上方运动（与AUV路径相交）
    rov_true_state = np.array([450.0, 50.0, -5.0, 5.0])  # ROV从(450,50)出发，速度为(-5,5)
    
    # 设置人类意图类型
    if human_intent_type is not None:
        cas.update_human_intent_type(np.array(human_intent_type))
        print(f"设置人类意图类型: 忽视={human_intent_type[0]:.2f}, 激进={human_intent_type[1]:.2f}, 礼貌={human_intent_type[2]:.2f}")
    
    # 获取当前人类意图类型描述
    intent_type_desc = cas.get_intent_type_description()
    print(f"当前主导意图类型: {intent_type_desc}")
    
    # 初始化键盘控制器（如果启用）
    keyboard_controller = None
    if keyboard_control:
        keyboard_controller = KeyboardController(accel_step=0.2, max_accel=2.0)
        keyboard_controller.start()
        print("键盘控制已启动，使用方向键控制ROV运动")
        # 重置ROV初始速度为0（等待键盘输入）
        rov_true_state[2:4] = 0.0
    
    # 初始化pygame显示（如果使用地图）
    if use_map and sim_map is not None:
        sim_map.init_pygame_display()
    
    # 模拟参数
    steps = int(sim_time / dt)
    
    # 存储轨迹
    auv_trajectory = [auv_state.copy()]
    rov_trajectory = [rov_true_state.copy()]
    
    # 存储控制输入和意图概率
    control_inputs = []
    intent_probs = []
    rov_controls = []  # 存储ROV的控制输入
    
    # 存储碰撞信息
    collision_occurred = False
    collision_time = None
    
    # 模拟人类意图变化
    # 初始时人类想要直接前往目标
    human_intent = np.array([1.0, 1.0]) / np.sqrt(2)
    cas.update_human_intent(human_intent)
    
    # 记录最小距离
    min_distance = float('inf')
    min_distance_time = 0
    
    try:
        # 模拟场景
        for i in range(steps):
            # 当前时间
            current_time = i * dt
            
            # 获取ROV控制输入
            if keyboard_control:
                # 从键盘控制器获取控制输入
                if keyboard_controller is not None:
                    rov_control = keyboard_controller.get_control()
                else:
                    rov_control = np.zeros(2)  # 如果控制器不可用，使用零控制
            else:
                # 预设轨迹，无控制输入
                rov_control = np.zeros(2)
                
                # 在指定时间点，模拟人类意图方向变化，希望向右绕行
                if current_time >= human_intent_change_time and current_time < human_intent_change_time + dt:
                    print(f"时间 {current_time:.1f}s: 人类意图方向变化为向右绕行")
                    human_intent = np.array([1.0, 0.0])
                    cas.update_human_intent(human_intent)
            
            # 存储ROV控制输入
            rov_controls.append(rov_control.copy())
            
            # 模拟ROV运动 - x(t+1)=x(t)+Δt⋅f(x(t),u(t))的动力学方程
            # 先更新位置: x(t+1) = x(t) + vx*dt, y(t+1) = y(t) + vy*dt
            rov_true_state[0] += rov_true_state[2] * dt  # x += vx*dt
            rov_true_state[1] += rov_true_state[3] * dt  # y += vy*dt
            
            # 更新ROV速度: vx(t+1) = vx(t) + ax*dt, vy(t+1) = vy(t) + ay*dt
            rov_true_state[2] += rov_control[0] * dt  # vx += ax*dt
            rov_true_state[3] += rov_control[1] * dt  # vy += ay*dt
            
            # 检查ROV是否与障碍物碰撞（如果启用地图）
            if use_map and sim_map is not None:
                if sim_map.check_collision(rov_true_state[0], rov_true_state[1], radius=3.0):
                    print(f"警告: 时间 {current_time:.1f}s - ROV与障碍物碰撞！")
                    # 碰撞后ROV速度减半（模拟碰撞效果）
                    rov_true_state[2] *= -0.5
                    rov_true_state[3] *= -0.5
            
            # 生成ROV观测
            rov_obs = rov_true_state[:2] + np.random.normal(0, 0.1, 2)
            
            # 处理观测
            cas.process_observation(rov_obs)
            
            # 规划轨迹
            control, _ = cas.plan_trajectory()
            control_inputs.append(control.copy())
            intent_probs.append(cas.dbn.intent_probs.copy())
            
            # 更新AUV状态 - 使用照片中的动力学方程
            # 先更新位置: x(t+1) = x(t) + vx*dt, y(t+1) = y(t) + vy*dt
            auv_state[0] += auv_state[2] * dt  # x += vx*dt
            auv_state[1] += auv_state[3] * dt  # y += vy*dt
            
            # 再更新速度: vx(t+1) = vx(t) + ax*dt, vy(t+1) = vy(t) + ay*dt
            auv_state[2] += control[0] * dt  # vx += ax*dt
            auv_state[3] += control[1] * dt  # vy += ay*dt
            
            # 检查AUV是否与障碍物碰撞（如果启用地图）
            if use_map and sim_map is not None:
                if sim_map.check_collision(auv_state[0], auv_state[1], radius=3.0):
                    print(f"警告: 时间 {current_time:.1f}s - AUV与障碍物碰撞！")
                    # 碰撞后AUV速度减半（模拟碰撞效果）
                    auv_state[2] *= -0.5
                    auv_state[3] *= -0.5
                    
                    # 记录碰撞信息
                    if not collision_occurred:
                        collision_occurred = True
                        collision_time = current_time
            
            # 更新系统中的AUV状态
            cas.update_auv_state(auv_state.copy())
            
            # 存储轨迹
            auv_trajectory.append(auv_state.copy())
            rov_trajectory.append(rov_true_state.copy())
            
            # 计算AUV和ROV之间的距离
            dist = np.sqrt((auv_state[0] - rov_true_state[0])**2 + 
                          (auv_state[1] - rov_true_state[1])**2)
            if dist < min_distance:
                min_distance = dist
                min_distance_time = current_time
            
            # 更新pygame显示（如果启用地图）
            if use_map and sim_map is not None:
                sim_map.draw_pygame(
                    auv_pos=auv_state[:2], 
                    rov_pos=rov_true_state[:2],
                    target_pos=target[:2]
                )
            
            # 打印进度
            if i % int(steps/10) == 0:
                print(f"模拟进度: {i/steps*100:.1f}%")
                
            # 如果使用键盘控制或地图，添加短暂延时以便于控制和观察
            if keyboard_control or use_map:
                time.sleep(dt)
    
    finally:
        # 如果使用键盘控制，确保停止控制器
        if keyboard_control and keyboard_controller is not None:
            keyboard_controller.stop()
        
        # 如果使用地图，确保关闭pygame
        if use_map and sim_map is not None:
            sim_map.close_pygame()
    
    # 转换为numpy数组
    auv_trajectory = np.array(auv_trajectory)
    rov_trajectory = np.array(rov_trajectory)
    control_inputs = np.array(control_inputs)
    intent_probs = np.array(intent_probs)
    rov_controls = np.array(rov_controls)
    
    # 保存轨迹数据
    np.save(os.path.join(save_dir, 'auv_trajectory.npy'), auv_trajectory)
    np.save(os.path.join(save_dir, 'rov_trajectory.npy'), rov_trajectory)
    np.save(os.path.join(save_dir, 'control_inputs.npy'), control_inputs)
    np.save(os.path.join(save_dir, 'intent_probs.npy'), intent_probs)
    np.save(os.path.join(save_dir, 'human_intent_type.npy'), cas.human_intent_type)
    np.save(os.path.join(save_dir, 'rov_controls.npy'), rov_controls)
    
    print(f"模拟完成，结果已保存至 {save_dir} 目录")
    print(f"人类意图类型: 忽视={cas.human_intent_type[0]:.2f}, 激进={cas.human_intent_type[1]:.2f}, 礼貌={cas.human_intent_type[2]:.2f}")
    print(f"主导意图类型: {cas.get_intent_type_description()}")
    print(f"AUV与ROV最小距离: {min_distance:.2f}m，发生时间: {min_distance_time:.1f}s")
    
    if collision_occurred:
        print(f"AUV与障碍物发生碰撞，碰撞时间: {collision_time:.1f}s")
    
    return auv_trajectory, rov_trajectory, control_inputs, intent_probs, cas.human_intent_type


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AUV与ROV碰撞避让策略模拟')
    
    parser.add_argument('--save_dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--sim_time', type=float, default=20.0,
                        help='模拟总时间(秒)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='时间步长(秒)')
    parser.add_argument('--intent_change_time', type=float, default=10.0,
                        help='人类意图改变的时间点(秒)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机数种子，用于复现结果')
    
    # 添加人类意图类型参数
    parser.add_argument('--ignore_prob', type=float, default=0.2,
                        help='人类忽视意图概率')
    parser.add_argument('--aggressive_prob', type=float, default=0.5,
                        help='人类激进意图概率')
    parser.add_argument('--polite_prob', type=float, default=0.3,
                        help='人类礼貌意图概率')
    
    # 添加键盘控制参数
    parser.add_argument('--keyboard_control', action='store_true',
                        help='启用键盘控制ROV运动')
    
    # 添加地图相关参数
    parser.add_argument('--use_map', action='store_true',
                        help='启用2D模拟场景地图')
    parser.add_argument('--num_obstacles', type=int, default=30,
                        help='障碍物数量')
    parser.add_argument('--map_width', type=int, default=500,
                        help='地图宽度')
    parser.add_argument('--map_height', type=int, default=500,
                        help='地图高度')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 设置人类意图类型
    human_intent_type = [args.ignore_prob, args.aggressive_prob, args.polite_prob]
    
    # 运行模拟
    run_simulation(
        save_dir=args.save_dir,
        sim_time=args.sim_time,
        dt=args.dt,
        human_intent_change_time=args.intent_change_time,
        human_intent_type=human_intent_type,
        seed=args.seed,
        keyboard_control=args.keyboard_control,
        use_map=args.use_map,
        num_obstacles=args.num_obstacles,
        map_width=args.map_width,
        map_height=args.map_height
    ) 