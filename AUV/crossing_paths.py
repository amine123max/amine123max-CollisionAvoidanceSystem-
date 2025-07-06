#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
交会路径模拟 - AUV和ROV从固定位置出发，沿交会路径行驶，测试碰撞避让效果
"""

import numpy as np
import pygame
import time
import argparse
import os
import sys
from AUV import CollisionAvoidanceSystem

# 导入键盘控制模块
try:
    from keyboard_control import KeyboardController
    KEYBOARD_CONTROL_AVAILABLE = True
except ImportError:
    KEYBOARD_CONTROL_AVAILABLE = False
    print("错误: 未找到键盘控制模块，请确保keyboard_control.py文件存在")
    sys.exit(1)


class CrossingPathsSimulation:
    """交会路径模拟类"""
    
    def __init__(self, map_width=500, map_height=500, 
                 dt=0.1, seed=None, save_dir='results'):
        """
        初始化交会路径模拟
        
        参数:
            map_width: 地图宽度
            map_height: 地图高度
            dt: 时间步长(秒)
            seed: 随机数种子
            save_dir: 结果保存目录
        """
        self.map_width = map_width
        self.map_height = map_height
        self.dt = dt
        self.seed = seed
        self.save_dir = save_dir
        
        # 设置随机数种子
        if seed is not None:
            np.random.seed(seed)
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 初始化pygame
        pygame.init()
        self.screen_width = 1600
        self.screen_height = 1000
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('AUV与ROV交会路径模拟')
        self.font = pygame.font.Font(None, 30)  # 字体也稍微放大
        self.clock = pygame.time.Clock()
        
        # 初始化键盘控制器
        self.keyboard_controller = KeyboardController(accel_step=0.2, max_accel=2.0)
        
        # 创建碰撞避让系统
        self.cas = CollisionAvoidanceSystem()
        
        # 初始化状态
        # AUV从左下角出发，向右上方运动
        self.auv_state = np.array([50.0, 50.0, 5.0, 5.0])  # 从(50,50)出发，速度为(5,5)
        # ROV从右下角出发，向左上方运动（与AUV路径相交）
        self.rov_state = np.array([450.0, 50.0, -5.0, 5.0])  # 从(450,50)出发，速度为(-5,5)
        self.target = np.array([450.0, 450.0, 0.0, 0.0])  # 目标位于地图右上角
        
        # 更新碰撞避让系统状态
        self.cas.update_auv_state(self.auv_state.copy())
        self.cas.update_target(self.target.copy())
        
        # 设置人类意图类型 (默认: 忽视=0.2, 激进=0.5, 礼貌=0.3)
        self.human_intent_type = np.array([0.2, 0.5, 0.3])
        self.cas.update_human_intent_type(self.human_intent_type)
        
        # 初始人类意图 (直接前往目标)
        self.human_intent = np.array([1.0, 1.0]) / np.sqrt(2)
        self.cas.update_human_intent(self.human_intent)
        
        # 存储轨迹
        self.auv_trajectory = [self.auv_state.copy()]
        self.rov_trajectory = [self.rov_state.copy()]
        self.control_inputs = []
        self.rov_controls = []
        self.intent_probs = []
        
        # 状态变量
        self.running = False
        self.paused = False
        self.current_time = 0.0
        self.step_count = 0
        self.show_help = True
        self.keyboard_mode = False  # 是否启用键盘控制
        
        # 记录最小距离
        self.min_distance = float('inf')
        self.min_distance_time = 0.0
        
        # 初始化键盘控制器
        self.keyboard_controller.start()
    
    def run(self):
        """运行交会路径模拟"""
        self.running = True
        self.paused = False
        
        print("交会路径模拟已启动")
        print("AUV和ROV将沿交会路径行驶，测试碰撞避让效果")
        print("按K切换键盘控制模式，按ESC退出，按空格暂停/继续，按H显示/隐藏帮助")
        
        try:
            while self.running:
                # 处理事件
                self._handle_events()
                
                # 如果暂停，跳过更新
                if self.paused:
                    self._render()
                    self.clock.tick(30)
                    continue
                
                # 更新状态
                self._update()
                
                # 渲染
                self._render()
                
                # 控制帧率
                self.clock.tick(int(1/self.dt))
        finally:
            # 清理资源
            self._cleanup()
    
    def _handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                    print("模拟已" + ("暂停" if self.paused else "继续"))
                elif event.key == pygame.K_h:
                    self.show_help = not self.show_help
                    print("帮助信息已" + ("显示" if self.show_help else "隐藏"))
                elif event.key == pygame.K_r:
                    # 重置ROV位置
                    self.rov_state = np.array([450.0, 50.0, -5.0, 5.0])
                    print("ROV位置已重置")
                elif event.key == pygame.K_a:
                    # 重置AUV位置
                    self.auv_state = np.array([50.0, 50.0, 5.0, 5.0])
                    self.cas.update_auv_state(self.auv_state.copy())
                    print("AUV位置已重置")
                elif event.key == pygame.K_k:
                    # 切换键盘控制模式
                    self.keyboard_mode = not self.keyboard_mode
                    print("键盘控制模式已" + ("启用" if self.keyboard_mode else "禁用"))
                    if self.keyboard_mode:
                        # 重置ROV速度为0（等待键盘输入）
                        self.rov_state[2:4] = 0.0
    
    def _update(self):
        """更新状态"""
        # 更新时间和步数
        self.current_time += self.dt
        self.step_count += 1
        
        # 获取ROV控制输入
        if self.keyboard_mode:
            # 从键盘控制器获取控制输入
            rov_control = self.keyboard_controller.get_control()
        else:
            # 预设轨迹，无控制输入
            rov_control = np.zeros(2)
        
        self.rov_controls.append(rov_control.copy())
        
        # 更新ROV状态
        # 先更新位置: x(t+1) = x(t) + vx*dt, y(t+1) = y(t) + vy*dt
        self.rov_state[0] += self.rov_state[2] * self.dt
        self.rov_state[1] += self.rov_state[3] * self.dt
        
        # 更新ROV速度: vx(t+1) = vx(t) + ax*dt, vy(t+1) = vy(t) + ay*dt
        self.rov_state[2] += rov_control[0] * self.dt
        self.rov_state[3] += rov_control[1] * self.dt
        
        # 确保ROV不会离开地图边界
        self._constrain_to_map(self.rov_state)
        
        # 生成ROV观测（添加少量噪声）
        rov_obs = self.rov_state[:2] + np.random.normal(0, 0.1, 2)
        
        # 处理观测
        self.cas.process_observation(rov_obs)
        
        # 规划轨迹
        control, _ = self.cas.plan_trajectory()
        self.control_inputs.append(control.copy())
        self.intent_probs.append(self.cas.dbn.intent_probs.copy())
        
        # 更新AUV状态
        # 先更新位置: x(t+1) = x(t) + vx*dt, y(t+1) = y(t) + vy*dt
        self.auv_state[0] += self.auv_state[2] * self.dt
        self.auv_state[1] += self.auv_state[3] * self.dt
        
        # 再更新速度: vx(t+1) = vx(t) + ax*dt, vy(t+1) = vy(t) + ay*dt
        self.auv_state[2] += control[0] * self.dt
        self.auv_state[3] += control[1] * self.dt
        
        # 确保AUV不会离开地图边界
        self._constrain_to_map(self.auv_state)
        
        # 更新碰撞避让系统中的AUV状态
        self.cas.update_auv_state(self.auv_state.copy())
        
        # 存储轨迹
        self.auv_trajectory.append(self.auv_state.copy())
        self.rov_trajectory.append(self.rov_state.copy())
        
        # 计算AUV和ROV之间的距离
        dist = np.sqrt((self.auv_state[0] - self.rov_state[0])**2 + 
                      (self.auv_state[1] - self.rov_state[1])**2)
        if dist < self.min_distance:
            self.min_distance = dist
            self.min_distance_time = self.current_time
        
        # 每5秒打印一次状态信息
        if self.step_count % int(5 / self.dt) == 0:
            self._print_status()
    
    def _constrain_to_map(self, state):
        """确保状态不会离开地图边界"""
        # 检查x坐标
        if state[0] < 0:
            state[0] = 0
            state[2] *= -0.5  # 反弹
        elif state[0] > self.map_width:
            state[0] = self.map_width
            state[2] *= -0.5  # 反弹
        
        # 检查y坐标
        if state[1] < 0:
            state[1] = 0
            state[3] *= -0.5  # 反弹
        elif state[1] > self.map_height:
            state[1] = self.map_height
            state[3] *= -0.5  # 反弹
    
    def _render(self):
        """渲染模拟场景"""
        # 清屏
        self.screen.fill((255, 255, 255))
        
        # 计算缩放因子
        scale_x = self.screen_width / self.map_width
        scale_y = self.screen_height / self.map_height
        
        # 绘制网格
        grid_spacing = 100  # 每100米一条网格线
        for x in range(0, self.map_width + 1, grid_spacing):
            screen_x = int(x * scale_x)
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (screen_x, 0), (screen_x, self.screen_height))
        
        for y in range(0, self.map_height + 1, grid_spacing):
            screen_y = int(y * scale_y)
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (0, screen_y), (self.screen_width, screen_y))
        
        # 绘制轨迹（最近100步）
        max_trail = 100
        start_idx = max(0, len(self.auv_trajectory) - max_trail)
        
        # AUV轨迹
        for i in range(start_idx, len(self.auv_trajectory) - 1):
            p1 = (int(self.auv_trajectory[i][0] * scale_x), 
                  int(self.auv_trajectory[i][1] * scale_y))
            p2 = (int(self.auv_trajectory[i+1][0] * scale_x), 
                  int(self.auv_trajectory[i+1][1] * scale_y))
            # 使用渐变色，越近的轨迹越亮
            alpha = 100 + 155 * (i - start_idx) / (len(self.auv_trajectory) - 1 - start_idx)
            pygame.draw.line(self.screen, (0, 0, int(alpha)), p1, p2, 2)
        
        # ROV轨迹
        for i in range(start_idx, len(self.rov_trajectory) - 1):
            p1 = (int(self.rov_trajectory[i][0] * scale_x), 
                  int(self.rov_trajectory[i][1] * scale_y))
            p2 = (int(self.rov_trajectory[i+1][0] * scale_x), 
                  int(self.rov_trajectory[i+1][1] * scale_y))
            # 使用渐变色，越近的轨迹越亮
            alpha = 100 + 155 * (i - start_idx) / (len(self.rov_trajectory) - 1 - start_idx)
            pygame.draw.line(self.screen, (int(alpha), 0, 0), p1, p2, 2)
        
        # 绘制AUV
        auv_pos = (int(self.auv_state[0] * scale_x), int(self.auv_state[1] * scale_y))
        pygame.draw.circle(self.screen, (0, 0, 255), auv_pos, 8)
        
        # 绘制AUV速度向量
        if np.linalg.norm(self.auv_state[2:4]) > 0.1:
            speed = np.linalg.norm(self.auv_state[2:4])
            direction = self.auv_state[2:4] / speed
            end_point = (int(auv_pos[0] + direction[0] * speed * 5), 
                         int(auv_pos[1] + direction[1] * speed * 5))
            pygame.draw.line(self.screen, (0, 0, 255), auv_pos, end_point, 2)
        
        # 绘制ROV
        rov_pos = (int(self.rov_state[0] * scale_x), int(self.rov_state[1] * scale_y))
        pygame.draw.circle(self.screen, (255, 0, 0), rov_pos, 8)
        
        # 绘制ROV速度向量
        if np.linalg.norm(self.rov_state[2:4]) > 0.1:
            speed = np.linalg.norm(self.rov_state[2:4])
            direction = self.rov_state[2:4] / speed
            end_point = (int(rov_pos[0] + direction[0] * speed * 5), 
                         int(rov_pos[1] + direction[1] * speed * 5))
            pygame.draw.line(self.screen, (255, 0, 0), rov_pos, end_point, 2)
        
        # 绘制目标点
        target_pos = (int(self.target[0] * scale_x), int(self.target[1] * scale_y))
        pygame.draw.circle(self.screen, (0, 255, 0), target_pos, 10)
        
        # 如果AUV和ROV之间的距离小于安全距离，绘制连线
        dist = np.sqrt((self.auv_state[0] - self.rov_state[0])**2 + 
                      (self.auv_state[1] - self.rov_state[1])**2)
        if dist < 50:  # 安全距离设为50米
            pygame.draw.line(self.screen, (255, 0, 0), auv_pos, rov_pos, 1)
            
            # 在连线中间显示距离
            mid_x = (auv_pos[0] + rov_pos[0]) // 2
            mid_y = (auv_pos[1] + rov_pos[1]) // 2
            dist_text = self.font.render(f"{dist:.1f}m", True, (255, 0, 0))
            self.screen.blit(dist_text, (mid_x, mid_y))
        
        # 绘制状态信息
        self._draw_status()
        
        # 绘制帮助信息
        if self.show_help:
            self._draw_help()
        
        # 更新显示
        pygame.display.flip()
    
    def _draw_status(self):
        """绘制状态信息"""
        # 状态信息背景
        pygame.draw.rect(self.screen, (240, 240, 240), (30, 30, 400, 220))
        
        # 时间信息
        time_text = self.font.render(f"时间: {self.current_time:.1f}s", True, (0, 0, 0))
        self.screen.blit(time_text, (50, 40))
        
        # AUV信息
        auv_pos_text = self.font.render(f"AUV位置: ({self.auv_state[0]:.1f}, {self.auv_state[1]:.1f})", 
                                       True, (0, 0, 0))
        self.screen.blit(auv_pos_text, (50, 80))
        
        auv_vel_text = self.font.render(f"AUV速度: ({self.auv_state[2]:.1f}, {self.auv_state[3]:.1f})", 
                                       True, (0, 0, 0))
        self.screen.blit(auv_vel_text, (50, 120))
        
        # ROV信息
        rov_pos_text = self.font.render(f"ROV位置: ({self.rov_state[0]:.1f}, {self.rov_state[1]:.1f})", 
                                       True, (0, 0, 0))
        self.screen.blit(rov_pos_text, (50, 160))
        
        rov_vel_text = self.font.render(f"ROV速度: ({self.rov_state[2]:.1f}, {self.rov_state[3]:.1f})", 
                                       True, (0, 0, 0))
        self.screen.blit(rov_vel_text, (50, 200))
        
        # 最小距离信息
        dist_text = self.font.render(f"最小距离: {self.min_distance:.1f}m @ {self.min_distance_time:.1f}s", 
                                    True, (255, 0, 0))
        self.screen.blit(dist_text, (50, 240))
        
        # 键盘控制模式状态
        mode_text = self.font.render(f"键盘控制: {'开启' if self.keyboard_mode else '关闭'}", 
                                    True, (0, 0, 0))
        self.screen.blit(mode_text, (50, 280))
    
    def _draw_help(self):
        """绘制帮助信息"""
        # 帮助信息背景
        pygame.draw.rect(self.screen, (240, 240, 240, 200), 
                       (self.screen_width - 400, 30, 370, 260))
        
        # 帮助标题
        help_title = self.font.render("控制帮助", True, (0, 0, 0))
        self.screen.blit(help_title, (self.screen_width - 380, 40))
        
        # 控制说明
        controls = [
            "K: 切换键盘控制模式",
            "↑/↓/←/→: 控制ROV加速度（键盘模式）",
            "空格: 暂停/继续模拟",
            "R: 重置ROV位置",
            "A: 重置AUV位置",
            "H: 显示/隐藏帮助",
            "ESC: 退出模拟"
        ]
        
        for i, control in enumerate(controls):
            control_text = self.font.render(control, True, (0, 0, 0))
            self.screen.blit(control_text, (self.screen_width - 380, 80 + i * 35))
    
    def _print_status(self):
        """打印状态信息"""
        print(f"时间: {self.current_time:.1f}s")
        print(f"AUV位置: ({self.auv_state[0]:.1f}, {self.auv_state[1]:.1f}), "
              f"速度: ({self.auv_state[2]:.1f}, {self.auv_state[3]:.1f})")
        print(f"ROV位置: ({self.rov_state[0]:.1f}, {self.rov_state[1]:.1f}), "
              f"速度: ({self.rov_state[2]:.1f}, {self.rov_state[3]:.1f})")
        print(f"当前距离: {np.sqrt((self.auv_state[0] - self.rov_state[0])**2 + (self.auv_state[1] - self.rov_state[1])**2):.1f}m")
        print(f"最小距离: {self.min_distance:.1f}m @ {self.min_distance_time:.1f}s")
        print(f"意图概率: 正常={self.cas.dbn.intent_probs[0]:.2f}, "
              f"碰撞风险={self.cas.dbn.intent_probs[1]:.2f}, "
              f"紧急避让={self.cas.dbn.intent_probs[2]:.2f}")
        print("---")
    
    def _cleanup(self):
        """清理资源"""
        # 停止键盘控制器
        self.keyboard_controller.stop()
        
        # 保存轨迹数据
        self._save_results()
        
        # 退出pygame
        pygame.quit()
        
        print("模拟已结束")
        print(f"AUV与ROV最小距离: {self.min_distance:.2f}m，发生时间: {self.min_distance_time:.1f}s")
    
    def _save_results(self):
        """保存结果"""
        # 转换为numpy数组
        auv_trajectory = np.array(self.auv_trajectory)
        rov_trajectory = np.array(self.rov_trajectory)
        control_inputs = np.array(self.control_inputs)
        rov_controls = np.array(self.rov_controls)
        intent_probs = np.array(self.intent_probs)
        
        # 保存数据
        np.save(os.path.join(self.save_dir, 'crossing_auv_trajectory.npy'), auv_trajectory)
        np.save(os.path.join(self.save_dir, 'crossing_rov_trajectory.npy'), rov_trajectory)
        np.save(os.path.join(self.save_dir, 'crossing_control_inputs.npy'), control_inputs)
        np.save(os.path.join(self.save_dir, 'crossing_rov_controls.npy'), rov_controls)
        np.save(os.path.join(self.save_dir, 'crossing_intent_probs.npy'), intent_probs)
        np.save(os.path.join(self.save_dir, 'crossing_human_intent_type.npy'), self.human_intent_type)
        
        print(f"结果已保存至 {self.save_dir} 目录")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AUV与ROV交会路径模拟')
    
    parser.add_argument('--save_dir', type=str, default='results',
                        help='结果保存目录')
    parser.add_argument('--map_width', type=int, default=500,
                        help='地图宽度')
    parser.add_argument('--map_height', type=int, default=500,
                        help='地图高度')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='时间步长(秒)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机数种子，用于复现结果')
    
    return parser.parse_args()


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建交会路径模拟
    simulation = CrossingPathsSimulation(
        map_width=args.map_width,
        map_height=args.map_height,
        dt=args.dt,
        seed=args.seed,
        save_dir=args.save_dir
    )
    
    # 运行模拟
    simulation.run() 