#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D模拟场景地图 - 用于AUV与ROV碰撞避让策略模拟
支持随机生成固定障碍物
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import pygame
import random

class SimulationMap:
    """2D模拟场景地图类"""
    
    def __init__(self, width=500, height=500, num_obstacles=30, 
                 min_obstacle_size=10, max_obstacle_size=30, 
                 obstacle_types=None, seed=None):
        """
        初始化2D模拟场景地图
        
        参数:
            width: 地图宽度
            height: 地图高度
            num_obstacles: 障碍物数量
            min_obstacle_size: 最小障碍物尺寸
            max_obstacle_size: 最大障碍物尺寸
            obstacle_types: 障碍物类型列表，默认为['circle', 'rectangle']
            seed: 随机数种子
        """
        self.width = width
        self.height = height
        self.num_obstacles = num_obstacles
        self.min_obstacle_size = min_obstacle_size
        self.max_obstacle_size = max_obstacle_size
        self.obstacle_types = obstacle_types if obstacle_types else ['circle', 'rectangle']
        
        # 设置随机数种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # 障碍物列表
        self.obstacles = []
        
        # 生成随机障碍物
        self._generate_obstacles()
        
        # pygame显示相关
        self.pygame_initialized = False
        self.screen = None
        self.font = None
    
    def _generate_obstacles(self):
        """生成随机障碍物"""
        self.obstacles = []
        
        for _ in range(self.num_obstacles):
            # 随机选择障碍物类型
            obstacle_type = random.choice(self.obstacle_types)
            
            # 随机生成障碍物尺寸
            size = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
            
            # 随机生成障碍物位置（避免靠近边缘）
            margin = max(size, self.max_obstacle_size)
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)
            
            if obstacle_type == 'circle':
                # 圆形障碍物: (x, y, radius)
                self.obstacles.append({
                    'type': 'circle',
                    'x': x,
                    'y': y,
                    'radius': size
                })
            elif obstacle_type == 'rectangle':
                # 矩形障碍物: (x, y, width, height)
                width = size
                height = random.uniform(self.min_obstacle_size, self.max_obstacle_size)
                self.obstacles.append({
                    'type': 'rectangle',
                    'x': x - width/2,  # 中心点转左上角
                    'y': y - height/2,  # 中心点转左上角
                    'width': width,
                    'height': height,
                    'angle': random.uniform(0, 360)  # 随机旋转角度
                })
        
        print(f"已生成 {len(self.obstacles)} 个随机障碍物")
    
    def check_collision(self, x, y, radius=2.0):
        """
        检查指定位置是否与障碍物碰撞
        
        参数:
            x, y: 位置坐标
            radius: 检测半径
            
        返回:
            bool: 是否碰撞
        """
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                # 计算到圆心的距离
                dist = np.sqrt((x - obstacle['x'])**2 + (y - obstacle['y'])**2)
                if dist < (radius + obstacle['radius']):
                    return True
            elif obstacle['type'] == 'rectangle':
                # 简化的矩形碰撞检测（不考虑旋转）
                # 计算点到矩形中心的距离
                rect_center_x = obstacle['x'] + obstacle['width']/2
                rect_center_y = obstacle['y'] + obstacle['height']/2
                
                # 计算矩形的"半宽"和"半高"
                half_width = obstacle['width']/2
                half_height = obstacle['height']/2
                
                # 计算点到矩形中心的距离
                dx = abs(x - rect_center_x)
                dy = abs(y - rect_center_y)
                
                # 如果点在矩形外
                if dx > half_width + radius or dy > half_height + radius:
                    continue
                
                # 如果点在矩形内
                if dx <= half_width or dy <= half_height:
                    return True
                
                # 检查角落
                corner_dist = (dx - half_width)**2 + (dy - half_height)**2
                if corner_dist <= radius**2:
                    return True
        
        return False
    
    def get_closest_obstacle_distance(self, x, y):
        """
        计算到最近障碍物的距离
        
        参数:
            x, y: 位置坐标
            
        返回:
            float: 到最近障碍物的距离
        """
        min_dist = float('inf')
        
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                # 计算到圆心的距离，减去半径
                dist = np.sqrt((x - obstacle['x'])**2 + (y - obstacle['y'])**2) - obstacle['radius']
                min_dist = min(min_dist, dist)
            elif obstacle['type'] == 'rectangle':
                # 简化计算，使用到矩形中心的距离
                rect_center_x = obstacle['x'] + obstacle['width']/2
                rect_center_y = obstacle['y'] + obstacle['height']/2
                
                # 计算到矩形中心的距离，减去半对角线长度
                diagonal = np.sqrt(obstacle['width']**2 + obstacle['height']**2) / 2
                dist = np.sqrt((x - rect_center_x)**2 + (y - rect_center_y)**2) - diagonal
                min_dist = min(min_dist, dist)
        
        return max(0, min_dist)  # 不允许负值
    
    def plot_map(self, ax=None, show=True):
        """
        使用matplotlib绘制地图
        
        参数:
            ax: matplotlib轴对象，如果为None则创建新的
            show: 是否显示图形
            
        返回:
            ax: matplotlib轴对象
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        # 设置地图边界
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                circle = Circle((obstacle['x'], obstacle['y']), 
                               obstacle['radius'], 
                               fill=True, 
                               color='gray', 
                               alpha=0.7)
                ax.add_patch(circle)
            elif obstacle['type'] == 'rectangle':
                # 创建矩形
                rect = Rectangle((obstacle['x'], obstacle['y']),
                                obstacle['width'],
                                obstacle['height'],
                                angle=obstacle['angle'],
                                fill=True,
                                color='gray',
                                alpha=0.7)
                ax.add_patch(rect)
        
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X坐标 (m)')
        ax.set_ylabel('Y坐标 (m)')
        ax.set_title('模拟场景地图')
        
        if show:
            plt.tight_layout()
            plt.show()
        
        return ax
    
    def init_pygame_display(self, screen_width=1600, screen_height=1000):
        """初始化pygame显示"""
        if not self.pygame_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
            pygame.display.set_caption('2D模拟场景地图')
            self.font = pygame.font.Font(None, 30)  # 字体也稍微放大
            self.pygame_initialized = True
    
    def draw_pygame(self, auv_pos=None, rov_pos=None, target_pos=None):
        """
        使用pygame绘制地图
        
        参数:
            auv_pos: AUV位置 [x, y]
            rov_pos: ROV位置 [x, y]
            target_pos: 目标位置 [x, y]
        """
        if not self.pygame_initialized:
            self.init_pygame_display()
        
        # 清屏
        self.screen.fill((255, 255, 255))
        
        # 计算缩放因子
        scale_x = self.screen.get_width() / self.width
        scale_y = self.screen.get_height() / self.height
        
        # 绘制障碍物
        for obstacle in self.obstacles:
            if obstacle['type'] == 'circle':
                # 转换坐标
                screen_x = int(obstacle['x'] * scale_x)
                screen_y = int(obstacle['y'] * scale_y)
                screen_radius = int(obstacle['radius'] * scale_x)
                
                # 绘制圆形
                pygame.draw.circle(self.screen, (100, 100, 100), 
                                  (screen_x, screen_y), screen_radius)
            elif obstacle['type'] == 'rectangle':
                # 转换坐标
                screen_x = int(obstacle['x'] * scale_x)
                screen_y = int(obstacle['y'] * scale_y)
                screen_width = int(obstacle['width'] * scale_x)
                screen_height = int(obstacle['height'] * scale_y)
                
                # 绘制矩形（简化，不考虑旋转）
                pygame.draw.rect(self.screen, (100, 100, 100), 
                               (screen_x, screen_y, screen_width, screen_height))
        
        # 绘制AUV
        if auv_pos is not None:
            screen_x = int(auv_pos[0] * scale_x)
            screen_y = int(auv_pos[1] * scale_y)
            pygame.draw.circle(self.screen, (0, 0, 255), (screen_x, screen_y), 5)
        
        # 绘制ROV
        if rov_pos is not None:
            screen_x = int(rov_pos[0] * scale_x)
            screen_y = int(rov_pos[1] * scale_y)
            pygame.draw.circle(self.screen, (255, 0, 0), (screen_x, screen_y), 5)
        
        # 绘制目标
        if target_pos is not None:
            screen_x = int(target_pos[0] * scale_x)
            screen_y = int(target_pos[1] * scale_y)
            pygame.draw.circle(self.screen, (0, 255, 0), (screen_x, screen_y), 7)
        
        # 绘制坐标网格
        grid_spacing = 50  # 每50米一条网格线
        for x in range(0, self.width + 1, grid_spacing):
            screen_x = int(x * scale_x)
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (screen_x, 0), (screen_x, self.screen.get_height()))
        
        for y in range(0, self.height + 1, grid_spacing):
            screen_y = int(y * scale_y)
            pygame.draw.line(self.screen, (200, 200, 200), 
                           (0, screen_y), (self.screen.get_width(), screen_y))
        
        # 更新显示
        pygame.display.flip()
    
    def close_pygame(self):
        """关闭pygame显示"""
        if self.pygame_initialized:
            pygame.quit()
            self.pygame_initialized = False


# 测试代码
if __name__ == "__main__":
    # 创建模拟地图
    sim_map = SimulationMap(width=500, height=500, num_obstacles=30, seed=42)
    
    # 使用matplotlib绘制地图
    sim_map.plot_map(show=True)
    
    # 使用pygame绘制地图
    sim_map.init_pygame_display(screen_width=1600, screen_height=1000)
    
    # 模拟AUV和ROV运动
    auv_pos = [50, 50]
    rov_pos = [450, 50]
    target_pos = [450, 450]
    
    try:
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # 绘制地图和物体
            sim_map.draw_pygame(auv_pos, rov_pos, target_pos)
            
            # 简单移动AUV（朝目标方向）
            dx = target_pos[0] - auv_pos[0]
            dy = target_pos[1] - auv_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist > 1:
                auv_pos[0] += dx / dist
                auv_pos[1] += dy / dist
            
            # 控制帧率
            clock.tick(30)
    finally:
        sim_map.close_pygame() 