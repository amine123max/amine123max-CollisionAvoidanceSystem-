#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
键盘控制模块 - 用于通过键盘上下左右键控制ROV运动
"""

import pygame
import numpy as np
import time
import threading

class KeyboardController:
    """键盘控制器类，用于捕获键盘输入并转换为ROV控制命令"""
    
    def __init__(self, accel_step=0.5, max_accel=2.0):
        """
        初始化键盘控制器
        
        参数:
            accel_step: 每次按键的加速度增量
            max_accel: 最大加速度
        """
        self.accel_step = accel_step
        self.max_accel = max_accel
        
        # 控制输入 [ax, ay]
        self.control_input = np.zeros(2)
        
        # 运行标志
        self.running = False
        self.thread = None
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((400, 300))
        pygame.display.set_caption('ROV键盘控制')
        self.font = pygame.font.Font(None, 36)
        
    def start(self):
        """启动键盘控制线程"""
        if self.thread is None or not self.thread.is_alive():
            self.running = True
            self.thread = threading.Thread(target=self._keyboard_loop)
            self.thread.daemon = True
            self.thread.start()
            print("键盘控制已启动，使用方向键控制ROV运动")
            print("↑: 向上加速, ↓: 向下加速, ←: 向左加速, →: 向右加速")
            print("空格键: 停止 (加速度归零)")
    
    def stop(self):
        """停止键盘控制线程"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        pygame.quit()
        print("键盘控制已停止")
    
    def get_control(self):
        """获取当前控制输入"""
        return self.control_input.copy()
    
    def _keyboard_loop(self):
        """键盘输入处理循环"""
        clock = pygame.time.Clock()
        
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    break
            
            # 获取按键状态
            keys = pygame.key.get_pressed()
            
            # 处理方向键输入
            if keys[pygame.K_UP]:
                self.control_input[1] -= self.accel_step  # 向上加速 (y轴负方向)
            if keys[pygame.K_DOWN]:
                self.control_input[1] += self.accel_step  # 向下加速 (y轴正方向)
            if keys[pygame.K_LEFT]:
                self.control_input[0] -= self.accel_step  # 向左加速 (x轴负方向)
            if keys[pygame.K_RIGHT]:
                self.control_input[0] += self.accel_step  # 向右加速 (x轴正方向)
            if keys[pygame.K_SPACE]:
                self.control_input = np.zeros(2)  # 停止 (加速度归零)
            
            # 限制加速度大小
            accel_mag = np.linalg.norm(self.control_input)
            if accel_mag > self.max_accel:
                self.control_input = self.control_input * self.max_accel / accel_mag
            
            # 更新显示
            self._update_display()
            
            # 控制循环频率
            clock.tick(10)  # 10 Hz
    
    def _update_display(self):
        """更新显示界面"""
        self.screen.fill((0, 0, 0))
        
        # 显示当前控制输入
        text = self.font.render(f"加速度: [{self.control_input[0]:.2f}, {self.control_input[1]:.2f}]", 
                               True, (255, 255, 255))
        self.screen.blit(text, (50, 50))
        
        # 显示控制提示
        text = self.font.render("方向键: 控制加速度", True, (200, 200, 200))
        self.screen.blit(text, (50, 100))
        text = self.font.render("空格键: 停止", True, (200, 200, 200))
        self.screen.blit(text, (50, 150))
        
        # 绘制简单的方向指示器
        center = (200, 200)
        radius = 50
        pygame.draw.circle(self.screen, (50, 50, 50), center, radius)
        
        if np.linalg.norm(self.control_input) > 0:
            direction = self.control_input / np.linalg.norm(self.control_input)
            end_point = (
                int(center[0] + direction[0] * radius),
                int(center[1] + direction[1] * radius)
            )
            pygame.draw.line(self.screen, (0, 255, 0), center, end_point, 3)
        
        pygame.display.flip()


# 测试代码
if __name__ == "__main__":
    controller = KeyboardController()
    controller.start()
    
    try:
        while True:
            control = controller.get_control()
            print(f"当前控制输入: {control}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("程序已中断")
    finally:
        controller.stop() 