#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""屏幕捕获模块 - 支持后台截图"""

import numpy as np
from typing import Optional, Tuple
import subprocess
import time


class GameWindow:
    """游戏窗口管理"""
    
    def __init__(self, window_title: str = "崩坏：星穹铁道"):
        self.title = window_title
        self.rect = None
        self.hwnd = None
        self.game_path = self._find_game_path()
    
    def _find_game_path(self) -> Optional[str]:
        """查找游戏路径"""
        paths = [
            r"C:\Program Files\StarRail\StarRail.exe",
            r"D:\Games\StarRail\StarRail.exe",
            r"C:\Games\Honkai Star Rail\StarRail.exe",
        ]
        import os
        for path in paths:
            if os.path.exists(path):
                return path
        return None
    
    def find_window(self) -> bool:
        """查找游戏窗口"""
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(self.title)
            if windows:
                window = windows[0]
                self.hwnd = window._hWnd
                self.rect = {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height
                }
                self.title = window.title
                return True
        except:
            pass
        return False
    
    def start_game(self):
        """启动游戏"""
        if self.game_path:
            subprocess.Popen([self.game_path])
        else:
            print("未找到游戏路径，请手动启动")
    
    def activate(self):
        """激活窗口（临时）"""
        if self.hwnd:
            try:
                import pygetwindow as gw
                window = gw.getWindowsWithTitle(self.title)[0]
                window.activate()
            except:
                pass
    
    def click(self, x: int, y: int):
        """后台点击"""
        try:
            import pywinauto
            from pywinauto import Application
            
            app = Application().connect(handle=self.hwnd)
            window = app.window(handle=self.hwnd)
            
            # 计算相对于窗口的坐标
            abs_x = self.rect["left"] + x
            abs_y = self.rect["top"] + y
            
            window.set_focus()
            import pyautogui
            pyautogui.click(abs_x, abs_y)
            
        except Exception as e:
            print(f"点击失败：{e}")
    
    def press_key(self, key: str):
        """后台按键"""
        try:
            import pywinauto
            from pywinauto import Application
            
            app = Application().connect(handle=self.hwnd)
            window = app.window(handle=self.hwnd)
            window.set_focus()
            
            import pyautogui
            pyautogui.press(key)
            
        except Exception as e:
            print(f"按键失败：{e}")


class ScreenCapture:
    """屏幕捕获"""
    
    def __init__(self, window: GameWindow = None):
        self.window = window
        self.sct = None
    
    def _init_mss(self):
        if self.sct is None:
            import mss
            self.sct = mss.mss()
    
    def capture(self) -> np.ndarray:
        """后台捕获游戏画面"""
        self._init_mss()
        
        if self.window and self.window.rect:
            rect = self.window.rect
            monitor = {
                "left": rect["left"],
                "top": rect["top"],
                "width": rect["width"],
                "height": rect["height"]
            }
        else:
            import mss
            monitor = self.sct.monitors[1]
        
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def get_resolution(self) -> Tuple[int, int]:
        if self.window and self.window.rect:
            return (self.window.rect["width"], self.window.rect["height"])
        return (1920, 1080)


if __name__ == "__main__":
    print("ScreenCapture 模块")
    window = GameWindow()
    if window.find_window():
        print(f"✓ 找到窗口：{window.title}")
        print(f"  位置：{window.rect}")
    else:
        print("✗ 未找到窗口")
