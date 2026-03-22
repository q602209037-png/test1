#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""屏幕捕获模块"""

import mss
import mss.tools
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import cv2


class ScreenCapture:
    """屏幕捕获类"""
    
    def __init__(self, monitor_id: int = 1):
        self.monitor_id = monitor_id
        self.sct = mss.mss()
        self._game_window: Optional[dict] = None
        
    def get_monitors(self) -> list:
        return list(self.sct.monitors)
    
    def find_game_window(self, window_title: str = "崩坏：星穹铁道") -> Optional[dict]:
        try:
            import pygetwindow as gw
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                window = windows[0]
                self._game_window = {
                    "left": window.left,
                    "top": window.top,
                    "width": window.width,
                    "height": window.height
                }
                return self._game_window
        except ImportError:
            print("警告：pygetwindow 未安装，使用全屏捕获")
        except Exception as e:
            print(f"查找窗口失败：{e}")
        return self.sct.monitors[self.monitor_id]
    
    def capture(self, region: Optional[dict] = None) -> np.ndarray:
        if region is None:
            region = self._game_window or self.sct.monitors[self.monitor_id]
        screenshot = self.sct.grab(region)
        img = np.array(screenshot)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img_bgr
    
    def save_screenshot(self, filepath: str, region: Optional[dict] = None) -> str:
        img = self.capture(region)
        cv2.imwrite(filepath, img)
        return filepath
    
    def get_screen_resolution(self) -> Tuple[int, int]:
        monitor = self.sct.monitors[self.monitor_id]
        return (monitor["width"], monitor["height"])


if __name__ == "__main__":
    capture = ScreenCapture()
    monitors = capture.get_monitors()
    print(f"可用显示器数量：{len(monitors)}")
    frame = capture.capture()
    print(f"捕获画面尺寸：{frame.shape}")
