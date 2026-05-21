"""
마우스 자동 이동 스크립트 — Kaggle 세션 idle 끊김 방지
Ctrl+C 로 종료
"""
import time
import math
import sys

try:
    import pyautogui
except ImportError:
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'pyautogui', '-q'])
    import pyautogui

pyautogui.FAILSAFE = True  # 마우스를 화면 좌상단으로 이동하면 즉시 종료

INTERVAL = 30  # 초마다 한 번 이동
RADIUS = 5     # 픽셀 반경 (작게 움직임)

print("마우스 자동 이동 시작 (Ctrl+C 또는 마우스를 화면 좌상단으로 이동하면 종료)")
print(f"이동 간격: {INTERVAL}초")

count = 0
try:
    while True:
        x, y = pyautogui.position()
        # 작은 원 궤도로 이동
        angle = (count % 4) * 90
        dx = int(RADIUS * math.cos(math.radians(angle)))
        dy = int(RADIUS * math.sin(math.radians(angle)))
        pyautogui.moveTo(x + dx, y + dy, duration=0.3)
        count += 1
        print(f"[{time.strftime('%H:%M:%S')}] 이동 {count}회 완료 (위치: {x+dx}, {y+dy})", end='\r')
        time.sleep(INTERVAL)
except KeyboardInterrupt:
    print(f"\n종료. 총 {count}회 이동.")
