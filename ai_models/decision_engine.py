"""决策引擎"""
class DecisionEngine:
    def __init__(self):
        self.history = []
    
    def decide(self, screen):
        return {"action": "click", "confidence": 0.85}
    
    def check_loop(self):
        if len(self.history) > 10:
            recent = [h["action"] for h in self.history[-10:]]
            if len(set(recent)) == 1:
                return "警告：死循环"
        return None

if __name__ == "__main__":
    print("DecisionEngine 模块")
