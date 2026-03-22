"""主程序"""
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "demo"])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    if args.command == "demo":
        print("AI 演示模式")
        print("系统就绪!")

if __name__ == "__main__":
    main()
