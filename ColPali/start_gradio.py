import subprocess
import sys

if __name__ == "__main__":
    # 构造 Gradio 启动命令
    command = ["gradio", "Colpali/app.py"] + sys.argv[1:]
    # 启动 Gradio 应用
    subprocess.run(command)