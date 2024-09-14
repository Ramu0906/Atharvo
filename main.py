import subprocess

def run_training():
    subprocess.run(["python", "scripts/train.py"])

if __name__ == '__main__':
    run_training()