import os
import sys
import subprocess
import shutil

def main():
    print(f"Building for OS: {sys.platform}")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Define the base command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
    ]

    # The separator for --add-data differs depending on the OS
    sep = ";" if sys.platform.startswith("win") else ":"

    # Data files to include
    data_files = [
        f"templates{sep}templates",
        f"ondertitels-486017-0ee48ab1ba8d.json{sep}.",
        f"BV.wav{sep}.",
        f".env{sep}."
    ]

    for data in data_files:
        cmd.extend(["--add-data", data])

    cmd.append("app.py")

    print(f"Running command:\\n{' '.join(cmd)}")
    
    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
        print("\\nBuild successful!")
    except subprocess.CalledProcessError as e:
        print(f"\\nBuild failed with error: {e}")
        sys.exit(1)
        
    # Provide the path to the executable
    executable_name = "app.exe" if sys.platform.startswith("win") else "app"
    exe_path = os.path.join("dist", executable_name)
    if os.path.exists(exe_path):
        print(f"Executable generated at: {os.path.abspath(exe_path)}")

if __name__ == "__main__":
    main()
