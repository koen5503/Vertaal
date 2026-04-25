import os
import sys
import subprocess
import shutil
import glob

def main():
    print(f"Building for OS: {sys.platform}")
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Create a custom hook to prevent webrtcvad metadata crash on Windows
    os.makedirs("hooks", exist_ok=True)
    with open(os.path.join("hooks", "hook-webrtcvad.py"), "w") as f:
        f.write("# Custom hook to bypass PyInstaller metadata crash for webrtcvad-wheels\n")
        f.write("from PyInstaller.utils.hooks import collect_dynamic_libs\n")
        f.write("binaries = collect_dynamic_libs('webrtcvad')\n")
        f.write("datas = []\n")

    # Define the base command
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--additional-hooks-dir=hooks",
        # Important hidden imports for the new libraries
        "--hidden-import=zeroconf",
        "--hidden-import=zeroconf._utils.ipaddress",
        "--hidden-import=zeroconf._services.browser",
        "--hidden-import=sounddevice",
        "--hidden-import=uvicorn.protocols.http.auto",
        "--hidden-import=uvicorn.protocols.websockets.auto",
        "--hidden-import=uvicorn.lifespan.on",
    ]

    # The separator for --add-data differs depending on the OS
    sep = ";" if sys.platform.startswith("win") else ":"

    # Data files to include
    data_files = [
        f"templates{sep}templates",
        f"SpGlos.txt{sep}.",
    ]
    
    # Optionally include credentials if they exist
    if os.path.exists("ondertitels-486017-0ee48ab1ba8d.json"):
        data_files.append(f"ondertitels-486017-0ee48ab1ba8d.json{sep}.")

    # Include all .wav files in the root directory
    wav_files = glob.glob("*.wav")
    for wav in wav_files:
        data_files.append(f"{wav}{sep}.")

    # We DON'T include .env in the EXE because the user needs to edit it!
    # But we can include .env.example as a reference
    if os.path.exists(".env.example"):
        data_files.append(f".env.example{sep}.")

    for data in data_files:
        cmd.extend(["--add-data", data])

    cmd.append("app.py")

    print(f"Running command:\n{' '.join(cmd)}")
    
    # Run PyInstaller
    try:
        subprocess.check_call(cmd)
        print("\nBuild successful!")
    except subprocess.CalledProcessError as e:
        print(f"\nBuild failed with error: {e}")
        sys.exit(1)
        
    # Provide the path to the executable
    executable_name = "app.exe" if sys.platform.startswith("win") else "app"
    exe_path = os.path.join("dist", executable_name)
    if os.path.exists(exe_path):
        print(f"Executable generated at: {os.path.abspath(exe_path)}")
        print("\nNOTE: Remember to keep your '.env' file in the same folder as the EXE!")

if __name__ == "__main__":
    main()
