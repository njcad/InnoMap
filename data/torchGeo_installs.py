import os
import subprocess
import torch

def install_torch_geometric():
    # Check if running in Gradescope environment
    if 'IS_GRADESCOPE_ENV' in os.environ:
        print("Running in Gradescope environment. Skipping installation.")
        return

    # Get PyTorch version
    torch_version = torch.__version__.split("+")[0]
    cuda_version = "cpu"  # Adjust for CUDA if necessary (e.g., "cu117")
    base_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_version}.html"

    # List of packages to install
    packages = [
        f"torch-scatter -f {base_url}",
        f"torch-sparse -f {base_url}",
        f"torch-cluster -f {base_url}",
        f"torch-spline-conv -f {base_url}",
        "torch-geometric",
        "git+https://github.com/snap-stanford/deepsnap.git",
        "PyDrive"
    ]

    # Install each package
    for package in packages:
        try:
            print(f"Installing: {package}")
            subprocess.check_call(
                f"pip install {package}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"Successfully installed: {package}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to install: {package}\nError: {e}")

if __name__ == "__main__":
    install_torch_geometric()
