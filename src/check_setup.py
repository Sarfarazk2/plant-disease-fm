"""
check_setup.py  -  Step 1 verification script
Run inside the 'plantfm' environment:   python check_setup.py
Copy the full output and send it back so we can confirm the setup.
"""
import sys
import platform

print("=" * 50)
print("Python  :", sys.version.split()[0], "|", platform.system(), platform.release())

try:
    import torch
except ImportError:
    print("PyTorch : NOT INSTALLED  -> install it from pytorch.org (Step 1.5)")
    sys.exit()

print("PyTorch :", torch.__version__)
print("CUDA OK :", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU     :", props.name)
    print("GPU RAM : %.1f GB" % (props.total_memory / 1024**3))
    print("Compute capability: %d.%d" % (props.major, props.minor))
    try:
        x = torch.randn(2000, 2000, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print("GPU test: PASSED")
    except RuntimeError as e:
        print("GPU test: FAILED ->", str(e)[:200])
        print("Your GPU may need an older CUDA build of PyTorch. Send me this output.")
else:
    print("GPU not detected. Training will be slow. Send me this output.")

print("-" * 50)
packages = ["torchvision", "timm", "open_clip", "sklearn", "pandas",
            "matplotlib", "seaborn", "tqdm"]
for name in packages:
    try:
        module = __import__(name)
        print("%-12s: %s" % (name, getattr(module, "__version__", "installed")))
    except ImportError:
        print("%-12s: NOT INSTALLED" % name)
print("=" * 50)
