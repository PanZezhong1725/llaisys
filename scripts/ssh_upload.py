"""Upload local NVIDIA implementation files to the remote Linux server via SFTP."""
import os
import paramiko

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-3BQMrtEHcRqkOMov"
PASSWORD = "Cyx_20020815"
REMOTE_BASE = "/data/llaisys-26s"
LOCAL_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files to upload (relative to project root)
FILES = [
    "xmake.lua",
    "xmake/nvidia.lua",
    "src/device/nvidia/nvidia_resource.cu",
    "src/device/nvidia/nvidia_resource.cuh",
    "src/device/nvidia/nvidia_runtime_api.cu",
    "src/ops/add/op.cpp",
    "src/ops/add/nvidia/add_nvidia.hpp",
    "src/ops/add/nvidia/add_nvidia.cu",
    "src/ops/argmax/op.cpp",
    "src/ops/argmax/nvidia/argmax_nvidia.hpp",
    "src/ops/argmax/nvidia/argmax_nvidia.cu",
    "src/ops/embedding/op.cpp",
    "src/ops/embedding/nvidia/embedding_nvidia.hpp",
    "src/ops/embedding/nvidia/embedding_nvidia.cu",
    "src/ops/linear/op.cpp",
    "src/ops/linear/nvidia/linear_nvidia.hpp",
    "src/ops/linear/nvidia/linear_nvidia.cu",
    "src/ops/rms_norm/op.cpp",
    "src/ops/rms_norm/nvidia/rms_norm_nvidia.hpp",
    "src/ops/rms_norm/nvidia/rms_norm_nvidia.cu",
    "src/ops/rope/op.cpp",
    "src/ops/rope/nvidia/rope_nvidia.hpp",
    "src/ops/rope/nvidia/rope_nvidia.cu",
    "src/ops/self_attention/op.cpp",
    "src/ops/self_attention/nvidia/self_attention_nvidia.hpp",
    "src/ops/self_attention/nvidia/self_attention_nvidia.cu",
    "src/ops/swiglu/op.cpp",
    "src/ops/swiglu/nvidia/swiglu_nvidia.hpp",
    "src/ops/swiglu/nvidia/swiglu_nvidia.cu",
    "test/ops/self_attention.py",
    "test/ops/rope.py",
    "scripts/diag_rope.py",
]



def main():
    for rel in FILES:
        local = os.path.join(LOCAL_BASE, rel)
        remote = REMOTE_BASE + "/" + rel
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
            sftp = client.open_sftp()
            sftp.put(local, remote)
            sftp.close()
            print(f"uploaded: {rel}")
        except Exception as e:
            print(f"SFTP ERROR on {rel}: {e}")
            import traceback
            traceback.print_exc()
            raise
        finally:
            client.close()
    print("=== ALL UPLOADED ===")

if __name__ == "__main__":
    main()
