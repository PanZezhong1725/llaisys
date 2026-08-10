"""Upload a single file to the remote server via SSH (no SFTP required)."""
import os
import sys
import paramiko

HOST = "42.123.114.169"
PORT = 32222
USER = "root+vm-ejaKuDWnyXvtMqPQ"
PASSWORD = "Cyx_20020815"
REMOTE_BASE = "/data/llaisys-26s"


def upload_file(local_path: str, remote_path: str):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
        sftp = client.open_sftp()
        # Ensure parent directory exists
        remote_dir = os.path.dirname(remote_path)
        try:
            sftp.stat(remote_dir)
        except IOError:
            # Create directories recursively
            parts = remote_dir.strip("/").split("/")
            path = ""
            for p in parts:
                path += "/" + p
                try:
                    sftp.stat(path)
                except IOError:
                    sftp.mkdir(path)
        sftp.put(local_path, remote_path)
        sftp.close()
        print(f"uploaded: {local_path} -> {remote_path}")
    except Exception as e:
        print(f"SFTP ERROR: {e}")
        # Fallback: use base64 over SSH
        print("Trying base64 fallback...")
        with open(local_path, "rb") as f:
            import base64
            b64 = base64.b64encode(f.read()).decode()
        # Split into chunks to avoid command-line length issues
        chunk_size = 32000
        chunks = [b64[i:i+chunk_size] for i in range(0, len(b64), chunk_size)]
        # Clear the file first
        stdin, stdout, stderr = client.exec_command(f"> /tmp/upload_b64.txt")
        stdout.channel.recv_exit_status()
        for chunk in chunks:
            cmd = f"printf '%s' '{chunk}' >> /tmp/upload_b64.txt"
            stdin, stdout, stderr = client.exec_command(cmd)
            rc = stdout.channel.recv_exit_status()
            if rc != 0:
                err = stderr.read().decode()
                print(f"ERROR writing chunk: {err}")
                return
        # Decode and move
        stdin, stdout, stderr = client.exec_command(
            f"base64 -d /tmp/upload_b64.txt > '{remote_path}' && rm /tmp/upload_b64.txt"
        )
        rc = stdout.channel.recv_exit_status()
        if rc == 0:
            print(f"uploaded (base64): {local_path} -> {remote_path}")
        else:
            print(f"ERROR decoding: {stderr.read().decode()}")
    finally:
        client.close()


def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <local_path> <remote_rel_path>")
        print(f"Example: python {sys.argv[0]} src/llaisys/qwen2_model.cpp src/llaisys/qwen2_model.cpp")
        sys.exit(1)

    local_rel = sys.argv[1]
    remote_rel = sys.argv[2]

    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), local_rel
    )
    remote_path = REMOTE_BASE + "/" + remote_rel

    upload_file(local_path, remote_path)


if __name__ == "__main__":
    main()