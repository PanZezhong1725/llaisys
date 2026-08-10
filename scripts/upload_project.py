"""Upload the entire project as a tarball to the remote server."""
import os
import paramiko
import tarfile
import tempfile

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-3BQMrtEHcRqkOMov"
PASSWORD = "Cyx_20020815"
REMOTE_DIR = "/data/llaisys-26s"
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # Create tar.gz
    tar_path = os.path.join(tempfile.gettempdir(), "llaisys-26s.tar.gz")
    print(f"Creating archive: {tar_path} ...")
    tar = tarfile.open(tar_path, "w:gz", compresslevel=1)
    tar.add(LOCAL_DIR, arcname="llaisys-26s",
            filter=lambda x: None if "__pycache__" in x.name or ".build_cache" in x.name or ".git" in x.name.split(os.sep) else x)
    tar.close()
    size_mb = os.stat(tar_path).st_size / 1024 / 1024
    print(f"Archive size: {size_mb:.1f} MB")

    # Upload
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=60)
        sftp = client.open_sftp()
        print(f"Uploading to {REMOTE_DIR}.tar.gz ...")
        sftp.put(tar_path, REMOTE_DIR + ".tar.gz")
        sftp.close()
        print("Upload complete.")

        # Extract on server
        stdin, stdout, stderr = client.exec_command(
            f"cd /data && rm -rf {REMOTE_DIR} && tar xzf llaisys-26s.tar.gz && rm llaisys-26s.tar.gz && echo 'EXTRACT OK'",
            timeout=120
        )
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        if out:
            print("=== STDOUT ===")
            print(out)
        if err:
            print("=== STDERR ===")
            print(err)
        code = stdout.channel.recv_exit_status()
        print(f"=== EXIT CODE: {code} ===")
    finally:
        client.close()
        if os.path.exists(tar_path):
            os.remove(tar_path)

if __name__ == "__main__":
    main()