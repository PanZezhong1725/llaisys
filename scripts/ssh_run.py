"""Run a command on the remote Linux server via SSH (password auth)."""
import sys
import paramiko

HOST = "140.207.205.81"
PORT = 32222
USER = "root+vm-3BQMrtEHcRqkOMov"
PASSWORD = "Cyx_20020815"

def main():
    if len(sys.argv) < 2:
        print("Usage: python ssh_run.py '<command>'")
        sys.exit(1)
    command = sys.argv[1]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
        stdin, stdout, stderr = client.exec_command(command, timeout=600)
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
    except Exception as e:
        print(f"SSH ERROR: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()
