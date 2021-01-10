import paramiko, os

def remote_scp(type, host_ip, remote_path, local_path, username, password):
    ssh_port = 22
    # try:
    conn = paramiko.Transport((host_ip, ssh_port))
    conn.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(conn)
    if type == 'remoteRead':
        if not local_path:
            fileName = os.path.split(remote_path)
            local_path = os.path.join('/tmp', fileName[1])
        sftp.get(remote_path, local_path)

    if type == "remoteWrite":
        sftp.put(local_path, remote_path)

    conn.close()
    return True

    # except Exception:
    #     print('error')
    #     return False

