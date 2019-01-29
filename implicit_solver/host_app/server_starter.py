"""
@author: Vincent Bonnet
@description : Server to be executed on another process
"""

import ipc
import socket
import os

def main():
    host = "localhost" # 127.0.0.1
    port = 5050
    server = ipc.BundleIPC()

    # Create server socket and get connection
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host,port))

    print('Start Server IPC :')
    print('   parent process:', os.getppid())
    print('   process id:', os.getpid())

    server_socket.listen(1)
    conn, addr = server_socket.accept()
    print ("Connection from: " + str(addr))

    # Wait for client data
    while True:
            # check received data
            data = conn.recv(1024).decode()
            valid_data = False
            if data == 'exit':
                break # Exit the loop
            elif data == "is_defined":
                valid_data = True
            elif data == "initialize":
                valid_data = True
            elif data == "step":
                valid_data = True
            elif data == "get_scene":
                valid_data = True
            elif data == "get_solver":
                valid_data = True
            elif data == "get_context":
                valid_data = True

            if valid_data:
                print ("from connected  user: " + str(data))
                data = str(data).upper()
                print ("sending: " + str(data))
                conn.send(data.encode())

    conn.close()


if __name__ == '__main__':
    main()
    input("Press Enter to exit server...")

