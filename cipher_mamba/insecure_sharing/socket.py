import socket
import pickle
import torch

class BetterSocket:
    def __init__(self, s):
        self.socket = s
        self.msg_len = 2 ** 14

    def sendall(self, obj):
        # print(f"Going to send obj {obj}")
        pkl = pickle.dumps(obj)
        l = len(pkl)
        init_l = len(pkl)
        lbs = l.to_bytes(4)
        # print(f"With LBS = {lbs}")
        self.socket.sendall(lbs)
        # print("Going to send...")
        while l > self.msg_len:
            self.socket.sendall(pkl[0:self.msg_len])
            pkl = pkl[self.msg_len:]
            l = l - self.msg_len
            # print(f"{l} bytes left...")
        self.socket.sendall(pkl)
        # print(f"Done. {4 + init_l} bytes have been sent.")
    
    def recv(self):
        # print("Waiting to receive...")
        lbs = self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1)
        # print(f"LBS {lbs} received.")
        l = int.from_bytes(lbs)
        # print(f"Length {l} received.")
        
        pkl = b''
        while l > 0:
            rev = self.socket.recv(min(self.msg_len, l))
            l = l - len(rev)
            pkl = pkl + rev
            # print(f"{l} bytes left...")
        obj = pickle.loads(pkl)
        # print(f"Object {obj} received.")
        return obj