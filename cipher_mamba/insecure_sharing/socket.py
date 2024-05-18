import socket
import pickle
import torch
from insecure_sharing.seal import *

class BetterSocket:
    def __init__(self, s):
        self.socket = s
        self.msg_len = 2 ** 14

    def sendall(self, obj):
        pkl = pickle.dumps(obj)
        l = len(pkl)
        init_l = len(pkl)
        lbs = l.to_bytes(4)
        self.socket.sendall(lbs)
        while l > self.msg_len:
            self.socket.sendall(pkl[0:self.msg_len])
            pkl = pkl[self.msg_len:]
            l = l - self.msg_len
        self.socket.sendall(pkl)
    
    def recv(self):
        lbs = self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1)
        l = int.from_bytes(lbs)
        
        pkl = b''
        while l > 0:
            rev = self.socket.recv(min(self.msg_len, l))
            l = l - len(rev)
            pkl = pkl + rev
        obj = pickle.loads(pkl)
        return obj