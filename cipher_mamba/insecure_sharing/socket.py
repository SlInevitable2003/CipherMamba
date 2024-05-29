import socket
import pickle
import torch
from insecure_sharing.seal import *

class BetterSocket:
    def __init__(self, s, fast_identity = None):
        self.socket = s

        self.fast_trans = False
        self.fast_identity = fast_identity
        self.fast_buffer_prefix = './cipher_mamba/insecure_sharing/socket_buffer/buffer'

        self.msg_len = 1 << 12

    def sendall(self, obj, already_bstr = False):
        if already_bstr == False:
            pkl = pickle.dumps(obj)
        else:
            pkl = obj
        l = len(pkl)
        lbs = l.to_bytes(4, 'big')
        self.socket.sendall(lbs)
        while l > 0:
            self.socket.sendall(pkl[0:min(l, self.msg_len)])
            pkl = pkl[min(l, self.msg_len):]
            l -= min(l, self.msg_len)
        self.socket.sendall(pkl)
    
    def recv(self, already_bstr = False):
        lbs = self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1)
        l = int.from_bytes(lbs, 'big')
        
        pkl = b''
        while l > 0:
            rev = self.socket.recv(l)
            l = l - len(rev)
            pkl = pkl + rev
        if already_bstr == False:
            obj = pickle.loads(pkl)
        else:
            obj = pkl
        return obj