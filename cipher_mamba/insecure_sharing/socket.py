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

        self.send_idx = 0
        self.recv_idx = 0

    def sendall(self, obj, already_bstr = False):
        if self.fast_trans == False:
            if already_bstr == False:
                pkl = pickle.dumps(obj)
            else:
                pkl = obj
            l = len(pkl)
            lbs = l.to_bytes(4)
            self.socket.sendall(lbs)
            self.socket.sendall(pkl)
        else:
            path = self.fast_buffer_prefix + '_' + self.fast_identity + '_' + str(self.send_idx) + '.pickle'
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            self.socket.sendall(b'ACK')
            self.send_idx = (self.send_idx + 1) % 32
    
    def recv(self, already_bstr = False):
        if self.fast_trans == False:
            lbs = self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1)
            l = int.from_bytes(lbs)
            
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
        else:
            recv_identity = 'c' if self.fast_identity == 's' else 's'
            path = self.fast_buffer_prefix + '_' + recv_identity + '_' + str(self.recv_idx) + '.pickle'
            ack = self.socket.recv(1) + self.socket.recv(1) + self.socket.recv(1)
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            self.recv_idx = (self.recv_idx + 1) % 32
            return obj