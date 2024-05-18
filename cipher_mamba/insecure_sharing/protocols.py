import copy
import torch
from insecure_sharing.socket import BetterSocket
from insecure_sharing.iahe import AHE, Polynomial
from insecure_sharing.seal import *

class CipherMambaProtocol:
    def set_socket(self, s, role):
        if role == 'C':
            self.socket_c = s
        else:
            self.socket_s = s
    
    def create_ahe(self, role):
        ahe = AHE()
        if role == 'C':
            self.ahe_c = ahe
        else:
            self.ahe_s = ahe

    def synchronize(self, role, message=None, input_ids=None, token=None):
        if role == 'C':
            s = self.socket_c
            msg = s.recv()
            if msg == 'embedding':
                self.insecure_embedding('C', input_ids=input_ids)
                return msg, None
            elif msg == 'ahe s2c':
                pk_str = s.recv()
                rks_str = s.recv()
                self.ahe_s = AHE(pk_str=pk_str, rks_str=rks_str)
                return msg, None
            elif msg == 'onemore':
                token = s.recv()
                return msg, token
            elif msg == 'break':
                return msg, None
        else:
            s = self.socket_s
            s.sendall(message)
            if message == 'onemore':
                s.sendall(token)
            elif message == 'ahe s2c':
                s.sendall(self.ahe_s.pk.to_string())
                s.sendall(self.ahe_s.rks.to_string())
            return None

    embedding_first_time = True
    def insecure_embedding(self, role, input_ids=None, W=None):
        if role == 'C':
            s = self.socket_c
            k = s.recv()
            m = s.recv()
            
            if self.embedding_first_time == True:
                self.Enc_Ws = []
                for i in range(k):
                    line = self.ahe_s.context.from_cipher_str(s.recv())
                    self.Enc_Ws.append(line)

            ids = input_ids.squeeze(0)
            n = ids.shape[0]
            
            Enc_ws = []
            r = torch.randn((n, m)) * (1 << 12)
            r = r.to(torch.int32)
            for i, j in enumerate(ids.tolist()):
                line = self.Enc_Ws[j]
                rdm = r[i].tolist()
                line = self.ahe_s.ahe_add_plain(line, rdm, is_list=True)
                Enc_ws.append(line)
            
            s.sendall(n)
            for i in Enc_ws:
                s.sendall(i.to_string())
            self.x_after_embedding_c = (-1) * r

            # insecure reveal

            s.sendall(self.x_after_embedding_c)
            self.embedding_first_time = False
            return None
        else:
            s = self.socket_s
            k, m = W.shape[0], W.shape[1]
            s.sendall(k)
            s.sendall(m)
            
            if self.embedding_first_time == True:
                for i in range(k):
                    line = W[i].tolist()
                    print(f'[{i}] going to send:', line[0:3] + ['...'] + line[-3:])
                    s.sendall(self.ahe_s.enc_list(line).to_string())

            n = s.recv()
            w_plus_r = torch.zeros((n, m))
            for i in range(n):
                line = self.ahe_s.context.from_cipher_str(s.recv())
                line_lst = self.ahe_s.dec_list(line)
                w_plus_r[i] = torch.as_tensor(line_lst)
            self.x_after_embedding_s = w_plus_r

            # insecure reveal

            x = s.recv()
            x = x + self.x_after_embedding_s
            self.embedding_first_time = False
            return x


protocol = CipherMambaProtocol()