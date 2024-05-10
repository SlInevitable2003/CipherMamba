import torch
from insecure_sharing.socket import BetterSocket
from insecure_sharing.iahe import Enc, Dec, AHE_add

class CipherMambaProtocol:
    def set_socket(self, s, role):
        if role == 'C':
            self.socket_c = s
        else:
            self.socket_s = s

    def synchronize(self, role, message=None, input_ids=None, token=None):
        if role == 'C':
            s = self.socket_c
            msg = s.recv()
            if msg == 'embedding':
                self.insecure_embedding('C', input_ids=input_ids)
                return msg, None
            elif msg == 'onemore':
                token = s.recv()
                return msg, torch.cat((input_ids, token), 1)
            elif msg == 'break':
                return msg, None
        else:
            s = self.socket_s
            s.sendall(message)
            if message == 'onemore':
                s.sendall(token)
            return None

    embedding_first_time = True
    def insecure_embedding(self, role, input_ids=None, W=None):
        if role == 'C':
            s = self.socket_c
            k = s.recv()
            m = s.recv()
            
            if self.embedding_first_time == True:
                self.Enc_W = torch.zeros((1, 1, k, m)).to('cuda')
                for i in range(k):
                    Enc_W_list = s.recv()
                    self.Enc_W[0][0][i] = torch.as_tensor(Enc_W_list)

            n = input_ids.shape[1]
            Enc_w = torch.zeros((1, 1, n, m)).to('cuda')
            Enc_w[0][0] = torch.index_select(self.Enc_W[0][0], 0, input_ids[0])
            r = torch.randn_like(Enc_w).to('cuda')
            Enc_w_plus_r = AHE_add(Enc_w, r)
            
            s.sendall(n)
            for i in range(n):
                Enc_w_plus_r_list = Enc_w_plus_r[0][0][i].tolist()
                s.sendall(Enc_w_plus_r_list)
            self.x_after_embedding_c = (-1) * r

            # insecure reveal

            s.sendall(self.x_after_embedding_c)
            self.embedding_first_time = False
            return None
        else:
            s = self.socket_s
            k = W.shape[2]
            m = W.shape[3]
            s.sendall(k)
            s.sendall(m)
            
            if self.embedding_first_time == True:
                for i in range(k):
                    Enc_W_list = Enc(W[0][0][i]).tolist()
                    s.sendall(Enc_W_list)
            n = s.recv()
            Enc_w_plus_r = torch.zeros((1, 1, n, m)).to('cuda')
            for i in range(n):
                Enc_w_plus_r_list = s.recv()
                Enc_w_plus_r[0][0][i] = torch.as_tensor(Enc_w_plus_r_list)
            w_plus_r = Dec(Enc_w_plus_r)
            self.x_after_embedding_s = w_plus_r

            # insecure reveal

            x = s.recv()
            x = x + self.x_after_embedding_s
            self.embedding_first_time = False
            return x[0].to(torch.float16)

protocol = CipherMambaProtocol()