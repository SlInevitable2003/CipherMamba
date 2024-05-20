import copy
import torch
import torch.nn.functional as F
from insecure_sharing.socket import BetterSocket
from insecure_sharing.iahe import AHE, Polynomial
from insecure_sharing.seal import *

class CipherOption:
    use_secure_protocol = False
    def secure_protocol_set(self):
        use_secure_protocol = True

options = CipherOption()

class CipherMambaProtocol:
    def set_socket(self, s):
        self.socket = s
    
    def create_ahe(self, role):
        ahe = AHE()
        if role == 'C':
            self.ahe_c = ahe
        else:
            self.ahe_s = ahe

    def synchronize(self, role, message=None, input_ids=None, token=None, x=None):
        if role == 'C':
            s = self.socket
            msg = s.recv()
            if msg == 'embedding':
                self.insecure_embedding('C', input_ids=input_ids)
                return msg, None
            elif msg == 'ahe s2c':
                pk_str = s.recv()
                rks_str = s.recv()
                self.ahe_s = AHE(pk_str=pk_str, rks_str=rks_str)
                return msg, None
            elif msg == 'ahe c2s':
                self.ahe_c = AHE()
                s.sendall(self.ahe_c.pk.to_string())
                s.sendall(self.ahe_c.rks.to_string())
                return msg, None
            elif msg == 'onemore':
                token = s.recv()
                return msg, token
            elif msg == 'linear_lmHead':
                x = s.recv()
                self.logits = self.insecure_matmul('C', X=x)
                return msg, None
            elif msg == 'conv':
                x = s.recv()
                self.x_after_conv = self.insecure_matmul('C', X=x)
                return msg, None
            elif msg == 'break':
                return msg, None
        else:
            s = self.socket
            s.sendall(message)
            if message == 'onemore':
                s.sendall(token)
            elif message == 'ahe s2c':
                s.sendall(self.ahe_s.pk.to_string())
                s.sendall(self.ahe_s.rks.to_string())
            elif message == 'ahe c2s':
                pk_str = s.recv()
                rks_str = s.recv()
                self.ahe_c = AHE(pk_str=pk_str, rks_str=rks_str)
            elif message in ['linear_lmHead', 'conv']:
                s.sendall(x)
            return None

    embedding_first_time = True
    def insecure_embedding(self, role, input_ids=None, W=None):
        if role == 'C':
            s = self.socket
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
            r = torch.randn((n, m)).to(torch.double) * (1 << 12)
            r = r.to(torch.int32)
            for i, j in enumerate(ids.tolist()):
                line = self.Enc_Ws[j]
                rdm = r[i].tolist()
                line = self.ahe_s.ahe_add_plain(line, rdm, is_list=True)
                Enc_ws.append(line)
            
            s.sendall(n)
            for i in Enc_ws:
                s.sendall(i.to_string())
            self.x_after_embedding = (-1) * r

            # insecure reveal

            s.sendall(self.x_after_embedding)
            self.embedding_first_time = False
            return None
        else:
            s = self.socket
            k, m = W.shape[0], W.shape[1]
            s.sendall(k)
            s.sendall(m)
            

            print('')
            if self.embedding_first_time == True:
                for i in range(k):
                    line = W[i].tolist()
                    print(f'\r[{i}] going to send:', line[0:3] + ['...'] + line[-3:], end='')
                    s.sendall(self.ahe_s.enc_list(line).to_string())

            n = s.recv()
            w_plus_r = torch.zeros((n, m))
            for i in range(n):
                line = self.ahe_s.context.from_cipher_str(s.recv())
                line_lst = self.ahe_s.dec_list(line, length=m)
                w_plus_r[i] = torch.as_tensor(line_lst)
            self.x_after_embedding = w_plus_r

            # insecure reveal

            x = s.recv()
            x = x + self.x_after_embedding
            self.embedding_first_time = False
            return x
        
    def insecure_matmul(self, role, X=None, Y=None):
        import math

        def cal_block(m, n, k, N):
            t = math.ceil(math.sqrt(m * n * k / N))
            assert m * max(1, math.floor(n / t)) * max(1, math.floor(k / t)) <= N
            return m, max(1, math.floor(n / t)), max(1, math.floor(k / t))
        
        def matmul_body(self : CipherMambaProtocol, role, shape, X=None, Y=None):
            if role == 'C':
                s = self.socket
                m, n, k = shape[0], shape[1], shape[2]

                pix_coeff = [0] * ((m - 1) * n * k + n)
                for i in range(m):
                    for j in range(n):
                        pix_coeff[i * n * k + (n - 1) - j] = X[i][j].item()
                ct = self.ahe_s.context.from_cipher_str(s.recv())
                self.ahe_s.ahe_mul_plain_inplace(ct, pix_coeff, is_list=True)

                r = torch.randn((m, k)).to(torch.double) * (1 << 12)
                r = r.to(torch.int64)
                piz_coeff = [0] * (m * n * k)
                for i in range(m):
                    for j in range(k):
                        piz_coeff[i * n * k + (j + 1) * n - 1] = r[i][j].item()
                self.ahe_s.ahe_add_plain(ct, piz_coeff, is_list=True)
                s.sendall(ct.to_string())
                return (-1) * r
            else:
                s = self.socket
                m, n, k = shape[0], shape[1], shape[2]

                piy_coeff = [0] * (k * n)
                for i in range(n):
                    for j in range(k):
                        piy_coeff[j * n + i] = Y[i][j].item()
                ct = self.ahe_s.enc_list(piy_coeff)
                s.sendall(ct.to_string())
                ct = self.ahe_s.context.from_cipher_str(s.recv())
                piz_coeff = self.ahe_s.dec_list(ct, length=m*n*k)
                Z = torch.zeros((m, k)).to(torch.int64)
                for i in range(m):
                    for j in range(k):
                        Z[i][j] = piz_coeff[i * n * k + (j + 1) * n - 1]
                return Z
        
        if role == 'C':
            s = self.socket

            m = X.shape[0]
            n = X.shape[1]
            s.sendall(m)
            k = s.recv()
            N = self.ahe_s.poly_mod_deg

            m_w, n_w, k_w = cal_block(m, n, k, N)
            X = F.pad(X, (0, math.ceil(n / n_w) * n_w - n, 0, 0))
            m_p, n_p, k_p = 1, math.ceil(n / n_w), math.ceil(k / k_w)

            Z = torch.zeros((m_p * m_w, k_p * k_w))
            for i in range(m_p):
                for j in range(k_p):
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        sum_ij += matmul_body(self, role=role, shape=[m_w, n_w, k_w], X=X[i*m_w:(i+1)*m_w, l*n_w:(l+1)*n_w])
                    Z[i*m_w:(i+1)*m_w, j*k_w:(j+1)*k_w] = sum_ij
            Z = Z[0:m, 0:k]

            # insecure revealing
            s.sendall(Z)
            return Z

        else:
            s = self.socket

            n = Y.shape[0]
            k = Y.shape[1]
            m = s.recv()
            s.sendall(k)
            N = self.ahe_s.poly_mod_deg

            m_w, n_w, k_w = cal_block(m, n, k, N)
            Y = F.pad(Y, (0, math.ceil(k / k_w) * k_w - k, 0, math.ceil(n / n_w) * n_w - n))
            m_p, n_p, k_p = 1, math.ceil(n / n_w), math.ceil(k / k_w)

            # print('')
            Z = torch.zeros((m_p * m_w, k_p * k_w))
            for i in range(m_p):
                for j in range(k_p):
                    # print(f'\rmatmul process: {int((i * k_p + j) * 100 / (m_p * k_p))}%', end='')
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        sum_ij += matmul_body(self, role=role, shape=[m_w, n_w, k_w], Y=Y[l*n_w:(l+1)*n_w, j*k_w:(j+1)*k_w])
                    Z[i*m_w:(i+1)*m_w, j*k_w:(j+1)*k_w] = sum_ij
            Z = Z[0:m, 0:k]

            # insecure revealing

            ZZ = s.recv()
            Z = Z + ZZ
            return Z

protocol = CipherMambaProtocol()