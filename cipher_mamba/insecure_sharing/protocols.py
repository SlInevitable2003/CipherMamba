import copy
import torch
import numpy as np
import torch.nn.functional as F
from insecure_sharing.socket import BetterSocket
from insecure_sharing.iahe import AHE, CKKS, Polynomial
from cipher_mamba.insecure_sharing.multi_processing import MultiProcessing
from insecure_sharing.seal import *

class CipherOption:
    use_secure_protocol = True
    use_secure_protocol_embedding = False
    
    def secure_protocol_set(self, **kwargs):
        self.use_secure_protocol = True

        if kwargs.get('embedding') is not None:
            self.use_secure_protocol_embedding = kwargs['embedding']

options = CipherOption()
options.secure_protocol_set()

class CipherMambaProtocol:
    def set_socket(self, s):
        self.socket = s
    
    def create_ahe(self, role):
        ahe = AHE()
        if role == 'C':
            self.ahe_c = ahe
        else:
            self.ahe_s = ahe

    def create_ckks(self, role):
        ckks = CKKS()
        if role == 'C':
            self.ckks_c = ckks
        else:
            self.ckks_s = ckks

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
            elif msg == 'residual_plus':
                self.residual = self.residual + self.hidden_states

            elif msg == 'linear_lmHead':
                x = s.recv()
                self.logits = self.insecure_matmul('C', X=x)
                return msg, None
            elif msg == 'conv':
                x = s.recv()
                self.x = self.insecure_conv('C', X=x)
                return msg, None
            elif msg == 'SiLU':
                x = s.recv()
                self.x = self.insecure_SiLU('C', input_array = x)
                return msg, None
            elif msg == 'Softplus':
                x = s.recv()
                self.x = self.insecure_Softplus('C', input_array = x)
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
            elif message in ['SiLU', 'Softplus']:
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
                    line = self.ahe_s.context.from_cipher_str(s.recv(already_bstr=True))
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
                s.sendall(i.to_string(), already_bstr=True)
            self.hidden_states = (-1) * r
            self.residual = torch.zeros_like(self.hidden_states)

            # insecure reveal

            s.sendall(self.hidden_states)
            self.embedding_first_time = False
            return None
        else:
            s = self.socket
            k, m = W.shape[0], W.shape[1]
            s.sendall(k)
            s.sendall(m)
            
            print('')
            if self.embedding_first_time == True:
                lines = [W[i].tolist() for i in range(k)]
                target = lambda lst : protocol.ahe_s.enc_list(lst).to_string()
                print('processes creating...')

                processes = MultiProcessing(target=target, args=lines, granularity=32)
                processes.start()
                processes.join()

                line_idx = 0
                for i in processes.ret_buffer():
                    print(f'\r[{line_idx}] going to send...', end='')
                    s.sendall(i, already_bstr=True)
                    line_idx += 1

            n = s.recv()
            w_plus_r = torch.zeros((n, m))
            for i in range(n):
                line = self.ahe_s.context.from_cipher_str(s.recv(already_bstr=True))
                line_lst = self.ahe_s.dec_list(line, length=m)
                w_plus_r[i] = torch.as_tensor(line_lst)
            self.hidden_states = w_plus_r
            self.residual = torch.zeros_like(self.hidden_states)

            # insecure reveal

            x = s.recv()
            x = x + self.hidden_states
            self.embedding_first_time = False
            return x

    def insecure_rmsnorm(self, role, w=None, eps=None):
        if role == 'C':
            s = self.socket
        else:
            s = self.socket
            self.residual = self.residual + self.hidden_states

    def insecure_mul(self, role, x=None, y=None):
        if role == 'C':
            pass
        else:
            pass

    def insecure_matmul(self, role, X=None, Y=None):
        import math

        def cal_block(m, n, k, N):
            t = math.ceil(math.sqrt(m * n * k / N))
            assert m * max(1, math.floor(n / t)) * max(1, math.floor(k / t)) <= N
            return m, max(1, math.floor(n / t)), max(1, math.floor(k / t))
        
        if role == 'C':
            s = self.socket

            m = X.shape[0]
            n = X.shape[1]
            s.sendall(m)
            k = s.recv()
            N = self.ahe_s.poly_mod_deg

            m_w, n_w, k_w = cal_block(m, n, k, N)
            X = F.pad(X, (0, math.ceil(n / n_w) * n_w - n, 0, 0)).to('cpu')
            m_p, n_p, k_p = 1, math.ceil(n / n_w), math.ceil(k / k_w)

            def get_target_poly(ijl):
                X_t = X[ijl[0]*m_w:(ijl[0]+1)*m_w, ijl[2]*n_w:(ijl[2]+1)*n_w]
                pix_coeff = [0] * ((m_w - 1) * n_w * k_w + n_w)
                for i in range(m_w):
                    for j in range(n_w):
                        pix_coeff[i * n_w * k_w + (n_w - 1) - j] = X_t[i][j].item()
                
                r = torch.randn((m_w, k_w)).to(torch.double) * (1 << 12)
                r = r.to(torch.int64)
                piz_coeff = [0] * (m_w * n_w * k_w)
                for i in range(m_w):
                    for j in range(k_w):
                        piz_coeff[i * n_w * k_w + (j + 1) * n_w - 1] = r[i][j].item()
                
                return [pix_coeff, piz_coeff, r.tolist()]
            
            args = [[i, j, l] for i in range(m_p) for j in range(k_p) for l in range(n_p)]
            processes = MultiProcessing(target=get_target_poly, args=args, role='c', granularity=32)
            processes.start()
            processes.join()

            # ct = self.ahe_s.context.from_cipher_str(s.recv(already_bstr=True))
            # self.ahe_s.ahe_mul_plain_inplace(ct, pix_coeff, is_list=True)
            # self.ahe_s.ahe_add_plain(ct, piz_coeff, is_list=True)
            # s.sendall(ct.to_string(), already_bstr=True)

            s.recv()

            def get_ans_poly(pid):
                import pickle
                common_prefix = './cipher_mamba/insecure_sharing/socket_buffer/buffer'
                path_s = common_prefix + 's' + str(pid) + '.pickle'
                path_c = common_prefix + 'c' + str(pid) + '.pickle'
                with open(path_c, 'rb') as f:
                    pix_piz_r_lst = pickle.load(f)
                with open(path_c, 'wb'): pass
                with open(path_s, 'rb') as f:
                    encpiy_lst = pickle.load(f)
                with open(path_s, 'wb'): pass

                assert len(pix_piz_r_lst) == len(encpiy_lst)
                l = len(pix_piz_r_lst)

                ret = []
                for i in range(l):
                    pix_piz_r = pix_piz_r_lst[i]
                    if pix_piz_r is None:
                        break
                    ct = self.ahe_s.context.from_cipher_str(encpiy_lst[i])
                    pix, piz, r = pix_piz_r[0], pix_piz_r[1], pix_piz_r[2]

                    self.ahe_s.ahe_mul_plain_inplace(ct, pix, is_list=True)
                    self.ahe_s.ahe_add_plain_inplace(ct, piz, is_list=True)
                    ret.append([ct.to_string(), r])
                return ret
            
            processes = MultiProcessing(target=get_ans_poly, args=[i for i in range(32)], role='c', granularity=32, extra_buffer=True)
            processes.start()
            processes.join()

            def ret_buffer(large_buffer):
                for i in large_buffer:
                    for j in i:
                        yield j
            ans_buffer = ret_buffer(processes.ret_buffer())

            s.sendall(b'OK')
            s.recv()

            Z = torch.zeros((m_p * m_w, k_p * k_w))
            for i in range(m_p):
                for j in range(k_p):
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        nxt = next(ans_buffer)
                        r = torch.as_tensor(nxt[1])
                        sum_ij += r
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
            Y = F.pad(Y, (0, math.ceil(k / k_w) * k_w - k, 0, math.ceil(n / n_w) * n_w - n)).to('cpu')
            m_p, n_p, k_p = 1, math.ceil(n / n_w), math.ceil(k / k_w)
            
            # optimization version since tensor operation is used instead of iteration
            target = lambda ijl : self.ahe_s.enc_list(Y[ijl[2]*n_w:(ijl[2]+1)*n_w, ijl[1]*k_w:(ijl[1]+1)*k_w].t().flatten().tolist()).to_string()
            args = [[i, j, l] for i in range(m_p) for j in range(k_p) for l in range(n_p)]
            processes = MultiProcessing(target=target, args=args, granularity=32, show_process=False)
            processes.start()
            processes.join()

            s.sendall(b'OK')
            s.recv()

            # ct = self.ahe_s.context.from_cipher_str(s.recv(already_bstr=True))
            # piz_coeff = self.ahe_s.dec_list(ct, length=m*n*k)
            # Z = torch.zeros((m, k)).to(torch.int64)
            # for i in range(m):
            #     for j in range(k):
            #         Z[i][j] = piz_coeff[i * n * k + (j + 1) * n - 1]

            def get_ans_poly(pid):
                import pickle
                path = './cipher_mamba/insecure_sharing/socket_buffer/ex_bufferc' + str(pid) + '.pickle'
                with open(path, 'rb') as f:
                    ct_r_lst = pickle.load(f)
                ct_r_lst = ct_r_lst[0]
                
                ret = []
                for ct_r in ct_r_lst:
                    ct = ct_r[0]
                    ct = self.ahe_s.context.from_cipher_str(ct)
                    
                    piz_coeff = self.ahe_s.dec_list(ct, length=m_w*n_w*k_w)
                    Z = torch.zeros((m_w, k_w)).to(torch.int64)
                    for i in range(m_w):
                        for j in range(k_w):
                            Z[i][j] = piz_coeff[i * n_w * k_w + (j + 1) * n_w - 1]
                    ret.append(Z.tolist())
                return ret
                
            processes = MultiProcessing(target=get_ans_poly, args=[i for i in range(32)], role='s', granularity=32)
            processes.start()
            processes.join()
            
            def ret_buffer(large_buffer):
                for i in large_buffer:
                    for j in i:
                        yield j
            ans_buffer = ret_buffer(processes.ret_buffer())

            s.sendall(b'OK')

            print('')
            Z = torch.zeros((m_p * m_w, k_p * k_w))
            for i in range(m_p):
                for j in range(k_p):
                    print(f'\rmatmul process: {int((i * k_p + j) * 100 / (m_p * k_p))}%', end='')
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        Z_l = next(ans_buffer)
                        Z_l = torch.as_tensor(Z_l)
                        sum_ij += Z_l
                    Z[i*m_w:(i+1)*m_w, j*k_w:(j+1)*k_w] = sum_ij
            Z = Z[0:m, 0:k]

            # insecure revealing

            ZZ = s.recv()
            Z = Z + ZZ
            return Z

    def insecure_conv(self, role, X=None, K=None):
        if role == 'C':
            s = self.socket

            d, n = X.shape[0], X.shape[1] - 6
            X = X.to('cpu')
            s.sendall(n)

            args = [x for x in range(d)]

            def get_target_poly(channel):
                m_w, n_w, k_w = 1, n + 6, n + 3
                X_t = X[channel, :].unsqueeze(0)
                pix_coeff = [0] * ((m_w - 1) * n_w * k_w + n_w)
                for i in range(m_w):
                    for j in range(n_w):
                        pix_coeff[i * n_w * k_w + (n_w - 1) - j] = X_t[i][j].item()
                
                r = torch.randn((m_w, k_w)).to(torch.double) * (1 << 12)
                r = r.to(torch.int64)
                piz_coeff = [0] * (m_w * n_w * k_w)
                for i in range(m_w):
                    for j in range(k_w):
                        piz_coeff[i * n_w * k_w + (j + 1) * n_w - 1] = r[i][j].item()
                
                return [pix_coeff, piz_coeff, r.tolist()]

            processes = MultiProcessing(target=get_target_poly, args=args, role='c', granularity=32)
            processes.start()
            processes.join()

            pix_piz_r_lst = [x for x in processes.ret_buffer()]
            pix_lst = [x[0] for x in pix_piz_r_lst]
            piz_lst = [x[1] for x in pix_piz_r_lst]
            r_lst = [x[2] for x in pix_piz_r_lst]
            self.x = torch.as_tensor(r_lst).squeeze()
            self.x = self.x[:, 0 : n].t()

            del pix_piz_r_lst
            
            enc_piy_lst = []
            for i in range(d):
                enc_piy_lst.append(s.recv(already_bstr=True))
            
            def get_ans_poly(channel):
                ct = self.ahe_s.context.from_cipher_str(enc_piy_lst[channel])
                self.ahe_s.ahe_mul_plain_inplace(ct, pix_lst[channel], is_list=True)
                self.ahe_s.ahe_add_plain_inplace(ct, piz_lst[channel], is_list=True)
                return ct.to_string()
            
            processes = MultiProcessing(target=get_ans_poly, args=args, role='c', granularity=32)
            processes.start()
            processes.join()
            
            for i in processes.ret_buffer():
                s.sendall(i, already_bstr=True)

            # insecure revealing

            s.sendall(self.x)

        else:
            s = self.socket

            d = K.shape[0]
            K = K.to('cpu')
            n = s.recv()

            args = [x for x in range(d)]

            def get_target_poly(channel):
                f_kernel = torch.zeros((n + 6, n + 3)).to(torch.int64)
                kernel = K[channel].t()
                for j in range(n + 3):
                    f_kernel[j : j + 4, j] = kernel
                return self.ahe_s.enc_list(f_kernel.t().flatten().tolist()).to_string()

            processes = MultiProcessing(target=get_target_poly, args=args, granularity=32)
            processes.start()
            processes.join()

            for i in processes.ret_buffer():
                s.sendall(i, already_bstr=True)

            ct_lst = []
            for i in range(d):
                ct_lst.append(s.recv(already_bstr=True))

            def get_ans_poly(channel):
                m_w, n_w, k_w = 1, n + 6, n + 3
                ct = self.ahe_s.context.from_cipher_str(ct_lst[channel])
                    
                piz_coeff = self.ahe_s.dec_list(ct, length=m_w*n_w*k_w)
                Z = torch.zeros((m_w, k_w)).to(torch.int64)
                for i in range(m_w):
                    for j in range(k_w):
                        Z[i][j] = piz_coeff[i * n_w * k_w + (j + 1) * n_w - 1]
                return Z.tolist()
            
            processes = MultiProcessing(target=get_ans_poly, args=args, granularity=32)
            processes.start()
            processes.join()

            self.x = torch.as_tensor([x for x in processes.ret_buffer()]).squeeze()
            self.x = self.x[:, 0 : n].t()

            # insecure revealing

            x = s.recv()
            x += self.x
            return x.t()
    def insecure_SiLU(self, role, input_array):
        import math

        if role == 'C':
            s = self.socket
            self.ckks_c = CKKS()
            s.sendall(self.ckks_c.pk.to_string())
            s.sendall(self.ckks_c.rks.to_string())

            length = 1
            for i in input_array.shape:
                length *= i
            input_a = input_array.to('cpu')
            input_cipher_C = self.ckks_c.enc_array(input_a.reshape(length).numpy())
            input_cipher_exp_C = self.ckks_c.enc_array(torch.exp(-input_a).reshape(length).numpy())
            s.sendall(input_cipher_C.to_string())
            s.sendall(input_cipher_exp_C.to_string())

            input_cipher_exp_C_str = s.recv()
            input_cipher_C_str = s.recv()
            input_cipher_exp_C = self.ckks_c.context.from_cipher_str(input_cipher_exp_C_str)
            input_cipher_C = self.ckks_c.context.from_cipher_str(input_cipher_C_str)
            
            input_exp_C = self.ckks_c.dec_array(input_cipher_exp_C, length)
            input_exp_C = 1/input_exp_C
            input_C = self.ckks_c.dec_array(input_cipher_C, length)

            SiLU_C_mul = input_C * input_exp_C
            s.sendall(SiLU_C_mul)
            return input_array
        else:
            s = self.socket
            pk = s.recv()
            rks = s.recv()
            self.ckks_s = CKKS(pk_str=pk, rks_str=rks)

            length = 1
            for i in input_array.shape:
                length *= i
            
            input_a_S = input_array.to('cpu')
            input_plain_S = self.ckks_s.encode.encode(input_a_S.reshape(length), self.ckks_s.scale)
            input_plain_exp_S = self.ckks_s.encode.encode(torch.exp(-input_a_S).reshape(length).numpy(),self.ckks_s.scale)

            input_cipher_C_str = s.recv()
            input_cipher_exp_C_str = s.recv()
            input_cipher_C = self.ckks_s.context.from_cipher_str(input_cipher_C_str)
            input_cipher_exp_C = self.ckks_s.context.from_cipher_str(input_cipher_exp_C_str)

            ran2 = np.random.random(length)
            ran2_plain_S = self.ckks_s.encode.encode(ran2, self.ckks_s.scale)
            ran3 = np.random.random(length)#S
            ran3_plain_S = self.ckks_s.encode.encode(ran3, self.ckks_s.scale)
            ones_plain_S = self.ckks_s.encode.encode(np.ones(length),self.ckks_s.scale)
            ones_cipher_S = self.ckks_s.enc_array(np.ones(length))

            self.ckks_s.eval.multiply_plain_inplace(input_cipher_exp_C, input_plain_exp_S)
            self.ckks_s.eval.multiply_plain_inplace(ones_cipher_S, ones_plain_S)
            self.ckks_s.eval.add_inplace(input_cipher_exp_C,ones_cipher_S)
            self.ckks_s.ckks_mul_plain_inplace(input_cipher_exp_C, ran2_plain_S)
            self.ckks_s.eval.rescale_to_next_inplace(input_cipher_exp_C)

            self.ckks_s.ckks_add_plain_inplace(input_cipher_C, input_plain_S)
            self.ckks_s.ckks_mul_plain_inplace(input_cipher_C, ran3_plain_S)

            s.sendall(input_cipher_exp_C.to_string())
            s.sendall(input_cipher_C.to_string())
            ran3 = 1/ran3

            SiLU_S_mul = ran3*ran2
            SiLU_C_mul = s.recv()

            output = SiLU_C_mul*SiLU_S_mul
            output = torch.tensor(output, dtype=input_array.dtype, device=input_array.device)
            return output
    def insecure_Softplus(self, role, input_array):
        if role == 'C':
            s = self.socket
            self.ckks_c = CKKS()
            s.sendall(self.ckks_c.pk.to_string())
            s.sendall(self.ckks_c.rks.to_string())

            length = 1
            for i in input_array.shape:
                length *= i

            input_a_C = input_array.to('cpu')
            input_cipher_exp_C = self.ckks_c.enc_array(torch.exp(input_a_C).reshape(length).numpy())
            s.sendall(input_cipher_exp_C.to_string())

            input_cipher_exp_C_str = s.recv()
            input_cipher_exp_C = self.ckks_c.context.from_cipher_str(input_cipher_exp_C_str)

            input_exp_C = self.ckks_c.dec_array(input_cipher_exp_C, length)
            SoftPlus_C = torch.log(torch.tensor(input_exp_C))

            s.sendall(SoftPlus_C)
            return SoftPlus_C
        else:
            s = self.socket
            pk = s.recv()
            rks = s.recv()
            self.ckks_s = CKKS(pk_str=pk, rks_str=rks)

            length = 1
            for i in input_array.shape:
                length *= i

            input_a_S = input_array.to('cpu')
            input_plain_exp_S = self.ckks_s.encode.encode(torch.exp(input_a_S).reshape(length).numpy(),self.ckks_s.scale)
            input_cipher_exp_C_str = s.recv()
            input_cipher_exp_C = self.ckks_s.context.from_cipher_str(input_cipher_exp_C_str)
            
            ran2 = np.random.random(length)
            ran2_plain_S = self.ckks_s.encode.encode(ran2, self.ckks_s.scale)

            ones_plain_S = self.ckks_s.encode.encode(np.ones(length),self.ckks_s.scale)
            ones_cipher_S = self.ckks_s.enc_array(np.ones(length))

            self.ckks_s.eval.multiply_plain_inplace(input_cipher_exp_C, input_plain_exp_S)
            self.ckks_s.eval.multiply_plain_inplace(ones_cipher_S, ones_plain_S)
            self.ckks_s.eval.add_inplace(input_cipher_exp_C, ones_cipher_S)
            self.ckks_s.ckks_mul_plain_inplace(input_cipher_exp_C, ran2_plain_S)
            self.ckks_s.eval.rescale_to_next_inplace(input_cipher_exp_C)

            s.sendall(input_cipher_exp_C.to_string())
            ran2 = 1/ran2

            Softplus_S = torch.log(torch.tensor(ran2))

            SoftPlus_C = s.recv()

            output = SoftPlus_C + Softplus_S
            return output


protocol = CipherMambaProtocol()