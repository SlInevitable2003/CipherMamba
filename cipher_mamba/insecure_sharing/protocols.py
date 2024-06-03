import copy
import torch
import torch.nn.functional as F
from insecure_sharing.socket import BetterSocket
from insecure_sharing.iahe import AHE, Polynomial
from cipher_mamba.insecure_sharing.multi_processing import MultiProcessing
from insecure_sharing.seal import *

class CipherOption:
    use_secure_protocol = False
    
    def secure_protocol_set(self):
        self.use_secure_protocol = True

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

    def synchronize(self, role, message=None, input_ids=None, token=None, x=None):
        if role == 'C':
            s = self.socket
            msg = s.recv()
            if msg == 'embedding':
                self.insecure_embedding('C', input_ids=input_ids)
                s.sendall(self.hidden_states)
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
                self.insecure_rmsnorm(role='C')
                return msg, None
            elif msg == 'linear_inProj':
                self.xz = self.insecure_matmul('C', X=self.hidden_states)
                s.sendall(self.xz)
                return msg, None
            elif msg == 'linear_lmHead':
                self.hidden_states = self.hidden_states[-1:]
                self.logits = self.insecure_matmul('C', X=self.hidden_states)
                s.sendall(self.logits)
                return msg, None
            elif msg == 'linear_xProj':
                self.x_dbl = self.insecure_matmul('C', X=self.x)
                s.sendall(self.x_dbl)
                return msg, None
            elif msg == 'linear_dtProj':
                dt_rank = s.recv()
                d_state = s.recv()
                self.dt, self.B, self.C = torch.split(self.x_dbl, [dt_rank, d_state, d_state], dim=-1)
                self.dt = self.insecure_matmul('C', X=self.dt)
                s.sendall(self.dt)
                return msg, None
            elif msg == 'linear_outProj':
                x = s.recv()
                self.hidden_states = self.insecure_matmul('C', X=x)
                s.sendall(self.hidden_states)
                return msg, None
            elif msg == 'conv':
                self.x, self.z = self.xz.chunk(2, dim=1)
                self.insecure_conv('C', X=F.pad(self.x.t(), (3, 3)))
                s.sendall(self.x)
                self.x = s.recv()
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
            elif message in ['linear_outProj']: # to be remove
                s.sendall(x)
            elif message == 'linear_dtProj':
                s.sendall(x[0])
                s.sendall(x[1])
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
                line = self.ahe_s.context.from_cipher_str(self.Enc_Ws[j].to_string())
                rdm = r[i].tolist()
                line = self.ahe_s.ahe_add_plain(line, rdm, is_list=True)
                Enc_ws.append(line)
            
            s.sendall(n)
            for i in Enc_ws:
                s.sendall(i.to_string(), already_bstr=True)
            self.hidden_states = (-1) * r
            self.residual = torch.zeros_like(self.hidden_states).to(torch.int32)

            self.embedding_first_time = False
            return None
        else:
            s = self.socket
            k, m = W.shape[0], W.shape[1]
            s.sendall(k)
            s.sendall(m)
            
            # print('')
            if self.embedding_first_time == True:
                lines = [W[i].tolist() for i in range(k)]
                target = lambda lst : protocol.ahe_s.enc_list(lst).to_string()

                processes = MultiProcessing(target=target, args=lines, granularity=32)
                processes.start()
                processes.join()

                line_idx = 0
                for i in processes.ret_buffer():
                    # print(f'\r[{line_idx}] going to send...', end='')
                    s.sendall(i, already_bstr=True)
                    line_idx += 1

            n = s.recv()
            w_plus_r = torch.zeros((n, m)).to(torch.int32)
            for i in range(n):
                line = self.ahe_s.context.from_cipher_str(s.recv(already_bstr=True))
                line_lst = self.ahe_s.dec_list(line, length=m)
                w_plus_r[i] = torch.as_tensor(line_lst)
            self.hidden_states = w_plus_r
            self.residual = torch.zeros_like(self.hidden_states).to(torch.int32)

            self.embedding_first_time = False
            return self.hidden_states

    def insecure_rmsnorm(self, role, w=None, eps=None):
        if role == 'C':
            s = self.socket
            s.sendall(self.residual)
            self.hidden_states = s.recv()

        else:
            import copy
            s = self.socket
            self.residual = self.residual + self.hidden_states
            residual = s.recv()
            residual += self.residual
            
            m = residual.shape[1]
            assert m == 768

            org_res = copy.deepcopy(residual)
            real_res = copy.deepcopy(org_res)
            residual = torch.sum(residual * residual, dim=1) >> 12
            residual = residual.to(torch.double) / (m << 12)
            residual += eps
            residual = residual ** (-0.5)
            residual = residual * (1 << 12)
            residual = residual.to(torch.int64)
            
            org_res = (org_res * residual.unsqueeze(0).t()) >> 12
            org_res = (org_res * w) >> 12

            self.hidden_states = org_res >> 1
            s.sendall(org_res - self.hidden_states)

            return org_res, real_res

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
            # print(f'C goint to do matmul with X = {X}')

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

            Z = torch.zeros((m_p * m_w, k_p * k_w)).to(torch.int64)
            for i in range(m_p):
                for j in range(k_p):
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        nxt = next(ans_buffer)
                        r = (-1) * (torch.as_tensor(nxt[1]) >> 12)
                        sum_ij += r
                    Z[i*m_w:(i+1)*m_w, j*k_w:(j+1)*k_w] = sum_ij
            Z = Z[0:m, 0:k]

            return Z

        else:
            s = self.socket
            # print(f'S goint to do matmul with Y = {Y}')

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
            processes = MultiProcessing(target=target, args=args, granularity=32)
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

            # print('')
            Z = torch.zeros((m_p * m_w, k_p * k_w)).to(torch.int64)
            for i in range(m_p):
                for j in range(k_p):
                    # print(f'\rmatmul process: {int((i * k_p + j) * 100 / (m_p * k_p))}%', end='')
                    sum_ij = torch.zeros((m_w, k_w))
                    for l in range(n_p):
                        Z_l = next(ans_buffer)
                        Z_l = torch.as_tensor(Z_l)
                        sum_ij += (Z_l >> 12)
                    Z[i*m_w:(i+1)*m_w, j*k_w:(j+1)*k_w] = sum_ij
            Z = Z[0:m, 0:k]

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
                        piz_coeff[i * n_w * k_w + (j + 1) * n_w - 1] = (-1) * r[i][j].item()
                
                return [pix_coeff, piz_coeff, r.tolist()]

            processes = MultiProcessing(target=get_target_poly, args=args, role='c', granularity=32)
            processes.start()
            processes.join()

            pix_piz_r_lst = [x for x in processes.ret_buffer()]
            pix_lst = [x[0] for x in pix_piz_r_lst]
            piz_lst = [x[1] for x in pix_piz_r_lst]
            r_lst = [x[2] for x in pix_piz_r_lst]
            self.x = torch.as_tensor(r_lst).squeeze()
            self.x = self.x[:, 0 : n].t() >> 12

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
            self.x = self.x[:, 0 : n].t() >> 12

            return self.x

protocol = CipherMambaProtocol()