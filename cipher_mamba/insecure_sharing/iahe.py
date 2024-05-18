from insecure_sharing.seal import *
import torch

class Polynomial:
    def __init__(self, deg, coeff, mod_bits):
        self.deg = deg
        self.coeff = coeff
        self.mod_bits = mod_bits

    def to_string(self):
        ret = ""
        first_item = True
        for i in range(self.deg, -1, -1):
            cf = self.coeff[i]
            if first_item:
                first_item = False
            else:
                ret += " + "
            if cf < 0:
                cf = (1 << self.mod_bits) + cf
            ret += hex(cf)[2:] + 'x^' + str(i)
        return ret
    
    def from_string(s, mod_bits):
        def from_hex_bit(c):
            if c in '0123456789':
                return int(c)
            ret = 10
            for i in 'ABCDEF':
                if i == c:
                    return ret
                ret += 1

        ret = Polynomial(0, [], mod_bits=mod_bits)

        if s == "":
            return ret
        stack = {}
        maxidx = 0
        ss = s.split(' + ')
        for item in ss:
            tmp_lst = item.split('x^')
            coeff, idx = tmp_lst[0], (int(tmp_lst[1]) if len(tmp_lst) > 1 else 0)
            cval = 0
            for c in coeff:
                cval = cval * 16 + from_hex_bit(c)
            if cval >= (1 << (ret.mod_bits - 1)):
                cval = - ((1 << ret.mod_bits) - cval)
            stack[idx] = cval
            maxidx = max(maxidx, idx)

        ret.deg = maxidx
        ret.coeff = [0] * (ret.deg + 1)
        for i, c in stack.items():
            ret.coeff[i] = c
        return ret
        

class AHE:
    def __init__(self, poly_modulus_degree = 4096, plain_modulus_bits = 37, pk_str=None, rks_str=None):
        self.poly_mod_deg = poly_modulus_degree
        self.plain_mod_bits = plain_modulus_bits

        parms = EncryptionParameters(scheme_type.bfv)
        parms.set_poly_modulus_degree(poly_modulus_degree)
        parms.set_coeff_modulus(CoeffModulus.BFVDefault(poly_modulus_degree))
        parms.set_plain_modulus(1 << plain_modulus_bits)

        self.context = SEALContext(parms)
        
        if pk_str == None:
            gen = KeyGenerator(self.context)
            self.sk = gen.secret_key()
            self.pk = gen.create_public_key()
            self.rks = gen.create_relin_keys()
            self.dec = Decryptor(self.context, self.sk)
        else:
            self.pk = self.context.from_public_str(pk_str)
            self.rks = self.context.from_relin_str(rks_str)

        self.enc = Encryptor(self.context, self.pk)
        self.eval = Evaluator(self.context)
        

    def enc_list(self, lst):
        p = Polynomial(deg=len(lst)-1, coeff=lst, mod_bits=self.plain_mod_bits)
        m = Plaintext(p.to_string())
        c = self.enc.encrypt(m)
        return c
    
    def dec_list(self, c, length = 0):
        m = self.dec.decrypt(c)
        p = Polynomial.from_string(m.to_string(), mod_bits=self.plain_mod_bits)
        lst = p.coeff
        if len(lst) < length:
            lst = lst + [0] * (length - len(lst))
        return lst
    
    def ahe_add_plain(self, c, p, is_list = False):
        if is_list:
            p = Polynomial(deg=len(p)-1, coeff=p, mod_bits=self.plain_mod_bits)
            m = Plaintext(p.to_string())
        else:
            m = p
        c2 = self.enc.encrypt(m)
        ret = self.eval.add(c, c2)
        self.eval.relinearize_inplace(ret, self.rks)
        return ret


def Enc(x):
    return x

def Dec(x):
    return x

def AHE_add(x, y):
    return x + y