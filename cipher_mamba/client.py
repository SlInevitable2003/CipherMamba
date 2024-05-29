import socket
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cipher_mamba.insecure_sharing.protocols import protocol
from cipher_mamba.insecure_sharing.socket import BetterSocket

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

HOST = "127.0.0.1"
PORT = 43267

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    ss = BetterSocket(s, fast_identity='c')
    protocol.set_socket(s=ss)
    print("Successfully connect to the cipher-mamba server")

    while True:
        prompt = input("Input the prompt here: ")
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        ss.sendall(input_ids) # actually, C should not send input_ids to S

        print("Response:")

        msg, ret = protocol.synchronize('C', input_ids=input_ids)
        while msg != 'break':
            if msg == 'onemore':
                input_ids = ret
                res = tokenizer.batch_decode(input_ids)
                print(res[0], end="")
            msg, ret = protocol.synchronize('C', input_ids=input_ids)

        print("")