import socket
from transformers import AutoTokenizer

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cipher_mamba.insecure_sharing.models import protocol
from cipher_mamba.insecure_sharing.socket import BetterSocket

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

HOST = "127.0.0.1"
PORT = 43222

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    ss = BetterSocket(s)
    protocol.set_socket(s=ss, role="C")
    print("Successfully connect to the cipher-mamba server")
    while True:
        prompt = input("Input the prompt here: ")
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        ss.sendall(input_ids) # actually, C should not send input_ids to S

        msg, ret = protocol.synchronize('C', input_ids=input_ids)
        while msg != 'break':
            if msg == 'onemore':
                input_ids = ret
            msg, ret = protocol.synchronize('C', input_ids=input_ids)

        # secure-sharing computation

        out = ss.recv()
        response = tokenizer.batch_decode(out)
        print("Response: ", response)