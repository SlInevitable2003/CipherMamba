import argparse
import time
import json

import torch
import torch.nn.functional as F

from einops import rearrange

from transformers import AutoModelForCausalLM

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


parser = argparse.ArgumentParser(description="Generation benchmarking")
parser.add_argument("--model-name", type=str, default="state-spaces/mamba-130m")
# parser.add_argument("--prompt", type=str, default=None)
parser.add_argument("--promptlen", type=int, default=100)
parser.add_argument("--genlen", type=int, default=100)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--topk", type=int, default=1)
parser.add_argument("--topp", type=float, default=1.0)
parser.add_argument("--minp", type=float, default=0.0)
parser.add_argument("--repetition-penalty", type=float, default=1.0)
parser.add_argument("--batch", type=int, default=1)
args = parser.parse_args()

repeats = 3
device = "cuda"
dtype = torch.float16

print(f"Loading model {args.model_name}")
model = MambaLMHeadModel.from_pretrained(args.model_name, device=device, dtype=dtype)
model.eval()
print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

torch.random.manual_seed(0)

import socket

HOST = "127.0.0.1"
PORT = 43233

from cipher_mamba.insecure_sharing.protocols import protocol
from cipher_mamba.insecure_sharing.socket import BetterSocket

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Ready to receive the message now.")
    conn, addr = s.accept()
    with conn:
        connn = BetterSocket(conn)
        protocol.set_socket(s=connn)

        while True:
            input_ids = connn.recv()
            protocol.create_ahe(role="S")
            protocol.synchronize('S', message='ahe s2c')
            # protocol.synchronize('S', message='ahe c2s')

            # inference
            max_length = input_ids.shape[1] + args.genlen

            fn = lambda: model.generate(
                input_ids=input_ids,
                max_length=max_length,
                cg=True,
                return_dict_in_generate=True,
                output_scores=True,
                enable_timing=False,
                temperature=args.temperature,
                top_k=args.topk,
                top_p=args.topp,
                min_p=args.minp,
                repetition_penalty=args.repetition_penalty,
            )
            print("Going to generate...")

            torch.cuda.synchronize()
            start = time.time()
            out = fn()
            torch.cuda.synchronize()

            protocol.synchronize('S', message="break")
            out_sequences = out.sequences
            # out = out.sequences.tolist()
            # connn.sendall(out)

            # torch.cuda.synchronize()
            # start = time.time()
            # for _ in range(repeats):
            #     fn()
            # torch.cuda.synchronize()
            print(f"Prompt length: {len(input_ids[0])}, generation length: {len(out_sequences[0]) - len(input_ids[0])}")
            print(f"{args.model_name} prompt processing + decoding time: {(time.time() - start) * 1000:.0f}ms")