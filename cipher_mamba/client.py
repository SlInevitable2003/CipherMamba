import socket
import pickle
from transformers import AutoTokenizer

device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

HOST = "127.0.0.1"
PORT = 43210

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    print("Successfully connect to the cipher-mamba server")
    while True:
        prompt = input("Input the prompt here: ")
        tokens = tokenizer(prompt, return_tensors="pt")
        input_ids = tokens.input_ids.to(device=device)
        s.sendall(pickle.dumps(input_ids)) # actually, C should not send input_ids to S

        # secure-sharing computation

        out = pickle.loads(s.recv(65536))
        response = tokenizer.batch_decode(out)
        print("Response: ", response)