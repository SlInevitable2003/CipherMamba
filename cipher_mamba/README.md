## RUN

### for server

excute:

```
export HF_ENDPOINT=https://hf-mirror.com
python cipher_mamba/server.py --topp 0.9 --temperature 0.7 --repetition-penalty 1.2
```

### for client

excute:

```
export HF_ENDPOINT=https://hf-mirror.com
python cipher_mamba/client.py
```