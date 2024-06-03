## RUN

### for server

excute:

```
export HF_ENDPOINT=https://hf-mirror.com
python cipher_mamba/server.py --topp 0.9 --temperature 0.7 --repetition-penalty 1.2 > out.txt
```

### for client

excute:

```
export HF_ENDPOINT=https://hf-mirror.com
export TOKENIZERS_PARALLELISM=true
python cipher_mamba/client.py < in.txt
```

### for test

try this prompt:

```
My cat wrote all this CUDA code for a new language model and
```