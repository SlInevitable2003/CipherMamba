import os

base_dir = os.path.dirname(__file__)

files = [os.path.join(base_dir, file) for file in os.listdir(base_dir) if file != 'clear_buffer.py']

for file in files:
    with open(file, 'wb'): pass

print('Done.')