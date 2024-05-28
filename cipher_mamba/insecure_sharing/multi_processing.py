import multiprocessing
import pickle
import math

class MultiProcessing:
    process_pool = []
    ret_buffer = []

    def __init__(self, target, args, granularity = 32, show_process = False, ):
        n = len(args)
        assert n > granularity

        self.buffer_prefix = './cipher_mamba/insecure_sharing/socket_buffer/buffer'
        
        t = math.ceil(n / granularity)
        if t * granularity - n > 0:
            args += [None] * (t * granularity - n)
        args = [args[j * t : (j + 1) * t] for j in range(granularity)]

        def modify_for_args(args):
            return [[i, args[i]] for i in range(granularity)]
        def modify_for_ret(func):
            def ret_func(process_id, arg):
                if show_process:
                    print(f'process {process_id} working...')
                my_buffer = [None] * t
                for i in range(t):
                    if arg[i] is None:
                        break
                    my_buffer[i] = func(arg[i])
                    if show_process and i % 100 == 0:
                        print(f'process {process_id} working on {i}/{t}...')

                path = self.buffer_prefix + str(process_id) + '.pickle'
                with open(path, 'wb') as f:
                    pickle.dump(my_buffer, f)

                if show_process:
                    print(f'process {process_id} finishing...')
            return ret_func
        
        args = modify_for_args(args)
        target = modify_for_ret(target)
        for i in range(granularity):
            self.process_pool.append(multiprocessing.Process(target=target, args=args[i]))

    def reset(self, target, args):
        for i in range(len(self.process_pool)):
            self.process_pool[i] = multiprocessing.Process(target=target, args=args[i])

    def start(self):
        for i in self.process_pool:
            i.start()

    def run(self):
        for i in self.process_pool:
            i.run()

    def join(self):
        for i in self.process_pool:
            i.join()
        for i in range(len(self.process_pool)):
            path = self.buffer_prefix + str(i) + '.pickle'
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            for j in obj:
                if j is None:
                    break
                self.ret_buffer.append(j)
        
    def terminate(self):
        for i in self.process_pool:
            i.terminate()