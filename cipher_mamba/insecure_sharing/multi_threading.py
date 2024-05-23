import threading
import math

class MultiThreading:
    threading_pool = []
    ret_buffer = []

    def __init__(self, target, args, granularity = 32, show_process = False):
        n = len(args)
        self.ret_buffer = [None] * n
        assert n > granularity
        
        t = math.ceil(n / granularity)
        if t * granularity - n > 0:
            args += [None] * (t * granularity - n)
        args = [args[j * t : (j + 1) * t] for j in range(granularity)]

        def modify_for_args(args):
            return [[i, args[i]] for i in range(granularity)]
        def modify_for_ret(func):
            def ret_func(thread_id, arg):
                if show_process:
                    print(f'thread {thread_id} working...')
                my_buffer = [None] * t
                for i in range(t):
                    if arg[i] is None:
                        break
                    my_buffer[i] = func(arg[i])
                    if show_process and i % 100 == 0:
                        print(f'thread {thread_id} working on {i}/{t}...')
                for i in range(t):
                    if my_buffer[i] is None:
                        break
                    self.ret_buffer[thread_id * t + i] = my_buffer[i]
                if show_process:
                    print(f'thread {thread_id} finishing...')
            return ret_func
        
        args = modify_for_args(args)
        target = modify_for_ret(target)
        for i in range(granularity):
            self.threading_pool.append(threading.Thread(target=target, args=args[i]))

    def reset(self, target, args):
        for i in range(len(self.threading_pool)):
            self.threading_pool[i] = threading.Thread(target=target, args=args[i])

    def start(self):
        for i in self.threading_pool:
            i.start()

    def run(self):
        for i in self.threading_pool:
            i.run()

    def join(self):
        for i in self.threading_pool:
            i.join()