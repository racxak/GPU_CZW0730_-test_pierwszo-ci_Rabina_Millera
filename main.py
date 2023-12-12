import numpy as np
import time
from multiprocessing import Pool
import pyopencl as cl
import numpy as np

# Ustalenie kontekstu OpenCL i kolejki
platforms = cl.get_platforms()
devices = platforms[0].get_devices()
context = cl.Context(devices)
queue = cl.CommandQueue(context)

# Przygotowanie kodu kernela OpenCL
kernel_code = """
long pow_mod(long base, long exponent, long modulus) {
    long result = 1;
    base = base % modulus;

    while (exponent > 0) {
        if (exponent % 2 == 1)
            result = (result * base) % modulus;

        exponent = exponent >> 1;
        base = (base * base) % modulus;
    }

    return result;
}

__kernel void is_prime_gpu(__global const long* numbers, __global char* results, const unsigned int k, __global const long* random_numbers) {
    int idx = get_global_id(0);
    long n = numbers[idx];
    if (n == 2 || n == 3) {
        results[idx] = 1;
        return;
    }
    if (n % 2 == 0 || n < 2) {
        results[idx] = 0;
        return;
    }

    long s = 0;
    long d = n - 1;
    while (d % 2 == 0) {
        s += 1;
        d /= 2;
    }

    for (int i = 0; i < k; ++i) {
        long a = random_numbers[i]; 
        long x = pow_mod(a, d, n);
        if (x == 1 || x == n - 1)
            continue;

        int should_continue = 0;
        for (long r = 0; r < s; ++r) {
            x = pow_mod(x, 2, n);
            if (x == n - 1) {
                should_continue = 1;
                break;
            }
        }

        if (!should_continue) {
            results[idx] = 0;
            return;
        }
    }

    results[idx] = 1;
}
"""


# Function to check if n is prime, used for the Rabin Miller test
def is_prime(n, k=7):  # number of tests = k
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n < 2:
        return False

    s, d = 0, n - 1
    while d % 2 == 0:
        s += 1
        d //= 2

    for _ in range(k):
        a = np.random.randint(2, n-1, dtype=np.int64).item()
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for r in range(s):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

# Function to parallelize with multiple processes
def parallel_prime_test(n):
    return is_prime(n)

# Function to test primes without multiprocessing
def sequential_prime_test(numbers):
    return [is_prime(n) for n in numbers]

if __name__ == '__main__':
    # List of numbers to test for primality
    numbers = [999331, 1111111111111111111, 982451653]

    # Using multiprocessing
    start_time_mp = time.time()
    with Pool(processes=None) as pool:
        results_mp = pool.map(parallel_prime_test, numbers)
    end_time_mp = time.time()

    # Sequential processing
    start_time_seq = time.time()
    results_seq = sequential_prime_test(numbers)
    end_time_seq = time.time()

    # Output the results and time taken for multiprocessing
    print("Multiprocessing results:")
    for number, result in zip(numbers, results_mp):
        print(f"{number} is {'prime' if result else 'not prime'}")
    print(f"Time taken with multiprocessing: {end_time_mp - start_time_mp} seconds")

    # Output the results and time taken for sequential processing
    print("\nSequential results:")
    for number, result in zip(numbers, results_seq):
        print(f"{number} is {'prime' if result else 'not prime'}")
    print(f"Time taken with sequential processing: {end_time_seq - start_time_seq} seconds")

    # Kompilacja kernela
    program = cl.Program(context, kernel_code).build()

    # Przygotowanie danych
    numbers = np.array([999331, 1111111111111111111, 982451653], dtype=np.int64)
    results = np.zeros_like(numbers, dtype=np.int8)
    k = 7

    # Generowanie liczb losowych dla testu Rabin-Miller
    np.random.seed(0)  # Dla powtarzalności wyników, w praktyce użyj np.random.default_rng()
    random_numbers = np.random.randint(2, numbers.max(), size=(len(numbers) * k,), dtype=np.int64)

    # Utworzenie buforów pamięci
    numbers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=numbers)
    results_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, results.nbytes)
    random_numbers_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=random_numbers)

    # Uruchomienie kernela i mierzenie czasu
    start_time_gpu = time.time()
    kernel_event = program.is_prime_gpu(queue, (len(numbers),), None, numbers_buf, results_buf, np.uint32(k),
                                        random_numbers_buf)
    kernel_event.wait()  # Czeka na zakończenie wykonania kernela
    end_time_gpu = time.time()

    # Odczytanie wyników
    cl.enqueue_copy(queue, results, results_buf)
    queue.finish()  # Czekaj na zakończenie wszystkich operacji w kolejce

    # Wypisanie wyników i czasu wykonania na GPU
    print("\nGPU results:")
    for number, result in zip(numbers, results):
        print(f"{number} is {'prime' if result else 'not prime'}")
    print(f"Time taken with GPU: {end_time_gpu - start_time_gpu} seconds")