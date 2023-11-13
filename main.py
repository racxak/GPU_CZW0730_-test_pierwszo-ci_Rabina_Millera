import numpy as np
import time

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

# Function to test primes sequentially
def sequential_prime_test(numbers):
    results = []
    for number in numbers:
        result = is_prime(number)
        results.append(result)
        print(f"{number} is {'prime' if result else 'not prime'}")
    return results

# List of numbers to test for primality
numbers = [999331, 1111111111111111111, 982451653]

# Start the timer
start_time = time.time()

# Test the numbers for primality sequentially
results = sequential_prime_test(numbers)

# End the timer
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
