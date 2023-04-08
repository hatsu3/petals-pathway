import queue
import time
import random
import multiprocessing
import socket
import concurrent.futures


# The priority queue that holds incoming requests
request_queue = queue.PriorityQueue()


# The worker function that executes requests in priority order
def execute_requests():
    while True:
        try:
            # Get the next request from the priority queue (blocking call)
            priority, request = request_queue.get()
            print(f"Executing request {request} with priority {priority}")

            # Simulate some processing time for the request
            time.sleep(random.uniform(0.5, 1.5))

            # Mark the request as done (for debugging purposes)
            print(f"Finished request {request}")

            # Notify the priority queue that the request has been processed
            request_queue.task_done()
        except KeyboardInterrupt:
            # Exit the worker thread if the program is interrupted
            break


# The connection handler function that puts incoming requests into the priority queue
def handle_connection(client_socket, client_address):
    # Receive the incoming request from the client
    request = client_socket.recv(1024).decode().strip()

    # Extract the request ID and priority from the request
    sender_pid, request_id, priority = request.split(':')
    request_id = int(request_id.strip())
    priority = int(priority.strip())

    # Add the request to the priority queue with the specified priority
    request_queue.put((priority, request_id))

    # Notify the client that the request has been received
    response = f"Request {request_id} from proc #{sender_pid} received with priority {priority}"
    client_socket.sendall(response.encode())

    # Close the client connection
    client_socket.close()


# The process that sends requests to the server at random intervals
def send_requests(process_id):
    while True:
        # Generate a random delay using Poisson distribution
        delay = random.expovariate(0.5)

        # Sleep for the specified delay
        time.sleep(delay)

        # Connect to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('localhost', 8000))

            # Generate a request ID and priority for the request
            request_id = random.randint(1, 1000)
            priority = random.randint(1, 10)

            # Send the request to the server
            request = f"{process_id}:{request_id}:{priority}"
            s.sendall(request.encode())

            # Receive the response from the server
            response = s.recv(1024)
            print(f"Process {process_id} received response: {response.decode()}")


# The main program that starts the executor thread, the connection handler threads, and the request sender processes
def main():
    # Start the request sender processes
    processes = []
    for i in range(2):
        p = multiprocessing.Process(target=send_requests, args=(i,))
        p.start()
        processes.append(p)

    # Create a thread pool for handling incoming requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Listen for incoming connections from clients
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', 8000))
            s.listen()

            try:
                while True:
                    # Accept an incoming connection from a client
                    client_socket, client_address = s.accept()
                    print(f"Received connection from {client_address}")

                    # Submit the request handling task to the thread pool
                    executor.submit(handle_connection, client_socket, client_address)
            except KeyboardInterrupt:
                # Stop the thread pool and exit the program gracefully
                print("Killing the request sender processes...")
                for p in processes:
                    p.terminate()
                    p.join()
                
                print("Stopping the server...")
                executor.shutdown(wait=False)
                s.close()


if __name__ == '__main__':
    main()
