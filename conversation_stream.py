import threading
import subprocess
import queue

def stream_cpp_output(cmd, label, output_queue):
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        for line in process.stdout:
            output_queue.put((label, line.strip()))
        process.stdout.close()
        process.wait()
    except Exception as e:
        output_queue.put(f"Error: {e}")

       
# Initialize a thread-safe queue
output_queue = queue.Queue()

cmd_out = ["./build/bin/Release/whisper-stream.exe", "-m", "./models/ggml-base.en.bin",
        "-t", "6", "--step", "0", "--length", "3000", "-vth", "0.8", "-ac", "0", "-c", "1", "--step", "1000"]
cmd_in = ["./build/bin/Release/whisper-stream.exe", "-m", "./models/ggml-base.en.bin",
        "-t", "6", "--step", "0", "--length", "3000", "-vth", "0.8", "-ac", "0", "--step", "1000"]

# Create and start threads for each subprocess
thread1 = threading.Thread(
    target=stream_cpp_output, args=(cmd_out, "Output", output_queue))
thread2 = threading.Thread(
    target=stream_cpp_output, args=(cmd_in, "Input", output_queue))

thread1.start()
thread2.start()

# Process output in real-time
while thread1.is_alive() or thread2.is_alive() or not output_queue.empty():
    try:
        label, line = output_queue.get()
        # Disable SDL2 init log
        if line[:5] == "init:":
            continue
        print(f"[{label}] {line}")
    except queue.Empty:
        continue

# Ensure all threads have completed
thread1.join()
thread2.join()