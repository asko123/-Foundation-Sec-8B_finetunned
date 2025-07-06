#!/usr/bin/env python3
"""
Memory Monitor - Track GPU memory usage during training
"""

import torch
import time
import psutil
import os
import signal
import sys
from datetime import datetime

class MemoryMonitor:
    def __init__(self, interval=5, log_file="memory_usage.log"):
        self.interval = interval
        self.log_file = log_file
        self.running = True
        
    def log_memory_usage(self):
        """Log current memory usage to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # CPU Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        cpu_memory_used = memory.used / (1024**3)  # GB
        cpu_memory_total = memory.total / (1024**3)  # GB
        cpu_memory_percent = memory.percent
        
        log_entry = f"[{timestamp}] CPU: {cpu_percent}%, RAM: {cpu_memory_used:.2f}/{cpu_memory_total:.2f}GB ({cpu_memory_percent:.1f}%)"
        
        # GPU Memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(i)
                
                log_entry += f", GPU{i} ({gpu_name}): {allocated:.2f}/{total:.2f}GB allocated, {reserved:.2f}GB reserved"
        else:
            log_entry += ", No GPU available"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
        
        # Print to console
        print(log_entry)
        
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\nMemory monitoring stopped.")
        self.running = False
        sys.exit(0)
        
    def start_monitoring(self):
        """Start the memory monitoring loop"""
        print(f"Starting memory monitoring every {self.interval} seconds...")
        print(f"Logging to: {self.log_file}")
        print("Press Ctrl+C to stop monitoring")
        
        # Set up signal handler
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Clear log file
        with open(self.log_file, 'w') as f:
            f.write(f"Memory monitoring started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        while self.running:
            try:
                self.log_memory_usage()
                time.sleep(self.interval)
            except KeyboardInterrupt:
                self.signal_handler(None, None)
            except Exception as e:
                print(f"Error in memory monitoring: {e}")
                time.sleep(self.interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory Monitor for Training")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", default="memory_usage.log", help="Log file path")
    
    args = parser.parse_args()
    
    monitor = MemoryMonitor(interval=args.interval, log_file=args.log_file)
    monitor.start_monitoring()

if __name__ == "__main__":
    main() 