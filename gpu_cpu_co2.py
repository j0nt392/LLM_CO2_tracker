"""
Carbon Footprint Tracker for AI Models

This script monitors and visualizes the environmental impact of running AI models,
specifically tracking CPU, GPU, and RAM usage along with their associated energy
consumption and CO2 emissions. It provides real-time graphs and measurements
through a GUI interface.

Key Features:
- Hardware detection for CPU, GPU, and RAM
- Real-time resource monitoring
- Energy consumption calculation
- CO2 emissions tracking
- Interactive GUI with live graphs
- Integration with Ollama LLM for testing

Dependencies:
- PySide6: GUI framework
- psutil: System monitoring
- pyqtgraph: Real-time plotting
- langchain_community: LLM integration
"""

import psutil
import platform
import subprocess
import re
from dataclasses import dataclass
import time
from typing import Optional
from langchain_community.llms import Ollama
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel
from PySide6.QtCore import QTimer
import pyqtgraph as pg
import sys

@dataclass
class SystemSpecs:
    """
    Dataclass to store system hardware specifications.
    
    Attributes:
        cpu_model (str): CPU model name
        cpu_tdp (Optional[float]): CPU Thermal Design Power in watts
        gpu_model (str): GPU model name
        gpu_type (str): GPU type ('nvidia', 'amd', or 'intel_integrated')
        gpu_tdp (Optional[float]): GPU Thermal Design Power in watts
        ram_total_gb (float): Total RAM in gigabytes
    """
    cpu_model: str
    cpu_tdp: Optional[float]
    gpu_model: str
    gpu_type: str
    gpu_tdp: Optional[float]
    ram_total_gb: float

class HardwareDetector:
    """
    Utility class for detecting and gathering system hardware information.
    Supports Windows, Linux, and macOS platforms.
    """
    
    @staticmethod
    def get_cpu_info() -> tuple[str, Optional[float]]:
        """
        Detect CPU model and TDP.
        
        Returns:
            tuple: (CPU model name, TDP in watts if available)
        """
        if platform.system() == "Windows":
            try:
                import wmi
                w = wmi.WMI()
                cpu = w.Win32_Processor()[0]
                return cpu.Name, None
            except:
                return platform.processor(), None
        
        elif platform.system() == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    info = f.read()
                model = re.search("model name.*: (.*)", info).group(1)
                return model, None
            except:
                return platform.processor(), None
        
        elif platform.system() == "Darwin":
            try:
                cmd = "sysctl -n machdep.cpu.brand_string"
                model = subprocess.check_output(cmd.split()).decode().strip()
                return model, None
            except:
                return platform.processor(), None
        
        return platform.processor(), None

    @staticmethod
    def get_gpu_info() -> tuple[str, str, Optional[float]]:
        """
        Detect GPU model, type, and TDP.
        
        Returns:
            tuple: (GPU model name, GPU type, TDP in watts if available)
        """
        # Check for NVIDIA GPU
        try:
            nvidia_smi = subprocess.check_output(["nvidia-smi", "-L"]).decode()
            model = re.search("GPU 0: (.*?) \(", nvidia_smi).group(1)
            power_info = subprocess.check_output([
                'nvidia-smi', 
                '--query-gpu=power.default_limit', 
                '--format=csv,noheader,nounits'
            ]).decode()
            tdp = float(power_info.strip())
            return model, "nvidia", tdp
        except:
            pass

        # Check for Intel integrated GPU
        if platform.system() == "Windows":
            try:
                import wmi
                w = wmi.WMI()
                gpu = w.Win32_VideoController()[0]
                if "Intel" in gpu.Name:
                    return gpu.Name, "intel_integrated", 15
            except:
                pass
        
        # Check for AMD GPU
        try:
            if platform.system() == "Linux":
                with open("/sys/class/drm/card0/device/vendor", "r") as f:
                    vendor_id = f.read().strip()
                    if vendor_id == "0x1002":  # AMD vendor ID
                        return "AMD GPU", "amd", None
        except:
            pass

        return "Integrated Graphics", "intel_integrated", 15

    @staticmethod
    def get_system_specs():
        cpu_model, cpu_tdp = HardwareDetector.get_cpu_info()
        gpu_model, gpu_type, gpu_tdp = HardwareDetector.get_gpu_info()
        ram_total_gb = psutil.virtual_memory().total / (1024**3)

        # Estimate TDP based on CPU model if not detected
        if not cpu_tdp:
            if any(x in cpu_model.lower() for x in ['i9', 'ryzen 9']):
                cpu_tdp = 125
            elif any(x in cpu_model.lower() for x in ['i7', 'ryzen 7']):
                cpu_tdp = 95
            elif any(x in cpu_model.lower() for x in ['i5', 'ryzen 5']):
                cpu_tdp = 65
            else:
                cpu_tdp = 45  # Conservative estimate

        return SystemSpecs(
            cpu_model=cpu_model,
            cpu_tdp=cpu_tdp,
            gpu_model=gpu_model,
            gpu_type=gpu_type,
            gpu_tdp=gpu_tdp,
            ram_total_gb=ram_total_gb
        )

    @staticmethod
    def get_intel_gpu_usage():
        """Get Intel GPU utilization percentage."""
        try:
            if platform.system() == "Linux":
                # Using intel_gpu_top if available
                cmd = "intel_gpu_top -J"
                output = subprocess.check_output(cmd.split(), timeout=1).decode()
                # Parse JSON output for GPU utilization
                import json
                data = json.loads(output)
                return float(data['engines']['render']['busy'])
            elif platform.system() == "Windows":
                # Using Windows Performance Counters
                import wmi
                w = wmi.WMI(namespace="root\\Intel\\Power")
                gpu_data = w.Intel_GPU()[0]
                return float(gpu_data.GPUUtilization)
        except:
            # If we can't get actual GPU usage, estimate based on CPU usage
            # since integrated GPUs share resources with CPU
            return psutil.cpu_percent() * 0.3  # Rough estimate
        return 0

class CarbonTracker:
    def __init__(self, carbon_intensity: float = 0.385):
        self.specs = HardwareDetector.get_system_specs()
        self.process = psutil.Process()
        self.carbon_intensity = carbon_intensity
        self.cooling_overhead = 1.1  # 10% cooling overhead
        self.ram_power_per_gb = 0.3  # Watts per GB
        self.get_gpu_usage = (
            HardwareDetector.get_intel_gpu_usage 
            if self.specs.gpu_type == "intel_integrated" 
            else lambda: 0
        )
        
        print("\n=== System Specifications ===")
        print(f"CPU: {self.specs.cpu_model} (TDP: {self.specs.cpu_tdp}W)")
        print(f"GPU: {self.specs.gpu_model} (Type: {self.specs.gpu_type}, TDP: {self.specs.gpu_tdp}W)")
        print(f"RAM: {self.specs.ram_total_gb:.1f} GB")
        print("===========================\n")

    def measure(self, func):
        def wrapper(*args, **kwargs):
            # Start measurements
            start_time = time.time()
            start_cpu = psutil.cpu_percent(interval=None)
            start_ram = self.process.memory_info().rss / (1024**3)
            start_gpu = self.get_gpu_usage()
            
            result = func(*args, **kwargs)
            
            # End measurements
            end_time = time.time()
            end_cpu = psutil.cpu_percent(interval=None)
            end_ram = self.process.memory_info().rss / (1024**3)
            end_gpu = self.get_gpu_usage()
            
            # Calculate metrics
            duration = end_time - start_time
            avg_cpu_percent = (end_cpu + start_cpu) / 2
            avg_ram_gb = (end_ram + start_ram) / 2
            avg_gpu_percent = (end_gpu + start_gpu) / 2
            
            # Energy calculations (Wh)
            cpu_energy = (self.specs.cpu_tdp * (avg_cpu_percent/100) * duration) / 3600
            ram_energy = (self.ram_power_per_gb * avg_ram_gb * duration) / 3600
            
            # GPU energy calculation based on type
            if self.specs.gpu_type == "intel_integrated":
                # Intel integrated GPU typically uses 10-15W under load
                gpu_energy = (15 * (avg_gpu_percent/100) * duration) / 3600
            elif self.specs.gpu_type == "nvidia" and self.specs.gpu_tdp:
                gpu_energy = (self.specs.gpu_tdp * (avg_gpu_percent/100) * duration) / 3600
            else:
                gpu_energy = 0
            
            # Total energy including cooling
            total_energy_kwh = (cpu_energy + ram_energy + gpu_energy) * self.cooling_overhead / 1000
            
            # Calculate CO2
            co2_emissions = total_energy_kwh * self.carbon_intensity
            
            print("\n=== Environmental Impact ===")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Average CPU: {avg_cpu_percent:.1f}%")
            print(f"Average GPU: {avg_gpu_percent:.1f}%")
            print(f"RAM Usage: {avg_ram_gb:.2f} GB")
            
            print("\n=== Energy & Emissions ===")
            print(f"CPU Energy: {cpu_energy*1000:.2f} Wh")
            print(f"GPU Energy: {gpu_energy*1000:.2f} Wh")
            print(f"RAM Energy: {ram_energy*1000:.2f} Wh")
            print(f"Total Energy: {total_energy_kwh*1000:.2f} Wh")
            print(f"CO2 Emissions: {co2_emissions*1000:.2f} gCO2")
            print("=========================\n")
            
            return result
        return wrapper

class CarbonTrackerGUI(QMainWindow):
    def __init__(self, tracker):
        super().__init__()
        self.setWindowTitle("AI Carbon Tracker")
        self.setMinimumSize(800, 600)
        
        self.tracker = tracker
        self.llm = Ollama(model="llama3.2:1b")
        
        # Setup data storage
        self.timestamps = []
        self.cpu_data = []
        self.gpu_data = []
        self.ram_data = []
        self.co2_data = []
        # Add energy data lists
        self.cpu_energy_data = []
        self.gpu_energy_data = []
        self.ram_energy_data = []
        
        self.setup_ui()
        
        # Update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_graphs)
        self.timer.start(1000)  # Update every second
        
        self.start_time = time.time()

    def setup_ui(self):
        """
        Sets up the GUI interface with three main graphs:
        1. Resource Usage (Red=CPU, Green=GPU, Blue=RAM)
        2. Energy Consumption (Red=CPU, Green=GPU, Blue=RAM)
        3. Cumulative CO2 Emissions (Yellow)
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # System info
        specs = self.tracker.specs
        info_label = QLabel(
            f"=== System Specifications ===\n"
            f"CPU: {specs.cpu_model} (TDP: {specs.cpu_tdp}W)\n"
            f"GPU: {specs.gpu_model} (Type: {specs.gpu_type}, TDP: {specs.gpu_tdp}W)\n"
            f"RAM: {specs.ram_total_gb:.1f} GB"
        )
        layout.addWidget(info_label)
        
        # Usage Graph
        self.usage_plot = pg.PlotWidget(title="Resource Usage")
        self.usage_plot.setLabel('left', 'Usage (%)')
        self.usage_plot.setLabel('bottom', 'Time (s)')
        self.usage_plot.addLegend()
        # Red for CPU, Green for GPU, Blue for RAM
        self.cpu_curve = self.usage_plot.plot(pen='r', name='CPU Usage')
        self.gpu_curve = self.usage_plot.plot(pen='g', name='GPU Usage')
        self.ram_curve = self.usage_plot.plot(pen='b', name='RAM Usage')
        layout.addWidget(self.usage_plot)
        
        # Energy Graph
        self.energy_plot = pg.PlotWidget(title="Energy Consumption")
        self.energy_plot.setLabel('left', 'Energy (Wh)')
        self.energy_plot.setLabel('bottom', 'Time (s)')
        self.energy_plot.addLegend()
        # Same color scheme as usage graph
        self.cpu_energy_curve = self.energy_plot.plot(pen='r', name='CPU Energy')
        self.gpu_energy_curve = self.energy_plot.plot(pen='g', name='GPU Energy')
        self.ram_energy_curve = self.energy_plot.plot(pen='b', name='RAM Energy')
        layout.addWidget(self.energy_plot)
        
        # CO2 Graph
        self.co2_plot = pg.PlotWidget(title="Cumulative CO2 Emissions")
        self.co2_plot.setLabel('left', 'CO2 (gCO2)')
        self.co2_plot.setLabel('bottom', 'Time (s)')
        self.co2_curve = self.co2_plot.plot(pen='y', name='Total CO2')
        layout.addWidget(self.co2_plot)
        
        # Test button
        self.test_button = QPushButton("Run LLM Test")
        self.test_button.clicked.connect(self.run_llm_test)
        layout.addWidget(self.test_button)
        
        # Current values label
        self.current_values = QLabel("Current Values:\nCPU: ---%\nGPU: ---%\nRAM: ---%\nEnergy: --- Wh\nCO2: --- gCO2")
        layout.addWidget(self.current_values)

    def update_graphs(self):
        current_time = time.time() - self.start_time
        
        # Get current values
        cpu_percent = psutil.cpu_percent()
        gpu_percent = self.tracker.get_gpu_usage()
        ram_percent = psutil.virtual_memory().percent
        
        # Calculate energy for the last second
        duration = 1.0
        cpu_energy = (self.tracker.specs.cpu_tdp * (cpu_percent/100) * duration) / 3600
        ram_energy = (self.tracker.ram_power_per_gb * self.tracker.specs.ram_total_gb * (ram_percent/100) * duration) / 3600
        
        # GPU energy calculation based on type
        if self.tracker.specs.gpu_type == "intel_integrated":
            gpu_energy = (15 * (gpu_percent/100) * duration) / 3600
        elif self.tracker.specs.gpu_type == "nvidia" and self.tracker.specs.gpu_tdp:
            gpu_energy = (self.tracker.specs.gpu_tdp * (gpu_percent/100) * duration) / 3600
        else:
            gpu_energy = 0
        
        # Total energy including cooling
        total_energy = (cpu_energy + ram_energy + gpu_energy) * self.tracker.cooling_overhead
        co2 = total_energy * self.tracker.carbon_intensity
        
        # Update data lists
        self.timestamps.append(current_time)
        self.cpu_data.append(cpu_percent)
        self.gpu_data.append(gpu_percent)
        self.ram_data.append(ram_percent)
        
        # Update energy data lists
        self.cpu_energy_data.append(cpu_energy * 1000)  # Convert to Wh
        self.gpu_energy_data.append(gpu_energy * 1000)
        self.ram_energy_data.append(ram_energy * 1000)
        
        self.co2_data.append(sum(self.co2_data[-1:] or [0]) + co2 * 1000)  # Cumulative CO2
        
        # Update graphs
        self.cpu_curve.setData(self.timestamps, self.cpu_data)
        self.gpu_curve.setData(self.timestamps, self.gpu_data)
        self.ram_curve.setData(self.timestamps, self.ram_data)
        
        # Update energy curves with full history
        self.cpu_energy_curve.setData(self.timestamps, self.cpu_energy_data)
        self.gpu_energy_curve.setData(self.timestamps, self.gpu_energy_data)
        self.ram_energy_curve.setData(self.timestamps, self.ram_energy_data)
        
        self.co2_curve.setData(self.timestamps, self.co2_data)
        
        # Update current values label
        self.current_values.setText(
            f"Current Values:\n"
            f"CPU: {cpu_percent:.1f}%\n"
            f"GPU: {gpu_percent:.1f}%\n"
            f"RAM: {ram_percent:.1f}%\n"
            f"Energy: {total_energy*1000:.2f} Wh\n"
            f"Total CO2: {self.co2_data[-1]:.2f} gCO2"
        )

    def run_llm_test(self):
        """
        Runs a test query to the Ollama LLM and measures its environmental impact.
        Includes error handling and connection verification.
        """
        @self.tracker.measure
        def run_test():
            try:
                print("\n=== Starting LLM Test ===")
                print("Checking Ollama connection...")
                
                # Test if Ollama server is running
                import requests
                try:
                    requests.get("http://localhost:11434")
                    print("✓ Ollama server is running")
                except requests.exceptions.ConnectionError:
                    print("✗ Error: Ollama server is not running!")
                    print("Please start Ollama server first.")
                    print("Run 'ollama serve' in a terminal")
                    return None

                # Verify model is available
                print(f"Checking if model 'llama3.2' is available...")
                try:
                    response = self.llm.invoke(
                        "Hello",
                        temperature=0,
                        max_tokens=10
                    )
                    print("✓ Model check successful")
                except Exception as e:
                    print("✗ Error: Model not available!")
                    print("Try running: 'ollama pull llama2'")
                    print(f"Error details: {str(e)}")
                    return None

                # Send actual query
                print("\nSending request to Ollama...")
                response = self.llm.invoke(
                    "What is the environmental impact of AI? (Please keep the answer short)",
                    temperature=0.7,
                    max_tokens=100
                )
                print("\nLLM Response:", response)
                return response

            except Exception as e:
                print(f"\n✗ Error running LLM test: {str(e)}")
                print("\nTroubleshooting steps:")
                print("1. Make sure Ollama is installed")
                print("2. Run 'ollama serve' in a terminal")
                print("3. Run 'ollama pull llama2'")
                print("4. Check if other Ollama commands work")
                return None

        run_test()

def main():
    app = QApplication(sys.argv)
    
    tracker = CarbonTracker()
    window = CarbonTrackerGUI(tracker)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()