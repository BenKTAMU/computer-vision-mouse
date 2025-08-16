import psutil
import time
import threading
import json
import os
from datetime import datetime
import logging
from collections import deque
import cv2
import numpy as np

class SystemMonitor:
    def __init__(self, log_interval=1.0, max_history=1000):
        self.log_interval = log_interval
        self.max_history = max_history
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'cpu': deque(maxlen=max_history),
            'memory': deque(maxlen=max_history),
            'gpu': deque(maxlen=max_history),
            'disk': deque(maxlen=max_history),
            'network': deque(maxlen=max_history),
            'camera': deque(maxlen=max_history),
            'hand_tracking': deque(maxlen=max_history)
        }
        
        # Current system state
        self.current_state = {}
        
        # Monitoring control
        self.monitoring = False
        self.monitor_thread = None
        
        # Alert thresholds
        self.thresholds = {
            'cpu_high': 80.0,      # CPU usage above 80%
            'memory_high': 85.0,   # Memory usage above 85%
            'fps_low': 15.0,       # FPS below 15
            'latency_high': 100.0  # Latency above 100ms
        }
        
        # Alert history
        self.alerts = deque(maxlen=100)
        
        # Performance analysis
        self.performance_analysis = {
            'bottlenecks': [],
            'optimization_suggestions': [],
            'trends': {}
        }
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("System monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                self._collect_metrics()
                self._analyze_performance()
                self._check_alerts()
                time.sleep(self.log_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self):
        """Collect current system metrics"""
        timestamp = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        
        cpu_data = {
            'timestamp': timestamp,
            'usage_percent': cpu_percent,
            'frequency_mhz': cpu_freq.current if cpu_freq else 0,
            'core_count': cpu_count,
            'load_avg': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }
        self.metrics['cpu'].append(cpu_data)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        memory_data = {
            'timestamp': timestamp,
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'usage_percent': memory.percent,
            'swap_percent': psutil.swap_memory().percent
        }
        self.metrics['memory'].append(memory_data)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        disk_data = {
            'timestamp': timestamp,
            'total_gb': disk.total / (1024**3),
            'used_gb': disk.used / (1024**3),
            'free_gb': disk.free / (1024**3),
            'usage_percent': (disk.used / disk.total) * 100,
            'read_mb_s': disk_io.read_bytes / (1024**2) if disk_io else 0,
            'write_mb_s': disk_io.write_bytes / (1024**2) if disk_io else 0
        }
        self.metrics['disk'].append(disk_data)
        
        # Network metrics
        network = psutil.net_io_counters()
        network_data = {
            'timestamp': timestamp,
            'bytes_sent_mb': network.bytes_sent / (1024**2),
            'bytes_recv_mb': network.bytes_recv / (1024**2),
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        self.metrics['network'].append(network_data)
        
        # Update current state
        self.current_state = {
            'cpu': cpu_data,
            'memory': memory_data,
            'disk': disk_data,
            'network': network_data,
            'timestamp': timestamp
        }
    
    def add_hand_tracking_metrics(self, fps, detection_time, processing_time, hand_detected):
        """Add hand tracking specific metrics"""
        timestamp = time.time()
        
        tracking_data = {
            'timestamp': timestamp,
            'fps': fps,
            'detection_time_ms': detection_time * 1000,
            'processing_time_ms': processing_time * 1000,
            'hand_detected': hand_detected,
            'latency_ms': (detection_time + processing_time) * 1000
        }
        
        self.metrics['hand_tracking'].append(tracking_data)
    
    def add_camera_metrics(self, frame_width, frame_height, frame_rate, frame_delay):
        """Add camera performance metrics"""
        timestamp = time.time()
        
        camera_data = {
            'timestamp': timestamp,
            'frame_width': frame_width,
            'frame_height': frame_height,
            'frame_rate': frame_rate,
            'frame_delay_ms': frame_delay * 1000,
            'resolution_megapixels': (frame_width * frame_height) / (1024**2)
        }
        
        self.metrics['camera'].append(camera_data)
    
    def _analyze_performance(self):
        """Analyze performance trends and identify bottlenecks"""
        if len(self.metrics['hand_tracking']) < 10:
            return
        
        # Analyze FPS trends
        recent_fps = [m['fps'] for m in list(self.metrics['hand_tracking'])[-10:]]
        avg_fps = sum(recent_fps) / len(recent_fps)
        fps_trend = 'stable'
        
        if len(recent_fps) >= 2:
            if recent_fps[-1] > recent_fps[0] * 1.1:
                fps_trend = 'improving'
            elif recent_fps[-1] < recent_fps[0] * 0.9:
                fps_trend = 'degrading'
        
        # Analyze latency
        recent_latency = [m['latency_ms'] for m in list(self.metrics['hand_tracking'])[-10:]]
        avg_latency = sum(recent_latency) / len(recent_latency)
        
        # Identify bottlenecks
        bottlenecks = []
        if avg_fps < 20:
            bottlenecks.append("Low FPS - Consider reducing camera resolution or model complexity")
        if avg_latency > 50:
            bottlenecks.append("High latency - Check camera settings and processing pipeline")
        
        # CPU analysis
        if len(self.metrics['cpu']) > 0:
            recent_cpu = [m['usage_percent'] for m in list(self.metrics['cpu'])[-10:]]
            avg_cpu = sum(recent_cpu) / len(recent_cpu)
            if avg_cpu > 80:
                bottlenecks.append("High CPU usage - Consider optimizing algorithms or reducing workload")
        
        # Memory analysis
        if len(self.metrics['memory']) > 0:
            recent_memory = [m['usage_percent'] for m in list(self.metrics['memory'])[-10:]]
            avg_memory = sum(recent_memory) / len(recent_memory)
            if avg_memory > 85:
                bottlenecks.append("High memory usage - Check for memory leaks or reduce buffer sizes")
        
        # Update analysis
        self.performance_analysis.update({
            'bottlenecks': bottlenecks,
            'fps_trend': fps_trend,
            'avg_fps': avg_fps,
            'avg_latency': avg_latency,
            'last_updated': datetime.now().isoformat()
        })
    
    def _check_alerts(self):
        """Check for performance alerts"""
        if not self.current_state:
            return
        
        # CPU alert
        if self.current_state['cpu']['usage_percent'] > self.thresholds['cpu_high']:
            self._add_alert('HIGH_CPU', f"CPU usage: {self.current_state['cpu']['usage_percent']:.1f}%")
        
        # Memory alert
        if self.current_state['memory']['usage_percent'] > self.thresholds['memory_high']:
            self._add_alert('HIGH_MEMORY', f"Memory usage: {self.current_state['memory']['usage_percent']:.1f}%")
        
        # FPS alert
        if len(self.metrics['hand_tracking']) > 0:
            current_fps = self.metrics['hand_tracking'][-1]['fps']
            if current_fps < self.thresholds['fps_low']:
                self._add_alert('LOW_FPS', f"FPS: {current_fps:.1f}")
        
        # Latency alert
        if len(self.metrics['hand_tracking']) > 0:
            current_latency = self.metrics['hand_tracking'][-1]['latency_ms']
            if current_latency > self.thresholds['latency_high']:
                self._add_alert('HIGH_LATENCY', f"Latency: {current_latency:.1f}ms")
    
    def _add_alert(self, alert_type, message):
        """Add a new alert"""
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'message': message,
            'severity': 'WARNING'
        }
        
        self.alerts.append(alert)
        self.logger.warning(f"Alert: {alert_type} - {message}")
    
    def get_current_status(self):
        """Get current system status summary"""
        if not self.current_state:
            return {"status": "No data available"}
        
        status = {
            "status": "OK",
            "timestamp": datetime.fromtimestamp(self.current_state['timestamp']).isoformat(),
            "cpu": {
                "usage": f"{self.current_state['cpu']['usage_percent']:.1f}%",
                "frequency": f"{self.current_state['cpu']['frequency_mhz']:.0f} MHz"
            },
            "memory": {
                "usage": f"{self.current_state['memory']['usage_percent']:.1f}%",
                "available": f"{self.current_state['memory']['available_gb']:.1f} GB"
            },
            "disk": {
                "usage": f"{self.current_state['disk']['usage_percent']:.1f}%",
                "free": f"{self.current_state['disk']['free_gb']:.1f} GB"
            }
        }
        
        # Add hand tracking metrics if available
        if len(self.metrics['hand_tracking']) > 0:
            latest_tracking = self.metrics['hand_tracking'][-1]
            status["hand_tracking"] = {
                "fps": f"{latest_tracking['fps']:.1f}",
                "latency": f"{latest_tracking['latency_ms']:.1f} ms",
                "hand_detected": latest_tracking['hand_detected']
            }
        
        # Check for warnings
        warnings = []
        if self.current_state['cpu']['usage_percent'] > 70:
            warnings.append("High CPU usage")
        if self.current_state['memory']['usage_percent'] > 80:
            warnings.append("High memory usage")
        if len(self.metrics['hand_tracking']) > 0:
            if self.metrics['hand_tracking'][-1]['fps'] < 20:
                warnings.append("Low FPS")
        
        if warnings:
            status["status"] = "WARNING"
            status["warnings"] = warnings
        
        return status
    
    def get_performance_report(self):
        """Generate a comprehensive performance report"""
        if len(self.metrics['hand_tracking']) < 5:
            return {"error": "Insufficient data for analysis"}
        
        # Calculate averages over last 50 samples
        sample_size = min(50, len(self.metrics['hand_tracking']))
        recent_tracking = list(self.metrics['hand_tracking'])[-sample_size:]
        
        fps_values = [m['fps'] for m in recent_tracking]
        latency_values = [m['latency_ms'] for m in recent_tracking]
        detection_times = [m['detection_time_ms'] for m in recent_tracking]
        
        report = {
            "summary": {
                "avg_fps": sum(fps_values) / len(fps_values),
                "min_fps": min(fps_values),
                "max_fps": max(fps_values),
                "avg_latency": sum(latency_values) / len(latency_values),
                "avg_detection_time": sum(detection_times) / len(detection_times)
            },
            "trends": self.performance_analysis,
            "alerts": list(self.alerts)[-10:],  # Last 10 alerts
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate optimization recommendations"""
        recommendations = []
        
        if len(self.metrics['hand_tracking']) > 0:
            latest_fps = self.metrics['hand_tracking'][-1]['fps']
            latest_latency = self.metrics['hand_tracking'][-1]['latency_ms']
            
            if latest_fps < 25:
                recommendations.append({
                    "priority": "HIGH",
                    "category": "Performance",
                    "suggestion": "Reduce camera resolution or model complexity to improve FPS",
                    "expected_improvement": "15-25% FPS increase"
                })
            
            if latest_latency > 50:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Responsiveness",
                    "suggestion": "Optimize hand detection pipeline and reduce processing overhead",
                    "expected_improvement": "20-40% latency reduction"
                })
        
        if len(self.metrics['cpu']) > 0:
            latest_cpu = self.metrics['cpu'][-1]['usage_percent']
            if latest_cpu > 80:
                recommendations.append({
                    "priority": "MEDIUM",
                    "category": "Resource Usage",
                    "suggestion": "Consider using lighter hand detection model or reducing frame rate",
                    "expected_improvement": "Lower CPU usage"
                })
        
        return recommendations
    
    def export_metrics(self, filepath):
        """Export all metrics to a JSON file"""
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "metrics": {k: list(v) for k, v in self.metrics.items()},
                "analysis": self.performance_analysis,
                "alerts": list(self.alerts)
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        for metric_type in self.metrics:
            self.metrics[metric_type].clear()
        self.alerts.clear()
        self.logger.info("All metrics cleared")

def main():
    """Test the system monitor"""
    monitor = SystemMonitor()
    
    print("Starting system monitor...")
    monitor.start_monitoring()
    
    try:
        # Simulate some metrics
        for i in range(5):
            monitor.add_hand_tracking_metrics(30 + i, 0.02, 0.01, True)
            monitor.add_camera_metrics(640, 480, 30, 0.033)
            time.sleep(1)
        
        # Get status and report
        print("\nCurrent Status:")
        status = monitor.get_current_status()
        print(json.dumps(status, indent=2))
        
        print("\nPerformance Report:")
        report = monitor.get_performance_report()
        print(json.dumps(report, indent=2))
        
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
