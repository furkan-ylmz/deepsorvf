import psutil
import GPUtil
import time
import threading
import os
import csv
from datetime import datetime
import numpy as np

class PerformanceMonitor:
    def __init__(self, log_file="performance_log.csv"):
        self.log_file = log_file
        self.monitoring = False
        self.monitor_thread = None
        self.start_time = None
        self.measurements = []
        
        # Initialize log file
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cpu_percent', 'memory_percent', 'memory_mb',
                'gpu_utilization', 'gpu_memory_used', 'gpu_memory_total',
                'disk_read_mb', 'disk_write_mb', 'network_sent_mb', 'network_recv_mb'
            ])
    
    def get_system_metrics(self):
        """Sistem metriklerini topla"""
        # CPU kullanımı
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory kullanımı
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_mb = memory.used / (1024 * 1024)
        
        # GPU kullanımı (varsa)
        gpu_utilization = 0
        gpu_memory_used = 0
        gpu_memory_total = 0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # İlk GPU'yu al
                gpu_utilization = gpu.load * 100
                gpu_memory_used = gpu.memoryUsed
                gpu_memory_total = gpu.memoryTotal
        except:
            pass
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
        disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
        network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'memory_mb': memory_mb,
            'gpu_utilization': gpu_utilization,
            'gpu_memory_used': gpu_memory_used,
            'gpu_memory_total': gpu_memory_total,
            'disk_read_mb': disk_read_mb,
            'disk_write_mb': disk_write_mb,
            'network_sent_mb': network_sent_mb,
            'network_recv_mb': network_recv_mb
        }
    
    def monitor_loop(self):
        """Monitoring döngüsü"""
        initial_metrics = self.get_system_metrics()
        
        while self.monitoring:
            metrics = self.get_system_metrics()
            
            # Baseline'dan farkı hesapla
            metrics['disk_read_mb'] -= initial_metrics['disk_read_mb']
            metrics['disk_write_mb'] -= initial_metrics['disk_write_mb']
            metrics['network_sent_mb'] -= initial_metrics['network_sent_mb']
            metrics['network_recv_mb'] -= initial_metrics['network_recv_mb']
            
            self.measurements.append(metrics)
            
            # Log dosyasına yaz
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    metrics['timestamp'], metrics['cpu_percent'], 
                    metrics['memory_percent'], metrics['memory_mb'],
                    metrics['gpu_utilization'], metrics['gpu_memory_used'], 
                    metrics['gpu_memory_total'], metrics['disk_read_mb'],
                    metrics['disk_write_mb'], metrics['network_sent_mb'], 
                    metrics['network_recv_mb']
                ])
            
            time.sleep(2)  # Her 2 saniyede bir ölç
    
    def start_monitoring(self):
        """Monitoring başlat"""
        print("🚀 Performans monitoring başlatıldı...")
        print(f"📊 Log dosyası: {self.log_file}")
        
        self.start_time = time.time()
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # Sistem bilgilerini göster
        self.print_system_info()
    
    def stop_monitoring(self):
        """Monitoring durdur"""
        if self.monitoring:
            self.monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join()
            
            total_time = time.time() - self.start_time
            self.generate_report(total_time)
    
    def print_system_info(self):
        """Sistem bilgilerini yazdır"""
        print("\n💻 Sistem Bilgileri:")
        print(f"CPU: {psutil.cpu_count()} çekirdek")
        print(f"RAM: {psutil.virtual_memory().total // (1024**3)} GB")
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                print(f"GPU: {gpu.name} ({gpu.memoryTotal} MB)")
            else:
                print("GPU: Bulunamadı")
        except:
            print("GPU: Bulunamadı")
    
    def generate_report(self, total_time):
        """Performans raporu oluştur"""
        if not self.measurements:
            print("❌ Ölçüm verisi bulunamadı!")
            return
        
        print("\n" + "="*60)
        print("📈 PERFORMANS RAPORU")
        print("="*60)
        
        print(f"⏱️  Toplam Süre: {total_time:.2f} saniye ({total_time/60:.2f} dakika)")
        
        # CPU istatistikleri
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        print(f"\n🔥 CPU Kullanımı:")
        print(f"   Ortalama: {np.mean(cpu_values):.1f}%")
        print(f"   Maksimum: {np.max(cpu_values):.1f}%")
        print(f"   Minimum:  {np.min(cpu_values):.1f}%")
        
        # Memory istatistikleri
        memory_values = [m['memory_mb'] for m in self.measurements]
        print(f"\n🧠 RAM Kullanımı:")
        print(f"   Ortalama: {np.mean(memory_values):.0f} MB")
        print(f"   Maksimum: {np.max(memory_values):.0f} MB")
        print(f"   Minimum:  {np.min(memory_values):.0f} MB")
        
        # GPU istatistikleri
        gpu_util_values = [m['gpu_utilization'] for m in self.measurements if m['gpu_utilization'] > 0]
        if gpu_util_values:
            gpu_memory_values = [m['gpu_memory_used'] for m in self.measurements if m['gpu_memory_used'] > 0]
            print(f"\n🎮 GPU Kullanımı:")
            print(f"   Ortalama Kullanım: {np.mean(gpu_util_values):.1f}%")
            print(f"   Maksimum Kullanım: {np.max(gpu_util_values):.1f}%")
            print(f"   Ortalama Memory: {np.mean(gpu_memory_values):.0f} MB")
            print(f"   Maksimum Memory: {np.max(gpu_memory_values):.0f} MB")
        
        # Disk I/O
        disk_read_total = max([m['disk_read_mb'] for m in self.measurements])
        disk_write_total = max([m['disk_write_mb'] for m in self.measurements])
        print(f"\n💾 Disk I/O:")
        print(f"   Toplam Okuma: {disk_read_total:.1f} MB")
        print(f"   Toplam Yazma: {disk_write_total:.1f} MB")
        
        # Network I/O
        network_sent_total = max([m['network_sent_mb'] for m in self.measurements])
        network_recv_total = max([m['network_recv_mb'] for m in self.measurements])
        print(f"\n🌐 Network I/O:")
        print(f"   Toplam Gönderilen: {network_sent_total:.1f} MB")
        print(f"   Toplam Alınan: {network_recv_total:.1f} MB")
        
        # Enerji tahmini (kabaca)
        avg_cpu = np.mean(cpu_values)
        estimated_power = self.estimate_power_consumption(avg_cpu, gpu_util_values)
        print(f"\n⚡ Tahmini Güç Tüketimi:")
        print(f"   Ortalama: {estimated_power:.1f} Watt")
        print(f"   Toplam Enerji: {estimated_power * total_time / 3600:.2f} Wh")
        
        print(f"\n📄 Detaylı log: {self.log_file}")
        print("="*60)
    
    def estimate_power_consumption(self, avg_cpu, gpu_util_values):
        """Güç tüketimi tahmini"""
        # Temel sistem: ~50W
        base_power = 50
        
        # CPU: %1 kullanım per ~1W
        cpu_power = avg_cpu * 1.0
        
        # GPU: varsa ortalama %1 per ~2W
        gpu_power = 0
        if gpu_util_values:
            gpu_power = np.mean(gpu_util_values) * 2.0
        
        return base_power + cpu_power + gpu_power

def main():
    """Test için kullanım örneği"""
    monitor = PerformanceMonitor("deepsorvf_performance.csv")
    
    print("DeepSORVF Performans Monitor")
    print("CTRL+C ile durdurun...")
    
    try:
        monitor.start_monitoring()
        
        # DeepSORVF'i burada çalıştırın
        # subprocess.run(["python", "main.py"])
        
        # Test için bekleme
        time.sleep(30)
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring durduruluyor...")
    finally:
        monitor.stop_monitoring()

if __name__ == "__main__":
    main()
