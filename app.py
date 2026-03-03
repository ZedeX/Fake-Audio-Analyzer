from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import io
import base64
import json
from datetime import datetime
import logging
import threading
from functools import wraps

app = Flask(__name__)

# 检测是否在Vercel环境中
IS_VERCEL = os.environ.get('VERCEL_ENV') == 'true'

# 配置Flask
if IS_VERCEL:
    # Vercel环境：使用临时目录
    import tempfile
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
    app.config['LOG_FOLDER'] = tempfile.gettempdir()
    
    # Vercel中禁用文件日志（Serverless无持久化存储）
    ENABLE_FILE_LOG = False
    
    # 限制文件大小（Vercel Serverless限制）
    app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit
else:
    # 本地环境
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['LOG_FOLDER'] = 'logs'
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit
    ENABLE_FILE_LOG = True
    
    # 创建必要的目录
    for folder in [app.config['UPLOAD_FOLDER'], app.config['LOG_FOLDER']]:
        if not os.path.exists(folder):
            os.makedirs(folder)

# 线程锁（用于保护共享资源）
file_lock = threading.Lock()
log_lock = threading.Lock()
analysis_lock = threading.Lock()

# 线程安全的文件操作装饰器
def thread_safe(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with analysis_lock:
            return func(*args, **kwargs)
    return wrapper

# 自定义日志格式化器（确保没有ANSI转义码）
class CleanFormatter(logging.Formatter):
    def format(self, record):
        # 移除消息中的ANSI转义码
        import re
        if isinstance(record.msg, str):
            record.msg = re.sub(r'\x1b\[[0-9;]*m', '', record.msg)
        return super().format(record)

# JSON编码器（处理numpy.float32等类型）
class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 配置日志 - 根据环境选择处理器（必须在环境检测之后）
def configure_logging():
    """配置日志系统"""
    log_handlers = [logging.StreamHandler()]
    
    if ENABLE_FILE_LOG:
        # 本地环境：写入文件
        try:
            log_file = os.path.join(app.config['LOG_FOLDER'], 'app.log')
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(CleanFormatter('%(asctime)s - %(levelname)s - %(message)s'))
            log_handlers.append(file_handler)
            print(f"[INFO] 文件日志已启用: {log_file}")
        except Exception as e:
            print(f"[WARNING] 无法创建文件日志: {e}")
    else:
        print("[INFO] 文件日志已禁用（Vercel环境）")
    
    # 配置基本日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=log_handlers
    )
    
    # 应用清理格式化器到所有StreamHandler
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(CleanFormatter('%(asctime)s - %(levelname)s - %(message)s'))

# 延迟配置日志，只在本地运行时执行
# Vercel环境中不执行日志配置
if not IS_VERCEL:
    configure_logging()

# 设置matplotlib支持中文（如果系统中文字体可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文标签
    logging.warning("中文字体加载失败，将使用英文标签")
    pass

class AudioAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.y, self.sr = librosa.load(file_path, sr=None)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)
        self.logger = logging.getLogger(__name__)
        
    def generate_spectrogram(self):
        """生成频谱图"""
        try:
            # 计算STFT，使用更合理的参数
            n_fft = 4096  # 提高频率分辨率
            hop_length = n_fft // 4
            
            # 生成频谱
            D = librosa.stft(self.y, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            
            # 转换为dB
            db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            # 创建图像 - 使用英文标签避免字体问题
            plt.figure(figsize=(14, 7))
            librosa.display.specshow(db, sr=self.sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='hz', 
                                    cmap='viridis')
            plt.colorbar(format='%+2.0f dB')
            plt.title('Audio Spectrogram', fontsize=14, fontweight='bold')
            plt.xlabel('Time (seconds)', fontsize=12)
            plt.ylabel('Frequency (Hz)', fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 保存到内存
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=120, 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plt.close()
            
            return buffer
        except Exception as e:
            self.logger.error(f"生成频谱图失败: {str(e)}")
            raise
    
    def analyze_quality(self):
        """分析音频质量 - 优化版"""
        self.logger.info(f"开始分析文件: {self.filename}")
        self.logger.info(f"采样率: {self.sr/1000:.1f}kHz, 时长: {self.duration:.2f}秒")
        
        try:
            # 计算频率分布 - 使用更大的n_fft提高精度
            n_fft = min(8192, len(self.y) // 5)  # 更高的频率分辨率
            D = librosa.stft(self.y, n_fft=n_fft)
            magnitude = np.abs(D)
            frequencies = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
            
            # 计算每个频率的平均能量
            mean_magnitude = np.mean(magnitude, axis=1)
            
            # 找出有效频率范围（能量高于阈值）
            threshold = np.max(mean_magnitude) * 0.005  # 降低到0.5%提高灵敏度
            effective_freq_idx = np.where(mean_magnitude > threshold)[0]
            
            if len(effective_freq_idx) == 0:
                max_effective_freq = 0
            else:
                max_effective_freq = frequencies[effective_freq_idx[-1]]
            
            # 分析高频特征
            result = {
                'sample_rate': self.sr,
                'duration': self.duration,
                'max_effective_freq': max_effective_freq,
                'is_lossless': True,
                'is_hi_res': False,
                'issues': [],
                'analysis': {},
                'confidence': 0.0,  # 添加置信度
                'detailed_metrics': {}
            }
            
            # 判断是否为假无损
            self._analyze_lossless(result, frequencies, mean_magnitude, magnitude)
            
            # 判断是否为假高解析度
            self._analyze_hi_res(result, frequencies, mean_magnitude)
            
            # 计算置信度
            self._calculate_confidence(result)
            
            # 记录详细分析结果
            self._log_analysis_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"分析失败: {str(e)}")
            raise
    
    def _analyze_lossless(self, result, frequencies, mean_magnitude, magnitude):
        """优化版：分析是否为真无损"""
        sr = result['sample_rate']
        
        if sr <= 48000:
            self.logger.info("检测CD音质音频...")
            
            # 方法1：检查16kHz截断特征（MP3转码）
            self._detect_mp3_transcode(result, frequencies, mean_magnitude)
            
            # 方法2：检查频谱滚降点
            self._detect_rolloff_point(result, frequencies, mean_magnitude)
            
            # 方法3：检查AAC毛刺特征
            self._analyze_aac_features(result, frequencies, mean_magnitude)
            
            # 方法4：检查频谱平坦度
            self._analyze_spectral_flatness(result)
    
    def _detect_mp3_transcode(self, result, frequencies, mean_magnitude):
        """检测MP3转码特征"""
        # 检查15.5-16.5kHz区域（128k MP3通常在16kHz截断）
        freq_15k5 = np.where(frequencies >= 15500)[0]
        freq_16k5 = np.where(frequencies >= 16500)[0]
        
        if len(freq_15k5) > 0 and len(freq_16k5) > 0:
            idx_start = freq_15k5[0]
            idx_end = min(freq_16k5[0], len(mean_magnitude))
            
            if idx_end > idx_start:
                # 计算这个区域的能量变化率
                energy_before = np.mean(mean_magnitude[max(0, idx_start-100):idx_start])
                energy_after = np.mean(mean_magnitude[idx_start:idx_end])
                max_energy = np.max(mean_magnitude)
                
                # MP3特征：在16kHz附近能量急剧下降
                if energy_before > max_energy * 0.01 and energy_after < max_energy * 0.0005:
                    cutoff_ratio = energy_after / (energy_before + 1e-10)
                    if cutoff_ratio < 0.05:  # 能量下降超过95%
                        result['is_lossless'] = False
                        result['issues'].append("⚠️ MP3转码特征：16kHz处有明显截断（能量下降95%+）")
                        result['analysis']['mp3_transcode'] = True
                        result['analysis']['cutoff_freq'] = 16000
                        result['analysis']['cutoff_ratio'] = cutoff_ratio
                        self.logger.warning(f"检测到MP3转码: 截断比例={cutoff_ratio:.3f}")
    
    def _detect_rolloff_point(self, result, frequencies, mean_magnitude):
        """检测频谱滚降点"""
        max_energy = np.max(mean_magnitude)
        threshold = max_energy * 0.001  # 0.1%阈值
        
        # 找到能量低于阈值的位置
        rolloff_idx = np.where(mean_magnitude < threshold)[0]
        
        if len(rolloff_idx) > 0:
            rolloff_freq = float(frequencies[rolloff_idx[0]])  # 转换为Python float
            result['detailed_metrics']['rolloff_frequency'] = rolloff_freq
            
            # 如果是CD音质，但滚降点在18kHz以下，可能是假无损
            if result['sample_rate'] <= 48000 and rolloff_freq < 18000:
                result['issues'].append(f"⚠️ 频谱过早滚降：在{rolloff_freq/1000:.1f}kHz处能量已低于0.1%")
                result['analysis']['early_rolloff'] = True
                self.logger.info(f"频谱滚降点: {rolloff_freq/1000:.1f}kHz")
    
    def _analyze_aac_features(self, result, frequencies, mean_magnitude):
        """优化版：分析AAC编码特征"""
        if result['sample_rate'] <= 48000:
            # 检查16-20kHz区域的能量变化
            freq_16k = np.where(frequencies >= 16000)[0]
            freq_20k = np.where(frequencies >= 20000)[0]
            
            if len(freq_16k) > 0 and len(freq_20k) > 0:
                high_freq_region = mean_magnitude[freq_16k[0]:freq_20k[0]]
                
                if len(high_freq_region) > 20:
                    # 计算局部变化率（毛刺特征）
                    local_variations = []
                    for i in range(0, len(high_freq_region)-5, 5):
                        window = high_freq_region[i:i+5]
                        if np.mean(window) > 0:
                            variation = np.std(window) / np.mean(window)
                            local_variations.append(variation)
                    
                    if local_variations:
                        avg_variation = np.mean(local_variations)
                        max_variation = np.max(local_variations)
                        
                        # AAC特征：能量变化剧烈
                        if avg_variation > 1.5 or max_variation > 3.0:
                            result['issues'].append("⚠️ AAC编码特征：高频区域能量分布不均（毛刺现象）")
                            result['analysis']['aac_features'] = True
                            result['analysis']['spectral_variation'] = avg_variation
                            self.logger.info(f"检测到AAC特征: 变异系数={avg_variation:.2f}")
    
    def _analyze_spectral_flatness(self, result):
        """分析频谱平坦度（检测压缩痕迹）"""
        try:
            # 计算频谱平坦度
            flatness = librosa.feature.spectral_flatness(y=self.y, S=np.abs(librosa.stft(self.y)))
            avg_flatness = float(np.mean(flatness))  # 转换为Python float
            
            result['detailed_metrics']['spectral_flatness'] = avg_flatness
            
            # 有损压缩通常会导致更高的频谱平坦度
            if avg_flatness > 0.3 and result['sample_rate'] <= 48000:
                result['issues'].append(f"⚠️ 频谱平坦度异常：{avg_flatness:.3f}（有损压缩痕迹）")
                result['analysis']['high_flatness'] = True
                self.logger.info(f"频谱平坦度: {avg_flatness:.3f}")
        except:
            pass
    
    def _analyze_hi_res(self, result, frequencies, mean_magnitude):
        """优化版：分析是否为真高解析度音频"""
        sr = result['sample_rate']
        max_freq = result['max_effective_freq']
        
        if sr > 48000:
            self.logger.info("检测高解析度音频...")
            
            # 方法1：检查有效频率
            if max_freq < 30000:
                result['issues'].append(f"❌ 疑似升频：有效频率仅{max_freq/1000:.1f}kHz（应>{30:.1f}kHz）")
                result['analysis']['upscaling'] = True
                result['analysis']['effective_freq'] = max_freq
                result['is_hi_res'] = False
                self.logger.warning(f"有效频率不足: {max_freq/1000:.1f}kHz")
            else:
                result['is_hi_res'] = True
                result['analysis']['real_hi_res'] = True
                self.logger.info(f"有效频率达标: {max_freq/1000:.1f}kHz")
            
            # 方法2：精细检查21kHz/24kHz边界
            self._check_frequency_boundary_detail(result, frequencies, mean_magnitude)
            
            # 方法3：检查高频噪音特征
            self._analyze_high_freq_noise(result, frequencies, mean_magnitude)
    
    def _check_frequency_boundary_detail(self, result, frequencies, mean_magnitude):
        """精细检查升频边界"""
        # 检查20-25kHz区域的能量变化
        freq_20k = np.where(frequencies >= 20000)[0]
        freq_25k = np.where(frequencies >= 25000)[0]
        
        if len(freq_20k) > 0 and len(freq_25k) > 0:
            # 计算边界区域的能量梯度
            region_before = mean_magnitude[freq_20k[0]-50:freq_20k[0]]
            region_after = mean_magnitude[freq_25k[0]:freq_25k[0]+50]
            
            if len(region_before) > 0 and len(region_after) > 0:
                energy_before = np.mean(region_before)
                energy_after = np.mean(region_after)
                
                # 升频特征：边界后能量急剧下降或变为噪音
                if energy_after > energy_before * 0.1:  # 边界后能量相对过高
                    result['issues'].append("⚠️ 疑似升频：21-24kHz边界后高频噪音异常")
                    result['analysis']['noise_after_boundary'] = True
                    self.logger.warning(f"边界能量异常: 前={energy_before:.6f}, 后={energy_after:.6f}")
    
    def _analyze_high_freq_noise(self, result, frequencies, mean_magnitude):
        """分析高频噪音特征"""
        # 检查30kHz以上的噪音
        freq_30k = np.where(frequencies >= 30000)[0]
        
        if len(freq_30k) > 0:
            noise_region = mean_magnitude[freq_30k[0]:freq_30k[0]+100]
            
            if len(noise_region) > 0:
                noise_level = np.mean(noise_region)
                max_energy = np.max(mean_magnitude)
                
                # 真高解析度：高频噪音应该很低
                noise_ratio = float(noise_level / (max_energy + 1e-10))  # 转换为Python float
                if noise_level > max_energy * 0.01:
                    result['issues'].append(f"⚠️ 高频噪音异常：30kHz处噪音过高（可能为升频引入）")
                    result['analysis']['high_freq_noise'] = True
                    result['detailed_metrics']['high_freq_noise_level'] = noise_ratio
                    self.logger.info(f"高频噪音水平: {noise_ratio:.6f}")
    
    def _calculate_confidence(self, result):
        """计算检测结果置信度"""
        confidence = 0.5  # 基础置信度
        
        if result['sample_rate'] <= 48000:
            # CD音质
            if not result['is_lossless']:
                # 如果有多个问题证据，提高置信度
                issues_count = len([k for k in result['analysis'].keys() 
                                  if k in ['mp3_transcode', 'early_rolloff', 'aac_features']])
                confidence = 0.6 + issues_count * 0.15
            else:
                # 如果真无损证据充分
                if result['max_effective_freq'] > 18000:
                    confidence = 0.7
        else:
            # 高解析度
            if not result['is_hi_res']:
                issues_count = len([k for k in result['analysis'].keys() 
                                  if k in ['upscaling', 'boundary_21k_24k', 'noise_after_boundary']])
                confidence = 0.6 + issues_count * 0.15
            else:
                if result['max_effective_freq'] > 35000:
                    confidence = 0.8
        
        result['confidence'] = min(confidence, 0.95)
        self.logger.info(f"检测置信度: {result['confidence']:.2%}")
    
    def _log_analysis_result(self, result):
        """记录分析结果到日志文件（线程安全）"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': self.filename,
            'sample_rate': float(self.sr),
            'duration': float(self.duration),
            'max_effective_freq': float(result['max_effective_freq']),
            'is_lossless': bool(result['is_lossless']),
            'is_hi_res': bool(result['is_hi_res']),
            'confidence': float(result['confidence']),
            'issues': result['issues'],
            'analysis': result['analysis'],
            'detailed_metrics': result['detailed_metrics']
        }
        
        # 保存到JSON日志文件（仅在非Vercel环境中）
        if ENABLE_FILE_LOG:
            log_file = os.path.join(app.config['LOG_FOLDER'], 'analysis_history.jsonl')
            with log_lock:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # 记录到标准日志
        issues_str = '; '.join(result['issues']) if result['issues'] else '无问题'
        self.logger.info(f"分析完成: {self.filename} | 置信度:{result['confidence']:.1%} | 问题:{issues_str}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return app.response_class(
            response=json.dumps({'error': '没有上传文件'}, cls=NumpyJSONEncoder, ensure_ascii=False),
            status=400,
            mimetype='application/json'
        )
    
    file = request.files['audio']
    if file.filename == '':
        return app.response_class(
            response=json.dumps({'error': '文件名为空'}, cls=NumpyJSONEncoder, ensure_ascii=False),
            status=400,
            mimetype='application/json'
        )
    
    original_filename = file.filename
    app.logger.info(f"收到上传请求: {original_filename}")
    
    try:
        # 保存文件（线程安全）
        filename = str(uuid.uuid4()) + '_' + original_filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        with file_lock:
            file.save(filepath)
        
        app.logger.info(f"文件已保存: {filepath}")
        
        # 分析音频
        analyzer = AudioAnalyzer(filepath)
        
        # 生成频谱图
        spec_buffer = analyzer.generate_spectrogram()
        spec_base64 = base64.b64encode(spec_buffer.getvalue()).decode()
        app.logger.info("频谱图生成完成")
        
        # 分析质量
        analysis = analyzer.analyze_quality()
        app.logger.info(f"音频分析完成: {original_filename}")
        
        # 清理文件（线程安全）
        with file_lock:
            if os.path.exists(filepath):
                os.remove(filepath)
                app.logger.info(f"临时文件已清理: {filepath}")
            else:
                app.logger.warning(f"文件不存在，无法清理: {filepath}")
        
        return app.response_class(
            response=json.dumps({
                'success': True,
                'spectrogram': spec_base64,
                'analysis': analysis
            }, cls=NumpyJSONEncoder, ensure_ascii=False),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        app.logger.error(f"分析失败 - 文件: {original_filename}, 错误: {str(e)}")
        app.logger.error(f"返回错误响应: {str(e)}")
        return app.response_class(
            response=json.dumps({'error': str(e)}, cls=NumpyJSONEncoder, ensure_ascii=False),
            status=400,
            mimetype='application/json'
        )

# WSGI handler for Vercel
def handler(event, context):
    """Vercel Serverless handler"""
    return app

if __name__ == '__main__':
    # 确保WERKZEUG_SERVER_FD环境变量不存在
    if 'WERKZEUG_SERVER_FD' in os.environ:
        del os.environ['WERKZEUG_SERVER_FD']
    
    # 使用reloader_options避免werkzeug错误
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
