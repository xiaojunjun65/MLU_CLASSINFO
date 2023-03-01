import logging
import time
class Logging(object):
    def __init__(self, filename=None, console_out=True, file_out=True):
        self.logger = logging.getLogger()

        if console_out:
            self.logger.setLevel(logging.INFO)  # Log等级总开关 
            stream_handler = logging.StreamHandler()  # 日志控制台输出
            # 控制台输出格式
            stream_format = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
            stream_handler.setFormatter(stream_format)
            self.logger.addHandler(stream_handler)

        if file_out:
            if filename is None:
                filename = time.strftime("%Y-%m-%d", time.localtime())+'.log'
            handler = logging.FileHandler(filename, mode='w', encoding='UTF-8')  # 日志文件输出
            handler.setLevel(logging.DEBUG)
            # 文件输出格式
            logging_format = logging.Formatter(
                '%(asctime)s - %(filename)s - %(funcName)s - %(lineno)s - %(levelname)s: %(message)s')
            handler.setFormatter(logging_format)  # 为改处理器handler 选择一个格式化器
            self.logger.addHandler(handler)  # 为记录器添加 处理方式Handler
        
        assert(console_out | file_out)

    # def info(self, msg, *args, **kwargs):
    #     self.logger.info(msg, *args, **kwargs)

    # def warn(self, msg, *args, **kwargs):
    #     self.logger.warn(msg, *args, **kwargs)

    # def error(self, msg, *args, **kwargs):
    #     self.logger.error(msg, *args, **kwargs)
    
    # def dubug(self, msg, *args, **kwargs):
    #     self.logger.debug(msg, *args, **kwargs)