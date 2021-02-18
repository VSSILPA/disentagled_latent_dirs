import time
import logging
import os

class PerfomanceLogger(object):
	def __init__(self):
		self.task_time_map = {}
		self.logger = logging.getLogger()
		logging.basicConfig(level=logging.INFO)

	def start_monitoring(self, task_name):
		self.task_time_map[task_name] = time.time()

	def stop_monitoring(self, task_name):
		if task_name in self.task_time_map:
			start_time = self.task_time_map[task_name]
			self.logger.info("PerfLog |"+task_name+"|TT:"+str(time.time()-start_time)+"s")
			self.task_time_map.pop(task_name)
		else:
			raise Exception ("Task" + task_name+" not found")

	@staticmethod
	def configure_logger(config):
		experiment_name = config['experiment_name']
		logger = logging.getLogger()
		logger.setLevel(logging.INFO)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
		ch.setFormatter(formatter)
		logger.addHandler(ch)
		directory = os.path.dirname(os.getcwd()) + f'/results/{experiment_name}'
		if not os.path.exists(directory):
			os.makedirs(directory)
		fh = logging.FileHandler(directory+'/logfile.txt')
		logger.addHandler(fh)

