import logging
import os
import sys

from cloud_handler import CloudLoggingHandler
from cron_executor import Executor

PROJECT = 'antsa-demo-devfest'
TOPIC = 'antsa-mnist-demo' #pub sub topic

script_path = os.path.abspath(os.path.join(os.getcwd(), 'predict.py'))

sample_task = "nohup python3 -u %s & \nexit" % script_path


root_logger = logging.getLogger('cron_executor')
root_logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root_logger.addHandler(ch)

cloud_handler = CloudLoggingHandler(on_gce=True, logname="task_runner")
root_logger.addHandler(cloud_handler)

# create the executor that watches the topic, and will run the job task
utility_executor = Executor(topic=TOPIC, project=PROJECT, task_cmd=sample_task, subname='predict_task')

# add a cloud logging handler and stderr logging handler
job_cloud_handler = CloudLoggingHandler(on_gce=True, logname=utility_executor.subname)
utility_executor.job_log.addHandler(job_cloud_handler)
utility_executor.job_log.addHandler(ch)
utility_executor.job_log.setLevel(logging.DEBUG)


# watches indefinitely
utility_executor.watch_topic()