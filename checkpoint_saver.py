"""
Andrin Jenal, 2017
ETH Zurich
"""

import os
import logging
from time import strftime, time
import json


class CheckpointSaver:
    def __init__(self, checkpoint_dir, experiment_name="DCGAN"):
        # checkpoint generation relevant parameters
        self.experiment_name = experiment_name + "_" + strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = os.path.join(checkpoint_dir, self.experiment_name)
        self.summary_dir = os.path.join(checkpoint_dir, self.experiment_name)
        self.last_epoch = 0
        self.last_time = time()
        self.total_time = 0

        # create experiment folder
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
            print('created experiment directory:', self.experiment_dir)

        # create summary folder
        if self.summary_dir and not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)
            print('created summary directory:', self.summary_dir)

        # logging configuration
        logging.basicConfig(format="%(message)s", filename=os.path.join(self.summary_dir, "output.log"), level=logging.INFO)

        # log experiment params
        self.audit_parameter("experiment_name", self.experiment_name)
        self.audit_parameter("experiment_directory", self.experiment_dir)
        self.audit_parameter("summary_directory", self.summary_dir)

    def _get_time(self, time_in_seconds):
        time_in_minutes = time_in_seconds / 60
        if time_in_minutes > 1.0:
            return "%.2f minutes" % time_in_minutes
        else:
            return "%.2f seconds" % time_in_seconds

    def audit_model_parameters(self, param_dict):
        print("TreeNet parameters:")
        logging.info("==========================TreeNet parameters==========================")
        for name, value in param_dict.items():
            print(name + ":", str(value))
            logging.info(name + ": " + str(value))

    def audit_parameter(self, param_name, param):
        logging.info(param_name + ": " + str(param))

    def audit_loss(self, msg):
        print(msg)
        logging.info(msg)

    def audit_time(self, epoch):
        current_time = time()
        print("elapsed time for epoch:", str(self.last_epoch) + " to " + str(epoch) + ": " + self._get_time(current_time - self.last_time))
        logging.info("elapsed time for epoch: " + str(self.last_epoch) + " to " + str(epoch) + ": " + self._get_time(current_time - self.last_time))
        self.total_time += (current_time - self.last_time)
        self.last_epoch = epoch
        self.last_time = current_time

    def save_checkpoint(self, saver, sess, epoch):
        saver.save(sess, os.path.join(self.experiment_dir, "draw_model"), global_step=epoch)
        print("saved model checkpoint to file...")
        logging.info("time by now: " + self._get_time(self.total_time))

    def save_experiment_config(self, model_params, name="config"):
        with open(os.path.join(self.summary_dir, name + ".json"), "w") as jfile:
            json.dump(model_params, jfile, indent=True)

    def get_experiment_dir(self):
        return self.experiment_dir

    def get_summary_dir(self):
        return self.summary_dir

    def get_logger(self):
        return logging
