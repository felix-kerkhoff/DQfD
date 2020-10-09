import imageio
from skimage.transform import resize
import os
import texttable as tt
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt

class QLearningLogger:
    """
        *********************
        ** QLearningLogger **
        *********************

        Class for logging and visualizing the learning process of a deep Q-agent.


        -----------
        Parameters:
        -----------
            save_path:  string; 
                        the path for saving the logging data

            restore:    bool; 
                        variable indicating whether to use the logs of a previous training session
    """

    def __init__(self, save_path = '', restore = False):
        self.save_path = save_path
        
        self.q_values = []
        self.losses = []
        self.training_scores = []
        self.validation_scores = []
        self.durations = []
        self.comp_times = []
        self.best_scores = [-np.inf]
        self.best_validation_scores = [-np.inf]
        
        self._ep_lag = 0
        if restore:
            self.restore_logging_data()
            self._ep_lag = len(self.training_scores) - 1
        self._previous_step = 0
        self._last_ep_time = time.time()
        self._make_directories()
        
        
    def _make_directories(self):
        try:
            if not os.path.isdir(self.save_path + "/trained_models/"):
                os.makedirs(self.save_path + "/trained_models/")
            if not os.path.isdir(self.save_path + "/logging_data/"):
                os.makedirs(self.save_path + "/logging_data/")
            if not os.path.isdir(self.save_path + "/plots/"):
                os.makedirs(self.save_path + "/plots/")
            if not os.path.isdir(self.save_path + "/videos/"):
                os.makedirs(self.save_path + "/videos/")
        
        except OSError:
            print ("Creation of the directory %s failed" %(self.save_path))
            

    def save_model(self, model):
        model.save(self.save_path + "/trained_models/current_model.h5")
        
    
    def save_logging_data(self):
        with open(self.save_path + "/logging_data/losses", 'wb') as file:
            pickle.dump(self.losses, file)
            
        with open(self.save_path + "/logging_data/q_values", 'wb') as file:
            pickle.dump(self.q_values, file)
            
        with open(self.save_path + "/logging_data/validation_scores", 'wb') as file:
            pickle.dump(self.validation_scores, file)
            
        with open(self.save_path + "/logging_data/training_scores", 'wb') as file:
            pickle.dump(self.training_scores, file)
            
        with open(self.save_path + "/logging_data/durations", 'wb') as file:
            pickle.dump(self.durations, file)
            
                
    def save_memory(self, memory):
        memory_dict = {'frames': np.asarray(memory.frames),
                       'actions': np.asarray(memory.actions),
                       'rewards': np.asarray(memory.rewards), 
                       'priorities': np.asarray(memory._priority_tree.keys[:memory.max_frame_num]),
                       'is_full': memory._is_full,
                       'current_index': memory._current_index}
        with open(self.save_path + "/logging_data/memory", 'wb') as file:
            pickle.dump(memory_dict, file, protocol=4)
            
    
    def restore_memory(self, memory, memory_path = None):
        if memory_path is None:
            memory_path = self.save_path + "/logging_data/memory"
            
        with open(memory_path, 'rb') as file:
            memory_dict = pickle.load(file)
            memory.frames = memory_dict['frames']
            memory.actions = memory_dict['actions']
            memory.rewards = memory_dict['rewards']
            memory._priority_tree._construct(memory_dict['priorities'])
            memory._is_full = memory_dict['is_full']
            memory._current_index

    def restore_logging_data(self):
        for value in ["losses", "q_values", "validation_scores", "training_scores", "durations"]:
            with open(self.save_path + "/logging_data/" + value, 'rb') as file:
                if value == "losses":
                    self.losses = pickle.load(file)
                elif value == "q_values":
                    self.q_values = pickle.load(file)
                elif value == "validation_scores":
                    self.validation_scores = pickle.load(file)
                elif value == "training_scores":
                    self.training_scores = pickle.load(file)
                elif value == "durations":
                    self.durations = pickle.load(file)
    
    def save_all(self, model, memory, store_memory = False):
        self.save_logging_data()
        if store_memory:
            self.save_memory(memory)
        self.save_model(model)
    
    
    def generate_gif(self, frames_for_gif, time_step, score, validation = ''):
        imageio.mimsave(self.save_path + "/videos/" + validation + "step_{0}_score_{1}.gif".format(time_step, score), frames_for_gif, duration=1/30)
        
    def _record_best_episode(self, frames_for_gif, time_step, score, validation = False):
        if not validation:
            if score > self.best_scores[-1]:
                self.best_scores.append(score)
                self.generate_gif(frames_for_gif, time_step, score)
        else:
            if score > self.best_validation_scores[-1]:
                self.best_validation_scores.append(score)
                self.generate_gif(frames_for_gif, time_step, score, 'validation_')
        
        
    def _get_moving_avg(self, array, n):
        moving = np.zeros((array.shape[0] - n, n))
        for i in range(n):
            moving[:, i] = array[i:(array.shape[0] - n + i)]

        moving_avg = np.mean(moving, axis = 1)
        return(moving_avg)
    
    
    def make_plots(self):
        for (value, title) in [(self.training_scores, 'Training-Score'), (self.validation_scores, 'Validation-Score'),
                               (self.losses, 'Loss'), (self.q_values, 'Action-Value')]:
            if len(value) > 0:
                moving_avg_value = self._get_moving_avg(np.array(value), min(len(value), 50))#int(0.1 * len(value)))
                moving_avg_step = np.arange(moving_avg_value.shape[0]) if title == 'Validation-Score' else self._get_moving_avg(np.cumsum(self.durations), min(len(value), 50))
                figure = plt.figure(figsize = (8, 5))
                plt.plot(moving_avg_step, moving_avg_value)
                plt.xlabel('Validation-Episode' if title == 'Validation-Score' else 'Step')
                plt.ylabel(title)
                plt.title(title)
                plt.savefig(self.save_path + "/plots/" + title + ".png")
                plt.close()
        
    
    def add_episode_logs(self, step, score, q_values, losses, frames_for_gif):
        self._record_best_episode(frames_for_gif, step, score)
        self.comp_times.append(time.time() - self._last_ep_time)
        self.durations.append(step - self._previous_step)
        self.training_scores.append(score)
        self.losses.append(np.mean(losses))
        if len(q_values) > 0:
            self.q_values.append(np.mean(q_values))
        else:
            self.q_values.append(0)
        self._previous_step = step
        self._last_ep_time = time.time()
        
    
    def show_progress(self, lower_idx, upper_idx, validation_score = None, validation_frames = None, validation_model = None, summary = False):
        lower_idx += self._ep_lag
        upper_idx += self._ep_lag
        if validation_score is not None:
            self.validation_scores.append(validation_score)
            if validation_score > self.best_validation_scores[-1]:
                self._record_best_episode(validation_frames, upper_idx, validation_score, validation = True)
                validation_model.save(self.save_path + "/trained_models/best_validation_model_{0}.h5".format(upper_idx))
        if summary:
            comp_output = ['Total Computation Time', '{0}'.format(time.strftime('%H:%M:%S', time.gmtime(np.sum(self.comp_times[(lower_idx - self._ep_lag):(upper_idx - self._ep_lag + 1)]))))]
            frame_output = ['Total Number of Frames', '{0}'.format(int(np.sum(self.durations[lower_idx:(upper_idx + 1)])))]
            validation_output = ['Average Validation Score', '{0}'.format(np.mean(self.validation_scores))]
        else:
            comp_output = ['Average Computation Time', '{0}'.format(time.strftime('%H:%M:%S', time.gmtime(np.mean(self.comp_times[lower_idx:(upper_idx + 1)]))))]
            frame_output = ['Average Number of Frames', '{0}'.format(int(np.mean(self.durations[lower_idx:(upper_idx + 1)])))]
            validation_output = ['Validation Score', '{0}'.format(self.validation_scores[-1])]
        
        output = [[],
                  comp_output,
                  frame_output,
                  ['Average Score', '{0}'.format(np.mean(self.training_scores[lower_idx:(upper_idx + 1)]))],
                  ['Maximum Score', '{0}'.format(np.amax(self.training_scores[lower_idx:(upper_idx + 1)]))],
                  ['Minimum Score', '{0}'.format(np.amin(self.training_scores[lower_idx:(upper_idx + 1)]))],
                  validation_output,
                  ['Average Action-Values', '{0}'.format(np.mean(self.q_values[lower_idx:(upper_idx + 1)]) if len(self.q_values[lower_idx:(upper_idx + 1)]) > 0 else 'N/A')],
                  ['Average Losses', '{0}'.format(np.mean(self.losses[lower_idx:(upper_idx + 1)]))],
                 ]
        
        tab = tt.Texttable()
        tab.add_rows(output)
        tab.set_cols_align(['c', 'c'])
        tab.header(['Episodes', '{0} - {1}'.format(lower_idx, upper_idx)])
        print(tab.draw())
        print('\n\n')
