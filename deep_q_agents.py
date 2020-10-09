from experience_replay import PrioritizedExperienceReplay
import numpy as np
from numpy import clip
from numpy.random import rand, randint
from atari_preprocessing import ProcessedAtariEnv
from deep_q_networks import DeepQNetwork
import tensorflow as tf
from tensorflow import math
argmax = math.argmax
import time
import os
from log_training import QLearningLogger

class BaseQAgent:
    """
    ******************
    ** BaseQAgent **
    ******************

        Base class for logged training and testing of a deep Q-learning agent.
        Methods for decision making and performing parameter updates will be defined in subclasses.
        
        -----------
        Parameters:
        -----------
            env:            object; 
                            OpenAI gym learning environment

            memory:         object; 
                            an instance of PrioritizedExperienceReplay for storage and sampling of an agent's experiences

            policy_network: object;
                            an instance of DeepQNetwork for parameter updates and decision making

            target_network: object;
                            an instance of DeepQNetwork for the estimation of target values for parameter updates

            frame_shape:    tuple;
                            the shape of one frame rendered by the environment

            save_path:      string;
                            the path in which the training logs will be stored

            logger:         object;
                            an instance of QLearningLogger for detailed documentation of the learning progress

    """
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None,
                 target_network = None, 
                 frame_shape = (84, 84),
                 save_path = None, 
                 logger = QLearningLogger):
        
        self.env = env
        self.memory = memory
        self.policy_network = policy_network
        self.target_network = target_network

        self.num_stacked_frames = memory.num_stacked_frames
        
        self.save_path = None
        if save_path is not None:
            self.save_path = save_path
            self.logger = logger(self.save_path)
        
        # internal variables
        self._current_state = np.zeros((1, self.num_stacked_frames, *frame_shape), dtype = np.uint8)
        self._step_counter = 0
        
        # logging variables
        self._q_values = []
        self._losses = []
        
        
    
    
    def _batch_update(self):
        pass
    
    
    def _predict_current_q_values(self):
        pass
    
    
    def _make_decision(self):
        pass
    
    
    def _random_warmup(self, num_steps):
        """Prefill the replay memory with num_steps random transitions"""
        new_frame = self.env.reset()
        reward = 0.0
        action = 0
        done = False
        self.memory.add_experience(action, reward, new_frame, 1, done)
        
        for i in range(num_steps):
            
            action = np.random.randint(self.num_actions)
            new_frame, reward, done, _ = self.env.step(action)
            self.memory.add_experience(action, reward, new_frame, 1, done)
            
            if done:
                new_frame = self.env.reset()
                self.memory.add_experience(0, 0.0, new_frame, 1, False)

        self.memory.add_experience(0, 0.0, new_frame, 1, True)
    
    
    def _start_episode(self, test = False):
        """Start episode with a number of num_stacked_frames no-ops 
        to get the first state of an episode"""
        self.env.true_reset()
        new_frame = self.env.reset()
        if not test:
            self.memory.add_experience(0, 0.0, new_frame, False)
        for i in range(self.num_stacked_frames):
            self._current_state[0, i] = new_frame
            new_frame, reward, done, _ = self.env.step(0)
            if not test:
                self.memory.add_experience(0, reward, new_frame, done)
            
    
    def _update_target_model(self):
        """Copy current weights of the policy network to the target network"""
        self.target_network.model.set_weights(self.policy_network.model.get_weights())
    
    
    def _pretrain(self, pretrain_steps, target_interval):
        """Pretrain the agent on its memory (prefilled with demonstrations)"""
        for step in range(pretrain_steps):
            self._batch_update(pretrain = True)
            if step % target_interval == 0:
                self.logger.save_model(self.policy_network.model)
                print('\nStep:    %i' %(step))
                self._update_target_model()
                print('Validation Score:    %f' %(self.test()[0]))
                print('\n')
    
    
    def train(self,
              num_episodes = 100,
              num_steps = 500000,
              max_steps_per_episode = 10000,
              target_interval = 10000,
              learning_interval = 4,
              frame_skip = 1,
              warmup_steps = None,
              pretrain_steps = None,
              output_freq = 50,
              save_freq = 5, 
              store_memory = False):
        """
            Train a deep Q-learning agent

            -----------
            Parameters:
            -----------
                num_episodes:          integer; 
                                       the minimum number of training episodes

                num_steps:             integer; 
                                       the minimum number of training steps

                max_steps_per_episode: integer; 
                                       the maximum number of steps per episode

                target_interval:       integer; 
                                       the length of the interval in which the target network is kept unchanged

                learning_interval:     integer; 
                                       in every learning_interval-th step a parameter update is performed

                frame_skip:            integer; 
                                       the number of frames in which chosen actions are repeated 
                                       (Attention: in many environments this technique is already handled, so the default is 1)

                warmup_steps:          integer; 
                                       the number of random transitions for prefilling the memory

                pretrain_steps:        integer; 
                                       the number of steps to pretrain the agent on its memory (prefilled with demonstrations)

                output_freq:           integer; 
                                       the frequency of outputting the current training logs

                save_freq:             integer; 
                                       the frequency of saving the current training logs

                store_memory:          boolean; 
                                       whether to store the whole memory of the agent for later use (Attention: this file may be very large)
        """
        
        # prefill memory with random transitions if requested
        if warmup_steps is not None:
            self._random_warmup(warmup_steps)
        
        # pretrain the agent on its on own memory
        if pretrain_steps is not None:
            self._pretrain(pretrain_steps, target_interval)
            
        # logging initialization
        score, self._q_values, self._losses = 0., [], []
        raw_frames = np.zeros(shape = (max_steps_per_episode, *self.env._unprocessed_frame.shape), dtype = np.uint8)

        episode_idx = 0
        while episode_idx < num_episodes or self._step_counter < num_steps:
            # reset environment and get first state
            self._start_episode()
            
            for i in range(max_steps_per_episode):
                
                #-------------------------------------------------------------------------------#
                #####################
                # Interactive Phase #
                #####################
                
                # choose an action, observe reactions of the environment and
                # add this experience to the agent's memory 
                if self._step_counter % frame_skip == 0:    
                    action = self._make_decision()
                new_frame, reward, done, _ = self.env.step(action)
                self.memory.add_experience(action, reward, new_frame, 1, done)
                
                # update current state
                self._current_state[0, :(self.num_stacked_frames-1)] = self._current_state[0, 1:]
                self._current_state[0, self.num_stacked_frames-1] = new_frame
                #-------------------------------------------------------------------------------#
                
                
                #-------------------------------------------------------------------------------#
                ##################
                # Learning Phase #
                ##################
                
                # perform a parameter update of the current policy model
                if self._step_counter % learning_interval == 0:
                    self._batch_update()
                
                # update the target model
                if self._step_counter % target_interval == 0:
                    self._update_target_model()
                #-------------------------------------------------------------------------------#
                
                # logging
                score += self.env._unprocessed_reward
                raw_frames[i] = self.env._unprocessed_frame
                
                
                self._step_counter += 1
                
                if self.env.was_real_done:
                    self.logger.add_episode_logs(self._step_counter, score, self._q_values, self._losses, raw_frames[:i])
                    score, self._q_values, self._losses = 0., [], []
                    break
                    
                if done:
                    self.env.reset()
                
                    
            if not self.env.was_real_done:
                self.memory.add_experience(action, reward, new_frame, 1, True)
                self.logger.add_episode_logs(self._step_counter, score, self._q_values, self._losses, raw_frames[:i])
                score, self._q_values, self._losses = 0., [], []
                
            if episode_idx%(num_episodes/output_freq)==0:
                validation_score, validation_frames = self.test(record = True, max_steps_per_episode = max_steps_per_episode)
                #validation_score, validation_frames = 0, []
                lower_idx = int(clip(episode_idx-(num_episodes/output_freq)+1, 0, num_episodes-1))
                self.logger.show_progress(lower_idx, episode_idx, validation_score, validation_frames, self.policy_network.model)
                
            if episode_idx%(num_episodes/save_freq)==0:
                self.logger.make_plots()
                self.logger.save_all(self.policy_network.model, self.memory, store_memory)
                
            

            episode_idx += 1    
        print('==========================\ntraining session completed\n==========================\n\n\n=======\nSummary\n======='
              )
        self.logger.show_progress(0, num_episodes, summary = True)
        self.logger.make_plots()
        self.logger.save_all(self.policy_network.model, self.memory, store_memory)
        
        
    
    
    def test(self, record = False, eps = 0.001, max_steps_per_episode = 10000):
        """
            Test a deep Q-learning agent in one episode
            using an epsilon-greedy strategy

            -----------
            Parameters:
            -----------
                record:                boolean; 
                                       whether to record the frames of the episode

                eps:                   float; 
                                       the probability of choosing a random action

                max_steps_per_episode: integer; 
                                       the maximum number of steps per episode

        """
        
        frames = []
        done = False
        total_reward = 0
        reward = 0
        self.env.true_reset()
        new_frame, _, _, _ = self.env.step(0)
        
        t = 0
        while not self.env.was_real_done and t<max_steps_per_episode:
            if done:
                self.env.reset()
            if record:
                frames.append(self.env._unprocessed_frame)

            action = self._make_decision(test_eps = eps)

            new_frame, reward, done, info = self.env.step(action)
        
            total_reward += self.env._unprocessed_reward
            
            # update current state
            self._current_state[0, :(self.num_stacked_frames-1)] = self._current_state[0, 1:]
            self._current_state[0, self.num_stacked_frames-1] = new_frame
            
            t += 1
           
        return(total_reward, frames)
                    
                    
                    
class DQNAgent(BaseQAgent):
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None,#DeepQNetwork(),
                 target_network = None,#DeepQNetwork(),
                 frame_shape = (84, 84),
                 save_path = None,
                 
                 discount_factor = 0.99,
                 n_step = None,
                 double_q = False,
                 expert_memory = None,
                 prioritized_replay = False
                 ):
        
        BaseQAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path)
        
        self._idx_range = np.arange(self.memory.batch_size, dtype = np.int32)
        
        self.discount_factor = discount_factor
        self.n_step = n_step
        self.expert_memory = expert_memory
        self.double_q = double_q
        self.prioritized_replay = prioritized_replay
        
                              
    def _get_mini_batch(self, expert = False):
        if expert:
            memory = self.expert_memory
        else:
            memory = self.memory
            
        if self.n_step is None:
            (self.state_1_batch, 
             self.action_batch, 
             self.reward_batch, 
             self.state_2_batch, 
             self.done_batch,
             self.batch_weights) = memory.get_mini_batch()
            
        else:
            (self.state_1_batch, 
             self.action_batch, 
             self.reward_batch, 
             self.state_2_batch, 
             self.done_batch,
             self.n_step_return_batch,
             self.state_n_batch,
             self.n_done_batch,
             self.batch_weights) = memory.get_mini_batch([self.n_step, self.discount_factor])
            
           
    def _get_target_action_values(self):
        # predict all action values for the next states using the target network
        target_values = self.target_network.predict(self.state_2_batch)
        if not self.double_q:
            optimal_actions = argmax(target_values, axis = 1)
        else:
            # get optimal actions in the next states with respect to the policy network
            optimal_actions = self.policy_network.get_optimal_actions(self.state_2_batch)
                
        # get target values corresponding to the optimal actions
        optimal_action_values = tf.reduce_sum(tf.multiply(target_values, tf.one_hot(optimal_actions, target_values.shape[1])), axis = 1)
            
        # compute the new target action values using the Bellman-equation
        target_action_values = self.reward_batch + (1 - self.done_batch) * self.discount_factor * optimal_action_values
        
        
        if self.n_step is None:
            n_step_target_action_values = None
        else:
            # predict all action values for the next states using the target network
            n_step_target_values = self.target_network.predict(self.state_n_batch)
            if not self.double_q:
                n_step_optimal_actions = argmax(n_step_target_values, axis = 1)
            else:
                # get optimal actions in the next states with respect to the policy network
                n_step_optimal_actions = self.policy_network.get_optimal_actions(self.state_n_batch)
                
            # get target values corresponding to the optimal actions
            n_step_optimal_action_values = tf.reduce_sum(tf.multiply(n_step_target_values, 
                                                                     tf.one_hot(n_step_optimal_actions, n_step_target_values.shape[1])), axis = 1)
            
            # compute the new target action values using the Bellman-equation
            n_step_target_action_values = self.n_step_return_batch + (1 - self.n_done_batch) * (self.discount_factor)**(self.n_step) * n_step_optimal_action_values
            
        return(target_action_values, n_step_target_action_values)
        
               
    def _choose_memory(self):
        memory_weight = self.memory._priority_tree.get_total_weight()
        expert_weight = self.expert_memory._priority_tree.get_total_weight()
        return(np.random.binomial(size = 1, n = 1, p = expert_weight/(expert_weight + memory_weight)) == 1)
        
    
    def _batch_update(self, pretrain = False):
        # decide, from which memory to sample
        expert = False
        if self.expert_memory is not None:
            expert = self._choose_memory()
        if pretrain and self.expert_memory is not None:
            expert = True
        # sample mini batch
        self._get_mini_batch(expert)
        # compute target values
        target_action_values, n_step_target_action_values = self._get_target_action_values()
        # train the policy network using the target values and obtain the loss
        mean_loss, losses = self.policy_network.train(self.state_1_batch, self.action_batch, target_action_values, n_step_target_action_values, expert, self.batch_weights)
        self._losses.append(mean_loss)
        # update priorities according to losses
        if self.prioritized_replay:
            if expert:
                self.expert_memory.update_mini_batch_priorities(losses.numpy())
            else:
                self.memory.update_mini_batch_priorities(losses.numpy())
            
          
    def _predict_current_q_values(self):
        self._current_predictions = self.policy_network.predict(self._current_state)[0]
            

            
        
class EpsilonGreedyAgent(BaseQAgent):
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None,#DeepQNetwork(),
                 target_network = None,#DeepQNetwork(),
                 frame_shape = (84, 84),
                 save_path = None,
                 
                 num_actions = 4,
                 epsilon = 0.05):
        
        BaseQAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path)
        self.epsilon = epsilon
        self.num_actions = num_actions
            
    
    def _make_decision(self):
        if rand() < self.epsilon:
            return(randint(self.num_actions))
        else:
            self._predict_current_q_values()
            print(self._current_predictions)
            return(argmax(self._current_predictions, axis = 1))
        
        
        
class EpsilonAnnealingAgent(BaseQAgent):
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None,#DeepQNetwork(),
                 target_network = None,#DeepQNetwork(),
                 frame_shape = (84, 84),
                 save_path = None,
                 
                 num_actions = 4,
                 eps_schedule = [[1.0, 0.1, 1000000],
                                 [0.1, 0.001, 5000000]]):
        
        BaseQAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path)
        
        self.eps_schedule = np.array(eps_schedule)
        self.eps_schedule[:,2] = np.cumsum(self.eps_schedule[:,2])
        self.eps_lag = 0
        self.num_actions = num_actions
        
    
    def _get_current_epsilon(self):
        if self._step_counter > self.eps_schedule[0, 2] and self.eps_schedule.shape[0] > 1:
            self.eps_schedule = np.delete(self.eps_schedule, 0, 0)
            self.eps_lag = self._step_counter
        max_eps, min_eps, eps_steps = self.eps_schedule[0]
        epsilon = max_eps - min(1, (self._step_counter - self.eps_lag) / (eps_steps - self.eps_lag)) * (max_eps - min_eps)
        return(epsilon)
    
    def _make_decision(self, test_eps = None):
        if test_eps is not None:
            epsilon = test_eps
        else:
            epsilon = self._get_current_epsilon()
        
        if rand() < epsilon:
            return(randint(self.num_actions))
        else:
            self._predict_current_q_values()
            action = int(argmax(self._current_predictions))
            self._q_values.append(self._current_predictions[action])
            return(action)
        
        
        
class EpsGreedyDQNAgent(EpsilonGreedyAgent, DQNAgent):
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None,#DeepQNetwork(),
                 target_network = None, #DeepQNetwork(),
                 frame_shape = (84, 84),
                 save_path = None,
                
                 discount_factor = 0.99,
                 n_step = None,
                 double_q = False,
                 expert_memory = None,
                 prioritized_replay = False,
                
                 num_actions = 4,
                 epsilon = 0.05):
        
        EpsilonGreedyAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path, num_actions, epsilon)
        DQNAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path, discount_factor, n_step, double_q, expert_memory, prioritized_replay)
        
        
        
class EpsAnnDQNAgent(EpsilonAnnealingAgent, DQNAgent):
    
    def __init__(self,
                 env = ProcessedAtariEnv(),
                 memory = PrioritizedExperienceReplay(), 
                 policy_network = None, #DeepQNetwork(),
                 target_network = None, #DeepQNetwork(),
                 frame_shape = (84, 84),
                 save_path = None,
                
                 discount_factor = 0.99,
                 n_step = None,
                 double_q = False,
                 expert_memory = None,
                 prioritized_replay = False,
                
                 num_actions = 4,
                 eps_schedule = [[1.0, 0.1, 1000000],
                                 [0.1, 0.001, 5000000]]
                 ):
        
        EpsilonAnnealingAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path, num_actions, eps_schedule)
        DQNAgent.__init__(self, env, memory, policy_network, target_network, frame_shape, save_path, discount_factor, n_step, double_q, expert_memory, prioritized_replay)
