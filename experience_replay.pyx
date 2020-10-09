# cython: language_level=3

import numpy as np
from libc.stdint cimport uint8_t
import pickle
import os

     
cdef class BinarySumTree:
    """
    ******************* 
    ** BinarySumTree **
    *******************
    
        Class for representing a (perfect) binary sum tree, i.e. a tree, 
        in which every node (except for the leaves) has got exactly two child nodes and
        the key of every node (except for the leaves) is equal to the sum of the keys of its child nodes.
        
        The tree is represented by an array, which is indexed from the leaves towards the root and from left to right.
        
        For example, a tree with 4 leaves is indexed in the following way:

        |            6           | (root)
        |         /     \        |
        |       4         5      |
        |     /   \     /   \    |
        |    0     1   2     3   | (leaves)
                                        
        
        In this setting the parent node, the left child and the right child node 
        of a node with index i in a tree with n leaves can be accessed in the following way:
            
            parent(i) = floor(i / 2) + n
            left_child(i) = 2 * (i - n)
            right_child(i) = 2 * (i - n) + 1
    
        -----------
        Parameters:
        -----------
            leaf_number: int; number of leaves (has to be a power of 2 as a perfect binary tree is expected)
            keys:        (memory view of) array of floats; array containing the keys of the binary sum tree
            
    
    """
    
    
    cdef public:
        unsigned int leaf_number
        float[::1] keys
        unsigned int sample_size
        unsigned int[::1] sampled_indices
        float[::1] _sampled_keys   
        unsigned int _height
        unsigned int _root_id
    
    def __cinit__(self, leaf_number = 2**20, leaf_keys = None, sample_size = 32, key_dtype = np.single):
        """Initialize parameters"""

        self.leaf_number = leaf_number
        self._height = int(np.log2(self.leaf_number) + 1)
        self._root_id = 2 * self.leaf_number - 2
        self.keys = np.zeros(2 * self.leaf_number - 1, dtype = key_dtype)
        if leaf_keys is not None: 
            self._construct(leaf_keys)
        self.sample_size = sample_size
        self.sampled_indices = np.zeros(sample_size, dtype = np.uintc)
        self._sampled_keys = np.single(self.keys[self._root_id] *  np.random.random(sample_size))

            
    def _construct(self, float[::1] leaf_keys):
        """Construct a binary sum tree with the given leaf_keys.
           For details on the tree structure, see the class description."""
        
        cdef unsigned int level, first_1, last_1, first_2, last_2
        cdef float[::1] level_view
        first_1 = 0
        last_1 = self.leaf_number - 1
        
        self.keys[first_1:(last_1)+1] = leaf_keys
        for level in range(1, self._height):
            # compute the first and last index of the current level
            first_2 = last_1 + 1 
            last_2 = first_2 + int(self.leaf_number / 2**(level)) - 1
            # compute the keys of the current level
            level_view = np.add(self.keys[first_1:(last_1 + 1):2], self.keys[(first_1 + 1):(last_1 + 1):2])
            self.keys[first_2:(last_2 + 1)] = level_view
            # update the level indices
            first_1 = first_2
            last_1 = last_2
    
    
    cdef unsigned int _parent(self, unsigned int index):
        """Return the parent index of the given index"""
        return(self.leaf_number + index // 2)
    
    
    cdef unsigned int _left(self, unsigned int index):
        """Return the left child index of the given index"""
        return(2 * (index - self.leaf_number))
    
    
    cdef unsigned int _right(self, unsigned int index):
        """Return the right child index of the given index"""
        return(2 * (index - self.leaf_number) + 1)
    
    
    cdef unsigned int _get_index_by_key(self, float key):
        """"Get the index such that: sum(self.keys[0:(index-1)]) < key <= sum(self.keys[0:index])"""
    
        cdef unsigned int index = self._root_id
        cdef unsigned int left
        while index >= self.leaf_number:
            left = self._left(index)
            if self.keys[left] >= key:
                index = left
            else: 
                index = left + 1 
                key = key - self.keys[left]
        return(index)
    
    
    def sample_indices(self, unsigned int sample_size = 0):
        """Sample a given number of (leaf-)indices with probabilty proportional to their keys."""
        
        if sample_size > 0:
            self.sample_size = sample_size
            self._sampled_keys = np.single(self.keys[self._root_id] *  np.random.random(sample_size))
            self.sampled_indices = np.zeros(sample_size, dtype = np.uintc)
                                           
        # sample a number of sample_size random keys out of the interval [0, self.keys[self._root_id]]
        self._sampled_keys = np.single(self.keys[self._root_id] *  np.random.random(self.sample_size))
        cdef unsigned int i = 0
        for i in range(self.sample_size):
            self.sampled_indices[i] = self._get_index_by_key(self._sampled_keys[i])
        #return(indices)
    
    
    def set_key(self, unsigned int index, float new_key):
        "Set the key at the given index to the given new key."

        # set the key of the given index to the new_key and calculate the change of the key value at this index 
        cdef float change = new_key - self.keys[index]
        self.keys[index] = new_key
        
        # propagate the change iteratively up to the root
        cdef unsigned int parent = index
        while parent != self._root_id:
            parent = self._parent(parent)
            self.keys[parent] += change
            
            
    def get_total_weight(self):
        return(self.keys[-1])
    
    
    
    
cdef class PrioritizedExperienceReplay:
    
    """
        ********************************* 
        ** PrioritizedExperienceReplay **
        *********************************
       
        Class for the representation of the prioritized experience replay memory in Deep Q-Learning algorithms.
        The class is maintaining experience-tuples of a Q-Learning agent of the following form:
        
               exp := (state_1, action, reward, state_2),
        
        where state_1 and state_2 consist of a specific number of stacked frames.
        
        Example: Let state_1 := (frame_1, frame_2,...,frame_k). 
                     state_2 := (frame_2, frame_3,...,frame_k+1)
        
        ==> storing state_1 and state_2 explicitly would be inefficient,
            as they have most of their frames in common
        ==> only the frames will be stored and the index of one experience-tuple is set to be
            equal to the index of the most recent frame of state_1 (in the example: frame_k)
       
        -----------
        Parameters:
        -----------
            max_frame_num:      int; 
                                maximum number of frames being stored in the replay memory
            
            num_stacked_frames: int;
                                number of frames that are stacked for one state 
                            
            frame_shape:        tuple of two ints; 
                                shape of one single frame
                            
            frames:             (memory view of) 3d-array of unsigned ints; 
                                array of all frames in the replay memory
                            
            actions:            (memory view of) 1d-array of ints; 
                                array of actions, which have been taken in the corresponding frames
            
            rewards:            (memory view of) 1d-array of floats; 
                                array of rewards, which have been observed in the corresponding frames
                            
            priorities:         (memory view of) 1d-array of floats; 
                                array of priorities for the corresponding frames
                            
            episode_endings:    1d-array of unsigned ints; 
                                array indicating the terminal frame of all episodes 
            
            batch_size:         int;
                                size of the mini-batch, which will be used for training the agent

            prio_coeff:         float;
                                coefficient of prioritization 
                                (alpha in 'Prioritized Experience Replay', Schaul et al. 2015)

            is_schedule:        list of lists of the form [start_value, end_value, num_steps]; 
                                schedule for (piecewise) linearly annealing the importance sampling coefficient 
                                (beta in 'Prioritized Experience Replay', Schaul et al. 2015)

            epsilon:            float;
                                constant beeing added to each priority to ensure non-zero priorities

            restore_path:       string;
                                if restore_path is not none, the internal state of the replay memory will be restored from a previous training session
    """
    
    
    cdef public:
        unsigned int max_frame_num
        unsigned int num_stacked_frames
        uint8_t[:, :, ::1] frames
        int[::1] actions
        float[::1] rewards
        unsigned int batch_size
        float prio_coeff
        float is_min_coeff
        float is_max_coeff
        float is_steps
        float epsilon
        object _priority_tree
        uint8_t[:, :, :, ::1] _batch_frames
        int[::1] _batch_actions
        float[::1] _batch_rewards
        float[::1] _batch_n_step_returns
        uint8_t[::1] _batch_dones
        uint8_t[::1] _batch_n_dones
        float[::1] _batch_weights
        int _current_index
        bint _is_full
        int _start_new_episode
        int _sample_counter
    
    
    def __cinit__(self, max_frame_num = 2**20, 
                  num_stacked_frames = 4,
                  frame_shape = (84, 84),
                  frames = None,
                  actions = None,
                  rewards = None,
                  priorities = None, 
                  episode_endings = None,
                  priority_dtype = np.single, 
                  batch_size = 32,
                  prio_coeff = 1.0,
                  is_schedule = [0.4, 1.0, 5000000],
                  epsilon = 0.0001,
                  restore_path = None):
        
        """Initialize parameters"""
        self.max_frame_num = max_frame_num
        self.num_stacked_frames = num_stacked_frames
        self.batch_size = batch_size
        self.prio_coeff = prio_coeff
        self.is_min_coeff = is_schedule[0]
        self.is_max_coeff = is_schedule[1]
        self.is_steps = is_schedule[2]
        self.epsilon = epsilon
        
        self.frames = np.zeros((max_frame_num, *frame_shape), dtype = np.uint8)
        self.actions = np.zeros(max_frame_num, dtype = np.intc)
        self.rewards = np.zeros(max_frame_num, dtype = np.single)
        
        self._priority_tree = BinarySumTree(leaf_number = max_frame_num, 
                                           leaf_keys = None, 
                                           key_dtype = priority_dtype,
                                           sample_size = batch_size)
        
        # allocate memory for a mini-batch
        self._batch_frames = np.zeros((batch_size, 2*num_stacked_frames+1, *frame_shape), dtype = np.uint8)
        self._batch_actions = np.zeros(batch_size, dtype = np.intc)
        self._batch_rewards = np.zeros(batch_size, dtype = np.single)
        self._batch_n_step_returns = np.zeros(batch_size, dtype = np.single)
        self._batch_dones = np.zeros(batch_size, dtype = np.uint8)
        self._batch_n_dones = np.zeros(batch_size, dtype = np.uint8)
        self._batch_weights = np.zeros(batch_size, dtype = np.single)
        
        # internal variables
        self._current_index = -1
        self._is_full = False
        self._start_new_episode = self.num_stacked_frames
        self._sample_counter = 0
        
        if (frames is not None and 
            actions is not None and 
            rewards is not None and 
            priorities is not None and
            episode_endings is not None):
            self._init_experiences(frames, actions, rewards, 
                                   priorities, episode_endings)
            
        if restore_path is not None:
            self._restore_internal_state(restore_path)
        
    
    
    def save_internal_state(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        memory_dict = {'frames': np.asarray(self.frames),
                       'actions': np.asarray(self.actions),
                       'rewards': np.asarray(self.rewards), 
                       'priorities': np.asarray(self._priority_tree.keys[:self.max_frame_num]),
                       'is_full': self._is_full,
                       'current_index': self._current_index}
        with open(save_path + "memory", 'wb') as file:
            pickle.dump(memory_dict, file)
    
    
    cdef void _restore_internal_state(self, restore_path):
        with open(restore_path, 'rb') as file:
            memory_dict = pickle.load(file)
            self.frames = memory_dict['frames']
            self.actions = memory_dict['actions']
            self.rewards = memory_dict['rewards']
            self._priority_tree._construct(memory_dict['priorities'])
            self._is_full = memory_dict['is_full']
            self._current_index = memory_dict['current_index']
    
    
    cdef void _init_experiences(self, uint8_t[:, :, ::1] frames, int[::1] actions, float[::1] rewards, 
                                float[::1] priorities, uint8_t[::1] episode_endings):
        cdef unsigned int index = 1
        self.add_experience(0, 0., frames[0], 0., episode_endings[0])
        for index in range(1, len(episode_endings)):
            self.add_experience(actions[index-1], rewards[index-1], frames[index], 
                                priorities[index-1], episode_endings[index])
            
            
    cdef void _set_batch_frames(self, unsigned int batch_index, int lower_index, unsigned int upper_index, bint n_step_frames):
        cdef unsigned int shift = 0
        if n_step_frames:
            shift = self.num_stacked_frames + 1
            
        # check, if both upper_index and lower_index are valid 
        if upper_index <= self.max_frame_num and lower_index >= 0:
            self._batch_frames[batch_index][shift:(self.num_stacked_frames + 1 + shift)] = self.frames[lower_index:upper_index]

        # if upper_index is invalid (i.e upper_index > self.max_frame_num),
        # then the frames range from the right to the left boundary of the memory  
        elif upper_index > self.max_frame_num and lower_index < int(self.max_frame_num):
            self._batch_frames[batch_index][shift:(self.max_frame_num - lower_index + shift)] = self.frames[lower_index:self.max_frame_num]
            self._batch_frames[batch_index][(self.max_frame_num - lower_index + shift):(self.num_stacked_frames + 1 + shift)] = self.frames[:(upper_index - self.max_frame_num)]
        
        # if both indices are out of the right range,
        # then the requested frames are located at the left boundary of the memory
        # (this case only occurs for the n-step mini-batch)         
        elif upper_index > self.max_frame_num and lower_index >= int(self.max_frame_num):
            self._batch_frames[batch_index][shift:(self.num_stacked_frames + 1 + shift)] = self.frames[(lower_index - self.max_frame_num):(upper_index - self.max_frame_num)]
                
        # if lower_index is invalid (i.e lower_index < 0),
        # then the frames range from the right to the left boundary of the memory  
        elif lower_index < 0:
            self._batch_frames[batch_index][shift:(-lower_index + shift)] = self.frames[lower_index:]
            self._batch_frames[batch_index][(-lower_index + shift):(self.num_stacked_frames + 1 + shift)] = self.frames[:upper_index]
       
    
    cdef int _move_index(self, unsigned int index, int amount):
        cdef int  new_index = index + amount
        if new_index >= int(self.max_frame_num):
            new_index -= self.max_frame_num
        elif new_index < 0:
            new_index += self.max_frame_num
        return new_index
    
    cdef void _set_n_step_returns(self, unsigned int start_index, unsigned int batch_index, unsigned int max_step_num, float discount_factor):
        self._batch_n_dones[batch_index] = 0        
        cdef unsigned int i = 0
        cdef float n_step_return = 0
        cdef int new_index = 0
        while i < max_step_num:
            new_index = self._move_index(start_index, i)
            if self._priority_tree.keys[new_index] == 0:
                self._batch_n_dones[batch_index] = 1
                break
            else:
                n_step_return += discount_factor**i * self.rewards[new_index]
                i += 1
                
        if self._priority_tree.keys[self._move_index(start_index, i)] == 0:
            self._batch_n_dones[batch_index] = 1
        self._batch_n_step_returns[batch_index] = n_step_return
        self._set_batch_frames(batch_index, start_index + i + 1 - self.num_stacked_frames, start_index + 1 + i, True)


    cdef void _get_batch_weights(self):
        # get current total weight and size of the replay memory
        cdef float current_size = self.max_frame_num if self._is_full else self._current_index
        cdef float current_total_weight = self._priority_tree.get_total_weight()

        # get current beta
        cdef float current_beta = min(self.is_max_coeff, self.is_min_coeff + self._sample_counter * (self.is_max_coeff - self.is_min_coeff) / self.is_steps)
        
        # compute sample weights of the samlped keys
        cdef float max_weight = 0.0000001
        cdef unsigned int i = 0
        for i in range(self.batch_size):
            self._batch_weights[i] = (current_size * self._priority_tree.keys[self._priority_tree.sampled_indices[i]] / current_total_weight + 0.0000001)**(-current_beta)
            max_weight = self._batch_weights[i] if self._batch_weights[i] > max_weight else max_weight

        # rescale weights
        i = 0
        for i in range(self.batch_size):
            self._batch_weights[i] = self._batch_weights[i] / max_weight
    
    
    def get_mini_batch(self, n_step_return = None):
        """Return a mini-batch of experiences (state_1, action, reward, state_2) with prioritization.
           
           ----------
           Parameter:
           ----------
               n_step_return: list of length 2;
                              the first entry specifies the number n of steps for computing the n-step returns
                              the second entry specifies the discount factor for computing the n-step returns,
                              default is None (in this case, no n-step returns will be computed)
                              
           -------
           Output:
           -------
               state_1: array of shape (batch_size, num_stacked_frames, *frame_shape);
                        mini-batch of first states
               
               actions: array of shape (batch_size);
                        mini-batch of actions taken after observing state_1
                        
               rewards: array of shape (batch_size);
                        mini-batch of rewards observed after taking the action specified in actions
               
               state_2: array of shape (batch_size, num_stacked_frames, *frame_shape);
                        mini-batch of states observed after taking the action specified in actions
                        
               dones:   array of shape (batch_size);
                        mini-batch of booleans indicating, whether state_2 is the terminal state of an episode

               weights: array of shape (batch_size);
                        mini-batch of importance sampling weights for the sampled experiences
                        
           --------------------------------------------------
           Output (additional, if n_step_return is not None):
           --------------------------------------------------
               n_step_returns: array of shape (batch_size);
                               mini-batch of discounted n-step returns
                               
               state_n: array of shape (batch_size, num_stacked_frames, *frame_shape);
                        mini-batch of states observed n steps after observing state_1
                        
               n_dones: array of shape (batch_size);
                        mini-batch of booleans indicating, whether state_n is the terminal state of an episode
        """
        # sample mini-batch indices
        self._priority_tree.sample_indices()
        self._sample_counter += 1
        
        # get experiences (state1, action, reward, state2) for the sampled indices
        cdef unsigned int i = 0
        cdef unsigned int sampled_index, upper_index
        cdef int lower_index
        for i in range(self.batch_size):
            sampled_index = self._priority_tree.sampled_indices[i]
            
            # check, if sampled_index corresponds to a terminal state
            if self._priority_tree.keys[self._move_index(sampled_index, 1)] == 0:
                self._batch_dones[i] = 1
            else:
                self._batch_dones[i] = 0
            
            # get actions and rewards corresponding to the sampled indices 
            self._batch_actions[i] = self.actions[sampled_index]
            self._batch_rewards[i] = self.rewards[sampled_index]
            
            # get frame indices corresponding to the sampled indices
            # (i.e. the frames with indices: lower_index,...,upper_index-1)
            lower_index = sampled_index + 1 - self.num_stacked_frames
            upper_index = sampled_index + 2 
            
            # get frames between lower and upper index and store them in _batch_frames
            self._set_batch_frames(i, lower_index, upper_index, False)
                
            # check, if n_step_return is None. If it is not None, it is assumed to be a list of 
            # an integer value specifying the maximum number of steps for the n-step return and
            # a float value specifying the discount factor
            if n_step_return is not None:
                self._set_n_step_returns(sampled_index, i, n_step_return[0], n_step_return[1])

        # get mini-batch importance sampling weights
        self._get_batch_weights()
                
        if n_step_return is None:
            return(np.asarray(self._batch_frames[:, :self.num_stacked_frames]), 
                   np.asarray(self._batch_actions), 
                   np.asarray(self._batch_rewards), 
                   np.asarray(self._batch_frames[:, 1:self.num_stacked_frames+1]),
                   np.asarray(self._batch_dones),
                   np.asarray(self._batch_weights))
        
        else:
            return(np.asarray(self._batch_frames[:, :self.num_stacked_frames]), 
                   np.asarray(self._batch_actions), 
                   np.asarray(self._batch_rewards), 
                   np.asarray(self._batch_frames[:, 1:self.num_stacked_frames+1]),
                   np.asarray(self._batch_dones),
                   np.asarray(self._batch_n_step_returns),
                   np.asarray(self._batch_frames[:, self.num_stacked_frames+1:]),
                   np.asarray(self._batch_n_dones),
                   np.asarray(self._batch_weights))
    
    
    cdef void _move_current_index(self):
        self._current_index += 1
        if self._current_index >= int(self.max_frame_num):
            self._is_full = True
            self._current_index -= self.max_frame_num
    
    
    cdef void _forget_experience(self):
        cdef unsigned int new_start_index = self._current_index + self.num_stacked_frames - 1
        if new_start_index >= self.max_frame_num:
            new_start_index -= self.max_frame_num
        self._priority_tree.set_key(new_start_index, 0)
    
    
    cdef void _set_new_frame(self, uint8_t[:, ::1] new_frame):
        self._move_current_index()
        if self._is_full:
            self._forget_experience()
        self.frames[self._current_index][:] = new_frame
        self._priority_tree.set_key(self._current_index, 0)
        
    
    def add_experience(self, int action, float reward, uint8_t[:, ::1] new_frame, float priority, bint episode_done = False):
        """Add an experience to the replay memory.

           -----------
           Parameters:
           -----------
               action:       int;
                             the action chosen in the considered experience

               reward:       float;
                             the reward received in the considered experience

               new_frame:    array of unsigned integers with shape equal to frame_shape;
                             the new frame observed in the considered experience

               priority:     float;
                             the priority of the considered experience

               episode_done: bool;
                             variable indicating whether the considered experience is the end of an episode

           
           Note: - if state_1 = (frame_1,frame_2,...,frame_k), where k = num_stacked_frames,
                   then state_2 = (frame_2,...,frame_k, new_frame)
                   ==> as frame_1,...,frame_k already exist in the memory,
                       it is only necessary to add new_frame to the memory
                 - as the first k-1 frames and the last frame of an episode can't be used 
                   for building a state, their priorities are set equal to zero
                 - if the memory is out of capacity the least recent experience will be overwritten
        """
        
        if self._start_new_episode == 0:
            self.actions[self._current_index] = action
            self.rewards[self._current_index] = reward
            self._priority_tree.set_key(self._current_index, (priority + self.epsilon)**(self.prio_coeff))
        else:    
            self._start_new_episode -= 1
        
        if episode_done: 
            self._start_new_episode = self.num_stacked_frames
            
        self._set_new_frame(new_frame)
        
        
    def update_mini_batch_priorities(self, float[::1] new_priorities):
        """Update the priorities of the recent mini-batch with the given new_priorities."""
        cdef unsigned int index = 0
        for index in range(self.batch_size):
            self._priority_tree.set_key(self._priority_tree.sampled_indices[index], new_priorities[index]**(self.prio_coeff))

            
        
