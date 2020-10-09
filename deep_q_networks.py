import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras import initializers
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Huber, MSE, MAE
import numpy as np

               
        
class DeepQNetwork:
    """
    ******************
    ** DeepQNetwork **
    ******************

        Class for the representation of a Deep Q-Network (DQN),
        which can be trained by a combination of one-step loss, n-step loss, 
        expert demonstration loss and l2 loss.
        
        -----------
        Parameters:
        -----------
            conv_layers:        dict; 
                                dictionary specifying the structure of the used convolutional layers of the DQN

            dense_layers:       dict; 
                                dictionary specifying the structure of the used fully connected layers of the DQN

            in_shape:           tuple; 
                                the shape of the input layer of the DQN

            num_actions:        int; 
                                the number of possible actions to choose (defining the shape of the output layer of the DQN)

            dueling:            bool; 
                                variable indicating whether to use a dueling neural network structure

            optimizer:          callable; 
                                the Keras optimizer used for applying a gradient descent step to the network weights

            lr_schedule:        list of lists of the form [start_value, end_value, num_steps]; 
                                schedule for (piecewise) linearly decreasing the learning rate

            one_step_loss_fn:   callable;
                                function for computing the one-step loss

            n_step_loss_fn:     callable;
                                function for computing the n-step loss

            expert_loss_fn:     callable;
                                function for computing the large margin classfication loss

            one_step_weight:    float;
                                the weight of the one-step loss in the computation of the total loss

            n_step_weight:      float;
                                the weight of the n-step loss in the computation of the total loss

            expert_weight:      float;
                                the weight of the large margin classfication loss in the computation of the total loss

            l2_weight:          float;
                                the weight of the l2 loss in the computation of the total loss

            large_margin_coeff: float;
                                the margin for the computation of the large margin classification loss
    """
    
    def __init__(self, 
                 conv_layers = {'filters': [32, 64, 64],#, 1024],
                                'kernel_sizes': [8, 4, 3],#, 7],
                                'strides': [4, 2, 1],#, 1],
                                'paddings': ['valid' for _ in range(3)],
                                'activations': ['relu' for _ in range(3)],
                                'initializers': [initializers.VarianceScaling(scale = 2.0) for _ in range(3)],
                                'names': ['conv_%i'%(i) for i in range(1,4)]
                               },
                 dense_layers = {'units': [512],
                                 'activations': ['relu'],
                                 'initializers': [initializers.VarianceScaling(scale = 2.0)],
                                 'names': ['dense_1']},
                 in_shape = (4, 84, 84),
                 num_actions = 6,
                 dueling = True,
                 optimizer = Adam,
                 lr_schedule = [[0.00025, 0.00005, 500000],
                                [0.00005, 0.00001, 1000000]],              
                 one_step_loss_fn = Huber(reduction=tf.keras.losses.Reduction.NONE), #lambda x, y: tf.math.square(x - y),
                 n_step_loss_fn = Huber(reduction=tf.keras.losses.Reduction.NONE), #lambda x, y: tf.math.square(x - y),
                 expert_loss_fn = lambda x, y: tf.math.abs(x - y),
                 one_step_weight = 1.0,
                 n_step_weight = 0.0,
                 expert_weight = 0.0,
                 l2_weight = 0.0,
                 large_margin_coeff = 0.8
                ):
        
        # define training parameters
        self.num_actions = num_actions
        
        self.optimizer = optimizer(learning_rate = lr_schedule[0][0])
        self.lr_schedule = np.array(lr_schedule)
        self.lr_schedule[:,2] = np.cumsum(self.lr_schedule[:,2])
        self.lr_lag = 0
        self._update_counter = 0
        
        # define loss functions
        self.one_step_loss_fn = one_step_loss_fn
        self.n_step_loss_fn = n_step_loss_fn
        self.expert_loss_fn = expert_loss_fn
        self.large_margin_coeff = large_margin_coeff
        
        # define loss weights
        self.one_step_weight = one_step_weight
        self.n_step_weight = n_step_weight
        self.expert_weight = expert_weight
        self.l2_weight = l2_weight
        
        # construct convolutional layers
        self.in_layer = Input(shape = in_shape, dtype = 'uint8', name = 'input_layer')
        self.scaled_in_layer = Lambda(lambda x: x/255.)(tf.transpose(tf.cast(self.in_layer, dtype = 'float32'), [0, 2, 3, 1]))
        
        self.conv_layers = []
        
        for layer_id in tf.range(len(conv_layers['filters'])):
            if layer_id == 0:
                conv_input = self.scaled_in_layer 
            else:
                conv_input = self.conv_layers[-1]
                
            self.conv_layers.append(Conv2D(filters = conv_layers['filters'][layer_id],
                                      kernel_size = conv_layers['kernel_sizes'][layer_id],
                                      strides = conv_layers['strides'][layer_id],
                                      padding = conv_layers['paddings'][layer_id],
                                      activation = conv_layers['activations'][layer_id],
                                      kernel_initializer = conv_layers['initializers'][layer_id],
                                      name = conv_layers['names'][layer_id],
                                      use_bias = False
                                     )(conv_input))

        # construct dueling architecture (if requested)
        if dueling:
            self.value_stream, self.advantage_stream = tf.split(self.conv_layers[-1], 2, 3)
            self.value_layer = Dense(units = 1,
                                     kernel_initializer = initializers.VarianceScaling(scale = 2.0),
                                     name = 'value_layer'
                                    )(Flatten()(self.value_stream))
            self.advantage_layer = Dense(units = self.num_actions,
                                         kernel_initializer = initializers.VarianceScaling(scale = 2.0),
                                         name = 'advantage_layer'
                                        )(Flatten()(self.advantage_stream))

            self.out_layer = self.value_layer + tf.math.subtract(self.advantage_layer, tf.reduce_mean(self.advantage_layer, axis = 1, keepdims = True))
            
        # construct final dense layers if requested
        else:
            self.dense_layers = [Flatten()(self.conv_layers[-1])]
            if dense_layers is not None:    
                for layer_id in tf.range(1, len(dense_layers['units'])):
                    dense_input = self.dense_layers[-1]    
                    self.dense_layers.append(Dense(units = dense_layers['units'][layer_id],
                                               activation = dense_layers['activations'][layer_id],
                                               kernel_initializer = dense_layers['initializers'][layer_id],
                                               name = dense_layers['names'][layer_id]
                                              )(dense_input))
                
            self.out_layer = Dense(units = self.num_actions,
                                   kernel_initializer = initializers.VarianceScaling(scale = 2.0),
                                   name = 'out_layer'
                                  )(self.dense_layers[-1])
        
        # define the model
        self.model = Model(inputs = [self.in_layer], outputs = [self.out_layer])
        

    def _get_current_lr(self):
        if self._update_counter > self.lr_schedule[0, 2] and self.lr_schedule.shape[0] > 1:
            self.lr_schedule = np.delete(self.lr_schedule, 0, 0)
            self.lr_lag = self._update_counter
        max_lr, min_lr, lr_steps = self.lr_schedule[0]
        lr = max_lr - min(1, (self._update_counter - self.lr_lag) / (lr_steps - self.lr_lag)) * (max_lr - min_lr)
        return(lr)

        
    def _update_lr(self):
        self.optimizer._set_hyper('learning_rate', self._get_current_lr())
    
    
    @tf.function
    def predict(self, states):
        """Perform a forward pass through the network"""
        predictions = self.model(states, training = False)
        return(predictions)
    
    @tf.function
    def get_optimal_actions(self, states):
        """Get the optimal actions for some states corresponding 
           to the current policy defined by the network parameters"""
        return(tf.math.argmax(self.predict(states), axis = 1))
    
    @tf.function
    def train(self, states, chosen_actions, target_action_values, n_step_target_action_values = None, expert = False, batch_weights = 1):
        """Perform a network parameter update for the specified 
           mini-batch of states, chosen actions and target values"""
        self._update_counter += 1
        self._update_lr()
        with tf.GradientTape() as tape:
            tape.watch(self.model.trainable_weights)
            predictions = self.model(states, training = False)
            chosen_action_values = tf.reduce_sum(tf.multiply(predictions, tf.one_hot(chosen_actions, self.num_actions, dtype = 'float32')), axis = 1)

            # one-step temporal difference loss
            losses = self.one_step_weight * self.one_step_loss_fn(chosen_action_values, target_action_values)
            
            # n-step temporal difference loss
            if n_step_target_action_values is not None:
                losses += self.n_step_weight * self.n_step_loss_fn(chosen_action_values, n_step_target_action_values)

            # large margin classification loss
            if expert:
                large_margin_targets = tf.reduce_max(predictions + tf.one_hot(chosen_actions, self.num_actions, on_value = 0., off_value = self.large_margin_coeff, dtype = 'float32'), axis = 1)
                losses += self.expert_weight * self.expert_loss_fn(chosen_action_values, large_margin_targets)

            # L2-regularization loss
            if self.l2_weight > 0:
                losses += self.l2_weight * tf.reduce_sum([tf.reduce_sum(tf.square(layer_weights)) for layer_weights in self.model.trainable_weights])
            
            
            mean_loss = tf.reduce_mean(losses * batch_weights)
            
        gradients = tape.gradient(mean_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return(mean_loss, losses)
