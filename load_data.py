# load_data.py

import urllib.request
import tempfile
import zipfile
import tarfile
import os
import numpy as np
import pandas as pd
import cv2
from experience_replay import PrioritizedExperienceReplay
from atari_preprocessing import atari_montezuma_processor


class LoadAtariHeadData:
    """
        ***********************
        ** LoadAtariHeadData **
        ***********************

        Class for loading the dataset of demonstrations in Atari 2600 games 
        described in 'Atari-HEAD: Atari Human Eye-Tracking and Demonstration Dataset' (Zhang et al. 2019)


        -----------
        Parameters:
        -----------
            game_name:        string; 
                              the name of the Atari 2600 game for which the demonstrations will be loaded

            archive_dir:      string; 
                              the directory to store and load the demonstration data

            frame_processor:  callable; 
                              function for processing the raw demonstration frames

            reward_processor: callable; 
                              function for processing the raw demonstration rewards
    """
    def __init__(self, game_name = 'montezuma_revenge', 
                 archive_dir = "AtariHEADArchives/",
                 atari_head_url = "https://zenodo.org/record/3451402/files/",
                 frame_processor = atari_montezuma_processor,
                 reward_processor = lambda x: np.sign(x)):
        
        self.game_name = game_name
        self.archive_dir = archive_dir
        self.frame_processor = frame_processor
        self.reward_processor = reward_processor
        self._zipfile_loc = self.archive_dir + self.game_name + ".zip"
        self._zipfile_url = atari_head_url + self.game_name + ".zip"
        self._act_rew_df = None
        self._png_name_df = None

        
    def _check_archive_dir(self):
        if not os.path.exists(self.archive_dir):
            print("\n %s will be added to the current directory as it does not exist yet." %(self.archive_dir))
            os.makedirs(self.archive_dir)
    
    def _check_game_archive(self):
        self._check_archive_dir()
        if not os.path.exists(self._zipfile_loc):
            print("\n %s will be downloaded from %s as it does not exist in %s." %(self.game_name + ".zip",
                                                                               self._zipfile_url,
                                                                               self.archive_dir))
            urllib.request.urlretrieve(self._zipfile_url, self._zipfile_loc)
    
    
    def _update_act_rew_df(self, txtfilepath):
        # read in the data from the given txtfilepath and convert it to a dataframe indexed by the frame_id
        series = pd.read_csv(txtfilepath, sep = '\n', squeeze = True)
        df = series.str.split(pat = ',', expand = True, n = 6)
        df.columns = series.name.split(',')
        df.set_index('frame_id', inplace = True)
        
        # remove columns, which won't be needed in this work
        df.drop(['episode_id', 'score', 'duration(ms)', 'gaze_positions'], axis = 1, inplace = True)
        
        # remove frames with unspecified actions or rewards
        df.replace(to_replace = 'null', value = np.nan, inplace = True)
        df.dropna(inplace = True)
        
        # append df to _act_rew_df
        self._act_rew_df = pd.concat([self._act_rew_df, df])
        
    
    def _update_png_name_df(self, namelist):
        namelist.sort(key = lambda s: (len(s), s)) 
        frame_ids = []
        png_names = []
        for png_name in namelist:
            if png_name.endswith('.png'):
                png_names.append(png_name)
                frame_ids.append(png_name.split('/')[-1].split('.')[0])
        
        self._png_name_df = pd.concat([self._png_name_df, pd.DataFrame({'png_names': png_names}, index = frame_ids)])
        
        
    def _get_frame_array(self, png_names, extract_dir, frame_shape = (84, 84)):
        frames = np.zeros(shape=(png_names.shape[0], *frame_shape), dtype = np.uint8)
        print('\n %i frames will be processed:' %(png_names.shape[0]))
        for i in range(png_names.shape[0]):
            # read image
            im = cv2.imread(extract_dir.name + "/" + png_names[i]).astype(np.uint8)
            # store processed image in frames
            frames[i] = self.frame_processor(im)
            if i%10000 == 0:
                print('%i done' %(i))
        return(frames)
    
    
    def _get_episode_endings(self, frames):
        ref_frames = (frames == frames[0]).all(axis=(1, 2))
        episode_endings = np.append((ref_frames[1:].astype(np.int) - 
                                     ref_frames[:-1].astype(np.int)) == 1, True)
        episode_endings[-1] = True
        return(episode_endings.astype(np.uint8))
    
    
    def _project_actions(self, actions):
        if self.game_name == 'breakout':
            # convert up to noop
            actions[actions == 2] = 0
            # convert down to noop
            actions[actions == 5] = 0
            # convert upright to right
            actions[actions == 6] = 3
            # convert upleft to left
            actions[actions == 7] = 4
            # convert downright to right
            actions[actions == 8] = 3
            # convert downleft to left
            actions[actions == 9] = 4
            # convert upfire to fire
            actions[actions == 10] = 1
            # convert rightfire to right
            actions[actions == 11] = 3
            # convert leftfire to left
            actions[actions == 12] = 4
            # convert downfire to fire
            actions[actions == 13] = 1
            # convert uprightfire to right
            actions[actions == 14] = 3
            # convert upleftfire to left
            actions[actions == 15] = 4
            # convert downrightfire to right
            actions[actions == 16] = 3
            # convert downleftfire to left
            actions[actions == 17] = 4
            # set index of right to 2 
            actions[actions == 3] = 2
            # set index of left to 3
            actions[actions == 4] = 3    
            
        if self.game_name == 'freeway':
            # convert fire to noop
            actions[actions == 1] = 0
            # convert right to noop
            actions[actions == 3] = 0
            # convert left to noop
            actions[actions == 4] = 0
            # convert upright to up
            actions[actions == 6] = 2
            # convert upleft to up
            actions[actions == 7] = 2
            # convert downright to down
            actions[actions == 8] = 5
            # convert downleft to down
            actions[actions == 9] = 5
            # convert upfire to up
            actions[actions == 10] = 2
            # convert rightfire to noop
            actions[actions == 11] = 0
            # convert leftfire to noop
            actions[actions == 12] = 0
            # convert downfire to down
            actions[actions == 13] = 5
            # convert uprightfire to up
            actions[actions == 14] = 2
            # convert upleftfire to up
            actions[actions == 15] = 2
            # convert downrightfire to down
            actions[actions == 16] = 5
            # convert downleftfire to down
            actions[actions == 17] = 5
            # set index of up to 1 
            actions[actions == 2] = 1
            # set index of down to 2
            actions[actions == 5] = 2

        if self.game_name == 'ms_pacman':
            for i in range(9):
                actions[actions == i+1] = i
                
            for i in range(10, 18):
                actions[actions == i] = i-9
            
        return(actions)
            
        
    def _skip_frames(self, frames, actions, rewards, episode_endings, frame_skip):
        n = frames.shape[0] + 1
        augmented_frames = frames[0:(n- frame_skip)][None]
        augmented_actions = actions[0:(n - frame_skip)][None]
        augmented_rewards = rewards[0:(n - frame_skip)][None]
        augmented_episode_endings = episode_endings[0:(n - frame_skip)][None]
        for i in range(1, frame_skip):
            augmented_frames = np.concatenate((augmented_frames, frames[i:(n - frame_skip + i)][None]), axis = 0)
            augmented_actions = np.concatenate((augmented_actions, actions[i:(n - frame_skip + i)][None]), axis = 0)
            augmented_rewards = np.concatenate((augmented_rewards, rewards[i:(n - frame_skip + i)][None]), axis = 0)
            augmented_episode_endings = np.concatenate((augmented_episode_endings, episode_endings[i:(n - frame_skip + i)][None]), axis = 0)
            
        reduced_frames = augmented_frames[-1]#np.amax(augmented_frames[-2:], axis = 0)
        reduced_actions = augmented_actions[-1]
        reduced_rewards = np.sum(augmented_rewards, axis = 0)
        reduced_episode_endings = np.amax(augmented_episode_endings, axis = 0)
        
        new_frames = reduced_frames[::frame_skip]
        new_actions = reduced_actions[::frame_skip]
        new_rewards = reduced_rewards[::frame_skip]
        new_episode_endings = reduced_episode_endings[::frame_skip]
        for i in range(1, frame_skip):
            new_frames = np.concatenate((new_frames, reduced_frames[i:][::frame_skip]), axis = 0)
            new_actions = np.concatenate((new_actions, reduced_actions[i:][::frame_skip]), axis = 0)
            new_rewards = np.concatenate((new_rewards, reduced_rewards[i:][::frame_skip]), axis = 0)
            new_episode_endings = np.concatenate((new_episode_endings, reduced_episode_endings[i:][::frame_skip]), axis = 0)
            
        return(new_frames, new_actions, new_rewards, new_episode_endings)
    
    
    def _save_demonstrations(self, frames, actions, rewards, episode_endings):
        np.save(self.archive_dir + self.game_name + "_frames", frames)
        np.save(self.archive_dir + self.game_name + "_actions", actions)
        np.save(self.archive_dir + self.game_name + "_rewards", rewards)
        np.save(self.archive_dir + self.game_name + "_episode_endings", episode_endings)
        print("\n Preprocessed demonstrations for the game %s have been saved to the directory %s"
              %(self.game_name, self.archive_dir))
        
    
    def _load_demonstrations(self):
        frames = np.load(self.archive_dir + self.game_name + "_frames.npy")
        actions = np.load(self.archive_dir + self.game_name + "_actions.npy")
        rewards = np.load(self.archive_dir + self.game_name + "_rewards.npy")
        episode_endings = np.load(self.archive_dir + self.game_name + "_episode_endings.npy")
        return(frames, actions, rewards, episode_endings)
    
    
    def get_demonstrations(self, frame_shape = (84, 84), recompute_demonstrations = False, only_highscore = False, exclude_highscore = True, frame_skip = 4):
        """Return demonstration data in the form: (frames, actions, rewards, episode_endings)"""
        # check if demonstrations already exist and load them if they do exist
        if (os.path.exists(self.archive_dir + self.game_name + "_frames.npy") and
            os.path.exists(self.archive_dir + self.game_name + "_actions.npy") and
            os.path.exists(self.archive_dir + self.game_name + "_rewards.npy") and
            os.path.exists(self.archive_dir + self.game_name + "_episode_endings.npy") and 
            not recompute_demonstrations):
            
            frames, actions, rewards, episode_endings = self._load_demonstrations()
        
        # if no demonstrations exist, they are loaded from a zip archive, 
        # which either already exists or will be loaded from the internet otherwise
        else:
            self._check_game_archive()
            extract_dir = tempfile.TemporaryDirectory()
            with zipfile.ZipFile(self._zipfile_loc, 'r') as zip_archive:
                zip_archive.extractall(extract_dir.name)
                print("\n %i files have been extracted from %s to a temporary dircetory and will now be processed:" %(len(zip_archive.namelist()), self._zipfile_loc))
                for filename in zip_archive.namelist():
                    if ((exclude_highscore and 'highscore' not in filename) or 
                        (only_highscore and 'highscore' in filename) or 
                        (not exclude_highscore and not only_highscore)):
                        if filename.endswith('.txt'):
                            self._update_act_rew_df(extract_dir.name + "/" + filename)

                        elif filename.endswith('.tar.bz2'):
                            with tarfile.open(extract_dir.name + "/" + filename, 'r') as tar_archive:
                                tar_archive.extractall(extract_dir.name)
                                self._update_png_name_df(tar_archive.getnames())
                        print('%s has been processed.' %(filename))
                        

            merged_df = pd.merge(self._act_rew_df, self._png_name_df, left_index=True, right_index=True)
            
            frames = self._get_frame_array(merged_df['png_names'].values, extract_dir, frame_shape)
            actions = self._project_actions(merged_df['action'].values.astype(np.intc))
            rewards = self.reward_processor(merged_df['unclipped_reward'].values.astype(np.single))
            episode_endings = self._get_episode_endings(frames)
            if frame_skip is not None:
                frames, actions, rewards, episode_endings = self._skip_frames(frames, actions, rewards, episode_endings, frame_skip)
            self._save_demonstrations(frames, actions, rewards, episode_endings)
            extract_dir.cleanup() 
            
        return(frames, actions, rewards, episode_endings)
    
    def demonstrations_to_per(self, 
                              max_frame_num = 2**20,
                              num_stacked_frames = 4,
                              frame_shape = (84, 84),
                              priority_dtype = np.single,
                              batch_size = 32,
                              prio_coeff = 0.0,
                              is_schedule = [0.4, 1.0, 5000000],
                              epsilon = 0.0001,
                              recompute_demonstrations = False,
                              only_highscore = False,
                              exclude_highscore = True,
                              frame_skip = 4):
        """Load demonstration data and return an instance of PrioritizedExperienceReplay,
        initialized with the demonstration data."""
        
        # get demonstrations
        frames, actions, rewards, episode_endings = self.get_demonstrations(frame_shape, recompute_demonstrations, only_highscore, exclude_highscore, frame_skip)
        
        # set all priorities of demonstrations to 1
        priorities = np.ones(actions.shape[0], dtype = priority_dtype)
        
        # iniatialize PrioritizedExperienceReplay object with the demonstrations
        replay_memory = PrioritizedExperienceReplay(max_frame_num = max_frame_num, 
                                                    num_stacked_frames = num_stacked_frames,
                                                    frame_shape = frame_shape,
                                                    frames = frames,
                                                    actions = actions,
                                                    rewards = rewards,
                                                    priorities = priorities, 
                                                    episode_endings = episode_endings,
                                                    priority_dtype = priority_dtype, 
                                                    batch_size = batch_size,
                                                    prio_coeff = prio_coeff,
                                                    is_schedule = is_schedule,
                                                    epsilon = epsilon)
        return(replay_memory)
        
