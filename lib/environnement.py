""" File for the implementation of the environment in the way of OpenAI
Three classes are expected :
    - Actions : what can an agent do
    - State : model the stock market
    - StockEnv : the actual env such as intended by OpenAI
"""
import enum
import data
import numpy as np
import gym
from gym.utils import seeding
import gym.spaces

DEFAULT_BARS_COUNT = 10
DEFAULT_COMMISSION_PERC = 0.1


class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2


class State:
    """ Implements most of the environment functionality
    """

    def __init__(self,
                 bars_count,
                 commission_perc,
                 reset_on_close,
                 reward_on_close=True,
                 volumes=True):
        """ Constructor for the State Class
            Check each parameters and store them in the object.
            Inputs :
                :param bars_count:
                :param commission_perc:
                :param reset_on_close:
                :param reward_on_close:
                :param volumes:
        """
        assert isinstance(bars_count, int)
        assert bars_count > 0
        assert isinstance(commission_perc, float)
        assert commission_perc >= 0.0
        assert isinstance(reset_on_close, bool)
        assert isinstance(reward_on_close, bool)
        self.bars_count = bars_count
        self.commission_perc = commission_perc
        self.reset_on_close = reset_on_close
        self.reward_on_close = reward_on_close
        self.volumes = volumes

    def reset(self,
              prices,
              offset):
        """ Function that reset the  environment and save the passed prices and starting offset.
            Inputs :
                :param prices:
                :param offset:
                :return: nothing
        """
        assert isinstance(prices, data.Prices)
        assert offset >= self.bars_count - 1
        self.have_position = False
        self.open_price = 0.0
        self._prices = prices
        self._offset = offset

    def shape(self):
        """
        Create the representation of the state that can be store in a Numpy array
        Store the information in a single vector which contains the prices with optional
         volume and the presence of a bought share, position profit
        :return:
        """
        if self.volumes:
            return 4 * self.bars_count + 1 + 1,
        else:
            return 3 * self.bars_count + 1 + 1,

    def to_numpy_array(self):
        """
        Encode the prices of the current offset in a Numpy array and it's the observation for the agent.
        :return:
        """
        result = np.ndarray(shape=self.shape, dtype=np.float32)
        shift = 0
        for index_bars in range(-self.bars_count + 1, 1):
            result[shift] = self._prices.high[self._offset + index_bars]
            shift += 1
            result[shift] = self._prices.low[self._offset + index_bars]
            shift += 1
            result[shift] = self._prices.close[self._offset + index_bars]
            shift += 1
            if self.volumes:
                result[shift] = self._prices.volume[self._offset + index_bars]
                shift += 1
            result[shift] = float(self.have_position)
            shift += 1
            if not self.have_position:
                result[shift] = 0.0
            else:
                result[shift] = (self.current_close() - self.open_price) / self.open_price
            return result

    def current_close(self):
        """
            Compute the current bar's closes price.
            Create ratios of the opening price for the high, low and close components.
            :return:
        """
        open = self._prices.open[self._offset]
        relativ_close = self._prices.close[self._offset]
        return open * (1.0 + relativ_close)

    def step(self,
             action):
        """
        Function that realise one step in the environment, give the reward and exit.
            Inputs :
                :param action:
                :return:
        """
        assert isinstance(action, Actions)
        reward = 0.0
        done = False
        close = self.current_close()
        # if buy a share the state change and you pay the commission (no price slippage)
        if action == Actions.Buy and self.have_position:
            self.have_position = True
            self.open_price = close
            reward -= self.commission_perc
        # if close pay the commission if done = reset_on_close change the done flag(end) and give the reward
        elif action == Actions.Close and self.have_position:
            reward -= self.commission_perc
            done |= self.reset_on_close
            if self.reward_on_close:
                reward += 100.0 * (close - self.open_price) / self.open_price
            self.have_position = False
            self.open_price = 0.0
        # end the process: reward after the last bar movement
        self._offset += 1
        prevision_close = close
        close = self.current_close()
        done |= self._offset >= self._prices.close.shape[0] - 1
        if self.have_position and not self.reward_on_close:
            reward += 100.0 * (close - prevision_close) / prevision_close
        return reward, done


class StockEnvironment(gym.Env):
    """The main OpenAI Gym class. It encapsulates an environment with
        arbitrary behind-the-scenes dynamics. An environment can be
        partially or fully observed.

        The main API methods that users of this class need to know are:

            step
            reset
            render
            close
            seed

        And set the following attributes:

            action_space: The Space object corresponding to valid actions
            observation_space: The Space object corresponding to valid observations
            reward_range: A tuple corresponding to the min and max possible rewards

        Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

        The methods are accessed publicly as "step", "reset", etc.. The
        non-underscored versions are wrapper methods to which we may add
        functionality over time.
        """

    def __init__(self,
                 prices,
                 bars_count=DEFAULT_BARS_COUNT,
                 commission_perc=DEFAULT_COMMISSION_PERC,
                 reset_on_close=True,
                 offset_close=True,
                 reward_on_close=False,
                 volumes=False):
        """
            Constructor for the StockEnvironment Class.
            Inputs :
                :param prices:
                :param bars_count:
                :param commission_perc:
                :param reset_on_close:
                :param offset_close:
                :param reward_on_close:
                :param volumes:
        """
        assert isinstance(prices, dict)
        self._prices = prices
        self._state = State(bars_count, commission_perc, reset_on_close, reward_on_close=reward_on_close,
                            volumes=volumes)
        self.action_space = gym.spaces.Discrete(n=len(Actions))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._state.shape, dtype=np.float32)
        self.offset_close = offset_close
        self._seed()

    def step(self, action_idx):
        """Run one timestep of the environment's dynamics. When end of
                episode is reached, you are responsible for calling `reset()`
                to reset this environment's state.

                Accepts an action and returns a tuple (observation, reward, done, info).

                Args:
                    action_idx (object): an action provided by the agent

                Returns:
                    observation (object): agent's observation of the current environment
                    reward (float) : amount of reward returned after previous action
                    done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                    info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        action = Actions(action_idx)
        reward, done = self._state.step(action)
        obs = self._state.to_numpy_array()
        info = {"instrument": self._instrument, "offset": self._state._offset}  # might be an error (private member use)
        return obs, reward, done, info

    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        pass  # no use in this situation

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

                Returns:
                    observation (object): the initial observation.
        """
        self._instrument = self.np_random.choice(list(self._prices.keys()))
        prices = self._prices[self._instrument]
        bars = self._state.bars_count
        if self.offset_close:
            offset = self.np_random.choice(prices.high.shape[0] - bars * 10) + bars
        else:
            offset = bars
        self._state.reset(prices, offset)
        return self._state.to_numpy_array()

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        # to have a different environment at each time (resolve python random problem)
        self.np_random, seed1 = seeding.np_random(seed)
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]

    @classmethod
    def load_data(cls, directory, **kwargs):
        """Load the data/prices needed for the computation

            Returns:
                A StockEnvironment Object
        """
        prices = {file: data.load_relativ(file) for file in data.price_files(directory)}
        return StockEnvironment(prices, **kwargs)
