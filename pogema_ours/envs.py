from typing import List, Optional

import numpy as np
import gymnasium
from gymnasium.error import ResetNeeded

from pogema_ours.grid import Grid, GridLifeLong, CooperativeGrid
from pogema_ours.grid_config import GridConfig
from pogema_ours.wrappers.metrics import LifeLongAverageThroughputMetric, NonDisappearEpLengthMetric, \
    NonDisappearCSRMetric, NonDisappearISRMetric, EpLengthMetric, ISRMetric, CSRMetric, OnGoalMetric
from pogema_ours.wrappers.multi_time_limit import MultiTimeLimit
from pogema_ours.generator import generate_new_target
from pogema_ours.wrappers.persistence import PersistentWrapper


class ActionsSampler:
    """
    Samples the random actions for the given number of agents using the given seed.
    """

    def __init__(self, num_actions, seed=42):
        self._num_actions = num_actions
        self._rnd = None
        self.update_seed(seed)

    def update_seed(self, seed=None):
        self._rnd = np.random.default_rng(seed)

    def sample_actions(self, dim=1):
        return self._rnd.integers(self._num_actions, size=dim)


class PogemaBase(gymnasium.Env):
    """
    Abstract class of the Pogema environment.
    """
    metadata = {"render_modes": ["ansi"], }

    def step(self, action):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None, ):
        raise NotImplementedError

    def __init__(self, grid_config: GridConfig = GridConfig()):
        # noinspection PyTypeChecker
        self.grid: Grid = None
        self.grid_config = grid_config

        self.action_space: gymnasium.spaces.Discrete = gymnasium.spaces.Discrete(len(self.grid_config.MOVES))
        self._multi_action_sampler = ActionsSampler(self.action_space.n, seed=self.grid_config.seed)

    def _get_agents_obs(self, agent_id=0):
        """
        Returns the observation of the agent with the given id.
        :param agent_id:
        :return:
        """
        return np.concatenate([
            self.grid.get_obstacles_for_agent(agent_id)[None],
            self.grid.get_positions(agent_id)[None],
            self.grid.get_square_target(agent_id)[None]
        ])

    def check_reset(self):
        """
        Checks if the reset needed.
        :return:
        """
        if self.grid is None:
            raise ResetNeeded("Please reset environment first!")

    def render(self, mode='human'):
        """
        Renders the environment using ascii graphics.
        :param mode:
        :return:
        """
        self.check_reset()
        return self.grid.render(mode=mode)

    def sample_actions(self):
        """
        Samples the random actions for the given number of agents.
        :return:
        """
        return self._multi_action_sampler.sample_actions(dim=self.grid_config.num_agents)

    def get_num_agents(self):
        """
        Returns the number of agents in the environment.
        :return:
        """
        return self.grid_config.num_agents


class Pogema(PogemaBase):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.was_on_goal = None
        full_size = self.grid_config.obs_radius * 2 + 1
        if self.grid_config.observation_type == 'default':
            self.observation_space = gymnasium.spaces.Box(-1.0, 1.0, shape=(3, full_size, full_size))
        elif self.grid_config.observation_type == 'POMAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
            )
        elif self.grid_config.observation_type == 'MAPF':
            self.observation_space: gymnasium.spaces.Dict = gymnasium.spaces.Dict(
                obstacles=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                agents=gymnasium.spaces.Box(0.0, 1.0, shape=(full_size, full_size)),
                xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                target_xy=gymnasium.spaces.Box(low=-1024, high=1024, shape=(2,), dtype=int),
                # global_obstacles=None, # todo define shapes of global state variables
                # global_xy=None,
                # global_target_xy=None,
            )
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def step(self, action: list): #reward can be changed here
        assert len(action) == self.grid_config.num_agents
        rewards = [0] * self.grid_config.num_agents

        terminated = []
        rewards = np.array(rewards,dtype=np.float64)
        rewards += self.move_agents(action)
        
        for agent_idx in range(self.grid_config.num_criminal, self.grid_config.num_agents):
            self.grid.finishes_xy[agent_idx] = self.grid.get_agents_xy()[0] #configuration on running
        self.update_was_on_goal()
        
        if any(self.was_on_goal[:self.grid_config.num_criminal]): #criminal won
            rewards += [100] * self.grid_config.num_criminal + [-100] * (self.grid_config.num_agents - self.grid_config.num_criminal)
        elif any(self.was_on_goal[self.grid_config.num_criminal:]): #police won
            rewards += [-100] * self.grid_config.num_criminal + [100] * (self.grid_config.num_agents - self.grid_config.num_criminal)
        
        c=self.grid_config
        for agent_idx in range(self.grid_config.num_agents):

            on_goal = self.grid.on_goal(agent_idx)
            # if on_goal and self.grid.is_active[agent_idx]:
            #     rewards.append(1.0)
            # else:
            #     rewards.append(0.0)
            terminated.append(on_goal)
            #TODO: BFS find shortest path: O(n)

        rewards += self.bfs_rewards(grid = self.grid.obstacles.copy(), moves = self.grid_config.MOVES,
                                        base_num = max(c.FREE, c.OBSTACLE) + 1, free_cell = c.FREE)
            
        for agent_idx in range(self.grid_config.num_agents):
            if self.grid.on_goal(agent_idx):
                # modified
                if agent_idx < self.grid_config.num_criminal:
                    self.grid.hide_agent(agent_idx)
                    self.grid.is_active[agent_idx] = False
                else:
                    self.grid.hide_agent(0)
                    self.grid.is_active[0] = False

        infos = self._get_infos()

        observations = self._obs()
        truncated = [False] * self.grid_config.num_agents
        # modified
        if any(terminated): terminated = [True] * self.grid_config.num_agents
        return observations, rewards, terminated, truncated, infos
    
    def bfs_rewards(self, grid, moves, base_num, free_cell):
        q = []
        rewards =[0] * self.grid_config.num_agents
        size_x = len(grid)
        size_y = len(grid[0])
        (x, y) = self.grid.get_agents_xy()[0]
        grid[x, y] = base_num
        q.append((x, y))

        while len(q):
            cx, cy = q.pop(0)
            for dx, dy in moves:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < size_x and 0 <= ny < size_y:
                    if grid[nx, ny] == free_cell:
                        grid[nx, ny] = grid[cx, cy] + 1
                        q.append((nx, ny))

        for agent_id in range(self.grid_config.num_agents):
            if agent_id < self.grid_config.num_criminal: #criminal
                rewards[agent_id] = - grid[self.grid.finishes_xy[agent_id]] + base_num
                rewards[agent_id] += self.grid_config.alpha * min(grid[self.grid.positions_xy[i]] for i in range(1, self.grid_config.num_agents)) - base_num
            else: #police
                rewards[agent_id] = - grid[self.grid.finishes_xy[agent_id]] + base_num
        return rewards #10 步之内有显著的影响，gamma^10=.95^10=0.6

    def _initialize_grid(self):
        self.grid: Grid = Grid(grid_config=self.grid_config)

    def update_was_on_goal(self):
        self.was_on_goal = [self.grid.on_goal(agent_idx) and self.grid.is_active[agent_idx]
                            for agent_idx in range(self.grid_config.num_agents)]

    def reset(self, seed: Optional[int] = None, return_info: bool = True, options: Optional[dict] = None, ):
        self._initialize_grid()
        self.update_was_on_goal()

        if seed is not None:
            self.grid.seed = seed

        if return_info:
            return self._obs(), self._get_infos()
        return self._obs()

    def _obs(self):
        if self.grid_config.observation_type == 'default':
            return [self._get_agents_obs(index) for index in range(self.grid_config.num_agents)]
        elif self.grid_config.observation_type == 'POMAPF':
            return self._pomapf_obs()

        elif self.grid_config.observation_type == 'MAPF':
            results, _ = self._pomapf_obs()
            global_obstacles = self.grid.get_obstacles()
            global_agents_xy = self.grid.get_agents_xy()
            global_targets_xy = self.grid.get_targets_xy()

            for agent_idx in range(self.grid_config.num_agents):
                result = results[agent_idx]
                result.update(global_obstacles=global_obstacles)
                result['global_xy'] = global_agents_xy[agent_idx]
                result['global_target_xy'] = global_targets_xy[agent_idx]

            return results
        else:
            raise ValueError(f"Unknown observation type: {self.grid.config.observation_type}")

    def _pomapf_obs(self):
        results = []
        dict_results = []
        agents_xy_relative = self.grid.get_agents_xy_relative()
        targets_xy_relative = self.grid.get_targets_xy_relative()

        for agent_idx in range(self.grid_config.num_agents):
            dict_result = {'obstacles': self.grid.get_obstacles_for_agent(agent_idx),
                      'agents': self.grid.get_positions(agent_idx),
                      'xy': agents_xy_relative[agent_idx],
                      'target_xy': targets_xy_relative[agent_idx]}
            result = np.concatenate([self.grid.get_obstacles_for_agent(agent_idx).flatten(),
                      self.grid.get_positions(agent_idx).flatten(),
                      agents_xy_relative[agent_idx],
                      targets_xy_relative[agent_idx]])

            results.append(result)
            dict_results.append(dict_result)
        return results, dict_results

    def _get_infos(self):
        infos = [dict() for _ in range(self.grid_config.num_agents)]
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]
        return infos

    def move_agents(self, actions) -> List[int]:
        if self.grid.config.collision_system == 'priority':
            rewards = []
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    rewards.append(self.grid.move(agent_idx, actions[agent_idx]))
                else:
                    rewards.append(0)
            return rewards
        elif self.grid.config.collision_system == 'block_both':
            used_cells = {}
            agents_xy = self.grid.get_agents_xy()
            for agent_idx, (x, y) in enumerate(agents_xy):
                if self.grid.is_active[agent_idx]:
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    used_cells[x + dx, y + dy] = 'blocked' if (x + dx, y + dy) in used_cells else 'visited'
                    used_cells[x, y] = 'blocked'
            for agent_idx in range(self.grid_config.num_agents):
                if self.grid.is_active[agent_idx]:
                    x, y = agents_xy[agent_idx]
                    dx, dy = self.grid_config.MOVES[actions[agent_idx]]
                    if used_cells.get((x + dx, y + dy), None) != 'blocked':
                        self.grid.move(agent_idx, actions[agent_idx])
        elif self.grid.config.collision_system == 'simultaneous':
            pass
            #TODO: queue, 先把目标地点是空的pos先标记，如果有两个智体的目标地点相同,贼车优先走，等价于priority把贼车放在前面
            #警车只要在贼车周围四个格子就说明抓到了
        else:
            raise ValueError('Unknown collision system: {}'.format(self.grid.config.collision_system))

    def get_agents_xy_relative(self):
        return self.grid.get_agents_xy_relative()

    def get_targets_xy_relative(self):
        return self.grid.get_targets_xy_relative()

    def get_obstacles(self, ignore_borders=False):
        return self.grid.get_obstacles(ignore_borders=ignore_borders)

    def get_agents_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_agents_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_targets_xy(self, only_active=False, ignore_borders=False):
        return self.grid.get_targets_xy(only_active=only_active, ignore_borders=ignore_borders)

    def get_state(self, ignore_borders=False, as_dict=False):
        return self.grid.get_state(ignore_borders=ignore_borders, as_dict=as_dict)


class PogemaLifeLong(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)

    def _initialize_grid(self):
        self.grid: GridLifeLong = GridLifeLong(grid_config=self.grid_config)

        main_rng = np.random.default_rng(self.grid_config.seed)
        seeds = main_rng.integers(np.iinfo(np.int32).max, size=self.grid_config.num_agents)
        self.random_generators = [np.random.default_rng(seed) for seed in seeds]

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents
        rewards = []

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        for agent_idx in range(self.grid_config.num_agents):
            on_goal = self.grid.on_goal(agent_idx)
            if on_goal and self.grid.is_active[agent_idx]:
                rewards.append(1.0)
            else:
                rewards.append(0.0)

            if self.grid.on_goal(agent_idx):
                self.grid.finishes_xy[agent_idx] = generate_new_target(self.random_generators[agent_idx],
                                                                       self.grid.point_to_component,
                                                                       self.grid.component_to_points,
                                                                       self.grid.positions_xy[agent_idx])

        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [False] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        return obs, rewards, terminated, truncated, infos


class PogemaCoopFinish(Pogema):
    def __init__(self, grid_config=GridConfig(num_agents=2)):
        super().__init__(grid_config)
        self.num_agents = self.grid_config.num_agents
        self.is_multiagent = True

    def _initialize_grid(self):
        self.grid: CooperativeGrid = CooperativeGrid(grid_config=self.grid_config)

    def step(self, action: list):
        assert len(action) == self.grid_config.num_agents

        infos = [dict() for _ in range(self.grid_config.num_agents)]

        self.move_agents(action)
        self.update_was_on_goal()

        is_task_solved = all(self.was_on_goal)
        for agent_idx in range(self.grid_config.num_agents):
            infos[agent_idx]['is_active'] = self.grid.is_active[agent_idx]

        obs = self._obs()

        terminated = [is_task_solved] * self.grid_config.num_agents
        truncated = [False] * self.grid_config.num_agents
        rewards = [1.0 if is_task_solved else 0.0 for _ in range(self.grid_config.num_agents)]
        return obs, rewards, terminated, truncated, infos


def _make_pogema(grid_config):
    if grid_config.on_target == 'restart':
        env = PogemaLifeLong(grid_config=grid_config)
    elif grid_config.on_target == 'nothing':
        env = PogemaCoopFinish(grid_config=grid_config)
    elif grid_config.on_target == 'finish':
        env = Pogema(grid_config=grid_config)
    else:
        raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    env = MultiTimeLimit(env, grid_config.max_episode_steps)
    if env.grid_config.persistent:
        env = PersistentWrapper(env)
    else:
        # adding metrics wrappers
        if grid_config.on_target == 'restart':
            env = LifeLongAverageThroughputMetric(env)
        elif grid_config.on_target == 'nothing':
            env = NonDisappearISRMetric(env)
            env = NonDisappearCSRMetric(env)
            env = NonDisappearEpLengthMetric(env)
        elif grid_config.on_target == 'finish':
            env = ISRMetric(env)
            env = CSRMetric(env)
            env = OnGoalMetric(env)
            env = EpLengthMetric(env)
        else:
            raise KeyError(f'Unknown on_target option: {grid_config.on_target}')

    return env
