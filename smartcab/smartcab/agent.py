import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
import math
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.index_to_action_words = {0:None, 1:'forward', 2:'left', 3:'right'}
        self.action_words_to_index = {None:0, 'forward':1, 'left':2, 'right':3}
        self.initialize_qtable()
        self.gamma = 0.0
        self.alpha_control = 125.0 #Increasing this increases learning
        self.epsilon = 0.01 # Exploration Rate
        # TODO: Initialize any additional variables here 
        self.trial_count = 0.0
        self.reward = 0
        self.reached_destination = False
        self.steps = 0
        self.penalty = 0

    def initialize_qtable(self):
        self.states = namedtuple("states",['light','oncoming','left','right','next_waypoint'])
        self.qtable = {}
        for light in ['red', 'green']:
            for oncoming in [None, 'forward', 'left', 'right']:
                for left in [None, 'forward', 'left', 'right']:
                    for right in [None, 'forward', 'left', 'right']:
                        for next_waypoint in [None, 'forward', 'left', 'right']:                        
                            self.qtable[self.states(light=light,oncoming=oncoming, right=right,
                                left=left, next_waypoint=next_waypoint)] = [0,0,0,0]
                            # [0,0,0,0] = [None, 'forward', 'left', 'right']
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial_count += 1
        self.reward = 0
        self.reached_destination = False
        self.steps = 0
        self.penalty = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.steps += 1
        

        # TODO: Update state
        state = self.states(light=inputs['light'],oncoming=inputs['oncoming'], left=inputs['left'], 
            right=inputs['right'], next_waypoint=self.next_waypoint)
        
        # TODO: Select action according to your policy
        maxq = max(self.qtable[state])
        max_indices = []
        for i, j in enumerate(self.qtable[state]):
            if j == maxq:
                max_indices.append(i)
                
        best_action = self.index_to_action_words[random.choice(max_indices)]
        action = np.random.choice(
            [best_action,random.choice([None, 'forward', 'left', 'right'])],
                p=[1-self.epsilon, self.epsilon])
                
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        #Q(state, action) = (1- alpha)*Q + alpha*(R(state, action) + Gamma * Max[Q(next state, all actions)])
        alpha = math.exp(-self.trial_count/self.alpha_control)
        next_state_sense = self.env.sense(self)
        next_state = self.states(light=next_state_sense['light'],oncoming=next_state_sense['oncoming'],
         right=next_state_sense['right'], left=next_state_sense['left'], next_waypoint=self.planner.next_waypoint())
        self.qtable[state][self.action_words_to_index[action]] = (
            (1 - alpha) * (self.qtable[state][self.action_words_to_index[action]]) + 
            (alpha) * (reward + self.gamma * max(self.qtable[next_state])))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.reward += reward


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    #print a.qtable.values()


if __name__ == '__main__':
    run()