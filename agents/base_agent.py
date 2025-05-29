from abc import ABC, abstractmethod

class BaseAgent:
    def run(self, state):
        raise NotImplementedError("Agents must implement the run method.")