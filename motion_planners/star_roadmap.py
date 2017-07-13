class StarRoadmap(object):

    def __init__(self, center, planner):
        self.center = center
        self.planner = planner
        self.roadmap = {}

    def grow(self, goal):
        if goal not in self.roadmap:
            self.roadmap[goal] = self.planner(self.center, goal)
        return self.roadmap[goal]

    def __call__(self, start, goal):
        start_traj = self.grow(start)
        if start_traj is None:
            return None
        goal_traj = self.grow(goal)
        if goal_traj is None:
            return None
        return (start_traj.reverse(), goal_traj)
