# motion-planners

Generic python implementations of several robotic motion planners.

## Citation

Caelan Reed Garrett. Motion Planners. https://github.com/caelan/motion-planners. 2017.

## Example


## Algorithms

## Single-Query

Sampling-Based:
* [Rapidly-Exploring Random Tree (RRT)](https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt.py)
* [Bidirectional RRT (BiRRT/RRT-Connect)](https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_connect.py)
* [MultiRRT](https://github.com/caelan/motion-planners/blob/master/motion_planners/multi_rrt.py)
* [RRT*](https://github.com/caelan/motion-planners/blob/master/motion_planners/rrt_star.py)

Grid Search
* [Breadth-First Search (BFS)](https://github.com/caelan/motion-planners/blob/691100867352db24535f29d1f4065b6da059ade3/motion_planners/discrete.py#L18)
* [Dijkstra/Uniform-Cost Search(UCS)](https://github.com/caelan/motion-planners/blob/691100867352db24535f29d1f4065b6da059ade3/motion_planners/discrete.py#L40)
* [A*](https://github.com/caelan/motion-planners/blob/691100867352db24535f29d1f4065b6da059ade3/motion_planners/discrete.py#L40)

Other
* [Straight-Line Path](https://github.com/caelan/motion-planners/blob/master/motion_planners/meta.py#L7)
* [Linear Shortcutting](https://github.com/caelan/motion-planners/blob/master/motion_planners/smoothing.py)
<!--* Diverse
* Random Restarts-->

## Multi-Query

Sampling-based:
* [Probabilistic Roadmap (PRM)](https://github.com/caelan/motion-planners/blob/master/motion_planners/prm.py)
* [Lazy PRM](https://github.com/caelan/motion-planners/blob/master/motion_planners/lazy_prm.py)
<!--* Star Roadmap-->

## Applications

* PyBullet Motion Planning - https://github.com/caelan/pybullet-planning
* PyBullet Task and Motion Planning (TAMP) - https://github.com/caelan/pddlstream
