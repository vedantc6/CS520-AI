<p>The goal of this project is to compare and analyse different search algorithms commonly used in AI. A grid is generated and presented as an environment to the agent.
				<ul>
					<li>Red squares repesent the source and destination. The agent starts from top-left corner and has to reach bottom-right corner.</li>
					<li>Black cells represent obstacles</li>
					<li>White cells represent explorable space</li>
				</ul> 
<!-- 				<img src="assets/images/ai_search_1.png" width="200px" height="200px"/><br> -->
				Three different algorithms are used:
				<ul>
					<li>Breadth-First Search</li>
					<li>Depth-First Search</li>
					<li>A* Algorithm with Euclidean and Manhattan distance heuristics</li>
				</ul>
				The project is divided into 4 stages:
				<ul>
					<li><b>Time analysis and comparison -</b> A comparison of the different algorithms in terms of time taken to find optimal paths, size of different mazes and which performs better</li>
					<li><b>Generating hard mazes -</b> Finding harder mazes based on length of shortest path, nodes expanded by an algorithm, max. time taken to find the path</li>
					<li><b>Thinning A* -</b> A variation of the A* algorithm to reduce number of redundant searches</li>
					<li><b>Fire in the Maze -</b> A variation of the search problem, which tests the intelligence of the agent when raced against time. With every new timestep, the fire starts spreading to adjacent nodes, and now the agent has to reach the destination while avoiding fire. This is accomplished by using a heuristic based DFS approach, where closeness to fire is penalized while closeness to target is rewarded</li>
				</ul>
