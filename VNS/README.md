# Variable Neighborhood Search
## Task: Implement General VNS Scheme for the Cell Formation Problem (Biclustering).
 
Datasets: 35 GT problems
20x20
24x40
30x50
37x53
30x90
Input format:
m p (number of machines and parts)
Next m rows: 
m(row number) list of parts processed by machine m separated by space
e.g:
1 9 17 19 31 33 
means machine 1 processes parts 9 17 19 31 33 

Feasible solutions:
Each cluster must contain at least 1 machine and 1 part.
Each machine must be assigned to exactly 1 cluster.
Each part must be assigned to exactly 1 cluster.

Output file: instancename.sol (e.g. 20x20.sol)
Output file format:
m1_clusterId m2_clusterId ...  - machines to clusters mapping
p1_clusterId p2_clusterId ...  - parts to clusters mapping
