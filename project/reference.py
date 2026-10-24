import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from tours import route_length


def connected_parts(adjacency):
    unseen = set(range(len(adjacency)))
    parts = []
    while unseen:
        stack = [min(unseen)]
        part = set()
        while stack:
            node = stack.pop()
            if node in part:
                continue
            part.add(node)
            stack.extend(neighbor for neighbor in adjacency[node] if neighbor not in part)
        unseen.difference_update(part)
        parts.append(part)
    return parts


def solve_exact(area):
    start = time.perf_counter()
    count = area.sites
    left,right = np.triu_indices(count,1)
    edge_count = len(left)
    costs = area.distance[left,right].astype(float)
    cuts, known_cuts = [],set()
    for round_number in range(1,101):
        row_ids = np.r_[left,right].tolist()
        columns = np.r_[np.arange(edge_count),np.arange(edge_count)].tolist()
        data = [1.0]*(2*edge_count)
        lower,upper = [2.0]*count,[2.0]*count
        for cut_number,part in enumerate(cuts):
            inside = [i for i,(u,v) in enumerate(zip(left,right)) if int(u) in part and int(v) in part]
            row_ids.extend([count+cut_number]*len(inside))
            columns.extend(inside)
            data.extend([1.0]*len(inside))
            lower.append(-np.inf)
            upper.append(len(part)-1)
        matrix = coo_matrix((data,(row_ids,columns)),shape=(len(lower),edge_count)).tocsc()
        answer = milp(costs,integrality=np.ones(edge_count),bounds=Bounds(0,1),
                      constraints=LinearConstraint(matrix,np.asarray(lower),np.asarray(upper)),
                      options={'mip_rel_gap':0.0,'time_limit':60.0})
        if answer.x is None or answer.status not in [0,1]:
            raise RuntimeError('The exact reference was not solved: '+str(answer.message))
        chosen = answer.x > .5
        if not np.all(np.minimum(abs(answer.x),abs(answer.x-1)) < 1e-5):
            raise RuntimeError('The reference solver has no integral incumbent')
        adjacency = [[] for _ in range(count)]
        for u,v in zip(left[chosen],right[chosen]):
            adjacency[int(u)].append(int(v))
            adjacency[int(v)].append(int(u))
        if any(len(neighbors)!=2 for neighbors in adjacency):
            raise RuntimeError('Invalid reference degrees')
        parts = connected_parts(adjacency)
        if len(parts)==1:
            tour,previous,current = [0],-1,0
            while len(tour)<count:
                neighbors=sorted(adjacency[current])
                next_site=neighbors[0] if neighbors[0]!=previous else neighbors[1]
                if next_site==0:
                    raise RuntimeError('The reference cycle closed too early')
                tour.append(next_site)
                previous,current=current,next_site
            tour=np.asarray(tour,dtype=np.int32)
            length=route_length(area,tour)
            if abs(float(answer.fun)-length)>1e-5:
                raise RuntimeError('Reference objective does not match its tour')
            bound=float(answer.mip_dual_bound)
            if int(np.ceil(bound-1e-5))!=length:
                raise RuntimeError('Optimality was not certified within the time limit')
            return {'tour':tour.tolist(),'length':length,'dual_bound':bound,
                    'status':int(answer.status),'cut_rounds':round_number,'cuts':len(cuts),
                    'seconds':time.perf_counter()-start}
        for part in parts:
            key=tuple(sorted(part))
            if key not in known_cuts:
                known_cuts.add(key)
                cuts.append(part)
    raise RuntimeError('Subtour elimination did not finish')
