import numpy as np
from tours import nearest_neighbor,evaluate_orders


def ordered_child(first,second,random):
    size=len(first)
    left,right=sorted(random.choice(size+1,2,replace=False))
    child=np.full(size,-1,dtype=np.int32)
    child[left:right]=first[left:right]
    fixed=set(first[left:right].tolist())
    sequence=np.r_[second[right:],second[:right]]
    missing=[int(site) for site in sequence if int(site) not in fixed]
    slots=np.r_[np.arange(right,size),np.arange(left)]
    child[slots]=missing
    return child


def genetic_routes(area,seed,population=32,iterations=160,crossover=.85,inversion=.25,tournament=4,elite=2):
    random=np.random.default_rng(seed)
    genes=np.arange(1,area.sites,dtype=np.int32)
    orders=np.asarray([random.permutation(genes) for _ in range(population)])
    orders[0]=nearest_neighbor(area)[1:]
    lengths=evaluate_orders(area,orders)
    winner=int(lengths.argmin())
    champion,best=orders[winner].copy(),int(lengths[winner])
    curve=[best]
    for _ in range(iterations):
        ranking=np.lexsort((np.arange(population),lengths))
        next_orders=[orders[index].copy() for index in ranking[:elite]]
        while len(next_orders)<population:
            parents=[]
            for _ in range(2):
                candidates=random.integers(0,population,size=tournament)
                parents.append(orders[candidates[lengths[candidates].argmin()]])
            for first,second in [(parents[0],parents[1]),(parents[1],parents[0])]:
                child=ordered_child(first,second,random) if random.random()<crossover else first.copy()
                if random.random()<inversion:
                    left,right=sorted(random.choice(len(genes),2,replace=False))
                    child[left:right+1]=child[left:right+1][::-1]
                next_orders.append(child)
                if len(next_orders)==population:break
        orders=np.asarray(next_orders,dtype=np.int32)
        lengths=evaluate_orders(area,orders)
        winner=int(lengths.argmin())
        if lengths[winner]<best:
            champion,best=orders[winner].copy(),int(lengths[winner])
        curve.append(best)
    return np.r_[0,champion],best,curve


def swaps_between(first,second):
    current=first.copy()
    locations=np.empty(len(current)+1,dtype=np.int32)
    locations[current]=np.arange(len(current))
    swaps=[]
    for place,desired in enumerate(second):
        if current[place]==desired:continue
        other=int(locations[desired])
        displaced=int(current[place])
        swaps.append((place,other))
        current[place],current[other]=current[other],current[place]
        locations[desired],locations[displaced]=place,other
    return swaps


def apply_swaps(order,swaps):
    for left,right in swaps:
        order[left],order[right]=order[right],order[left]


def swarm_routes(area,seed,population=32,iterations=160,inertia=.4,personal=.6,social=.8,shake=.12):
    random=np.random.default_rng(seed)
    genes=np.arange(1,area.sites,dtype=np.int32)
    positions=np.asarray([random.permutation(genes) for _ in range(population)])
    positions[0]=nearest_neighbor(area)[1:]
    lengths=evaluate_orders(area,positions)
    personal_orders,personal_lengths=positions.copy(),lengths.copy()
    velocities=[[] for _ in range(population)]
    winner=int(lengths.argmin())
    champion,best=positions[winner].copy(),int(lengths[winner])
    curve=[best]
    for _ in range(iterations):
        for particle in range(population):
            order=positions[particle].copy()
            movement=[swap for swap in velocities[particle] if random.random()<inertia]
            apply_swaps(order,movement)
            for target,probability in [(personal_orders[particle],personal),(champion,social)]:
                difference=swaps_between(order,target)
                prefix=int(random.binomial(len(difference),probability))
                chosen=difference[:prefix]
                apply_swaps(order,chosen)
                movement.extend(chosen)
            if random.random()<shake:
                swap=tuple(int(i) for i in random.choice(len(genes),2,replace=False))
                apply_swaps(order,[swap])
                movement.append(swap)
            positions[particle]=order
            velocities[particle]=movement
        lengths=evaluate_orders(area,positions)
        improved=lengths<personal_lengths
        personal_lengths[improved]=lengths[improved]
        personal_orders[improved]=positions[improved]
        winner=int(lengths.argmin())
        if lengths[winner]<best:
            champion,best=positions[winner].copy(),int(lengths[winner])
        curve.append(best)
    return np.r_[0,champion],best,curve


def ant_routes(area,seed,population=32,iterations=160,alpha=1.0,beta=3.0,evaporation=.15):
    random=np.random.default_rng(seed)
    size=area.sites
    pheromone=np.full((size,size),1.0/size)
    np.fill_diagonal(pheromone,0)
    visibility=np.zeros((size,size),dtype=float)
    np.divide(1.0,area.distance,out=visibility,where=area.distance>0)
    champion,best=None,np.iinfo(np.int64).max
    curve=[]
    ant_indices=np.arange(population)
    for step in range(iterations+1):
        routes=np.zeros((population,size),dtype=np.int32)
        visited=np.zeros((population,size),dtype=bool)
        visited[:,0]=True
        for stop in range(1,size):
            previous=routes[:,stop-1]
            attraction=pheromone[previous]**alpha * visibility[previous]**beta
            attraction[visited]=0
            cumulative=attraction.cumsum(axis=1)
            totals=cumulative[:,-1]
            threshold=random.random(population)*totals
            threshold=np.minimum(threshold,np.nextafter(totals,-np.inf))
            chosen=(cumulative<=threshold[:,None]).sum(axis=1)
            routes[:,stop]=chosen
            visited[ant_indices,chosen]=True
        if step==0:routes[0]=nearest_neighbor(area)
        lengths=evaluate_orders(area,routes[:,1:])
        winner=int(lengths.argmin())
        current=int(lengths[winner])
        if current<best:
            champion,best=routes[winner].copy(),current
        pheromone*=1-evaporation
        for route,length in [(routes[winner],current),(champion,best)]:
            following=np.roll(route,-1)
            np.add.at(pheromone,(route,following),1.0/length)
            np.add.at(pheromone,(following,route),1.0/length)
        pheromone=np.maximum(pheromone,1e-12)
        np.fill_diagonal(pheromone,0)
        curve.append(best)
    return champion,best,curve
