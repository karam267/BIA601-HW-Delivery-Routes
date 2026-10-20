from dataclasses import dataclass
import numpy as np


@dataclass
class DeliveryMap:
    code: str
    points: np.ndarray

    def __post_init__(self):
        self.points = np.asarray(self.points, dtype=np.int64)
        if self.points.ndim != 2 or self.points.shape[1] != 2 or len(self.points) < 3:
            raise ValueError('At least three distinct planar sites are required')
        if len(np.unique(self.points, axis=0)) != len(self.points):
            raise ValueError('Sites must be distinct')
        delta = self.points[:, None, :] - self.points[None, :, :]
        self.distance = np.rint(np.sqrt(np.sum(delta.astype(float)**2, axis=2))).astype(np.int64)
        np.fill_diagonal(self.distance, 0)

    @property
    def sites(self):
        return len(self.points)


def make_map(sites, seed, layout, code):
    random = np.random.default_rng(seed)
    if layout == 'uniform':
        cells = random.choice(1000000, size=sites, replace=False)
        points = np.column_stack((cells//1000, cells%1000))
    elif layout == 'clustered':
        centers = np.array([[230,250],[760,300],[500,770]])
        points = []
        while len(points) < sites:
            candidate = np.rint(random.normal(centers[len(points)%3], 65)).clip(0,999).astype(int)
            if tuple(candidate) not in points:
                points.append(tuple(candidate))
        points = np.asarray(points)
    elif layout == 'ring':
        angles = np.linspace(0,2*np.pi,sites,endpoint=False) + random.uniform(-.035,.035,size=sites)
        angles += random.uniform(0,2*np.pi)
        radius = random.integers(345,416,size=sites)
        points = np.rint(500 + radius[:,None]*np.column_stack((np.cos(angles),np.sin(angles)))).astype(int)
    else:
        raise ValueError(layout)
    return DeliveryMap(code,points)


def nearest_neighbor(area):
    remaining = set(range(1,area.sites))
    route = [0]
    while remaining:
        next_site = min(remaining,key=lambda node:(int(area.distance[route[-1],node]),node))
        route.append(next_site)
        remaining.remove(next_site)
    return np.asarray(route,dtype=np.int32)


def route_length(area,route):
    route = np.asarray(route)
    if route.shape != (area.sites,) or route.dtype.kind not in 'iu':
        raise ValueError('The route must be an integer permutation')
    if route[0] != 0 or not np.array_equal(np.sort(route),np.arange(area.sites)):
        raise ValueError('Start at the depot and visit every site exactly once')
    return int(area.distance[route,np.roll(route,-1)].sum())


def evaluate_orders(area,orders):
    routes = np.column_stack((np.zeros(len(orders),dtype=np.int32),orders))
    return area.distance[routes,np.roll(routes,-1,axis=1)].sum(axis=1)
