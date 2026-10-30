import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from tours import make_map,nearest_neighbor,route_length
from search import genetic_routes,swarm_routes,ant_routes
from reference import solve_exact

HERE=Path(__file__).resolve().parent
METHODS={'GA':genetic_routes,'PSO':swarm_routes,'ACO':ant_routes}


def read_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def held_karp(distance):
    count=len(distance)-1
    values={(1<<node,node):int(distance[0,node+1]) for node in range(count)}
    for mask in range(1,1<<count):
        for last in range(count):
            if not mask & (1<<last) or mask==(1<<last):continue
            previous=mask^(1<<last)
            values[mask,last]=min(values[previous,node]+int(distance[node+1,last+1])
                                  for node in range(count) if previous & (1<<node))
    return min(values[(1<<count)-1,last]+int(distance[last+1,0]) for last in range(count))


def close(actual,expected):
    assert np.isclose(float(actual),float(expected),rtol=1e-10,atol=1e-10),(actual,expected)


def check(folder,rerun=False):
    context=json.loads((folder/'environment.json').read_text())
    protocol=json.loads((HERE/'protocol.json').read_text())
    for name,digest in context['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest,('Changed source',name)
    cases=read_csv(folder/'cases.csv')
    trials=read_csv(folder/'trials.csv')
    nodes=read_csv(folder/'nodes.csv')
    totals=read_csv(folder/'aggregate.csv')
    proof=json.loads((folder/'reference.json').read_text())
    repetitions=context['repetitions']
    budget=protocol['population']*(protocol['iterations']+1)
    assert len(cases)==9 and len(trials)==9*3*repetitions and len(totals)==12
    assert context['evaluations_per_trial']==budget
    assert len({row['case'] for row in cases})==9
    assert len({(row['case'],row['method'],row['trial']) for row in trials})==len(trials)
    areas={}
    for case in cases:
        code=case['case']
        area=make_map(int(case['sites']),int(case['seed']),case['layout'],code)
        areas[code]=area
        saved=sorted([row for row in nodes if row['case']==code],key=lambda row:int(row['site']))
        assert [int(row['site']) for row in saved]==list(range(area.sites))
        assert np.array_equal(area.points,[[int(row['x']),int(row['y'])] for row in saved])
        reference=proof[code]
        optimum=int(case['optimum'])
        assert route_length(area,np.asarray(reference['tour']))==optimum==reference['length']
        assert reference['status'] in [0,1]
        assert int(np.ceil(reference['dual_bound']-1e-5))==optimum
        close(case['dual_bound'],reference['dual_bound'])
        assert int(case['cut_rounds'])==reference['cut_rounds']
        assert route_length(area,nearest_neighbor(area))==int(case['nn_length'])
        if area.sites==10:
            assert held_karp(area.distance)==optimum,('Independent exact reference',code)
        if rerun:
            assert solve_exact(area)['length']==optimum
    assert len(nodes)==sum(area.sites for area in areas.values())
    with np.load(folder/'routes.npz',allow_pickle=False) as routes,np.load(folder/'curves.npz',allow_pickle=False) as stored:
        curves=stored['curve']
        assert curves.shape==(len(trials),protocol['iterations']+1) and len(routes.files)==len(trials)
        for index,row in enumerate(trials):
            area=areas[row['case']]
            case=next(case for case in cases if case['case']==row['case'])
            tour=routes[f'trial_{index}']
            length=route_length(area,tour)
            independent=sum(int(area.distance[int(tour[k]),int(tour[(k+1)%area.sites])]) for k in range(area.sites))
            assert length==independent==int(row['length'])
            optimum=int(case['optimum'])
            assert optimum==int(row['optimum']) and optimum<=length<=int(case['nn_length'])
            close(row['gap'],100*(length-optimum)/optimum)
            assert float(row['seconds'])>0 and int(row['evaluations'])==budget
            curve=curves[index]
            assert curve[-1]==length and np.all(np.diff(curve)<=0) and np.all(curve>=optimum)
            target=np.flatnonzero(curve<=1.01*optimum)
            assert (row['reached']=='True')==bool(len(target))
            assert row['target_step']==(str(int(target[0])) if len(target) else '')
            if rerun and int(row['trial'])==1:
                again,score,history=METHODS[row['method']](area,seed=int(row['seed']),
                    population=protocol['population'],iterations=protocol['iterations'],**protocol[row['method']])
                assert np.array_equal(again,tour) and score==length and np.array_equal(history,curve)
    for summary in totals:
        scale,method=summary['scale'],summary['method']
        members=[case for case in cases if case['scale']==scale]
        if method=='NN':
            gap=np.mean([100*(int(case['nn_length'])-int(case['optimum']))/int(case['optimum']) for case in members])
            close(summary['mean_gap'],gap)
            close(summary['mean_std_gap'],0)
            close(summary['mean_seconds'],np.mean([float(case['nn_seconds']) for case in members]))
            close(summary['mean_kib'],np.mean([float(case['nn_kib']) for case in members]))
            assert summary['exact_hits']==f"{sum(int(case['nn_length'])==int(case['optimum']) for case in members)}/3"
            assert summary['target_hits']=='-' and summary['mean_target_step']==''
            continue
        chosen=[row for row in trials if row['scale']==scale and row['method']==method]
        assert len(chosen)==3*repetitions
        for case in members:
            group=[row for row in chosen if row['case']==case['case']]
            assert sorted(int(row['trial']) for row in group)==list(range(1,repetitions+1))
        close(summary['mean_gap'],np.mean([float(row['gap']) for row in chosen]))
        deviations=[np.std([float(row['gap']) for row in chosen if row['case']==case['case']],ddof=1)
                    if repetitions>1 else 0 for case in members]
        close(summary['mean_std_gap'],np.mean(deviations))
        close(summary['mean_seconds'],np.mean([float(row['seconds']) for row in chosen]))
        close(summary['mean_kib'],np.mean([float(case[method+'_kib']) for case in members]))
        assert summary['exact_hits']==f"{sum(int(row['length'])==int(row['optimum']) for row in chosen)}/{len(chosen)}"
        targets=[int(row['target_step']) for row in chosen if row['reached']=='True']
        assert summary['target_hits']==f'{len(targets)}/{len(chosen)}'
        if targets:close(summary['mean_target_step'],np.mean(targets))
        else:assert summary['mean_target_step']==''
    print(f'Passed: {len(cases)} references, {len(trials)} complete tours and histories, and all aggregate statistics.')
    print('The three 10-site references also match independent Held-Karp dynamic programming.')
    if rerun:print('Reproduced all nine exact values and 27 complete seeded search histories.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=HERE/'results')
    parser.add_argument('--rerun',action='store_true')
    arguments=parser.parse_args()
    check(arguments.results.resolve(),arguments.rerun)
