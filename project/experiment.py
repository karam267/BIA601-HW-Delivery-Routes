import argparse
import csv
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tours import make_map,nearest_neighbor,route_length
from search import genetic_routes,swarm_routes,ant_routes
from reference import solve_exact

HERE=Path(__file__).resolve().parent
SCALES={'small':(10,7000),'medium':(20,8000),'large':(40,9000)}
LAYOUTS=['uniform','clustered','ring']
METHODS={'GA':genetic_routes,'PSO':swarm_routes,'ACO':ant_routes}
PALETTE={'NN':'#65737e','GA':'#7656a4','PSO':'#c15050','ACO':'#208a86'}


def save_rows(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def traced_peak(function,*args,**kwargs):
    tracemalloc.start()
    try:
        function(*args,**kwargs)
        return tracemalloc.get_traced_memory()[1]/1024
    finally:
        tracemalloc.stop()


def baseline(area):
    tour=nearest_neighbor(area)
    return tour,route_length(area,tour)


def overview(trials,cases):
    result=[]
    for scale in SCALES:
        members=[case for case in cases if case['scale']==scale]
        result.append({'scale':scale,'method':'NN',
                       'mean_gap':statistics.mean(100*(c['nn_length']-c['optimum'])/c['optimum'] for c in members),
                       'mean_std_gap':0.0,'exact_hits':f"{sum(c['nn_length']==c['optimum'] for c in members)}/3",
                       'mean_seconds':statistics.mean(c['nn_seconds'] for c in members),
                       'mean_kib':statistics.mean(c['nn_kib'] for c in members),
                       'target_hits':'-','mean_target_step':None})
        for method in METHODS:
            rows=[row for row in trials if row['scale']==scale and row['method']==method]
            dispersions=[]
            for case in members:
                gaps=[row['gap'] for row in rows if row['case']==case['case']]
                dispersions.append(statistics.stdev(gaps) if len(gaps)>1 else 0.0)
            successes=[row['target_step'] for row in rows if row['reached']]
            result.append({'scale':scale,'method':method,
                           'mean_gap':statistics.mean(row['gap'] for row in rows),
                           'mean_std_gap':statistics.mean(dispersions),
                           'exact_hits':f"{sum(row['length']==row['optimum'] for row in rows)}/{len(rows)}",
                           'mean_seconds':statistics.mean(row['seconds'] for row in rows),
                           'mean_kib':statistics.mean(c[method+'_kib'] for c in members),
                           'target_hits':f'{len(successes)}/{len(rows)}',
                           'mean_target_step':statistics.mean(successes) if successes else None})
    return result


def route_panel(axis,area,tour,title,color):
    closed=np.r_[tour,tour[0]]
    points=area.points[closed]
    axis.plot(points[:,0],points[:,1],color=color,lw=1)
    axis.scatter(area.points[:,0],area.points[:,1],s=12,color=color)
    axis.scatter(area.points[0,0],area.points[0,1],marker='s',s=40,color='#d03c32',zorder=5)
    axis.set_title(title,fontsize=9)
    axis.set_aspect('equal',adjustable='box')
    axis.tick_params(labelsize=7)
    axis.grid(alpha=.15)


def draw(output,trials,cases,curves,tours,areas,proofs):
    figure,axes=plt.subplots(3,3,figsize=(11.4,10.3))
    for row,scale in enumerate(SCALES):
        for column,layout in enumerate(LAYOUTS):
            code=f'{scale}_{layout}'
            area=areas[code]
            route_panel(axes[row,column],area,np.array(proofs[code]['tour']),
                        f'{area.sites} sites / {layout}', '#40556d')
    figure.tight_layout()
    figure.savefig(output/'reference_routes.png',dpi=190)
    plt.close(figure)

    figure,axes=plt.subplots(1,3,figsize=(12.5,3.8))
    for axis,scale in zip(axes,SCALES):
        for method in METHODS:
            rows=[i for i,trial in enumerate(trials) if trial['scale']==scale and trial['method']==method]
            normalized=[100*(curves[i]-trials[i]['optimum'])/trials[i]['optimum'] for i in rows]
            axis.plot(np.mean(normalized,axis=0),label=method,color=PALETTE[method])
        axis.axhline(1,color='gray',lw=.8,ls=':')
        axis.set_title(scale.capitalize())
        axis.set_xlabel('Iteration')
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Mean excess distance over optimum (%)')
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output/'search_curves.png',dpi=190)
    plt.close(figure)

    figure,axes=plt.subplots(1,3,figsize=(12,3.7))
    for axis,scale in zip(axes,SCALES):
        groups=[[row['gap'] for row in trials if row['scale']==scale and row['method']==method] for method in METHODS]
        axis.boxplot(groups,tick_labels=list(METHODS))
        axis.set_title(scale.capitalize())
        axis.grid(alpha=.25)
    axes[0].set_ylabel('Excess distance over optimum (%)')
    figure.tight_layout()
    figure.savefig(output/'repeatability.png',dpi=190)
    plt.close(figure)

    totals=overview(trials,cases)
    figure,axes=plt.subplots(1,2,figsize=(10.4,4))
    positions=np.arange(3)
    for index,method in enumerate(['NN']+list(METHODS)):
        rows=[next(row for row in totals if row['scale']==scale and row['method']==method) for scale in SCALES]
        for axis,key in zip(axes,['mean_seconds','mean_kib']):
            axis.bar(positions+(index-1.5)*.2,[row[key] for row in rows],.2,label=method,color=PALETTE[method])
    for axis in axes:
        axis.set_xticks(positions,[name.capitalize() for name in SCALES])
        axis.set_yscale('log')
        axis.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Mean search time (s)')
    axes[1].set_ylabel('Mean traced peak (KiB)')
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output/'search_cost.png',dpi=190)
    plt.close(figure)

    code='large_uniform'
    area=areas[code]
    figure,axes=plt.subplots(2,2,figsize=(9,8))
    for axis,method in zip(axes.flat,['NN']+list(METHODS)):
        if method=='NN':
            tour=nearest_neighbor(area)
        else:
            eligible=[i for i,row in enumerate(trials) if row['case']==code and row['method']==method]
            tour=tours[min(eligible,key=lambda i:trials[i]['length'])]
        length=route_length(area,tour)
        route_panel(axis,area,tour,f'{method}: length={length}',PALETTE[method])
    figure.tight_layout()
    figure.savefig(output/'tour_comparison.png',dpi=190)
    plt.close(figure)


def experiment(repetitions,output):
    protocol=json.loads((HERE/'protocol.json').read_text())
    population,iterations=protocol['population'],protocol['iterations']
    evaluations=population*(iterations+1)
    output.mkdir(parents=True,exist_ok=True)
    trials,cases,nodes=[],[],[]
    tours,curves,areas,proofs=[],[],{},{}
    for scale_index,(scale,(sites,base_seed)) in enumerate(SCALES.items()):
        for layout_index,layout in enumerate(LAYOUTS):
            code=f'{scale}_{layout}'
            area=make_map(sites,base_seed+layout_index,layout,code)
            areas[code]=area
            proofs[code]=solve_exact(area)
            optimum=proofs[code]['length']
            _,nn_length=baseline(area)
            times=[]
            for _ in range(repetitions):
                start=time.perf_counter()
                baseline(area)
                times.append(time.perf_counter()-start)
            case={'case':code,'scale':scale,'layout':layout,'sites':sites,'seed':base_seed+layout_index,
                  'optimum':optimum,'dual_bound':proofs[code]['dual_bound'],
                  'cut_rounds':proofs[code]['cut_rounds'],'reference_seconds':proofs[code]['seconds'],
                  'nn_length':nn_length,'nn_seconds':statistics.mean(times),'nn_kib':traced_peak(baseline,area)}
            for site,(x,y) in enumerate(area.points):
                nodes.append({'case':code,'site':site,'x':int(x),'y':int(y)})
            print(code,'certified optimum',optimum,'NN',nn_length,flush=True)
            for method,solver in METHODS.items():
                for repetition in range(repetitions):
                    seed=500000+scale_index*10000+layout_index*100+repetition
                    start=time.perf_counter()
                    tour,length,curve=solver(area,seed=seed,population=population,iterations=iterations,**protocol[method])
                    elapsed=time.perf_counter()-start
                    assert length==route_length(area,tour) and optimum<=length<=nn_length
                    assert len(curve)==iterations+1 and curve[-1]==length and np.all(np.diff(curve)<=0)
                    target=next((i for i,v in enumerate(curve) if v<=1.01*optimum),None)
                    trials.append({'case':code,'scale':scale,'layout':layout,'method':method,
                                   'trial':repetition+1,'seed':seed,'length':length,'optimum':optimum,
                                   'gap':100*(length-optimum)/optimum,'seconds':elapsed,'evaluations':evaluations,
                                   'reached':target is not None,'target_step':target})
                    tours.append(np.asarray(tour,dtype=np.int16))
                    curves.append(np.asarray(curve,dtype=np.int64))
                case[method+'_kib']=traced_peak(solver,area,seed=500000+scale_index*10000+layout_index*100,
                                               population=population,iterations=iterations,**protocol[method])
                print(code,method,'complete',flush=True)
            cases.append(case)
    save_rows(output/'trials.csv',trials)
    save_rows(output/'cases.csv',cases)
    save_rows(output/'nodes.csv',nodes)
    save_rows(output/'aggregate.csv',overview(trials,cases))
    np.savez_compressed(output/'curves.npz',curve=np.asarray(curves))
    np.savez_compressed(output/'routes.npz',**{f'trial_{i}':tour for i,tour in enumerate(tours)})
    (output/'reference.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
    draw(output,trials,cases,curves,tours,areas,proofs)
    fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py'))}
    for name in ['protocol.json','requirements.txt']:
        fingerprints[name]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    context={'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,
             'matplotlib':matplotlib.__version__,'system':platform.platform(),'processor':platform.processor(),
             'repetitions':repetitions,'evaluations_per_trial':evaluations,'source_sha256':fingerprints}
    (output/'environment.json').write_text(json.dumps(context,indent=2),encoding='utf-8')
    print('Saved',len(trials),'verified-feasible search trials.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repeat',type=int,default=20)
    parser.add_argument('--output',type=Path,default=HERE/'results')
    arguments=parser.parse_args()
    if arguments.repeat<1:parser.error('--repeat must be positive')
    experiment(arguments.repeat,arguments.output.resolve())
