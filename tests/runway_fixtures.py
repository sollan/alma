"""Generated motion with known support, movement, pause and visibility states."""
import numpy as np
import pandas as pd


def synthetic_walk(fps=100, noise=.15, scale=1., reverse=False, lift=15., seed=7, pause=None):
    rng = np.random.default_rng(seed)
    time = np.arange(10*fps+1)/fps
    delayed = np.clip(time-pause[0],0.,pause[1]-pause[0]) if pause else 0.
    active = np.clip(time-2.-delayed,0.,6.)
    period, stance, speed = .4,.25,100.
    cycle = np.floor(active/period + 1e-9)
    local = active-cycle*period
    fraction = np.clip((local-stance)/(period-stance),0,1)
    ease = fraction*fraction*(3-2*fraction)
    toe_x = speed*period*cycle+20.+speed*period*ease
    toe_y = 100.-lift*np.sin(np.pi*fraction)
    body_x = speed*active
    coords = {'toe':(toe_x,toe_y), 'mtp':(toe_x+3,toe_y-4),
              'ankle':(body_x+5,np.full(len(time),82.)),
              'knee':(body_x-4,np.full(len(time),72.)),
              'hip':(body_x,np.full(len(time),62.)),
              'iliac crest':(body_x-12,np.full(len(time),50.))}
    data = pd.DataFrame({'bodyparts coords':np.arange(len(time))})
    for bp,(x,y) in coords.items():
        data[f'{bp} x'] = (x+rng.normal(0,noise,len(time)))*scale*(-1 if reverse else 1)
        data[f'{bp} y'] = (y+rng.normal(0,noise,len(time)))*scale
        data[f'{bp} likelihood'] = 1.
        data.loc[(time<1.)|(time>9.),f'{bp} likelihood'] = .1
    return data
