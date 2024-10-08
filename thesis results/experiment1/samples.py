import torch

# exclusive for threebody

SAMPLES = {  
            "yarn": 
                {"name" : "YARN",
                 "M":  [1,1,1],
                 "q1": [-1,0],
                 "q2": [1,0],
                 "q3": [0,0],
                 "v1":[0.559064,0.349192],
                 "v2":[0.559064,0.349192],
                 "v3":[-1.118128,-0.698384],
                 "T": 55.501762,
                 "H": -1.196537
                 },
            "googles":
                {"name": "GOOGLES",
                 "M":  [1,1,1],
                 "q1": [-1,0],
                 "q2": [1,0],
                 "q3": [0,0],
                 "v1":[0.083300,0.127889],
                 "v2":[0.083300,0.127889],
                 "v3":[-0.166600,-0.255778],
                 "T": 10.466818,
                 "H": -2.430116 
                 },
            "butterfly":
                {"name": "BUTTERFLY4",
                 "M":  [1,1,1],
                 "q1": [-1,0],
                 "q2": [1,0],
                 "q3": [0,0],
                 "v1":[0.405916,0.230163],
                 "v2":[0.405916,0.230163],
                 "v3":[-0.811832,-0.460326],
                 "T": 13.865763,
                 "H": -1.846772 
                 },
            "fig8":
                {"name": "FIGURE8",
                 "M":  [1,1,1],
                 "q1": [-1,0],
                 "q2": [1,0],
                 "q3": [0,0],
                 "v1":[0.347111,0.532728],
                 "v2":[0.347111,0.532728],
                 "v3":[-0.694222,-1.065456],
                 "T": 6.324449,
                 "H": -1.287146 
                 },
            "v810":
                {"name": "VIII 10",
                 "M":  [1,1,1],
                 "q1": [-1,0],
                 "q2": [1,0],
                 "q3": [0,0],
                 "v1":[0.268073,0.443797],
                 "v2":[0.268073,0.443797],
                 "v3":[-0.536146,-0.887594],
                 "T": 48.894527,
                 "H": -1.693544
                 },
            "a15":
                {"name": "Broucke A 15",
                 "M":  [1,1,1],
                 "q1": [-1.1889693067,0.0000000000],
                 "q2": [3.8201881837,0.0000000000],
                 "q3": [-2.6312188770,0.0000000000],
                 "v1":[0.0000000000,0.8042120498],
                 "v2":[0.0000000000,0.0212794833],
                 "v3":[0.0000000000,-0.8254915331],
                 "T": 92.056119,
                 "H": -0.383678
                 },
            "a6":
                {"name": "Broucke A 6",
                 "M":  [1,1,1],
                 "q1": [0.1432778606,0.0000000000],
                 "q2": [-1.2989717097,0.0000000000],
                 "q3": [-2.6312188770,0.0000000000],
                 "v1": [0.0000000000,1.1577475241],
                 "v2": [0.0000000000,-0.2974667752],
                 "v3": [0.0000000000,-0.8602807489],
                 "T": 9.593323,
                 "H": -1.004011
                 },
            "R10":
                {"name": "Broucke R 10",
                 "M":  [1,1,1],
                 "q1": [0.8822391241,0.0000000000],
                 "q2": [-0.6432718586,0.0000000000],
                 "q3": [-0.2389672654,0.0000000000],
                 "v1": [0.0000000000,1.0042424155],
                 "v2": [0.0000000000,-1.6491842814],
                 "v3": [0.0000000000,0.6449418659],
                 "T": 10.948278,
                 "H": -1.948666
                 },
        }


if __name__ == "__main__":
    list_samples = list(SAMPLES.keys())
    print(list_samples)
    list_spec =list(SAMPLES["yarn"].keys())
    print(list_spec)