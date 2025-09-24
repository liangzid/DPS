# VSer: Vulnerability Scanning on LLMs

This repository contains the source code of our paper `Decision Potential Surface: A Theoretical and Practical Approximation of LLM's Decision Boundary`.

## Envoriment


```sh
pip install -r dps_re.txt
```

## Explanation of Files



## Reproduce Experiments


Step 1: Generate preprared data.

```sh
bash scripts/1.1.parallel_kdps_varyK.sh
bash scripts/1.2.parallel_kdps_different_dataset.sh
bash scripts/1.1.parallel_kdps_different_model.sh
```

Step 2: Reproduce experiments of Section 5.2

```sh
bash scripts/2.1.compute_draw_varyK.sh
```

Step 3: Reproduce experiments of Section 5.3 & Appendix

```sh
bash scripts/2.2.compute_draw_concentration.sh
```

Step 4: Reproduce experiments of Section 5.4

```sh
python 5.run_kdps_varyDataset.py
```

Step 4: Reproduce experiments on Appendix

```sh
python 5.1.run_kdps_varyModels.py
```



