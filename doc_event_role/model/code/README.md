
# code for multi-granularity reader

- Step 0, for each template, mkdir

```
mkdir model_out_attack
mkdir model_save_attack
```


- Step 1, generate seq_tag pairs,

```
bash generate_data.sh
```

- Step 2, training and decoding,

```
python main.py --config config/attack.config
```

- Step 3, change predicted seq_tag pairs to eval format, generate ```pred.json```.

```
python seq_to_extracts.py --seqfile model_out/multi_bert.out --template_type attack
```


our implementation is built upon [ncrf++](https://arxiv.org/abs/1806.05626)


