# SR

```
parser = argparse.ArgumentParser()
parser.add_argument('--train-file', type=str, default='../dataset/DIV2K_train')
parser.add_argument('--eval-file', type=str, default='../dataset/DIV2K_valid')
parser.add_argument('--outputs-dir', type=str, default='output')
parser.add_argument('--weights-file', type=str)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=16)
parser.add_argument('--num-epochs', type=int, default=50)
parser.add_argument('--num-workers', type=int, default=1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--log', action='store_true')
parser.add_argument('--test-only', action='store_true')
parser.add_argument('--model-name', type=str, default='FSRCNN')
parser.add_argument('--clip', type=float, default=0.4)
parser.add_argument('--color_channels', type=int, default=1)
parser.add_argument('--add_loss', action='store_true')
```

# train:
```bash
bash sr_train.sh
```

# test:
```bash
bash sr_test.sh
```
