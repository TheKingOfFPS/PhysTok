#!/usr/bin/env python3
import argparse,json
from phystok.evaluation.audits import paired_predictions
p=argparse.ArgumentParser(description='Compare complete, ordered local prediction supports; never drop unpaired rows.')
p.add_argument('left');p.add_argument('right');a=p.parse_args()
print(json.dumps(paired_predictions(a.left,a.right),indent=2))
