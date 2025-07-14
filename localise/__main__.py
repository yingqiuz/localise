#!/usr/bin/env python
from localise.modes import predict_mode, train_mode
from localise.args import parse_arguments

def main():
    args = parse_arguments()
    if args.predict:
        predict_mode(args)
    elif args.train:
        train_mode(args)
    
if __name__ == "__main__":
    main()
